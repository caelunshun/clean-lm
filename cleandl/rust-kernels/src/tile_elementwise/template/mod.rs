//! WIP
//!
//! Template for building tile_elementwise kernels.
//!
//! The kernels are built around a warp-specialized design:
//! - Warps 0..=3 - Worker warps performing computations.
//! - Warp 4: TMA load warp.
//! - Warp 5: TMA store warp.
//! - Warp 6: cluster synchronization warp.
//! - Warp 7: CLC scheduler warp.
//!
//! The template assumes we are operating on TM (rows) x TN (columns) tiles of some iteration space.
//! Data within a tile is evenly partitioned among worker threads over the N dimension (each thread holds
//! all rows for its assigned columns). Out-of-bounds entries are set to 0.
//!
//! When a cluster size >1 is used,
//! the CTAs in a cluster are arranged along the M dimension, i.e. they hold different rows of the same
//! columns. TM remains the per-CTA tile size.
//! Currently, the only supported use case for a cluster is enabling columnwise MXFP8 quantization when
//! TM < 32.
//!
//! The consumer must specify three components of the kernel:
//! 1. Prologue, during which we may load data
//! 2. Tile processor, which specifies the sequence of loads and stores
//!    done per tile, plus the arithmetic code connecting them
//! 3. Epilogue, during which we may store data
//!
//! Loaded data can be bf16 or f32. Output data can be f32,
//! bf16, or MXFP8 (rowwise or columnwise).
//!
//! Loaded and stored values can either be M x N tiles, M x 1 column slices,
//! or 1 x N row slices. The latter two only supported bf16 and f32.
//! Loads and stores currently only support row-major layouts (except MXFP8 colwise store, which only supports col-major layout).
//!
//! bf16 and f32 stores can be reductions.
//!
//! The tile processor can perform intra-CTA reductions through shared memory.
//! Currently, only per-row reductions are supported in this way. (Per-column is easily implemented
//! within a lane's arithmetic code.)
//!
//! The template uses cluster launch control (CLC) to implement a persistent, dynamically load-balanced
//! kernel. Each logical CTA processes a "macroblock" of tiles. Larger macroblock sizes allow for lower scheduling overhead
//! but higher risk of wave quantization effects.
//!
//! Note that the partitioning of tile columns among worker threads is unspecified,
//! and may vary with tile size. Rowwise reductions should use KernelVisitor::rowwise_reduce.

use crate::{
    tile_elementwise::template::layouts::{CopyDir, Layouts},
    utils::clc::{ClcScheduler, ClcSmem},
};
use core::{cell::UnsafeCell, mem::MaybeUninit};
use ptx_std::{
    Dim3, MBarrier, PhaseParity, Scope, bf16, dynamic_smem_size, dynamic_smem_start,
    fence::{ProxyKind, fence_proxy},
    launch::{decrease_max_registers, increase_max_registers},
    memory::CachePolicy,
    thread_id,
    tma::{
        self, BulkWait, TmaDescriptor, TmaReduceOp, tma_load_tile, tma_reduce_tile, tma_store_tile,
    },
    warp_collectives::{LaneMask, elect, shuffle, shuffle_xor},
    warp_id,
};

mod layouts;

/// Specifies kernel description used to instantiate the kernel template.
///
/// Soundness: every thread must do the same calls on the KernelVisitor in the same order.
pub unsafe trait TileElementwiseKernel: Sized {
    type Params: Params;
    type WorkerState;

    /// Maps logical CTA ID (1D) and index within that macroblock
    /// to the Dim3 coord passed to Self::process_tile.
    fn map_coord(&self, macroblock: u32, idx_in_macroblock: u32) -> Dim3;

    fn init_worker_state(&self) -> Self::WorkerState;

    fn prologue(&mut self, v: impl KernelVisitor<Self, Self::Params>);

    fn process_tile(&mut self, coord: Dim3, v: impl KernelVisitor<Self, Self::Params>);

    fn epilogue(&mut self, v: impl KernelVisitor<Self, Self::Params>);
}

pub trait KernelVisitor<K: TileElementwiseKernel<Params = P>, P: Params> {
    /// Loads the specified tile. If this is a worker thread, invokes
    /// `load_callback` with this thread's results for the load.
    fn load<M>(
        &mut self,
        desc: LoadDescriptor<M>,
        load_callback: impl FnOnce(&mut K::WorkerState, M::LaneVals),
    ) where
        M: LoadStoreMode<P>;

    /// Performs intra-tile rowwise reduction using the specified reduction operator.
    /// If this is a worker thread, invokes `get_vals` to extract the values to reduce
    /// from the worker state, then calls `reduce_callback` with the output of the reduction.
    ///
    /// The reduction is performed using f32 precision. The reduction operator is assumed
    /// to be associative and commutative.
    fn rowwise_reduce(
        &mut self,
        reduction: impl Fn(f32, f32) -> f32,
        get_vals: impl FnOnce(&mut K::WorkerState) -> [f32; <K::Params as Params>::THREAD_M],
        reduce_callback: impl FnOnce(&mut K::WorkerState, [f32; <K::Params as Params>::THREAD_M]),
    );

    /// Stores the specified tile. If this is a worker thread, invokes
    /// `get_vals` to extract the values to store from the worker state.
    fn store<M, T>(
        &mut self,
        desc: StoreDescriptor<M>,
        get_vals: impl FnOnce(&mut K::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>;
}

/// Required swizzle mode:
/// Mx1 ops : no swizzle
/// MxN and 1xN ops :
/// - bf16: no swizzle
/// - f32: 32B swizzle
pub struct LoadDescriptor<'a, M> {
    pub mode: M,

    pub tma_desc: &'a TmaDescriptor,
    /// Note that for 1xN and Mx1 load modes,
    /// the "1" dimension's coordinate is ignored.
    pub coord_mn: [u32; 2],
    pub cache_policy: Option<CachePolicy>,
}

pub struct StoreDescriptor<'a, M> {
    pub mode: M,
    pub ty: StoreTy,
    pub tma_desc: &'a TmaDescriptor,
    /// Note that for 1xN and Mx1 store modes,
    /// the "1" dimension's coordinate is ignored.
    pub coord_mn: [u32; 2],
    pub reduction: Option<TmaReduceOp>,
    pub cache_policy: Option<CachePolicy>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoadTy {
    F32,
    Bf16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StoreTy {
    F32,
    Bf16,
    Mxfp8Rowwise,
    Mxfp8Colwise,
}

pub trait LoadStoreMode<P: Params> {
    type LaneVals;
    const NUM_TMA_COORDS: usize;

    fn num_bytes(&self) -> usize;
    fn gen_tma_coords(&self, coord_mn: [u32; 2]) -> [i32; Self::NUM_TMA_COORDS];

    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals;
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals;

    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8);
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8);
}

pub struct ModeMxN;

impl<P: Params> LoadStoreMode<P> for ModeMxN {
    type LaneVals = [[f32; P::THREAD_N]; P::THREAD_M];
    const NUM_TMA_COORDS: usize = 2;

    #[inline(always)]
    fn num_bytes(&self) -> usize {
        P::LOAD_SLOT_BYTES
    }

    #[inline(always)]
    fn gen_tma_coords(&self, coord_mn: [u32; 2]) -> [i32; Self::NUM_TMA_COORDS] {
        [coord_mn[1] as i32, coord_mn[0] as i32]
            .as_slice()
            .try_into()
            .unwrap()
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [[0.0; P::THREAD_N]; P::THREAD_M];
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                Layouts::<P>::new().copy_bf16_row(
                    &mut thr_vals[row] as *mut f32,
                    smem.cast::<bf16>().add(row * P::TN).cast_mut(),
                    thread_id().x,
                    CopyDir::SmemToReg,
                );
            }
            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [[0.0; P::THREAD_N]; P::THREAD_M];
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                Layouts::<P>::new().copy_f32_row(
                    &mut thr_vals[row] as *mut f32,
                    smem.cast::<f32>().add(row * P::TN).cast_mut(),
                    thread_id().x,
                    CopyDir::SmemToReg,
                );
            }
            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                Layouts::<P>::new().copy_bf16_row(
                    &thr_vals[row] as *const f32 as *mut f32,
                    smem.cast::<bf16>().add(row * P::TN),
                    thread_id().x,
                    CopyDir::RegToSmem,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                Layouts::<P>::new().copy_f32_row(
                    &thr_vals[row] as *const f32 as *mut f32,
                    smem.cast::<f32>().add(row * P::TN),
                    thread_id().x,
                    CopyDir::RegToSmem,
                );
            }
        }
    }
}

pub struct Mode1xN;

impl<P: Params> LoadStoreMode<P> for Mode1xN {
    type LaneVals = [f32; P::THREAD_N];
    const NUM_TMA_COORDS: usize = 1;

    #[inline(always)]
    fn num_bytes(&self) -> usize {
        size_of::<bf16>() * P::TN
    }

    #[inline(always)]
    fn gen_tma_coords(&self, coord_mn: [u32; 2]) -> [i32; Self::NUM_TMA_COORDS] {
        [coord_mn[1] as i32].as_slice().try_into().unwrap()
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [0.0; P::THREAD_N];

            Layouts::<P>::new().copy_bf16_row(
                &mut thr_vals as *mut f32,
                smem.cast::<bf16>().cast_mut(),
                thread_id().x,
                CopyDir::SmemToReg,
            );

            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [0.0; P::THREAD_N];

            Layouts::<P>::new().copy_f32_row(
                &mut thr_vals as *mut f32,
                smem.cast::<f32>().cast_mut(),
                thread_id().x,
                CopyDir::SmemToReg,
            );

            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            Layouts::<P>::new().copy_bf16_row(
                thr_vals as *const f32 as *mut f32,
                smem.cast::<bf16>(),
                thread_id().x,
                CopyDir::RegToSmem,
            );
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            Layouts::<P>::new().copy_f32_row(
                thr_vals as *const f32 as *mut f32,
                smem.cast::<f32>(),
                thread_id().x,
                CopyDir::RegToSmem,
            );
        }
    }
}

pub struct ModeMx1;

impl<P: Params> LoadStoreMode<P> for ModeMx1 {
    type LaneVals = [f32; P::THREAD_M];
    const NUM_TMA_COORDS: usize = 1;

    #[inline(always)]
    fn num_bytes(&self) -> usize {
        size_of::<bf16>() * P::TM
    }

    #[inline(always)]
    fn gen_tma_coords(&self, coord_mn: [u32; 2]) -> [i32; Self::NUM_TMA_COORDS] {
        [coord_mn[0] as i32].as_slice().try_into().unwrap()
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, _smem: *const u8) -> Self::LaneVals {
        todo!()
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, _smem: *const u8) -> Self::LaneVals {
        todo!()
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_bf16(&self, _thr_vals: &Self::LaneVals, _smem: *mut u8) {
        todo!()
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_f32(&self, _thr_vals: &Self::LaneVals, _smem: *mut u8) {
        todo!()
    }
}

pub const NUM_WORKER_WARPS: usize = 4;
pub const NUM_WORKER_THREADS: usize = 32 * NUM_WORKER_WARPS;

pub trait Params {
    /// Supports powers of two up to 8192 and at least 128.
    const TM: usize;
    /// Supports powers of two up to 64.
    const TN: usize;
    const CLUSTER_SIZE: usize;
    const MACROBLOCK_SIZE: usize;

    /// A load slot holds M x N bf16 elements. 1 x N or M x 1 loads
    /// still consume a full slot.
    const NUM_LOAD_SLOTS: usize;

    const LARGEST_STORE_BITS_PER_ELEMENT: usize;
    const DOES_STORE_ROWWISE_BLOCK_SCALES: bool;
    const DOES_STORE_COLWISE_BLOCK_SCALES: bool;

    // --- Derived constants --- //

    const TILE_ELEMENTS: usize = Self::TM * Self::TN;
    const TILE_ELEMENTS_DIV_8: usize = Self::TILE_ELEMENTS / 8;
    const TILE_ELEMENTS_DIV_4: usize = Self::TILE_ELEMENTS / 4;

    const THREAD_N: usize = Self::TN.div_ceil(NUM_WORKER_THREADS);
    const THREAD_N_DIV_8: usize = Self::THREAD_N / 8;
    const THREAD_M: usize = Self::TM;

    const LOAD_SLOT_BYTES: usize = (size_of::<bf16>() * Self::TILE_ELEMENTS).next_multiple_of(256);
    const STORE_SLOT_BYTES: usize =
        (Self::LARGEST_STORE_BITS_PER_ELEMENT * Self::TILE_ELEMENTS / 8).next_multiple_of(256);

    const CLUSTER_ENABLED: bool = Self::CLUSTER_SIZE > 1;
    const CLUSTER_SCOPE: Scope = const {
        if Self::CLUSTER_ENABLED {
            Scope::Cluster
        } else {
            Scope::Cta
        }
    };
}

#[repr(C)]
struct Barriers<P: Params> {
    load_slot_produced: [MBarrier; P::NUM_LOAD_SLOTS],
    load_slot_consumed: [MBarrier; P::NUM_LOAD_SLOTS],

    store_slot_produced: MBarrier,
    store_slot_consumed: MBarrier,

    rowwise_reduction_produced: MBarrier,
    rowwise_reduction_consumed: MBarrier,
}

#[repr(C, align(1024))]
struct Smem<P: Params> {
    load_slots: [[UnsafeCell<MaybeUninit<u8>>; P::LOAD_SLOT_BYTES]; P::NUM_LOAD_SLOTS],
    store_slot: [UnsafeCell<MaybeUninit<u8>>; P::STORE_SLOT_BYTES],
    rowwise_reduction_scratchpad: [[UnsafeCell<MaybeUninit<f32>>; P::TM]; NUM_WORKER_WARPS],
    bars: Barriers<P>,
    clc: UnsafeCell<ClcSmem>,
}

// --- TileScheduler --- //

struct TileScheduler<K: TileElementwiseKernel> {
    clc: ClcScheduler<{ <K::Params as Params>::CLUSTER_ENABLED }>,
    current_macroblock: Option<u32>,
    current_idx_in_macroblock: u32,
}

impl<K: TileElementwiseKernel> TileScheduler<K> {
    #[inline(always)]
    pub unsafe fn next_coord(&mut self, kernel: &K) -> Option<Dim3> {
        unsafe {
            let mb = match self.current_macroblock {
                Some(mb) => {
                    if self.current_idx_in_macroblock
                        == <K::Params as Params>::MACROBLOCK_SIZE as u32
                    {
                        let next = self.clc.get_next_block()?;
                        self.current_idx_in_macroblock = 0;
                        self.current_macroblock = Some(next.x);
                        next.x
                    } else {
                        mb
                    }
                }
                None => {
                    self.current_macroblock = self.clc.get_next_block().map(|d| d.x);
                    self.current_macroblock?
                }
            };

            let coord = kernel.map_coord(mb, self.current_idx_in_macroblock);
            self.current_idx_in_macroblock += 1;
            Some(coord)
        }
    }
}

// --- TmaLoadWarp --- //

struct TmaLoadWarp<'a, K: TileElementwiseKernel, P: Params> {
    smem: &'a Smem<P>,
    sched: TileScheduler<K>,
    load_idx: u32,
}

impl<'a, K: TileElementwiseKernel<Params = P>, P: Params> TmaLoadWarp<'a, K, P> {
    #[inline(always)]
    pub fn run(&mut self, kernel: &mut K) {
        unsafe {
            kernel.prologue(&mut *self);
            while let Some(coord) = self.sched.next_coord(kernel) {
                kernel.process_tile(coord, &mut *self);
            }
            kernel.epilogue(&mut *self);
        }
    }
}

impl<'a, 'tmp, K: TileElementwiseKernel<Params = P>, P: Params> KernelVisitor<K, P>
    for &'tmp mut TmaLoadWarp<'a, K, P>
{
    #[inline(always)]
    fn load<M>(
        &mut self,
        desc: LoadDescriptor<M>,
        _load_callback: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState, M::LaneVals),
    ) where
        M: LoadStoreMode<P>,
    {
        unsafe {
            let num_load_slots = <K::Params as Params>::NUM_LOAD_SLOTS;

            let slot_idx = self.load_idx as usize % num_load_slots;

            if self.load_idx >= num_load_slots as u32 {
                self.smem.bars.load_slot_consumed[slot_idx]
                    .wait(PhaseParity::of_phase(self.load_idx / num_load_slots as u32));
            }

            if elect(LaneMask::FULL).is_leader {
                self.smem.bars.load_slot_produced[slot_idx]
                    .arrive_expect_tx(desc.mode.num_bytes() as u32);
                tma_load_tile(
                    self.smem.load_slots[slot_idx]
                        .as_ptr()
                        .cast::<u8>()
                        .cast_mut(),
                    &self.smem.bars.load_slot_produced[slot_idx],
                    desc.tma_desc,
                    desc.mode.gen_tma_coords(desc.coord_mn),
                    desc.cache_policy,
                );
            }

            self.load_idx += 1;
        }
    }

    #[inline(always)]
    fn rowwise_reduce(
        &mut self,
        _reduction: impl Fn(f32, f32) -> f32,
        _get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        ) -> [f32; <K::Params as Params>::THREAD_M],
        _reduce_callback: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
            [f32; <K::Params as Params>::THREAD_M],
        ),
    ) {
        // No-op
    }

    #[inline(always)]
    fn store<M, T>(
        &mut self,
        _desc: StoreDescriptor<M>,
        _get_vals: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>,
    {
        // No-op
    }
}

// --- TmaStoreWarp --- //

struct TmaStoreWarp<'a, K: TileElementwiseKernel, P: Params> {
    smem: &'a Smem<P>,
    sched: TileScheduler<K>,
    store_idx: u32,
}

impl<'a, K: TileElementwiseKernel<Params = P>, P: Params> TmaStoreWarp<'a, K, P> {
    #[inline(always)]
    pub fn run(&mut self, kernel: &mut K) {
        unsafe {
            kernel.prologue(&mut *self);
            while let Some(coord) = self.sched.next_coord(kernel) {
                kernel.process_tile(coord, &mut *self);
            }
            kernel.epilogue(&mut *self);
        }
    }
}

impl<'a, 'tmp, K: TileElementwiseKernel<Params = P>, P: Params> KernelVisitor<K, P>
    for &'tmp mut TmaStoreWarp<'a, K, P>
{
    #[inline(always)]
    fn load<M>(
        &mut self,
        _desc: LoadDescriptor<M>,
        _load_callback: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState, M::LaneVals),
    ) where
        M: LoadStoreMode<P>,
    {
        // No-op
    }

    #[inline(always)]
    fn rowwise_reduce(
        &mut self,
        _reduction: impl Fn(f32, f32) -> f32,
        _get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        ) -> [f32; <K::Params as Params>::THREAD_M],
        _reduce_callback: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
            [f32; <K::Params as Params>::THREAD_M],
        ),
    ) {
        // No-op
    }

    #[inline(always)]
    fn store<M, T>(
        &mut self,
        desc: StoreDescriptor<M>,
        _get_vals: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>,
    {
        unsafe {
            self.smem
                .bars
                .store_slot_produced
                .wait(PhaseParity::of_phase(self.store_idx));

            if elect(LaneMask::FULL).is_leader {
                match desc.reduction {
                    None => {
                        tma_store_tile(
                            self.smem.store_slot.as_ptr().cast(),
                            desc.tma_desc,
                            desc.mode.gen_tma_coords(desc.coord_mn),
                            desc.cache_policy,
                        );
                    }
                    Some(TmaReduceOp::Add) => {
                        tma_reduce_tile::<_, { TmaReduceOp::Add }>(
                            self.smem.store_slot.as_ptr().cast(),
                            desc.tma_desc,
                            desc.mode.gen_tma_coords(desc.coord_mn),
                            desc.cache_policy,
                        );
                    }
                    _ => unimplemented!(),
                }

                tma::bulk_commit_group();
                tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                self.smem.bars.store_slot_consumed.arrive();
            }

            self.store_idx += 1;
        }
    }
}

// --- WorkerWarp --- //

struct WorkerWarp<'a, K: TileElementwiseKernel, P: Params> {
    smem: &'a Smem<P>,
    sched: TileScheduler<K>,
    state: K::WorkerState,
    load_idx: u32,
    store_idx: u32,
    reduction_idx: u32,
}

impl<'a, K: TileElementwiseKernel<Params = P>, P: Params> WorkerWarp<'a, K, P> {
    #[inline(always)]
    pub fn run(&mut self, kernel: &mut K) {
        unsafe {
            kernel.prologue(&mut *self);
            while let Some(coord) = self.sched.next_coord(kernel) {
                kernel.process_tile(coord, &mut *self);
            }
            kernel.epilogue(&mut *self);
        }
    }
}

impl<'a, 'tmp, K: TileElementwiseKernel<Params = P>, P: Params> KernelVisitor<K, P>
    for &'tmp mut WorkerWarp<'a, K, P>
{
    #[inline(always)]
    fn load<M>(
        &mut self,
        desc: LoadDescriptor<M>,
        load_callback: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState, M::LaneVals),
    ) where
        M: LoadStoreMode<P>,
    {
        let num_load_slots = P::NUM_LOAD_SLOTS;
        let slot_idx = self.load_idx as usize % num_load_slots;
        let parity = PhaseParity::of_phase(self.load_idx / num_load_slots as u32);

        unsafe {
            self.smem.bars.load_slot_produced[slot_idx].wait(parity);

            let vals = desc
                .mode
                .worker_load_from_smem_bf16(self.smem.load_slots[slot_idx].as_ptr().cast());
            self.smem.bars.load_slot_consumed[slot_idx].arrive();

            load_callback(&mut self.state, vals);

            self.load_idx += 1;
        }
    }

    #[inline(always)]
    fn rowwise_reduce(
        &mut self,
        reduction: impl Fn(f32, f32) -> f32,
        get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        ) -> [f32; <K::Params as Params>::THREAD_M],
        reduce_callback: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
            [f32; <K::Params as Params>::THREAD_M],
        ),
    ) {
        unsafe {
            if self.reduction_idx > 0 {
                self.smem
                    .bars
                    .rowwise_reduction_consumed
                    .wait(PhaseParity::of_phase(self.reduction_idx).flip());
            }

            let mut thr_vals = get_vals(&mut self.state);

            // Intra-warp reduction
            #[rustc_unroll(full)]
            for v in thr_vals.iter_mut() {
                #[rustc_unroll(full)]
                for bit in 0..5 {
                    *v = reduction(*v, shuffle_xor(LaneMask::FULL, *v, 1 << bit));
                }
            }

            // Intra-CTA reduction through SMEM
            if elect(LaneMask::FULL).is_leader {
                self.smem.rowwise_reduction_scratchpad[warp_id() as usize]
                    .as_ptr()
                    .cast::<[f32; <K::Params as Params>::THREAD_M]>()
                    .cast_mut()
                    .write(thr_vals);
            }

            self.smem.bars.rowwise_reduction_produced.arrive();
            self.smem
                .bars
                .rowwise_reduction_produced
                .wait(PhaseParity::of_phase(self.reduction_idx));

            #[rustc_unroll(full)]
            for i in 0..NUM_WORKER_WARPS {
                if i != warp_id() as usize {
                    #[rustc_unroll(full)]
                    for j in 0..<K::Params as Params>::THREAD_M {
                        thr_vals[j] = reduction(
                            thr_vals[j],
                            self.smem.rowwise_reduction_scratchpad[i][j]
                                .get()
                                .read()
                                .assume_init(),
                        );
                    }
                }
            }

            self.smem.bars.rowwise_reduction_consumed.arrive();

            reduce_callback(&mut self.state, thr_vals);

            self.reduction_idx += 1;
        }

        todo!()
    }

    #[inline(always)]
    fn store<M, T>(
        &mut self,
        desc: StoreDescriptor<M>,
        get_vals: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>,
    {
        unsafe {
            if self.store_idx > 0 {
                self.smem
                    .bars
                    .store_slot_consumed
                    .wait(PhaseParity::of_phase(self.store_idx).flip());
            }

            let vals = get_vals(&mut self.state);

            match desc.ty {
                StoreTy::F32 => desc.mode.worker_store_to_smem_f32(
                    &vals,
                    self.smem.store_slot.as_ptr().cast::<u8>().cast_mut(),
                ),
                StoreTy::Bf16 => desc.mode.worker_store_to_smem_bf16(
                    &vals,
                    self.smem.store_slot.as_ptr().cast::<u8>().cast_mut(),
                ),
                StoreTy::Mxfp8Rowwise => todo!(),
                StoreTy::Mxfp8Colwise => todo!(),
            }

            fence_proxy::<{ ProxyKind::AsyncSharedCta }>();
            self.smem.bars.store_slot_produced.arrive();

            self.store_idx += 1;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpRole {
    Worker,
    TmaLoad,
    TmaStore,
    ClusterSync,
    Clc,
}

impl WarpRole {
    pub fn get() -> Self {
        // Marks warp index as uniform for ptxas.
        let warp_idx = unsafe { shuffle(LaneMask::FULL, warp_id(), 0) };
        match warp_idx {
            0..4 => WarpRole::Worker,
            4 => WarpRole::TmaLoad,
            5 => WarpRole::TmaStore,
            6 => WarpRole::ClusterSync,
            7 => WarpRole::Clc,
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn run<K: TileElementwiseKernel<Params = P>, P: Params>(mut kernel: K) {
    unsafe {
        assert_eq!(dynamic_smem_size() as usize, size_of::<Smem<P>>());
        let smem = &*dynamic_smem_start().cast::<Smem<P>>();
        let clc = ClcScheduler::<{ P::CLUSTER_ENABLED }>::new(smem.clc.get());

        let mut sched = TileScheduler::<K> {
            clc,
            current_macroblock: None,
            current_idx_in_macroblock: 0,
        };

        let role = WarpRole::get();
        match role {
            WarpRole::Worker => {
                increase_max_registers::<224>();
            }
            _ => {
                decrease_max_registers::<32>();
                match role {
                    WarpRole::Worker => unreachable!(),
                    WarpRole::TmaLoad => {
                        TmaLoadWarp {
                            smem,
                            sched,
                            load_idx: 0,
                        }
                        .run(&mut kernel);
                    }
                    WarpRole::TmaStore => {
                        TmaStoreWarp {
                            smem,
                            sched,
                            store_idx: 0,
                        }
                        .run(&mut kernel);
                    }
                    WarpRole::ClusterSync => {
                        // TODO
                    }
                    WarpRole::Clc => {
                        sched.clc.run_scheduler();
                    }
                }
            }
        }
    }
}
