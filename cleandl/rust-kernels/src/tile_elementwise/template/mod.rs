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
//! A cluster size >1 allows one tile to span multiple CTAs. In this case, tiles and macroblocks
//! are cluster-wide, not per-CTA. Rowwise reductions will use cluster synchronization automatically.
//! Currently, columnwise reductions where M-cluster-size >1 are not supported. The only use for M-cluster-size>1
//! is to attain TM=32 to allow for columnwise MXFP8 output.
//!
//! The consumer must specify three components of the kernel:
//! 1. Prologue, during which we may load data
//! 2. Tile processor, which specifies the sequence of loads and stores
//!    done per tile, plus the arithmetic code connecting them
//! 3. Epilogue, during which we may store data
//!
//! Loaded data can be bf16 or f32. Output data can be f32,
//! bf16, or MXFP8 (rowwise or columnwise; the latter requires TM a multiple of 32).
//!
//! Loaded and stored values can either be M x N tiles, M x 1 column slices,
//! or 1 x N row slices. The latter two only support bf16 and f32.
//! Loads and stores currently only support row-major layouts (except MXFP8 colwise store, which only supports col-major layout).
//!
//! bf16 and f32 stores can be reductions.
//!
//! The tile processor can perform intra-CTA reductions through shared memory.
//! Currently, only per-row reductions are supported in this way. (Per-column is easily implemented
//! within a lane's arithmetic code if the M-cluster-size is 1.)
//!
//! The template uses cluster launch control (CLC) to implement a persistent, dynamically load-balanced
//! kernel. Each logical CGA processes a "macroblock" of tiles. Larger macroblock sizes reduce scheduling overhead
//! but increase risk of wave quantization effects.
//!
//! The grid must be launched with dimensions `(num_macroblocks * CLUSTER_SIZE_M, CLUSTER_SIZE_N, 1)`
//! and cluster dimensions `(CLUSTER_SIZE_M, CLUSTER_SIZE_N, 1)`. The macroblock index of a CGA
//! is then `block_id().x / CLUSTER_SIZE_M`.
//!
//! Note that the partitioning of tile columns among worker threads is unspecified,
//! and may vary with tile size. Rowwise reductions should use KernelVisitor::rowwise_reduce.

use crate::{
    tile_elementwise::template::layouts::{CopyDir, Layouts, iter_tma_panels},
    utils::{
        block_scale::{BlockScaledFormat, Mxfp8, SF_TILE_NCOLS, SF_TILE_NROWS, scale_factor_pos},
        clc::{ClcScheduler, ClcSmem},
        compiler_magic::{array_map_unrolled, array_zip_unrolled, uniform_warp_id},
        tma::{
            CompoundTmaDescriptor, GMEM_TMA_ALIGNMENT, SMEM_TMA_ALIGNMENT_NO_SWIZZLE,
            tma_store_tile_with_reduction,
        },
    },
};
use core::{cell::UnsafeCell, mem::MaybeUninit};
use ptx_std::{
    Denormals, Dim3, F32Ext, MBarrier, MemoryOrdering, NanHandling, PhaseParity, Scope, Vector,
    barrier::{BarrierAlignment, sync_threads},
    bf16, bf16x2, block_id_in_cluster, block_size,
    cluster::{self, map_to_cta, st_async},
    cluster_dim, cluster_size,
    cp_async::{self, CacheOperator},
    dynamic_smem_size, dynamic_smem_start, f32x2, f32x8, f32xN,
    fence::{ProxyKind, fence_proxy, fence_sync_restrict},
    grid_dim, lane_id,
    launch::{decrease_max_registers, increase_max_registers},
    memory::CachePolicy,
    thread_id,
    tma::{self, BulkWait, TmaReduceOp},
    warp_collectives::{LaneMask, elect, shuffle_xor},
    warp_id,
};

mod layouts;

pub const NUM_WORKER_WARPS: usize = 4;
pub const NUM_WORKER_THREADS: usize = 32 * NUM_WORKER_WARPS;

pub trait Params {
    /// Supports powers of two.
    ///
    /// Tile size across the full CGA.
    const CGA_TM: usize;
    /// Supports powers of two.
    ///
    /// Tile size across the full CGA.
    const CGA_TN: usize;
    const CLUSTER_SIZE_M: usize;
    const CLUSTER_SIZE_N: usize;
    const MACROBLOCK_SIZE: usize;

    /// A load slot is sized for the largest of the M x N, 1 x N, and M x 1 loads
    /// (see the LARGEST_*_LOAD_BITS_PER_ELEMENT constants). Every load consumes a full slot.
    const NUM_LOAD_SLOTS: usize;

    const LARGEST_MXN_LOAD_BITS_PER_ELEMENT: usize;
    const LARGEST_1XN_LOAD_BITS_PER_ELEMENT: usize;
    const LARGEST_MX1_LOAD_BITS_PER_ELEMENT: usize;

    const LARGEST_MXN_STORE_BITS_PER_NON_SF_ELEMENT: usize;
    const LARGEST_1XN_STORE_BITS_PER_NON_SF_ELEMENT: usize;
    const LARGEST_MX1_STORE_BITS_PER_NON_SF_ELEMENT: usize;

    const DOES_STORE_COLWISE_BLOCK_SCALED_DATA: bool;

    // --- Derived constants --- //

    const CLUSTER_SIZE: usize = Self::CLUSTER_SIZE_M * Self::CLUSTER_SIZE_N;

    const CTA_TM: usize = Self::CGA_TM / Self::CLUSTER_SIZE_M;
    const CTA_TN: usize = Self::CGA_TN / Self::CLUSTER_SIZE_N;

    /// Width of each TMA panel in SMEM, set by TMA box size constraints (at most 256).
    const TMA_PANEL_COLS: usize = if Self::CTA_TN < 256 {
        Self::CTA_TN
    } else {
        256
    };
    const NUM_TMA_PANELS: usize = Self::CTA_TN / Self::TMA_PANEL_COLS;

    const THREAD_N: usize = Self::CTA_TN.div_ceil(NUM_WORKER_THREADS);
    const THREAD_N_DIV_8: usize = Self::THREAD_N / 8;
    const THREAD_M: usize = Self::CTA_TM;

    const MXN_LOAD_SLOT_BYTES: usize = Self::LARGEST_MXN_LOAD_BITS_PER_ELEMENT
        * Self::CTA_TM
        * Self::TMA_PANEL_COLS
        * Self::NUM_TMA_PANELS
        / 8;
    const MX1_LOAD_SLOT_BYTES: usize = Self::LARGEST_MX1_LOAD_BITS_PER_ELEMENT * Self::CTA_TM / 8;
    const _1XN_LOAD_SLOT_BYTES: usize =
        Self::LARGEST_1XN_LOAD_BITS_PER_ELEMENT * Self::TMA_PANEL_COLS * Self::NUM_TMA_PANELS / 8;

    const MXN_STORE_SLOT_BYTES: usize = Self::LARGEST_MXN_STORE_BITS_PER_NON_SF_ELEMENT
        * Self::CTA_TM
        * Self::TMA_PANEL_COLS
        * Self::NUM_TMA_PANELS
        / 8;
    const MX1_STORE_SLOT_BYTES: usize =
        Self::LARGEST_MX1_STORE_BITS_PER_NON_SF_ELEMENT * Self::CTA_TM / 8;
    const _1XN_STORE_SLOT_BYTES: usize = Self::LARGEST_1XN_STORE_BITS_PER_NON_SF_ELEMENT
        * Self::TMA_PANEL_COLS
        * Self::NUM_TMA_PANELS
        / 8;

    const LOAD_SLOT_BYTES: usize = Self::MXN_LOAD_SLOT_BYTES
        .max(Self::MX1_LOAD_SLOT_BYTES)
        .max(Self::_1XN_LOAD_SLOT_BYTES)
        .next_multiple_of(SMEM_TMA_ALIGNMENT_NO_SWIZZLE);

    const STORE_SLOT_BYTES: usize = Self::MXN_STORE_SLOT_BYTES
        .max(Self::MX1_STORE_SLOT_BYTES)
        .max(Self::_1XN_STORE_SLOT_BYTES)
        .next_multiple_of(SMEM_TMA_ALIGNMENT_NO_SWIZZLE);

    const CLUSTER_ENABLED: bool = Self::CLUSTER_SIZE > 1;
    const CLUSTER_SCOPE: Scope = const {
        if Self::CLUSTER_ENABLED {
            Scope::Cluster
        } else {
            Scope::Cta
        }
    };

    /// Cluster-wide colwise reduction currently only used for writing colwise block-scaled
    /// data.
    const CLUSTER_COL_REDUCTION_SCRATCHPAD_BYTES: usize = const {
        if Self::DOES_STORE_COLWISE_BLOCK_SCALED_DATA && Self::CLUSTER_SIZE_M > 1 {
            size_of::<f32>() * Self::CTA_TN
        } else {
            0
        }
    };
}

/// Specifies kernel description used to instantiate the kernel template.
///
/// Soundness: every thread must do the same calls on the KernelVisitor in the same order.
pub unsafe trait TileElementwiseKernel: Sized {
    type Params: Params;
    type WorkerState;

    /// Maps logical CGA ID (1D) and index within that macroblock
    /// to the Dim3 coord passed to Self::process_tile.
    ///
    /// The CGA ID is `block_id().x / CLUSTER_SIZE_M` of the CGA's CTAs; see the module docs
    /// for the required grid layout.
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
    fn store<M>(
        &mut self,
        desc: StoreDescriptor<M>,
        get_vals: impl FnOnce(&mut K::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>;

    /// Stores an MxN tile with block scaling, either in rowwise or columnwise
    /// format.
    fn store_block_scaled<BS: BlockScaledFormat>(
        &mut self,
        desc: BlockScaledStoreDescriptor,
        get_vals: impl FnOnce(
            &mut K::WorkerState,
        ) -> [[f32; <K::Params as Params>::THREAD_N];
            <K::Params as Params>::THREAD_M],
    );
}

// --- Load/StoreDescriptor --- //

pub struct LoadDescriptor<'a, M> {
    pub mode: M,
    pub ty: LoadTy,
    pub tma_desc: &'a CompoundTmaDescriptor,
    /// Coordinates of the top-left of the tile.
    /// Note this must be the same across all threads
    /// in a CTA and all CTAs in a CGA.
    ///
    /// Note that for 1xN and Mx1 load modes,
    /// the "1" dimension's coordinate is ignored.
    pub coord_mn: [u32; 2],
    pub cache_policy: Option<CachePolicy>,
}

pub struct StoreDescriptor<'a, M> {
    pub mode: M,
    pub ty: StoreTy,
    pub tma_desc: &'a CompoundTmaDescriptor,
    /// Coordinates of the top-left of the tile.
    /// Note this must be the same across all threads
    /// in a CTA and all CTAs in a CGA.
    ///
    /// Note that for 1xN and Mx1 store modes,
    /// the "1" dimension's coordinate is ignored.
    pub coord_mn: [u32; 2],
    pub reduction: Option<TmaReduceOp>,
    pub cache_policy: Option<CachePolicy>,
}

pub struct BlockScaledStoreDescriptor<'a> {
    pub sf_dim: BlockScaleDim,
    pub tma_desc: &'a CompoundTmaDescriptor,
    pub sf_ptr: *mut u8,
    pub coord_mn: [u32; 2],
    pub cache_policy: Option<CachePolicy>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoadTy {
    F32,
    Bf16,
}

impl LoadTy {
    #[inline(always)]
    pub fn bits_per_element(self) -> usize {
        match self {
            LoadTy::F32 => 32,
            LoadTy::Bf16 => 16,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StoreTy {
    F32,
    Bf16,
}

impl StoreTy {
    #[inline(always)]
    pub fn bits_per_element(self) -> usize {
        match self {
            StoreTy::F32 => 32,
            StoreTy::Bf16 => 16,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlockScaledStoreTy {
    Mxfp8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlockScaleDim {
    Rowwise,
    Colwise,
}

// --- Load/StoreMode --- //

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LoadStoreModeEnum {
    ModeMxN,
    Mode1xN,
    ModeMx1,
}

pub trait LoadStoreMode<P: Params> {
    type LaneVals;

    fn mode_enum(&self) -> LoadStoreModeEnum;

    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals;
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals;

    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8);
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8);
}

pub struct ModeMxN;

impl<P: Params> LoadStoreMode<P> for ModeMxN {
    type LaneVals = [[f32; P::THREAD_N]; P::THREAD_M];

    #[inline(always)]
    fn mode_enum(&self) -> LoadStoreModeEnum {
        LoadStoreModeEnum::ModeMxN
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [[0.0; P::THREAD_N]; P::THREAD_M];
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                let mut raw = [bf16::ZERO; P::THREAD_N];
                Layouts::<P>::new().copy_row(
                    &mut raw as *mut bf16,
                    smem.cast::<bf16>().cast_mut(),
                    row,
                    thread_id().x,
                    P::CTA_TM,
                    CopyDir::SmemToReg,
                );
                thr_vals[row] = array_map_unrolled(raw, bf16::to_f32);
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
                Layouts::<P>::new().copy_row(
                    &mut thr_vals[row] as *mut f32,
                    smem.cast::<f32>().cast_mut(),
                    row,
                    thread_id().x,
                    P::CTA_TM,
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
                let raw = array_map_unrolled(thr_vals[row], bf16::from_f32);
                Layouts::<P>::new().copy_row(
                    &raw as *const bf16 as *mut bf16,
                    smem.cast::<bf16>(),
                    row,
                    thread_id().x,
                    P::CTA_TM,
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
                Layouts::<P>::new().copy_row(
                    &thr_vals[row] as *const f32 as *mut f32,
                    smem.cast::<f32>(),
                    row,
                    thread_id().x,
                    P::CTA_TM,
                    CopyDir::RegToSmem,
                );
            }
        }
    }
}

pub struct Mode1xN;

impl<P: Params> LoadStoreMode<P> for Mode1xN {
    type LaneVals = [f32; P::THREAD_N];

    #[inline(always)]
    fn mode_enum(&self) -> LoadStoreModeEnum {
        LoadStoreModeEnum::Mode1xN
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut raw = [bf16::ZERO; P::THREAD_N];

            Layouts::<P>::new().copy_row(
                &mut raw as *mut bf16,
                smem.cast::<bf16>().cast_mut(),
                0,
                thread_id().x,
                1,
                CopyDir::SmemToReg,
            );

            array_map_unrolled(raw, bf16::to_f32)
        }
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [0.0; P::THREAD_N];

            Layouts::<P>::new().copy_row(
                &mut thr_vals as *mut f32,
                smem.cast::<f32>().cast_mut(),
                0,
                thread_id().x,
                1,
                CopyDir::SmemToReg,
            );

            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            let raw = array_map_unrolled(*thr_vals, bf16::from_f32);
            Layouts::<P>::new().copy_row(
                &raw as *const bf16 as *mut bf16,
                smem.cast::<bf16>(),
                0,
                thread_id().x,
                1,
                CopyDir::RegToSmem,
            );
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            Layouts::<P>::new().copy_row(
                thr_vals as *const f32 as *mut f32,
                smem.cast::<f32>(),
                0,
                thread_id().x,
                1,
                CopyDir::RegToSmem,
            );
        }
    }
}

/// For stores, the values must be identical across all worker threads of all CTAs
/// sharing an M-coordinate (e.g. the output of `KernelVisitor::rowwise_reduce`),
/// as only one thread's values are stored.
pub struct ModeMx1;

impl<P: Params> LoadStoreMode<P> for ModeMx1 {
    type LaneVals = [f32; P::THREAD_M];

    #[inline(always)]
    fn mode_enum(&self) -> LoadStoreModeEnum {
        LoadStoreModeEnum::ModeMx1
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        let smem = smem.cast::<bf16>();
        let mut regs = [0.0; P::THREAD_M];

        unsafe {
            if P::THREAD_M.is_multiple_of(2) {
                // Vectorized conversions
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_M / 2 {
                    let slice = smem.cast::<bf16x2>().add(i).read().to_f32().to_array();
                    #[rustc_unroll(full)]
                    for j in 0..2 {
                        regs[i * 2 + j] = slice[j];
                    }
                }
            } else {
                // Unvectorized
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_M {
                    regs[i] = smem.add(i).read().to_f32();
                }
            }
        }

        regs
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals {
        unsafe { smem.cast::<Self::LaneVals>().read() }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_bf16(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                let smem = smem.cast::<bf16>();

                if P::THREAD_M.is_multiple_of(2) {
                    // Vectorized conversions
                    #[rustc_unroll(full)]
                    for i in 0..P::THREAD_M / 2 {
                        let slice = f32x2::new(thr_vals[i * 2], thr_vals[i * 2 + 1]);
                        smem.cast::<bf16x2>().add(i).write(bf16x2::from_f32(slice));
                    }
                } else {
                    // Unvectorized
                    #[rustc_unroll(full)]
                    for i in 0..P::THREAD_M {
                        smem.add(i).write(bf16::from_f32(thr_vals[i]));
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_smem_f32(&self, thr_vals: &Self::LaneVals, smem: *mut u8) {
        unsafe {
            if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                smem.cast::<Self::LaneVals>().write(*thr_vals);
            }
        }
    }
}

// --- Smem --- //

#[repr(C, align(1024))]
struct Smem<P: Params> {
    load_slots: [[UnsafeCell<MaybeUninit<u8>>; P::LOAD_SLOT_BYTES]; P::NUM_LOAD_SLOTS],
    store_slot: [UnsafeCell<MaybeUninit<u8>>; P::STORE_SLOT_BYTES],

    cluster_col_reduction_scratchpads:
        [[UnsafeCell<MaybeUninit<u8>>; P::CLUSTER_COL_REDUCTION_SCRATCHPAD_BYTES]; 2],

    /// These slots are double-buffered to reduce needed barrier waits.
    cta_reduction_scratchpads: [[[UnsafeCell<MaybeUninit<f32>>; P::CTA_TM]; NUM_WORKER_WARPS]; 2],
    cluster_row_reduction_scratchpads:
        [[[UnsafeCell<MaybeUninit<f32>>; P::CTA_TM]; P::CLUSTER_SIZE_N]; 2],

    bars: Barriers<P>,
    clc: UnsafeCell<ClcSmem>,
}

#[repr(C)]
struct Barriers<P: Params> {
    load_slot_produced: [MBarrier; P::NUM_LOAD_SLOTS],
    load_slot_consumed: [MBarrier; P::NUM_LOAD_SLOTS],

    store_slot_produced: MBarrier,
    store_slot_consumed: MBarrier,

    cta_reduction_produced: [MBarrier; 2],
    cluster_reduction_produced: [MBarrier; 2],
}

// --- TileScheduler --- //

struct TileScheduler<K: TileElementwiseKernel> {
    clc: ClcScheduler<{ <K::Params as Params>::CLUSTER_ENABLED }>,
    current_macroblock: Option<u32>,
    current_idx_in_macroblock: u32,
}

impl<K: TileElementwiseKernel> TileScheduler<K> {
    /// Maps a CTA block ID to the macroblock index of its CGA.
    #[inline(always)]
    fn block_to_macroblock(block: Dim3) -> u32 {
        block.x / <K::Params as Params>::CLUSTER_SIZE_M as u32
    }

    #[inline(always)]
    pub unsafe fn next_coord(&mut self, kernel: &K) -> Option<Dim3> {
        unsafe {
            let mb = match self.current_macroblock {
                Some(mb) => {
                    if self.current_idx_in_macroblock
                        == <K::Params as Params>::MACROBLOCK_SIZE as u32
                    {
                        let next = Self::block_to_macroblock(self.clc.get_next_block()?);
                        self.current_idx_in_macroblock = 0;
                        self.current_macroblock = Some(next);
                        next
                    } else {
                        mb
                    }
                }
                None => {
                    self.current_macroblock =
                        self.clc.get_next_block().map(Self::block_to_macroblock);
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
                    .wait(PhaseParity::of_phase(self.load_idx / num_load_slots as u32).flip());
            }

            let cta_offset_mn = [
                block_id_in_cluster().x as usize * P::CTA_TM,
                block_id_in_cluster().y as usize * P::CTA_TN,
            ];

            if elect(LaneMask::FULL).is_leader {
                match desc.mode.mode_enum() {
                    LoadStoreModeEnum::ModeMxN => {
                        assert!(P::LARGEST_MXN_LOAD_BITS_PER_ELEMENT >= desc.ty.bits_per_element());

                        let num_bytes = desc.ty.bits_per_element()
                            * P::CTA_TM
                            * P::TMA_PANEL_COLS
                            * P::NUM_TMA_PANELS
                            / 8;
                        self.smem.bars.load_slot_produced[slot_idx]
                            .arrive_expect_tx(num_bytes as u32);

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in
                            iter_tma_panels::<P>(P::CTA_TM)
                        {
                            let dst = self.smem.load_slots[slot_idx]
                                .as_ptr()
                                .cast::<u8>()
                                .cast_mut()
                                .add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma::tma_load_tile(
                                dst,
                                &self.smem.bars.load_slot_produced[slot_idx],
                                &desc.tma_desc.tma,
                                [
                                    (desc.coord_mn[1] as usize
                                        + panel_coord_offset
                                        + cta_offset_mn[1])
                                        as i32,
                                    (desc.coord_mn[0] as usize + cta_offset_mn[0]) as i32,
                                ],
                                desc.cache_policy,
                            );
                        }
                    }
                    LoadStoreModeEnum::Mode1xN => {
                        assert!(P::LARGEST_1XN_LOAD_BITS_PER_ELEMENT >= desc.ty.bits_per_element());

                        let num_bytes =
                            desc.ty.bits_per_element() * P::TMA_PANEL_COLS * P::NUM_TMA_PANELS / 8;
                        self.smem.bars.load_slot_produced[slot_idx]
                            .arrive_expect_tx(num_bytes as u32);

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in iter_tma_panels::<P>(1) {
                            let dst = self.smem.load_slots[slot_idx]
                                .as_ptr()
                                .cast::<u8>()
                                .cast_mut()
                                .add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma::tma_load_tile(
                                dst,
                                &self.smem.bars.load_slot_produced[slot_idx],
                                &desc.tma_desc.tma,
                                [
                                    (desc.coord_mn[1] as usize
                                        + panel_coord_offset
                                        + cta_offset_mn[1])
                                        as i32,
                                ],
                                desc.cache_policy,
                            );
                        }
                    }
                    LoadStoreModeEnum::ModeMx1 => {
                        assert!(P::LARGEST_MX1_LOAD_BITS_PER_ELEMENT >= desc.ty.bits_per_element());

                        let num_bytes = P::CTA_TM * desc.ty.bits_per_element() / 8;

                        if num_bytes.is_multiple_of(GMEM_TMA_ALIGNMENT) {
                            // (1) Access is TMA-aligned => use TMA directly.
                            self.smem.bars.load_slot_produced[slot_idx]
                                .arrive_expect_tx(num_bytes as u32);
                            tma::tma_load_tile(
                                self.smem.load_slots[slot_idx]
                                    .as_ptr()
                                    .cast::<u8>()
                                    .cast_mut(),
                                &self.smem.bars.load_slot_produced[slot_idx],
                                &desc.tma_desc.tma,
                                [(desc.coord_mn[0] as usize + cta_offset_mn[0]) as i32],
                                desc.cache_policy,
                            );
                        } else if num_bytes.is_multiple_of(4) {
                            // (2) Aligned to 4 bytes => use cp.async/LDGSTS.
                            let coord = desc.coord_mn[0] as usize + cta_offset_mn[0];
                            let tile_byte_offset = coord * desc.ty.bits_per_element() / 8;
                            let global_bytes = desc.tma_desc.global_size[0] as usize
                                * desc.ty.bits_per_element()
                                / 8;

                            #[rustc_unroll(full)]
                            for i in 0..num_bytes / 4 {
                                let dst = self.smem.load_slots[slot_idx]
                                    .as_ptr()
                                    .cast::<u32>()
                                    .cast_mut()
                                    .add(i);

                                let byte_offset = tile_byte_offset + i * 4;

                                // Technically this is UB when out-of-bounds, but copy_zfill
                                // needs a pointer even when out-of-bounds...
                                let src = desc.tma_desc.global_address.add(byte_offset);
                                let src_bytes = global_bytes.saturating_sub(byte_offset).min(4);

                                cp_async::copy_zfill::<u32, { CacheOperator::AllLevels }>(
                                    dst,
                                    src.cast(),
                                    src_bytes as u32,
                                );
                            }

                            self.smem.bars.load_slot_produced[slot_idx].cp_async_arrive_no_inc();
                        } else {
                            // <4 byte alignment is unsupported for now. Could technically be supported
                            // with some bitshifting tricks on the worker warps.
                            unimplemented!()
                        }
                    }
                }
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
    fn store<M>(
        &mut self,
        _desc: StoreDescriptor<M>,
        _get_vals: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>,
    {
        // No-op
    }

    #[inline(always)]
    fn store_block_scaled<BS: BlockScaledFormat>(
        &mut self,
        _desc: BlockScaledStoreDescriptor,
        _get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        )
            -> [[f32; <<K as TileElementwiseKernel>::Params as Params>::THREAD_N];
            <<K as TileElementwiseKernel>::Params as Params>::THREAD_M],
    ) {
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
    fn store<M>(
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
                let cta_offset_mn = [
                    block_id_in_cluster().x as usize * P::CTA_TM,
                    block_id_in_cluster().y as usize * P::CTA_TN,
                ];

                match desc.mode.mode_enum() {
                    LoadStoreModeEnum::ModeMxN => {
                        assert!(
                            P::LARGEST_MXN_STORE_BITS_PER_NON_SF_ELEMENT
                                >= desc.ty.bits_per_element()
                        );
                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in
                            iter_tma_panels::<P>(P::CTA_TM)
                        {
                            let src = self
                                .smem
                                .store_slot
                                .as_ptr()
                                .cast::<u8>()
                                .add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma_store_tile_with_reduction(
                                src,
                                &desc.tma_desc.tma,
                                [
                                    (desc.coord_mn[1] as usize
                                        + panel_coord_offset
                                        + cta_offset_mn[1])
                                        as i32,
                                    (desc.coord_mn[0] as usize + cta_offset_mn[0]) as i32,
                                ],
                                desc.cache_policy,
                                desc.reduction,
                            );
                        }

                        tma::bulk_commit_group();
                        tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                    }
                    LoadStoreModeEnum::Mode1xN => {
                        assert!(
                            P::LARGEST_1XN_STORE_BITS_PER_NON_SF_ELEMENT
                                >= desc.ty.bits_per_element()
                        );

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in iter_tma_panels::<P>(1) {
                            let src = self
                                .smem
                                .store_slot
                                .as_ptr()
                                .cast::<u8>()
                                .add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma_store_tile_with_reduction(
                                src,
                                &desc.tma_desc.tma,
                                [
                                    (desc.coord_mn[1] as usize
                                        + panel_coord_offset
                                        + cta_offset_mn[1])
                                        as i32,
                                ],
                                desc.cache_policy,
                                desc.reduction,
                            );
                        }

                        tma::bulk_commit_group();
                        tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                    }
                    LoadStoreModeEnum::ModeMx1 => {
                        assert!(
                            P::LARGEST_MX1_STORE_BITS_PER_NON_SF_ELEMENT
                                >= desc.ty.bits_per_element()
                        );

                        let num_bytes = P::CTA_TM * desc.ty.bits_per_element() / 8;

                        // Mx1 values are identical across CTAs sharing an M-coordinate,
                        // so only the first CTA in each cluster row stores them.
                        if block_id_in_cluster().y == 0 {
                            if num_bytes.is_multiple_of(GMEM_TMA_ALIGNMENT) {
                                // (1) TMA-aligned => use TMA directly.
                                let src = self.smem.store_slot.as_ptr().cast();
                                tma_store_tile_with_reduction(
                                    src,
                                    &desc.tma_desc.tma,
                                    [(desc.coord_mn[0] as usize + cta_offset_mn[0]) as i32],
                                    desc.cache_policy,
                                    desc.reduction,
                                );

                                tma::bulk_commit_group();
                                tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                            } else if desc.ty.bits_per_element() == 32 {
                                // (2) 32-bit elements for bounds-check granularity => go through registers.
                                assert!(
                                    desc.reduction.is_none(),
                                    "reduction not supported for <16 byte aligned Mx1 stores"
                                );
                                #[rustc_unroll(full)]
                                for i in 0..num_bytes / 4 {
                                    let val =
                                        self.smem.store_slot.as_ptr().cast::<u32>().add(i).read();
                                    let coord = i + desc.coord_mn[0] as usize + cta_offset_mn[0];
                                    if coord < desc.tma_desc.global_size[0] as usize {
                                        desc.tma_desc
                                            .global_address
                                            .cast::<u32>()
                                            .add(coord)
                                            .write(val);
                                    }
                                }
                            } else {
                                // <4 alignment unsupported for now.
                                unimplemented!()
                            }
                        }
                    }
                }

                self.smem.bars.store_slot_consumed.arrive();
            }

            self.store_idx += 1;
        }
    }

    #[inline(always)]
    fn store_block_scaled<BS: BlockScaledFormat>(
        &mut self,
        desc: BlockScaledStoreDescriptor,
        _get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        )
            -> [[f32; <<K as TileElementwiseKernel>::Params as Params>::THREAD_N];
            <<K as TileElementwiseKernel>::Params as Params>::THREAD_M],
    ) {
        unsafe {
            self.smem
                .bars
                .store_slot_produced
                .wait(PhaseParity::of_phase(self.store_idx));

            if elect(LaneMask::FULL).is_leader {
                let cta_offset_mn = [
                    block_id_in_cluster().x as usize * P::CTA_TM,
                    block_id_in_cluster().y as usize * P::CTA_TN,
                ];

                match desc.sf_dim {
                    BlockScaleDim::Rowwise => {
                        let bits_per_element = size_of::<BS::QuantizedVal<{ P::THREAD_N }>>() * 8;

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in
                            iter_tma_panels::<P>(P::CTA_TM)
                        {
                            let src = self
                                .smem
                                .store_slot
                                .as_ptr()
                                .cast::<u8>()
                                .add(panel_smem_offset * bits_per_element / 8);
                            tma::tma_store_tile(
                                src,
                                &desc.tma_desc.tma,
                                [
                                    (desc.coord_mn[1] as usize
                                        + panel_coord_offset
                                        + cta_offset_mn[1])
                                        as i32,
                                    (desc.coord_mn[0] as usize + cta_offset_mn[0]) as i32,
                                ],
                                desc.cache_policy,
                            );
                        }

                        tma::bulk_commit_group();
                        tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                    }
                    BlockScaleDim::Colwise => todo!(),
                }

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

            let vals = match desc.ty {
                LoadTy::F32 => desc
                    .mode
                    .worker_load_from_smem_f32(self.smem.load_slots[slot_idx].as_ptr().cast()),
                LoadTy::Bf16 => desc
                    .mode
                    .worker_load_from_smem_bf16(self.smem.load_slots[slot_idx].as_ptr().cast()),
            };
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
            let mut thr_vals = get_vals(&mut self.state);

            // Intra-warp reduction
            #[rustc_unroll(full)]
            for v in thr_vals.iter_mut() {
                #[rustc_unroll(full)]
                for bit in 0..5 {
                    *v = reduction(*v, shuffle_xor(LaneMask::FULL, *v, 1 << bit));
                }
            }

            let slot_idx = self.reduction_idx as usize % 2;

            // Intra-CTA reduction through SMEM
            if elect(LaneMask::FULL).is_leader {
                self.smem.cta_reduction_scratchpads[slot_idx]
                    .get_unchecked(warp_id() as usize)
                    .as_ptr()
                    .cast::<[f32; <K::Params as Params>::THREAD_M]>()
                    .cast_mut()
                    .write(thr_vals);
            }

            let token = self.smem.bars.cta_reduction_produced[slot_idx].arrive();
            self.smem.bars.cta_reduction_produced[slot_idx].wait(token);

            #[rustc_unroll(full)]
            for i in 0..NUM_WORKER_WARPS {
                #[rustc_unroll(full)]
                for j in 0..<K::Params as Params>::THREAD_M {
                    let smem_val = self.smem.cta_reduction_scratchpads[slot_idx][i][j]
                        .get()
                        .read()
                        .assume_init();
                    thr_vals[j] = if i == 0 {
                        smem_val
                    } else {
                        reduction(thr_vals[j], smem_val)
                    };
                }
            }

            if P::CLUSTER_SIZE_N > 1 {
                // Perform intra-CGA reduction through DSMEM.
                // Each CTA sends its reduced column to every CTA
                // having the same M-coordinate.
                if uniform_warp_id() == 0 && (lane_id() as usize) < P::CLUSTER_SIZE_N {
                    let dst_cta_n = lane_id();

                    let dst_cta_rank =
                        block_id_in_cluster().x + dst_cta_n * P::CLUSTER_SIZE_M as u32;
                    let dst = map_to_cta(
                        self.smem.cluster_row_reduction_scratchpads[slot_idx]
                            .get_unchecked(block_id_in_cluster().y as usize)
                            .as_ptr()
                            .cast_mut()
                            .cast::<f32>(),
                        dst_cta_rank,
                    );
                    let bar = map_to_cta(
                        &self.smem.bars.cluster_reduction_produced[slot_idx] as *const MBarrier
                            as *mut MBarrier,
                        dst_cta_rank,
                    );

                    #[rustc_unroll(full)]
                    for row in 0..P::THREAD_M {
                        st_async(dst.add(row), thr_vals[row], bar);
                    }
                }

                // Wait on data to arrive from peer CTAs.
                self.smem.bars.cluster_reduction_produced[slot_idx]
                    .wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                        PhaseParity::of_phase(self.reduction_idx / 2),
                    );
                fence_sync_restrict::<{ MemoryOrdering::Acquire }>();

                // Re-arm the barrier for when this slot is reused (two reductions in the future).
                if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                    self.smem.bars.cluster_reduction_produced[slot_idx]
                        .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                            (size_of::<f32>() * P::CLUSTER_SIZE_N * P::CTA_TM) as u32,
                        );
                }

                // Now every lane computes the final reduced value.
                #[rustc_unroll(full)]
                for row in 0..P::THREAD_M {
                    #[rustc_unroll(full)]
                    for cta_n in 0..P::CLUSTER_SIZE_N {
                        let cta_val = self.smem.cluster_row_reduction_scratchpads[slot_idx]
                            .get_unchecked(cta_n)
                            .as_ptr()
                            .add(row)
                            .read()
                            .into_inner()
                            .assume_init();
                        thr_vals[row] = if cta_n == 0 {
                            cta_val
                        } else {
                            reduction(thr_vals[row], cta_val)
                        };
                    }
                }
            }

            reduce_callback(&mut self.state, thr_vals);

            self.reduction_idx += 1;
        }
    }

    #[inline(always)]
    fn store<M>(
        &mut self,
        desc: StoreDescriptor<M>,
        get_vals: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState) -> M::LaneVals,
    ) where
        M: LoadStoreMode<P>,
    {
        unsafe {
            let vals = get_vals(&mut self.state);

            if self.store_idx > 0 {
                self.smem
                    .bars
                    .store_slot_consumed
                    .wait(PhaseParity::of_phase(self.store_idx).flip());
            }

            match desc.ty {
                StoreTy::F32 => desc.mode.worker_store_to_smem_f32(
                    &vals,
                    self.smem.store_slot.as_ptr().cast::<u8>().cast_mut(),
                ),
                StoreTy::Bf16 => desc.mode.worker_store_to_smem_bf16(
                    &vals,
                    self.smem.store_slot.as_ptr().cast::<u8>().cast_mut(),
                ),
            }

            fence_proxy::<{ ProxyKind::AsyncSharedCta }>();
            self.smem.bars.store_slot_produced.arrive();

            self.store_idx += 1;
        }
    }

    #[inline(always)]
    fn store_block_scaled<BS: BlockScaledFormat>(
        &mut self,
        desc: BlockScaledStoreDescriptor,
        get_vals: impl FnOnce(
            &mut <K as TileElementwiseKernel>::WorkerState,
        )
            -> [[f32; <<K as TileElementwiseKernel>::Params as Params>::THREAD_N];
            <<K as TileElementwiseKernel>::Params as Params>::THREAD_M],
    ) {
        assert!(
            P::LARGEST_MXN_STORE_BITS_PER_NON_SF_ELEMENT
                >= size_of::<BS::QuantizedVal<{ P::THREAD_N }>>() * 8
        );
        unsafe {
            let thr_vals = get_vals(&mut self.state);

            match desc.sf_dim {
                BlockScaleDim::Rowwise => {
                    let thr_amaxs = Layouts::<P>::new().compute_rowwise_thr_amaxs::<BS>(&thr_vals);

                    let thr_sfs = array_map_unrolled(
                        thr_amaxs,
                        #[inline(always)]
                        |row| {
                            let row = f32xN::from_array(row);
                            BS::compute_scale_factor(row)
                        },
                    );

                    let mut thr_quantized_vals = array_zip_unrolled(
                        thr_vals,
                        thr_sfs,
                        #[inline(always)]
                        |row, sf| {
                            let row = f32xN::from_array(row);
                            BS::quantize_f32(row, sf)
                        },
                    );

                    // Scale factors go directly to GMEM.
                    let sf_cols = desc.tma_desc.global_size[0]
                        .div_ceil(BS::BLOCK_SIZE as u32)
                        .next_multiple_of(SF_TILE_NCOLS);
                    let sf_rows = desc.tma_desc.global_size[1].next_multiple_of(SF_TILE_NROWS);

                    let tile_m = desc.coord_mn[0] + block_id_in_cluster().x * P::CTA_TM as u32;
                    let tile_n = desc.coord_mn[1] + block_id_in_cluster().y * P::CTA_TN as u32;

                    // Only store with one lane per group of lanes sharing a block.
                    let is_sf_writer = (lane_id() as usize)
                        .is_multiple_of(BS::BLOCK_SIZE / Layouts::<P>::VECTOR_SIZE);

                    #[rustc_unroll(full)]
                    for i in 0..P::THREAD_M {
                        let row = tile_m + i as u32;
                        let row_sfs = thr_sfs[i].to_array();

                        #[rustc_unroll(full)]
                        for chunk_idx in 0..Layouts::<P>::VECTORS_PER_THREAD_PER_ROW {
                            let element_idx = chunk_idx * Layouts::<P>::VECTOR_SIZE;
                            let n = tile_n
                                + Layouts::<P>::new()
                                    .thread_element_n_index(thread_id().x, element_idx)
                                    as u32;
                            let sf_col = n / BS::BLOCK_SIZE as u32;

                            if is_sf_writer && row < sf_rows && sf_col < sf_cols {
                                desc.sf_ptr
                                    .cast::<BS::ScaleFactor<{ P::THREAD_N }>>()
                                    .add(scale_factor_pos(
                                        row,
                                        sf_col,
                                        BS::BLOCK_SIZE as u32,
                                        sf_cols,
                                    ))
                                    .write(row_sfs[element_idx]);
                            }
                        }
                    }

                    if self.store_idx > 0 {
                        self.smem
                            .bars
                            .store_slot_consumed
                            .wait(PhaseParity::of_phase(self.store_idx).flip());
                    }

                    #[rustc_unroll(full)]
                    for i in 0..P::THREAD_M {
                        Layouts::<P>::new().copy_row(
                            (&mut thr_quantized_vals[i]) as *mut _
                                as *mut BS::QuantizedVal<{ P::THREAD_N }>,
                            self.smem.store_slot.as_ptr().cast_mut().cast(),
                            i,
                            thread_id().x,
                            P::CTA_TM,
                            CopyDir::RegToSmem,
                        );
                    }
                }
                BlockScaleDim::Colwise => {
                    assert!(P::DOES_STORE_COLWISE_BLOCK_SCALED_DATA);
                }
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
    pub unsafe fn get() -> Self {
        match unsafe { uniform_warp_id() } {
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
unsafe fn do_setup<P: Params>(smem: &Smem<P>) {
    unsafe {
        assert_eq!(dynamic_smem_size() as usize, size_of::<Smem<P>>());
        assert_eq!(block_size(), 256);
        assert_eq!(cluster_size() as usize, P::CLUSTER_SIZE);
        assert_eq!(cluster_dim().x as usize, P::CLUSTER_SIZE_M);
        assert_eq!(cluster_dim().y as usize, P::CLUSTER_SIZE_N);
        assert_eq!(grid_dim().y as usize, P::CLUSTER_SIZE_N);

        #[rustc_unroll(full)]
        for slot_idx in 0..P::NUM_LOAD_SLOTS {
            smem.bars.load_slot_produced[slot_idx].init(1);
            smem.bars.load_slot_consumed[slot_idx].init(NUM_WORKER_THREADS as u32);
        }

        #[rustc_unroll(full)]
        for slot_idx in 0..2 {
            smem.bars.cta_reduction_produced[slot_idx].init(NUM_WORKER_THREADS as u32);
            smem.bars.cluster_reduction_produced[slot_idx].init(1);
            smem.bars.cluster_reduction_produced[slot_idx]
                .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                    (size_of::<f32>() * P::CLUSTER_SIZE_N * P::CTA_TM) as u32,
                );
        }

        smem.bars
            .store_slot_produced
            .init(NUM_WORKER_THREADS as u32);
        smem.bars.store_slot_consumed.init(1);

        MBarrier::fence_init();
    }
}

#[inline(always)]
pub unsafe fn run<K: TileElementwiseKernel<Params = P>, P: Params>(mut kernel: K) {
    unsafe {
        const {
            assert!(P::CLUSTER_SIZE_M > 0);
            assert!(P::CLUSTER_SIZE_N > 0);
            // Cluster row reductions use one lane of warp 0 per peer CTA.
            // Note that clusters >16 are not possible on hardware as of this writing.
            assert!(P::CLUSTER_SIZE_N <= 32);
            assert!(P::CGA_TM.is_multiple_of(P::CLUSTER_SIZE_M));
            assert!(P::CGA_TN.is_multiple_of(P::CLUSTER_SIZE_N));

            assert!(P::NUM_LOAD_SLOTS > 0);
            assert!(P::MACROBLOCK_SIZE > 0);

            #[cfg(has_227kb_smem)]
            assert!(
                size_of::<Smem<P>>() <= 227 * 1024,
                "SMEM requirement exceeds hardware capacity; try reducing load slots"
            );
            #[cfg(not(has_227kb_smem))]
            assert!(
                size_of::<Smem<P>>() <= 99 * 1024,
                "SMEM requirement exceeds hardware capacity; try reducing load slots"
            );

            assert!(P::CTA_TN >= 128);
            assert!(P::CTA_TN.is_power_of_two());

            assert!(P::CTA_TM <= 256);
            assert!(P::CTA_TM.is_power_of_two());
        }

        let smem = &*dynamic_smem_start().cast::<Smem<P>>();
        let clc = ClcScheduler::<{ P::CLUSTER_ENABLED }>::new(smem.clc.get());

        if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
            do_setup(smem);
        }

        if P::CLUSTER_ENABLED {
            cluster::barrier_sync::<{ BarrierAlignment::Aligned }>();
        } else {
            sync_threads();
        }

        let mut sched = TileScheduler::<K> {
            clc,
            current_macroblock: None,
            current_idx_in_macroblock: 0,
        };

        let role = WarpRole::get();
        match role {
            WarpRole::Worker => {
                increase_max_registers::<224>();
                WorkerWarp {
                    smem,
                    sched,
                    state: kernel.init_worker_state(),
                    load_idx: 0,
                    store_idx: 0,
                    reduction_idx: 0,
                }
                .run(&mut kernel);
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
                        // TODO, placeholder for now
                        while sched.clc.get_next_block().is_some() {}
                    }
                    WarpRole::Clc => {
                        sched.clc.run_scheduler();
                    }
                }
            }
        }

        if P::CLUSTER_ENABLED {
            cluster::barrier_sync::<{ BarrierAlignment::Unaligned }>();
        }
    }
}
