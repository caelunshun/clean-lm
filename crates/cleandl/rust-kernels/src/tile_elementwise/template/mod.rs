//! WIP
//!
//! Template for building tile_elementwise kernels.
//!
//! The kernels are built around a warp-specialized design, with `W = Params::NUM_WORKER_WARPS` (4 or 8):
//! - Warps 0..W - Worker warps performing computations. These also store
//!   results directly from registers to GMEM.
//! - Warps W, W+1, W+2: load warps for M x N, 1 x N, and M x 1 loads respectively. Each load shape
//!   has its own ring of SMEM load slots (see `Params::NUM_*_LOAD_SLOTS`).
//! - Warp W+3: CLC scheduler warp.
//!
//! The non-worker warps form a single auxiliary warpgroup. Registers are rebalanced between
//! worker and auxiliary warps using `setmaxnreg` (see `Params::WORKER_REGISTERS`).
//!
//! The template assumes we are operating on TM (rows) x TN (columns) tiles of some iteration space.
//! Data within a tile is evenly partitioned among worker threads over the N dimension (each thread holds
//! all rows for its assigned columns). Out-of-bounds entries are set to 0 on loads and skipped on stores.
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
//! bf16 and f32 stores can be reductions, which use `red.global` (see `StoreReduceOp`).
//!
//! The tile processor can perform intra-CTA reductions through shared memory.
//! Currently, only per-row reductions are supported in this way. (Per-column is easily implemented
//! within a lane's arithmetic code if the M-cluster-size is 1.)
//!
//! The template uses cluster launch control (CLC) to implement a persistent, dynamically load-balanced
//! kernel. Each logical CGA processes a "macroblock" of tiles. Larger macroblock sizes reduce scheduling overhead
//! but increase risk of wave quantization effects.
//!
//! The grid must be launched with dimensions `(num_macroblocks * CLUSTER_SIZE_M, CLUSTER_SIZE_N, 1)`,
//! block dimensions `(BLOCK_SIZE, 1, 1)`, and cluster dimensions `(CLUSTER_SIZE_M, CLUSTER_SIZE_N, 1)`. The macroblock index of a CGA
//! is then `block_id().x / CLUSTER_SIZE_M`.
//!
//! Note that the partitioning of tile columns among worker threads is unspecified,
//! and may vary with tile size. Rowwise reductions should use KernelVisitor::rowwise_reduce.

use crate::{
    tile_elementwise::template::partitioning::{
        CopyDir, GmemRow, GmemTile, Partitioning, SmemRow, iter_tma_panels,
        store_contiguous_to_gmem,
    },
    utils::{
        block_scale::{BlockScaledFormat, Mxfp8, SF_TILE_NCOLS, SF_TILE_NROWS, scale_factor_pos},
        clc::{ClcScheduler, ClcSmem},
        compiler_magic::{array_map_unrolled, array_zip_unrolled, uniform_warp_id},
        tma::{CompoundTmaDescriptor, GMEM_TMA_ALIGNMENT, SMEM_TMA_ALIGNMENT_NO_SWIZZLE},
    },
};
use core::{cell::UnsafeCell, marker::ConstParamTy, mem::MaybeUninit};
use ptx_std::{
    Dim3, MBarrier, MemoryOrdering, PhaseParity, Scope,
    barrier::{BarrierAlignment, sync_threads},
    bf16, bf16x2, bf16xN, block_id_in_cluster, block_size,
    cluster::{self, map_to_cta, st_async},
    cluster_dim, cluster_size,
    cp_async::{self, CacheOperator},
    dynamic_smem_size, dynamic_smem_start, f32xN,
    fence::fence_sync_restrict,
    grid_dim, lane_id,
    launch::{decrease_max_registers, increase_max_registers},
    memory::CachePolicy,
    thread_id, tma,
    warp_collectives::{LaneMask, elect, shuffle_xor},
    warp_id,
};

mod blockscaled_colwise_store;
mod partitioning;

pub const NUM_AUX_WARPS: usize = 4;
pub const NUM_AUX_THREADS: usize = 32 * NUM_AUX_WARPS;

pub trait Params {
    /// Either 4 or 8.
    const NUM_WORKER_WARPS: usize;

    /// Registers per thread allocated to worker warps via `setmaxnreg`.
    ///
    /// `WORKER_REGISTERS * NUM_WORKER_THREADS + AUX_REGISTERS * NUM_AUX_THREADS` must not exceed
    /// the CTA's register pool, i.e. the kernel's launch `max_registers` times `BLOCK_SIZE`;
    /// otherwise the kernel deadlocks.
    const WORKER_REGISTERS: u32;
    /// Registers per thread allocated to auxiliary (non-worker) warps via `setmaxnreg`.
    const AUX_REGISTERS: u32;

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

    /// Number of slots in the ring of SMEM load slots for each load shape. Each slot is sized for
    /// the largest load of its shape (see the LARGEST_*_LOAD_BITS_PER_ELEMENT constants).
    ///
    /// May be 0 if the kernel never issues loads of that shape.
    const NUM_MXN_LOAD_SLOTS: usize;
    const NUM_1XN_LOAD_SLOTS: usize;
    const NUM_MX1_LOAD_SLOTS: usize;

    const LARGEST_MXN_LOAD_BITS_PER_ELEMENT: usize;
    const LARGEST_1XN_LOAD_BITS_PER_ELEMENT: usize;
    const LARGEST_MX1_LOAD_BITS_PER_ELEMENT: usize;

    const DOES_STORE_COLWISE_BLOCK_SCALED_DATA: bool;

    /// Whether bounds checks are needed on global memory operations.
    ///
    /// If `false`, then tensor sizes must be divisible by CGA_TM*MACROBLOCK_SIZE and CGA_TN resp.,
    /// and the load/store coordinates in their descriptors must produce addresses
    /// that are in-bounds for the entire tile and aligned to 32 bytes. Otherwise,
    /// the behavior is undefined.
    const ENABLE_BOUNDS_CHECKS: bool;

    // --- Derived constants --- //

    const NUM_WORKER_THREADS: usize = 32 * Self::NUM_WORKER_WARPS;
    const BLOCK_SIZE: usize = Self::NUM_WORKER_THREADS + NUM_AUX_THREADS;

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

    const THREAD_N: usize = Self::CTA_TN.div_ceil(Self::NUM_WORKER_THREADS);
    const THREAD_M: usize = Self::CTA_TM;

    const MXN_LOAD_SLOT_BYTES: usize = (Self::LARGEST_MXN_LOAD_BITS_PER_ELEMENT
        * Self::CTA_TM
        * Self::TMA_PANEL_COLS
        * Self::NUM_TMA_PANELS
        / 8)
    .next_multiple_of(SMEM_TMA_ALIGNMENT_NO_SWIZZLE);
    const MX1_LOAD_SLOT_BYTES: usize = (Self::LARGEST_MX1_LOAD_BITS_PER_ELEMENT * Self::CTA_TM / 8)
        .next_multiple_of(SMEM_TMA_ALIGNMENT_NO_SWIZZLE);
    const _1XN_LOAD_SLOT_BYTES: usize =
        (Self::LARGEST_1XN_LOAD_BITS_PER_ELEMENT * Self::TMA_PANEL_COLS * Self::NUM_TMA_PANELS / 8)
            .next_multiple_of(SMEM_TMA_ALIGNMENT_NO_SWIZZLE);

    const CLUSTER_ENABLED: bool = Self::CLUSTER_SIZE > 1;
    const CLUSTER_SCOPE: Scope = const {
        if Self::CLUSTER_ENABLED {
            Scope::Cluster
        } else {
            Scope::Cta
        }
    };

    /// Used to re-gather data in row-sharded format among peer CTAs
    /// for MXFP8 columnwise stores. This buffer holds pre-quantization
    /// `bf16` values received from peer CTAs.
    const CLUSTER_RESHARD_BUFFER_ELEMENTS: usize = const {
        if Self::DOES_STORE_COLWISE_BLOCK_SCALED_DATA && Self::CTA_TM < Mxfp8::BLOCK_SIZE {
            Mxfp8::BLOCK_SIZE * Self::CLUSTER_RESHARDED_CTA_TN
        } else {
            0
        }
    };
    /// Number of CTAs (adjacent along M) that reshard data among each other so that
    /// each holds full `BLOCK_SIZE`-row columns. 1 when `CTA_TM >= BLOCK_SIZE`.
    const CLUSTER_RESHARD_GROUP_WIDTH: usize = Mxfp8::BLOCK_SIZE.div_ceil(Self::CTA_TM);
    /// Number of columns each CTA holds after resharding.
    const CLUSTER_RESHARDED_CTA_TN: usize = Self::CTA_TN / Self::CLUSTER_RESHARD_GROUP_WIDTH;
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
    pub reduction: Option<StoreReduceOp>,
    pub cache_policy: Option<CachePolicy>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StoreReduceOp {
    Add,
}

#[derive(Copy, Clone)]
pub struct GmemStore<'a> {
    pub tma_desc: &'a CompoundTmaDescriptor,
    pub coord_mn: [u32; 2],
    pub cache_policy: Option<CachePolicy>,
    pub reduction: Option<StoreReduceOp>,
}

impl<'a> GmemStore<'a> {
    #[inline(always)]
    fn new<P: Params>(
        tma_desc: &'a CompoundTmaDescriptor,
        tile_coord_mn: [u32; 2],
        cache_policy: Option<CachePolicy>,
        reduction: Option<StoreReduceOp>,
    ) -> Self {
        Self {
            tma_desc,
            coord_mn: [
                tile_coord_mn[0] + block_id_in_cluster().x * P::CTA_TM as u32,
                tile_coord_mn[1] + block_id_in_cluster().y * P::CTA_TN as u32,
            ],
            cache_policy,
            reduction,
        }
    }

    #[inline(always)]
    fn mxn_tile<T>(&self) -> GmemTile<T> {
        let [m, n] = self.coord_mn.map(|x| x as usize);
        let row_stride = self.tma_desc.global_strides[0] as usize / size_of::<T>();
        GmemTile {
            start: self
                .tma_desc
                .global_address
                .cast::<T>()
                .wrapping_add(m * row_stride + n),
            row_stride,
            valid_rows: (self.tma_desc.global_size[1] as usize).saturating_sub(m),
            valid_cols: (self.tma_desc.global_size[0] as usize).saturating_sub(n),
            cache_policy: self.cache_policy,
            reduction: self.reduction,
        }
    }

    #[inline(always)]
    fn slice_1d<T>(&self, coord: u32) -> GmemRow<T> {
        GmemRow {
            start: self
                .tma_desc
                .global_address
                .cast::<T>()
                .wrapping_add(coord as usize),
            valid_cols: (self.tma_desc.global_size[0] as usize).saturating_sub(coord as usize),
            cache_policy: self.cache_policy,
            reduction: self.reduction,
        }
    }
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum LoadStoreModeEnum {
    ModeMxN,
    Mode1xN,
    ModeMx1,
}

impl LoadStoreModeEnum {
    pub const ALL: [Self; 3] = [Self::ModeMxN, Self::Mode1xN, Self::ModeMx1];

    #[inline(always)]
    const fn num_load_slots<P: Params>(self) -> usize {
        match self {
            Self::ModeMxN => P::NUM_MXN_LOAD_SLOTS,
            Self::Mode1xN => P::NUM_1XN_LOAD_SLOTS,
            Self::ModeMx1 => P::NUM_MX1_LOAD_SLOTS,
        }
    }
}

pub trait LoadStoreMode<P: Params> {
    type LaneVals;

    const MODE: LoadStoreModeEnum;

    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals;
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals;

    unsafe fn worker_store_to_gmem_bf16(&self, thr_vals: &Self::LaneVals, dst: &GmemStore);
    unsafe fn worker_store_to_gmem_f32(&self, thr_vals: &Self::LaneVals, dst: &GmemStore);
}

pub struct ModeMxN;

impl<P: Params> LoadStoreMode<P> for ModeMxN {
    type LaneVals = [[f32; P::THREAD_N]; P::THREAD_M];

    const MODE: LoadStoreModeEnum = LoadStoreModeEnum::ModeMxN;

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [[0.0; P::THREAD_N]; P::THREAD_M];
            #[rustc_unroll(full)]
            for row in 0..P::THREAD_M {
                let mut raw = [bf16::ZERO; P::THREAD_N];
                Partitioning::<P>::new().copy_row(
                    &mut raw as *mut bf16,
                    thread_id().x,
                    CopyDir::SmemToReg(SmemRow {
                        slot: smem.cast::<bf16>().cast_mut(),
                        row,
                        panel_rows: P::CTA_TM,
                    }),
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
                Partitioning::<P>::new().copy_row(
                    &mut thr_vals[row] as *mut f32,
                    thread_id().x,
                    CopyDir::SmemToReg(SmemRow {
                        slot: smem.cast::<f32>().cast_mut(),
                        row,
                        panel_rows: P::CTA_TM,
                    }),
                );
            }
            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_gmem_bf16(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe {
            let raw = array_map_unrolled(
                *thr_vals,
                #[inline(always)]
                |row| bf16xN::from_f32(f32xN::from_array(row)).to_array(),
            );
            Partitioning::<P>::new().store_rows_to_gmem(&raw, thread_id().x, dst.mxn_tile());
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_gmem_f32(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe {
            Partitioning::<P>::new().store_rows_to_gmem(thr_vals, thread_id().x, dst.mxn_tile());
        }
    }
}

pub struct Mode1xN;

impl<P: Params> LoadStoreMode<P> for Mode1xN {
    type LaneVals = [f32; P::THREAD_N];

    const MODE: LoadStoreModeEnum = LoadStoreModeEnum::Mode1xN;

    #[inline(always)]
    unsafe fn worker_load_from_smem_bf16(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut raw = [bf16::ZERO; P::THREAD_N];

            Partitioning::<P>::new().copy_row(
                &mut raw as *mut bf16,
                thread_id().x,
                CopyDir::SmemToReg(SmemRow {
                    slot: smem.cast::<bf16>().cast_mut(),
                    row: 0,
                    panel_rows: 1,
                }),
            );

            array_map_unrolled(raw, bf16::to_f32)
        }
    }

    #[inline(always)]
    unsafe fn worker_load_from_smem_f32(&self, smem: *const u8) -> Self::LaneVals {
        unsafe {
            let mut thr_vals = [0.0; P::THREAD_N];

            Partitioning::<P>::new().copy_row(
                &mut thr_vals as *mut f32,
                thread_id().x,
                CopyDir::SmemToReg(SmemRow {
                    slot: smem.cast::<f32>().cast_mut(),
                    row: 0,
                    panel_rows: 1,
                }),
            );

            thr_vals
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_gmem_bf16(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe {
            let raw = array_map_unrolled(*thr_vals, bf16::from_f32);
            Partitioning::<P>::new().store_rows_to_gmem(
                &[raw],
                thread_id().x,
                dst.slice_1d(dst.coord_mn[1]).into(),
            );
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_gmem_f32(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe {
            Partitioning::<P>::new().store_rows_to_gmem(
                core::array::from_ref(thr_vals),
                thread_id().x,
                dst.slice_1d(dst.coord_mn[1]).into(),
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

    const MODE: LoadStoreModeEnum = LoadStoreModeEnum::ModeMx1;

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
    unsafe fn worker_store_to_gmem_bf16(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe {
            let thr_vals_bf16 = bf16xN::from_f32(f32xN::from_array(*thr_vals)).to_array();
            store_mx1(&thr_vals_bf16, dst);
        }
    }

    #[inline(always)]
    unsafe fn worker_store_to_gmem_f32(&self, thr_vals: &Self::LaneVals, dst: &GmemStore) {
        unsafe { store_mx1(thr_vals, dst) }
    }
}

#[inline(always)]
unsafe fn store_mx1<T: Copy, const N: usize>(vals: &[T; N], dst: &GmemStore) {
    unsafe {
        if block_id_in_cluster().y == 0 && uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader
        {
            store_contiguous_to_gmem(vals, dst.slice_1d(dst.coord_mn[0]));
        }
    }
}

// --- Smem --- //

/// Double-buffered `[[f32; ROWS]; N]` reduction scratchpads.
type RowReductionScratchpads<const ROWS: usize, const N: usize> =
    [[[UnsafeCell<MaybeUninit<f32>>; ROWS]; N]; 2];

type LoadSlots<const SLOT_BYTES: usize, const N: usize> =
    [[UnsafeCell<MaybeUninit<u8>>; SLOT_BYTES]; N];

#[repr(C, align(1024))]
struct Smem<P: Params> {
    /// Slot sizes are multiples of the TMA alignment, so every slot is aligned.
    mxn_load_slots: LoadSlots<{ P::MXN_LOAD_SLOT_BYTES }, { P::NUM_MXN_LOAD_SLOTS }>,
    _1xn_load_slots: LoadSlots<{ P::_1XN_LOAD_SLOT_BYTES }, { P::NUM_1XN_LOAD_SLOTS }>,
    mx1_load_slots: LoadSlots<{ P::MX1_LOAD_SLOT_BYTES }, { P::NUM_MX1_LOAD_SLOTS }>,

    /// These slots are double-buffered to reduce needed barrier waits.
    cta_reduction_scratchpads: RowReductionScratchpads<{ P::CTA_TM }, { P::NUM_WORKER_WARPS }>,
    cluster_row_reduction_scratchpads:
        RowReductionScratchpads<{ P::CTA_TM }, { P::CLUSTER_SIZE_N }>,
    cluster_reshard_buffers:
        [[UnsafeCell<MaybeUninit<bf16>>; P::CLUSTER_RESHARD_BUFFER_ELEMENTS]; 2],

    bars: Barriers<P>,
    clc: UnsafeCell<ClcSmem>,
}

#[repr(C)]
struct Barriers<P: Params> {
    mxn_load_slot_produced: [MBarrier; P::NUM_MXN_LOAD_SLOTS],
    mxn_load_slot_consumed: [MBarrier; P::NUM_MXN_LOAD_SLOTS],
    _1xn_load_slot_produced: [MBarrier; P::NUM_1XN_LOAD_SLOTS],
    _1xn_load_slot_consumed: [MBarrier; P::NUM_1XN_LOAD_SLOTS],
    mx1_load_slot_produced: [MBarrier; P::NUM_MX1_LOAD_SLOTS],
    mx1_load_slot_consumed: [MBarrier; P::NUM_MX1_LOAD_SLOTS],

    cta_reduction_produced: [MBarrier; 2],
    cluster_row_reduction_produced: [MBarrier; 2],

    cluster_reshard_buffer_produced: [MBarrier; 2],
    cluster_reshard_buffer_consumed: [MBarrier; 2],
}

/// View of the ring of load slots for one load shape.
#[derive(Copy, Clone)]
struct LoadRing<'a> {
    slots: *mut u8,
    slot_bytes: usize,
    produced: &'a [MBarrier],
    consumed: &'a [MBarrier],
}

impl LoadRing<'_> {
    #[inline(always)]
    fn num_slots(&self) -> usize {
        self.produced.len()
    }

    #[inline(always)]
    fn slot(&self, slot_idx: usize) -> *mut u8 {
        self.slots.wrapping_add(slot_idx * self.slot_bytes)
    }
}

impl<P: Params> Smem<P> {
    #[inline(always)]
    fn load_ring(&self, mode: LoadStoreModeEnum) -> LoadRing<'_> {
        match mode {
            LoadStoreModeEnum::ModeMxN => LoadRing {
                slots: self.mxn_load_slots.as_ptr().cast::<u8>().cast_mut(),
                slot_bytes: P::MXN_LOAD_SLOT_BYTES,
                produced: &self.bars.mxn_load_slot_produced,
                consumed: &self.bars.mxn_load_slot_consumed,
            },
            LoadStoreModeEnum::Mode1xN => LoadRing {
                slots: self._1xn_load_slots.as_ptr().cast::<u8>().cast_mut(),
                slot_bytes: P::_1XN_LOAD_SLOT_BYTES,
                produced: &self.bars._1xn_load_slot_produced,
                consumed: &self.bars._1xn_load_slot_consumed,
            },
            LoadStoreModeEnum::ModeMx1 => LoadRing {
                slots: self.mx1_load_slots.as_ptr().cast::<u8>().cast_mut(),
                slot_bytes: P::MX1_LOAD_SLOT_BYTES,
                produced: &self.bars.mx1_load_slot_produced,
                consumed: &self.bars.mx1_load_slot_consumed,
            },
        }
    }
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

/// Issues the loads of shape `MODE`, ignoring all others.
struct TmaLoadWarp<'a, K: TileElementwiseKernel, P: Params, const MODE: LoadStoreModeEnum> {
    smem: &'a Smem<P>,
    sched: TileScheduler<K>,
    load_idx: u32,
}

impl<'a, K: TileElementwiseKernel<Params = P>, P: Params, const MODE: LoadStoreModeEnum>
    TmaLoadWarp<'a, K, P, MODE>
{
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

impl<'a, 'tmp, K: TileElementwiseKernel<Params = P>, P: Params, const MODE: LoadStoreModeEnum>
    KernelVisitor<K, P> for &'tmp mut TmaLoadWarp<'a, K, P, MODE>
{
    #[inline(always)]
    fn load<M>(
        &mut self,
        desc: LoadDescriptor<M>,
        _load_callback: impl FnOnce(&mut <K as TileElementwiseKernel>::WorkerState, M::LaneVals),
    ) where
        M: LoadStoreMode<P>,
    {
        if M::MODE != MODE {
            return;
        }

        unsafe {
            let ring = self.smem.load_ring(MODE);
            let num_load_slots = ring.num_slots();

            let slot_idx = self.load_idx as usize % num_load_slots;

            if self.load_idx >= num_load_slots as u32 {
                ring.consumed[slot_idx]
                    .wait(PhaseParity::of_phase(self.load_idx / num_load_slots as u32).flip());
            }

            let slot = ring.slot(slot_idx);
            let produced_bar = &ring.produced[slot_idx];

            let cta_offset_mn = [
                block_id_in_cluster().x as usize * P::CTA_TM,
                block_id_in_cluster().y as usize * P::CTA_TN,
            ];

            if elect(LaneMask::FULL).is_leader {
                match MODE {
                    LoadStoreModeEnum::ModeMxN => {
                        assert!(P::LARGEST_MXN_LOAD_BITS_PER_ELEMENT >= desc.ty.bits_per_element());

                        let num_bytes = desc.ty.bits_per_element()
                            * P::CTA_TM
                            * P::TMA_PANEL_COLS
                            * P::NUM_TMA_PANELS
                            / 8;
                        produced_bar.arrive_expect_tx(num_bytes as u32);

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in
                            iter_tma_panels::<P>(P::CTA_TM)
                        {
                            let dst = slot.add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma::tma_load_tile(
                                dst,
                                produced_bar,
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
                        produced_bar.arrive_expect_tx(num_bytes as u32);

                        #[rustc_unroll(full)]
                        for (panel_coord_offset, panel_smem_offset) in iter_tma_panels::<P>(1) {
                            let dst = slot.add(panel_smem_offset * desc.ty.bits_per_element() / 8);
                            tma::tma_load_tile(
                                dst,
                                produced_bar,
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
                            produced_bar.arrive_expect_tx(num_bytes as u32);
                            tma::tma_load_tile(
                                slot,
                                produced_bar,
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
                                let dst = slot.cast::<u32>().add(i);

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

                            produced_bar.cp_async_arrive_no_inc();
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

// --- WorkerWarp --- //

struct WorkerWarp<'a, K: TileElementwiseKernel, P: Params> {
    smem: &'a Smem<P>,
    sched: TileScheduler<K>,
    state: K::WorkerState,
    /// Indexed by `LoadStoreModeEnum`.
    load_idxs: [u32; 3],
    row_reduction_idx: u32,
    cluster_resharding_idx: u32,
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
        const {
            assert!(
                M::MODE.num_load_slots::<P>() > 0,
                "kernel issues a load of a shape that has no load slots"
            );
        }

        let ring = self.smem.load_ring(M::MODE);
        let load_idx = &mut self.load_idxs[M::MODE as usize];
        let num_load_slots = ring.num_slots();
        let slot_idx = *load_idx as usize % num_load_slots;
        let parity = PhaseParity::of_phase(*load_idx / num_load_slots as u32);

        unsafe {
            ring.produced[slot_idx].wait(parity);

            let slot = ring.slot(slot_idx);
            let vals = match desc.ty {
                LoadTy::F32 => desc.mode.worker_load_from_smem_f32(slot),
                LoadTy::Bf16 => desc.mode.worker_load_from_smem_bf16(slot),
            };
            ring.consumed[slot_idx].arrive();

            load_callback(&mut self.state, vals);

            *load_idx += 1;
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

            let slot_idx = self.row_reduction_idx as usize % 2;

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
            for i in 0..P::NUM_WORKER_WARPS {
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
                        &self.smem.bars.cluster_row_reduction_produced[slot_idx] as *const MBarrier
                            as *mut MBarrier,
                        dst_cta_rank,
                    );

                    #[rustc_unroll(full)]
                    for row in 0..P::THREAD_M {
                        st_async(dst.add(row), thr_vals[row], bar);
                    }
                }

                // Wait on data to arrive from peer CTAs.
                self.smem.bars.cluster_row_reduction_produced[slot_idx]
                    .wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                        PhaseParity::of_phase(self.row_reduction_idx / 2),
                    );
                fence_sync_restrict::<{ MemoryOrdering::Acquire }>();

                // Re-arm the barrier for when this slot is reused (two reductions in the future).
                if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                    self.smem.bars.cluster_row_reduction_produced[slot_idx]
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

            self.row_reduction_idx += 1;
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

            let dst = GmemStore::new::<P>(
                desc.tma_desc,
                desc.coord_mn,
                desc.cache_policy,
                desc.reduction,
            );
            match desc.ty {
                StoreTy::F32 => desc.mode.worker_store_to_gmem_f32(&vals, &dst),
                StoreTy::Bf16 => desc.mode.worker_store_to_gmem_bf16(&vals, &dst),
            }
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
        unsafe {
            let thr_vals = get_vals(&mut self.state);

            match desc.sf_dim {
                BlockScaleDim::Rowwise => {
                    let thr_amaxs =
                        Partitioning::<P>::new().compute_rowwise_thr_amaxs::<BS>(&thr_vals);

                    let thr_sfs = array_map_unrolled(
                        thr_amaxs,
                        #[inline(always)]
                        |row| {
                            let row = f32xN::from_array(row);
                            BS::compute_scale_factor(row)
                        },
                    );

                    let thr_quantized_vals = array_zip_unrolled(
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
                        .is_multiple_of(BS::BLOCK_SIZE / Partitioning::<P>::VECTOR_SIZE);

                    #[rustc_unroll(full)]
                    for i in 0..P::THREAD_M {
                        let row = tile_m + i as u32;
                        let row_sfs = thr_sfs[i].to_array();

                        #[rustc_unroll(full)]
                        for chunk_idx in 0..Partitioning::<P>::VECTORS_PER_THREAD_PER_ROW {
                            let element_idx = chunk_idx * Partitioning::<P>::VECTOR_SIZE;
                            let n = tile_n
                                + Partitioning::<P>::new()
                                    .thread_element_n_index(thread_id().x, element_idx)
                                    as u32;
                            let sf_col = n / BS::BLOCK_SIZE as u32;

                            if is_sf_writer
                                && (!P::ENABLE_BOUNDS_CHECKS || (row < sf_rows && sf_col < sf_cols))
                            {
                                desc.sf_ptr
                                    .cast::<BS::ScaleFactor<{ P::THREAD_N }>>()
                                    .add(scale_factor_pos(row, sf_col, sf_cols))
                                    .write(row_sfs[element_idx]);
                            }
                        }
                    }

                    let thr_quantized_vals = array_map_unrolled(
                        thr_quantized_vals,
                        #[inline(always)]
                        |row| row.to_array(),
                    );
                    Partitioning::<P>::new().store_rows_to_gmem(
                        &thr_quantized_vals,
                        thread_id().x,
                        GmemStore::new::<P>(desc.tma_desc, desc.coord_mn, desc.cache_policy, None)
                            .mxn_tile(),
                    );
                }
                BlockScaleDim::Colwise => {
                    blockscaled_colwise_store::worker_warp_blockscaled_colwise_store::<P, BS>(
                        self.smem,
                        thr_vals,
                        &desc,
                        &mut self.cluster_resharding_idx,
                    );
                }
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpRole {
    Worker,
    TmaLoad(LoadStoreModeEnum),
    Clc,
}

impl WarpRole {
    pub unsafe fn get<P: Params>() -> Self {
        let warp = unsafe { uniform_warp_id() } as usize;
        if warp < P::NUM_WORKER_WARPS {
            return WarpRole::Worker;
        }
        match warp - P::NUM_WORKER_WARPS {
            0 => WarpRole::TmaLoad(LoadStoreModeEnum::ModeMxN),
            1 => WarpRole::TmaLoad(LoadStoreModeEnum::Mode1xN),
            2 => WarpRole::TmaLoad(LoadStoreModeEnum::ModeMx1),
            3 => WarpRole::Clc,
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn do_setup<P: Params>(smem: &Smem<P>) {
    unsafe {
        assert_eq!(dynamic_smem_size() as usize, size_of::<Smem<P>>());
        assert_eq!(block_size() as usize, P::BLOCK_SIZE);
        assert_eq!(cluster_size() as usize, P::CLUSTER_SIZE);
        assert_eq!(cluster_dim().x as usize, P::CLUSTER_SIZE_M);
        assert_eq!(cluster_dim().y as usize, P::CLUSTER_SIZE_N);
        assert_eq!(grid_dim().y as usize, P::CLUSTER_SIZE_N);

        #[rustc_unroll(full)]
        for mode in LoadStoreModeEnum::ALL {
            let ring = smem.load_ring(mode);
            #[rustc_unroll(full)]
            for slot_idx in 0..ring.num_slots() {
                ring.produced[slot_idx].init(1);
                ring.consumed[slot_idx].init(P::NUM_WORKER_THREADS as u32);
            }
        }

        #[rustc_unroll(full)]
        for slot_idx in 0..2 {
            smem.bars.cta_reduction_produced[slot_idx].init(P::NUM_WORKER_THREADS as u32);
            smem.bars.cluster_row_reduction_produced[slot_idx].init(1);
            smem.bars.cluster_row_reduction_produced[slot_idx]
                .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                    (size_of::<f32>() * P::CLUSTER_SIZE_N * P::CTA_TM) as u32,
                );

            smem.bars.cluster_reshard_buffer_produced[slot_idx].init(1);
            smem.bars.cluster_reshard_buffer_produced[slot_idx]
                .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                    (size_of::<bf16>() * P::CLUSTER_RESHARD_BUFFER_ELEMENTS) as u32,
                );
            smem.bars.cluster_reshard_buffer_consumed[slot_idx].init(P::NUM_WORKER_THREADS as u32);
        }

        MBarrier::fence_init();
    }
}

#[inline(always)]
pub unsafe fn run<K: TileElementwiseKernel<Params = P>, P: Params>(mut kernel: K) {
    unsafe {
        const {
            assert!(P::NUM_WORKER_WARPS == 4 || P::NUM_WORKER_WARPS == 8);
            assert!(
                P::WORKER_REGISTERS as usize * P::NUM_WORKER_THREADS
                    + P::AUX_REGISTERS as usize * NUM_AUX_THREADS
                    <= 64 * 1024,
                "register allocation exceeds hardware capacity"
            );

            assert!(P::CLUSTER_SIZE_M > 0);
            assert!(P::CLUSTER_SIZE_N > 0);
            // Cluster row reductions use one lane of warp 0 per peer CTA.
            // Note that clusters >16 are not possible on hardware as of this writing.
            assert!(P::CLUSTER_SIZE_N <= 32);
            assert!(P::CGA_TM.is_multiple_of(P::CLUSTER_SIZE_M));
            assert!(P::CGA_TN.is_multiple_of(P::CLUSTER_SIZE_N));

            assert!(P::MACROBLOCK_SIZE > 0);

            #[cfg(has_227kb_smem)]
            assert!(
                size_of::<Smem<P>>() <= 227 * 1024,
                "SMEM requirement exceeds hardware capacity; try reducing load slots",
            );
            #[cfg(not(has_227kb_smem))]
            assert!(
                size_of::<Smem<P>>() <= 99 * 1024,
                "SMEM requirement exceeds hardware capacity; try reducing load slots"
            );

            assert!(P::CTA_TN >= P::NUM_WORKER_THREADS);
            assert!(P::CTA_TN.is_power_of_two());

            assert!(P::CTA_TM <= 256);
            assert!(P::CTA_TM.is_power_of_two());

            if P::DOES_STORE_COLWISE_BLOCK_SCALED_DATA {
                assert!(P::CGA_TM.is_multiple_of(Mxfp8::BLOCK_SIZE));
                // Resharded data must be evenly partitioned across worker threads.
                assert!(P::CLUSTER_RESHARDED_CTA_TN.is_multiple_of(P::NUM_WORKER_THREADS));
            }
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

        let role = WarpRole::get::<P>();
        match role {
            WarpRole::Worker => {
                increase_max_registers::<{ P::WORKER_REGISTERS }>();
                WorkerWarp {
                    smem,
                    sched,
                    state: kernel.init_worker_state(),
                    load_idxs: [0; 3],
                    row_reduction_idx: 0,
                    cluster_resharding_idx: 0,
                }
                .run(&mut kernel);
            }
            _ => {
                decrease_max_registers::<{ P::AUX_REGISTERS }>();
                match role {
                    WarpRole::Worker => unreachable!(),
                    WarpRole::TmaLoad(LoadStoreModeEnum::ModeMxN) => {
                        TmaLoadWarp::<K, P, { LoadStoreModeEnum::ModeMxN }> {
                            smem,
                            sched,
                            load_idx: 0,
                        }
                        .run(&mut kernel);
                    }
                    WarpRole::TmaLoad(LoadStoreModeEnum::Mode1xN) => {
                        TmaLoadWarp::<K, P, { LoadStoreModeEnum::Mode1xN }> {
                            smem,
                            sched,
                            load_idx: 0,
                        }
                        .run(&mut kernel);
                    }
                    WarpRole::TmaLoad(LoadStoreModeEnum::ModeMx1) => {
                        TmaLoadWarp::<K, P, { LoadStoreModeEnum::ModeMx1 }> {
                            smem,
                            sched,
                            load_idx: 0,
                        }
                        .run(&mut kernel);
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
