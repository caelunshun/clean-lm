//! Helpers for loading from and storing to the shared memory layouts
//! used by the tile_elementwise template in various situations, to/from
//! the corresponding optimized thread partitioning layout.
//!
//! There are two concepts:
//! - Thread partitioning layout: assignment of columns within the tile to worker threads. Determined based on vector size.
//! - SMEM layout: placement of elements for loads/stores in SMEM. Depends on element type size in memory.
//!
//! The kernel statically selects a thread partitioning layout based on `THREAD_N = TN.div_ceil(NUM_WORKER_THREADS)`.
//!
//! # Vectorized layouts
//! Elements are partitioned across threads along the N dimension in chunks of `VECTOR_SIZE` contiguous elements,
//! where `VECTOR_SIZE` is the largest of 8, 4 and 2 dividing `THREAD_N`. This allows each chunk to be loaded/stored
//! using a single vector instruction (e.g. 128-bit instructions on `bf16x8` vectors when `VECTOR_SIZE` is 8 and
//! the `bf16` type is used). For f32, there are extra considerations (see below).
//!
//! For example, with `VECTOR_SIZE` 8, elements are partitioned according to this pattern:
//!
//! ```
//! N      | 0..8 | 8..16 | 16..24 | ... | 1016..1024 | 1024..1032 | 1032..1040 | etc.
//! Thread | 0    | 1     | 2      | ... | 127        | 0          | 1          | etc.
//! ```
//! The SMEM layout is unswizzled and row-major, tiled using Mx256 tiles (1x256 for 1xN row slices)
//! to align with the TMA box size. Since each warp accesses `32 * VECTOR_SIZE <= 256` contiguous elements,
//! this access pattern has zero bank conflicts. By contrast,
//! the naive partitioning (where thread `i` gets a contiguous range of elements `i*(N/128)..(i+1)*(N/128)`)
//! would produce bank conflicts when N>1024.
//!
//! Each chunk is accessed in SMEM using a single `VECTOR_SIZE * size_of::<T>()`-byte load/store
//! (e.g. 64-bit for 8-bit elements when `VECTOR_SIZE` is 8), with one exception:
//! for f32 with `VECTOR_SIZE` 8, we cannot use f32x8 as the load/store type, because 256-bit SMEM operations do not exist.
//! We must use two back-to-back f32x4 operations instead.
//! However, doing such f32x4 loads/stores naively would create a 2-way bank conflict on every access,
//! due to the stride between threads. Our solution is to alternate which f32x4 component is accessed
//! by each lane:
//!
//! ```
//! tid % 8                     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
//! First accessed f32x4-index  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1
//! Second accessed f32x4-index | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0
//! ```
//!
//! We accomplish this through a register select, making it relatively cheap.
//!
//! # Fallback layout
//! For when `THREAD_N` is odd (`VECTOR_SIZE` 1). This is the same pattern with single-element chunks:
//! ```
//! N      | 0  | 1 | 2 | ... | 127 | 128 | 129 | etc.
//! Thread | 0  | 1 | 2 | ... | 127 | 0   | 1   | etc.
//! ```
//!
//! SMEM layouts are the same as for the vectorized layouts, so that host-side TMA encoding code is unaffected.
//! All SMEM operations are unvectorized.
//!
//! # GMEM stores
//! Worker threads store directly from registers to GMEM using the same thread partitioning.
//! Each chunk is stored with a single store of up to 256 bits (these only exist for GMEM, so the f32 select trick
//! is not needed). If any lane of a warp would store out-of-bounds or misaligned chunks, the whole warp instead uses
//! bounds-checked elementwise stores.

use crate::{
    tile_elementwise::template::{NUM_WORKER_THREADS, Params, StoreReduceOp},
    utils::block_scale::BlockScaledFormat,
};
use core::{
    hint::select_unpredictable,
    marker::PhantomData,
    simd::{u32x2, u32x4, u32x8},
};
use ptx_std::{
    F32Ext, MemoryOrdering, Scope,
    atomic::{AtomicAdd, AtomicSpace, red_add, red_add_with_cache_hint},
    bf16, bf16x2, bf16x4, bf16x8, f32x2, f32x4, f32xN, lane_id,
    memory::{CachePolicy, GlobalStoreValue, store_global},
    select,
    warp_collectives::{LaneMask, shuffle_xor, vote_any},
};

pub struct Partitioning<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> Partitioning<P> {
    /// How many row-contiguous elements within a lane's registers?
    pub const VECTOR_SIZE: usize = const { vector_size_for(P::THREAD_N) };
    pub const VECTORS_PER_THREAD_PER_ROW: usize = const { P::THREAD_N / Self::VECTOR_SIZE };

    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Maps a tuple (thread index, element index within a row for a thread) to its logical N-index.
    #[inline(always)]
    pub fn thread_element_n_index(&self, thread_id: u32, element_index: usize) -> usize {
        interleaved_n_index(thread_id as usize, element_index, Self::VECTOR_SIZE)
    }

    /// Copies one row-slice of elements between memory and this thread's registers,
    /// with both memory and registers holding `T`. `regs` points to the row in registers.
    #[inline(always)]
    pub unsafe fn copy_row<T: Copy>(&self, regs: *mut T, thread_id: u32, dir: CopyDir<T>) {
        const {
            assert!(size_of::<T>().is_power_of_two());
            assert!(size_of::<T>() <= 4);
        }

        unsafe {
            let (smem, to_smem) = match dir {
                CopyDir::SmemToReg(smem) => (smem, false),
                CopyDir::RegToSmem(smem) => (smem, true),
                CopyDir::RegToGmem { dst, predicated } => {
                    self.copy_row_to_gmem(regs, thread_id, dst, predicated);
                    return;
                }
            };

            #[rustc_unroll(full)]
            for i in 0..Self::VECTORS_PER_THREAD_PER_ROW {
                let element_idx = i * Self::VECTOR_SIZE;
                let n = self.thread_element_n_index(thread_id, element_idx);

                let smem_chunk = smem
                    .slot
                    .add(smem_offset::<P>(n, smem.row, smem.panel_rows));
                let regs_chunk = regs.add(element_idx);

                match Self::VECTOR_SIZE * size_of::<T>() {
                    1 => copy_unit::<u8>(regs_chunk.cast(), smem_chunk.cast(), to_smem),
                    2 => copy_unit::<u16>(regs_chunk.cast(), smem_chunk.cast(), to_smem),
                    4 => copy_unit::<u32>(regs_chunk.cast(), smem_chunk.cast(), to_smem),
                    8 => copy_unit::<u32x2>(regs_chunk.cast(), smem_chunk.cast(), to_smem),
                    16 => copy_unit::<u32x4>(regs_chunk.cast(), smem_chunk.cast(), to_smem),
                    32 => {
                        let smem0 = smem_chunk.cast::<u32x4>();
                        let smem1 = smem0.add(1);

                        let lane_parity = ((lane_id() % 8) / 4) == 1;
                        let (first_smem, second_smem) =
                            select_unpredictable(lane_parity, (smem1, smem0), (smem0, smem1));

                        let regs = regs_chunk.cast::<u32x4>();

                        if to_smem {
                            let val0 = regs.read_unaligned();
                            let val1 = regs.add(1).read_unaligned();

                            let [first, second] = select(lane_parity, [val1, val0], [val0, val1]);

                            first_smem.write(first);
                            second_smem.write(second);
                        } else {
                            let first = first_smem.read();
                            let second = second_smem.read();

                            let [val0, val1] =
                                select(lane_parity, [second, first], [first, second]);

                            regs.write_unaligned(val0);
                            regs.add(1).write_unaligned(val1);
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn copy_row_to_gmem<T: Copy>(
        &self,
        regs: *const T,
        thread_id: u32,
        dst: GmemRow<T>,
        predicated: bool,
    ) {
        unsafe {
            if !predicated {
                #[rustc_unroll(full)]
                for i in 0..Self::VECTORS_PER_THREAD_PER_ROW {
                    let element_idx = i * Self::VECTOR_SIZE;
                    let n = self.thread_element_n_index(thread_id, element_idx);
                    store_chunk_to_gmem::<T, { Self::VECTOR_SIZE }>(
                        regs.add(element_idx),
                        dst.start.add(n),
                        dst.cache_policy,
                        dst.reduction,
                    );
                }
            } else {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N {
                    let n = self.thread_element_n_index(thread_id, i);
                    if n < dst.valid_cols {
                        store_unit_to_gmem(
                            regs.add(i),
                            dst.start.add(n),
                            dst.cache_policy,
                            dst.reduction,
                        );
                    }
                }
            }
        }
    }

    /// Stores `R` rows of this thread's registers to a GMEM tile, skipping out-of-bounds elements.
    ///
    /// The warp uses vectorized stores only if every lane's chunks are in bounds and aligned.
    #[inline(always)]
    pub unsafe fn store_rows_to_gmem<T: Copy, const R: usize>(
        &self,
        regs: &[[T; P::THREAD_N]; R],
        thread_id: u32,
        dst: GmemTile<T>,
    ) {
        unsafe {
            let aligned = dst
                .start
                .addr()
                .is_multiple_of(Self::VECTOR_SIZE * size_of::<T>())
                && (R == 1 || dst.row_stride.is_multiple_of(Self::VECTOR_SIZE));
            let in_bounds = R <= dst.valid_rows
                && self.thread_element_n_index(thread_id, P::THREAD_N - 1) < dst.valid_cols;
            let predicated =
                P::ENABLE_BOUNDS_CHECKS && vote_any(LaneMask::FULL, !(aligned && in_bounds));

            let store = #[inline(always)]
            |predicated: bool| {
                #[rustc_unroll(full)]
                for r in 0..R {
                    self.copy_row(
                        regs[r].as_ptr().cast_mut(),
                        thread_id,
                        CopyDir::RegToGmem {
                            dst: dst.row(r),
                            predicated,
                        },
                    );
                }
            };
            if predicated {
                store(true)
            } else {
                store(false)
            }
        }
    }

    /// Computes rowwise scale factors on worker threads for the given block-scaled format,
    /// based on the chosen thread partitioning. The amaxes are broadcast in the returned
    /// array to match the size of thr_vals; the compiler avoids duplicating the registers.
    #[inline(always)]
    pub unsafe fn compute_rowwise_thr_amaxs<BS: BlockScaledFormat>(
        &self,
        thr_vals: &[[f32; P::THREAD_N]; P::THREAD_M],
    ) -> [[f32; P::THREAD_N]; P::THREAD_M] {
        unsafe {
            let mut amaxs = [[0.0; P::THREAD_N]; P::THREAD_M];

            const {
                assert!(BS::BLOCK_SIZE <= 32);
                assert!(BS::BLOCK_SIZE >= Self::VECTOR_SIZE);
                assert!(BS::BLOCK_SIZE.is_power_of_two());
            }

            #[rustc_unroll(full)]
            for i in 0..P::THREAD_M {
                #[rustc_unroll(full)]
                for chunk_idx in 0..P::THREAD_N / Self::VECTOR_SIZE {
                    let chunk = f32xN::<{ Self::VECTOR_SIZE }>::from_array(
                        thr_vals[i][chunk_idx * Self::VECTOR_SIZE..][..Self::VECTOR_SIZE]
                            .try_into()
                            .unwrap(),
                    );
                    let chunk_amax = chunk.abs().reduce_max();

                    // Intra-warp reduction
                    let reduction_group_size = BS::BLOCK_SIZE / Self::VECTOR_SIZE;
                    let block_amax = if reduction_group_size == 32 {
                        warp_reduce_max_f32(chunk_amax)
                    } else {
                        let mut block_amax = chunk_amax;
                        #[rustc_unroll(full)]
                        for bit in 0..reduction_group_size.ilog2() {
                            block_amax =
                                block_amax.max(shuffle_xor(LaneMask::FULL, block_amax, 1 << bit));
                        }
                        block_amax
                    };

                    #[rustc_unroll(full)]
                    for j in 0..Self::VECTOR_SIZE {
                        amaxs[i][chunk_idx * Self::VECTOR_SIZE + j] = block_amax;
                    }
                }
            }

            amaxs
        }
    }
}

#[inline(always)]
#[cfg(target_feature = "sm_100f")]
unsafe fn warp_reduce_max_f32(val: f32) -> f32 {
    use ptx_std::warp_collectives::{LaneMask, reduce_f32};

    unsafe {
        reduce_f32::<
            {
                use ptx_std::warp_collectives::ReduxOp;

                ReduxOp::Max
            },
            {
                use ptx_std::warp_collectives::Absolute;
                Absolute::Disabled
            },
            {
                use ptx_std::NanHandling;
                NanHandling::Ignore
            },
            _,
        >(LaneMask::FULL, val)
    }
}

#[inline(always)]
#[cfg(not(target_feature = "sm_100f"))]
unsafe fn warp_reduce_max_f32(mut val: f32) -> f32 {
    unsafe {
        use ptx_std::WARP_SIZE;

        for i in 0..WARP_SIZE.ilog2() {
            use ptx_std::warp_collectives::{LaneMask, shuffle_xor};

            val = val.max(shuffle_xor(LaneMask::FULL, val, 1 << i));
        }

        val
    }
}

/// Direction and memory location of a `Layouts::copy_row`.
#[derive(Copy, Clone)]
pub enum CopyDir<T> {
    SmemToReg(SmemRow<T>),
    RegToSmem(SmemRow<T>),
    RegToGmem {
        dst: GmemRow<T>,
        /// Use bounds-checked elementwise stores rather than vectorized ones,
        /// which require the whole row to be in bounds and aligned to `Layouts::VECTOR_SIZE` elements.
        predicated: bool,
    },
}

/// A row within an SMEM slot.
#[derive(Copy, Clone)]
pub struct SmemRow<T> {
    /// Base of the slot.
    pub slot: *mut T,
    pub row: usize,
    /// Rows per TMA panel (see `iter_tma_panels`).
    pub panel_rows: usize,
}

/// A row of a tile in GMEM.
#[derive(Copy, Clone)]
pub struct GmemRow<T> {
    /// Address of the element in the tile's first column. May be out of bounds (pedantically UB but not avoidable)
    pub start: *mut T,
    /// Number of in-bounds elements starting at `start`.
    pub valid_cols: usize,
    pub cache_policy: Option<CachePolicy>,
    /// Reductions interpret 16-bit elements as bf16 and 32-bit elements as f32.
    pub reduction: Option<StoreReduceOp>,
}

/// A row-major tile in GMEM.
#[derive(Copy, Clone)]
pub struct GmemTile<T> {
    /// Address of the tile's top-left element. May be out of bounds (pedantically UB but not avoidable)
    pub start: *mut T,
    /// Distance between rows in elements.
    pub row_stride: usize,
    /// Number of in-bounds rows/columns starting at `start`.
    pub valid_rows: usize,
    pub valid_cols: usize,
    pub cache_policy: Option<CachePolicy>,
    /// Reductions interpret 16-bit elements as bf16 and 32-bit elements as f32.
    pub reduction: Option<StoreReduceOp>,
}

impl<T> GmemTile<T> {
    #[inline(always)]
    pub fn row(&self, row: usize) -> GmemRow<T> {
        GmemRow {
            start: self.start.wrapping_add(row * self.row_stride),
            valid_cols: if row < self.valid_rows {
                self.valid_cols
            } else {
                0
            },
            cache_policy: self.cache_policy,
            reduction: self.reduction,
        }
    }
}

/// A single-row tile.
impl<T> From<GmemRow<T>> for GmemTile<T> {
    #[inline(always)]
    fn from(row: GmemRow<T>) -> Self {
        Self {
            start: row.start,
            row_stride: 0,
            valid_rows: 1,
            valid_cols: row.valid_cols,
            cache_policy: row.cache_policy,
            reduction: row.reduction,
        }
    }
}

/// Copies a single value between an aligned SMEM location and a (possibly unaligned) register location.
#[inline(always)]
unsafe fn copy_unit<U: Copy>(regs: *mut U, smem: *mut U, to_smem: bool) {
    unsafe {
        if to_smem {
            smem.write(regs.read_unaligned());
        } else {
            regs.write_unaligned(smem.read());
        }
    }
}

/// Stores a contiguous array from a single thread to a GMEM row,
/// using vectorized stores if the whole row is in bounds and aligned.
#[inline(always)]
pub unsafe fn store_contiguous_to_gmem<T: Copy, const N: usize>(vals: &[T; N], dst: GmemRow<T>) {
    unsafe {
        let vectorized = N.is_multiple_of(8)
            && N <= dst.valid_cols
            && dst.start.addr().is_multiple_of(8 * size_of::<T>());
        if vectorized {
            #[rustc_unroll(full)]
            for i in 0..N / 8 {
                store_chunk_to_gmem::<T, 8>(
                    vals.as_ptr().add(i * 8),
                    dst.start.add(i * 8),
                    dst.cache_policy,
                    dst.reduction,
                );
            }
        } else {
            #[rustc_unroll(full)]
            for i in 0..N {
                if i < dst.valid_cols {
                    store_unit_to_gmem(
                        vals.as_ptr().add(i),
                        dst.start.add(i),
                        dst.cache_policy,
                        dst.reduction,
                    );
                }
            }
        }
    }
}

/// Stores `N` contiguous elements from (possibly unaligned) registers to GMEM aligned to `N` elements.
#[inline(always)]
unsafe fn store_chunk_to_gmem<T: Copy, const N: usize>(
    regs: *const T,
    ptr: *mut T,
    cache_policy: Option<CachePolicy>,
    reduction: Option<StoreReduceOp>,
) {
    unsafe {
        match reduction {
            None => match N * size_of::<T>() {
                1 => store_chunk_units::<T, u8, N>(regs, ptr, cache_policy),
                2 => store_chunk_units::<T, u16, N>(regs, ptr, cache_policy),
                4 => store_chunk_units::<T, u32, N>(regs, ptr, cache_policy),
                8 => store_chunk_units::<T, u32x2, N>(regs, ptr, cache_policy),
                16 => store_chunk_units::<T, u32x4, N>(regs, ptr, cache_policy),
                32 => store_chunk_units::<T, u32x8, N>(regs, ptr, cache_policy),
                _ => unreachable!(),
            },
            Some(StoreReduceOp::Add) => match (size_of::<T>(), N) {
                (2, 1) => red_add_chunk_units::<T, bf16, N>(regs, ptr, cache_policy),
                (2, 2) => red_add_chunk_units::<T, bf16x2, N>(regs, ptr, cache_policy),
                (2, 4) => red_add_chunk_units::<T, bf16x4, N>(regs, ptr, cache_policy),
                (2, 8) => red_add_chunk_units::<T, bf16x8, N>(regs, ptr, cache_policy),
                (4, 1) => red_add_chunk_units::<T, f32, N>(regs, ptr, cache_policy),
                (4, 2) => red_add_chunk_units::<T, f32x2, N>(regs, ptr, cache_policy),
                (4, 4 | 8) => red_add_chunk_units::<T, f32x4, N>(regs, ptr, cache_policy),
                _ => unimplemented!("reductions only support bf16 and f32"),
            },
        }
    }
}

/// Stores a single element from registers to GMEM.
#[inline(always)]
pub unsafe fn store_unit_to_gmem<T: Copy>(
    regs: *const T,
    ptr: *mut T,
    cache_policy: Option<CachePolicy>,
    reduction: Option<StoreReduceOp>,
) {
    unsafe { store_chunk_to_gmem::<T, 1>(regs, ptr, cache_policy, reduction) }
}

/// Stores `N` contiguous elements as a sequence of `U`s.
#[inline(always)]
unsafe fn store_chunk_units<T, U: GlobalStoreValue, const N: usize>(
    regs: *const T,
    ptr: *mut T,
    cache_policy: Option<CachePolicy>,
) {
    unsafe {
        #[rustc_unroll(full)]
        for i in 0..N * size_of::<T>() / size_of::<U>() {
            store_global(
                ptr.cast::<U>().add(i),
                regs.cast::<U>().add(i).read_unaligned(),
                cache_policy,
            );
        }
    }
}

/// Reduces `N` contiguous elements into GMEM as a sequence of `U`s.
#[inline(always)]
unsafe fn red_add_chunk_units<T, U: AtomicAdd, const N: usize>(
    regs: *const T,
    ptr: *mut T,
    cache_policy: Option<CachePolicy>,
) {
    unsafe {
        #[rustc_unroll(full)]
        for i in 0..N * size_of::<T>() / size_of::<U>() {
            red_add_global(
                ptr.cast::<U>().add(i),
                regs.cast::<U>().add(i).read_unaligned(),
                cache_policy,
            );
        }
    }
}

#[inline(always)]
unsafe fn red_add_global<T: AtomicAdd>(ptr: *mut T, val: T, cache_policy: Option<CachePolicy>) {
    unsafe {
        match cache_policy {
            None => {
                red_add::<T, { MemoryOrdering::Relaxed }, { Scope::Gpu }, { AtomicSpace::Global }>(
                    ptr, val,
                )
            }
            Some(cache_policy) => red_add_with_cache_hint::<
                T,
                { MemoryOrdering::Relaxed },
                { Scope::Gpu },
                { AtomicSpace::Global },
            >(ptr, val, cache_policy),
        }
    }
}

/// Vector size of the thread partitioning layout with `thread_n` elements per thread per row:
/// the largest of 8, 4, 2 and 1 dividing it.
pub const fn vector_size_for(thread_n: usize) -> usize {
    if thread_n.is_multiple_of(8) {
        8
    } else if thread_n.is_multiple_of(4) {
        4
    } else if thread_n.is_multiple_of(2) {
        2
    } else {
        1
    }
}

/// N-index of a thread's `element_index`th element within a row, in the thread partitioning layout
/// with the given vector size.
#[inline(always)]
pub fn interleaved_n_index(thread_id: usize, element_index: usize, vector_size: usize) -> usize {
    (element_index / vector_size) * NUM_WORKER_THREADS * vector_size
        + thread_id * vector_size
        + element_index % vector_size
}

/// Offset (in elements) within an SMEM slot of the element at (`row`, `n`).
#[inline(always)]
fn smem_offset<P: Params>(n: usize, row: usize, panel_rows: usize) -> usize {
    (n / P::TMA_PANEL_COLS) * panel_rows * P::TMA_PANEL_COLS
        + P::TMA_PANEL_COLS * row
        + n % P::TMA_PANEL_COLS
}

/// Iterates over each TMA panel, its offset along the N dimension,
/// and its offset (in elements) within an SMEM slot.
/// `panel_rows` is the number of rows per panel (see `Layouts::new`).
#[inline(always)]
pub fn iter_tma_panels<P: Params>(panel_rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..P::NUM_TMA_PANELS).map(move |idx| {
        (
            idx * P::TMA_PANEL_COLS,
            idx * P::TMA_PANEL_COLS * panel_rows,
        )
    })
}
