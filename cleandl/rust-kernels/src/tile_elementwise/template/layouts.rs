//! Helpers for loading from and storing to the shared memory layouts
//! used by the tile_elementwise template in various situations, to/from
//! the corresponding optimized thread partitioning layout.
//!
//! There are two concepts:
//! - Thread partitioning layout: assignment of columns within the tile to worker threads. Determined based on TN.
//! - SMEM layout: placement of elements for loads/stores in SMEM. Depends on element type size in memory.
//!
//! Currently, the kernel statically selects one of two layout classes based on the value of TN:
//!
//! # Vectorized layout
//! Selected when `TN.div_ceil(NUM_WORKER_THREADS)` is a multiple of 8. This allows elements
//! to be loaded/stored using 128-bit instructions on `bf16x8` vectors, where `bf16` type is used.
//! For other types, there are extra considerations (see below).
//!
//! Elements are partitioned across threads along the N dimension according to this pattern:
//!
//! ```
//! N      | 0..8 | 8..16 | 16..24 | ... | 1016..1024 | 1024..1032 | 1032..1040 | etc.
//! Thread | 0    | 1     | 2      | ... | 127        | 0          | 1          | etc.
//! ```
//! For bf16x8 loads and stores in SMEM, the SMEM layout is unswizzled and row-major,
//! tiled using Mx256 tiles (1x256 for 1xN row slices) to align with the TMA box size
//! this access pattern has zero bank conflicts. By contrast,
//! the naive partitioning (where thread `i` gets a contiguous range of elements `i*(N/128)..(i+1)*(N/128)`)
//! would produce bank conflicts when N>1024.
//!
//! For 8-bit elements, we use the same SMEM layout as bf16, accessing it with 64-bit loads/stores.
//!
//! For f32, we cannot use f32x8 as the load/store type, because 256-bit SMEM operations do not exist.
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
//! For when the vectorized layout cannot be used.
//!
//! Partitioning pattern:
//! ```
//! N      | 0  | 1 | 2 | ... | 127 | 128 | 129 | etc.
//! Thread | 0  | 1 | 2 | ... | 127 | 0   | 1   | etc.
//! ```
//!
//! For SMEM layouts, we use the same layouts as the vectorized variants,
//! so that host-side TMA encoding code is unaffected
//! bf16 operations are done
//! using 16-bit accesses, f32 with 32-bit accesses, and f8 with 8-bit accesses,
//! i.e. all SMEM operations are unvectorized.

use crate::{
    tile_elementwise::template::{NUM_WORKER_THREADS, Params},
    utils::block_scale::BlockScaledFormat,
};
use core::{
    hint::select_unpredictable,
    marker::PhantomData,
    simd::{u32x2, u32x4},
};
use ptx_std::{
    F32Ext, f32xN, lane_id,
    warp_collectives::{LaneMask, shuffle_xor},
};

pub struct Layouts<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> Layouts<P> {
    /// How many row-contiguous elements within a lane's registers?
    pub const VECTOR_SIZE: usize = const { Layouts::<P>::new().vector_size() };
    pub const VECTORS_PER_THREAD_PER_ROW: usize = const { P::THREAD_N / Self::VECTOR_SIZE };

    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn is_8x_vectorized(&self) -> bool {
        P::THREAD_N > 0 && P::THREAD_N.is_multiple_of(8)
    }

    #[inline(always)]
    pub const fn vector_size(&self) -> usize {
        if self.is_8x_vectorized() { 8 } else { 1 }
    }

    /// Maps a tuple (thread index, element index within a row for a thread) to its logical N-index.
    #[inline(always)]
    pub fn thread_element_n_index(&self, thread_id: u32, element_index: usize) -> usize {
        if self.is_8x_vectorized() {
            (element_index / 8) * NUM_WORKER_THREADS * 8
                + thread_id as usize * 8
                + element_index % 8
        } else {
            element_index * NUM_WORKER_THREADS + thread_id as usize
        }
    }

    /// Copies one row-slice of elements between SMEM and this thread's registers,
    /// with both SMEM and registers holding `T`.
    /// `smem` points to the slot base; `regs` points to the row in registers.
    #[inline(always)]
    pub unsafe fn copy_row<T: Copy>(
        &mut self,
        regs: *mut T,
        smem: *mut T,
        row: usize,
        thread_id: u32,
        panel_rows: usize,
        dir: CopyDir,
    ) {
        const {
            assert!(size_of::<T>().is_power_of_two());
            assert!(size_of::<T>() <= 4);
        }

        unsafe {
            if self.is_8x_vectorized() {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N_DIV_8 {
                    let n = thread_id as usize * 8 + i * 8 * NUM_WORKER_THREADS;

                    let smem_tile =
                        smem.add((n / P::TMA_PANEL_COLS) * panel_rows * P::TMA_PANEL_COLS);
                    let smem_row = smem_tile.add(P::TMA_PANEL_COLS * row);

                    let smem_chunk = smem_row.add(n % P::TMA_PANEL_COLS);
                    let regs_chunk = regs.add(i * 8);

                    match size_of::<T>() {
                        1 => copy_unit::<u32x2>(regs_chunk.cast(), smem_chunk.cast(), dir),
                        2 => copy_unit::<u32x4>(regs_chunk.cast(), smem_chunk.cast(), dir),
                        4 => {
                            let smem0 = smem_chunk.cast::<u32x4>();
                            let smem1 = smem0.add(1);

                            let lane_parity = ((lane_id() % 8) / 4) == 1;
                            let (first_smem, second_smem) =
                                select_unpredictable(lane_parity, (smem1, smem0), (smem0, smem1));

                            let regs = regs_chunk.cast::<u32x4>();

                            match dir {
                                CopyDir::SmemToReg => {
                                    let first = first_smem.read();
                                    let second = second_smem.read();

                                    let (val0, val1) = select_unpredictable(
                                        lane_parity,
                                        (second, first),
                                        (first, second),
                                    );

                                    regs.write_unaligned(val0);
                                    regs.add(1).write_unaligned(val1);
                                }
                                CopyDir::RegToSmem => {
                                    let val0 = regs.read_unaligned();
                                    let val1 = regs.add(1).read_unaligned();

                                    let (first, second) = select_unpredictable(
                                        lane_parity,
                                        (val1, val0),
                                        (val0, val1),
                                    );

                                    first_smem.write(first);
                                    second_smem.write(second);
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            } else {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N {
                    let n = thread_id as usize + i * NUM_WORKER_THREADS;

                    let smem_tile =
                        smem.add((n / P::TMA_PANEL_COLS) * panel_rows * P::TMA_PANEL_COLS);
                    let smem_row = smem_tile.add(P::TMA_PANEL_COLS * row);

                    let offset = n % P::TMA_PANEL_COLS;

                    copy_unit(regs.add(i), smem_row.add(offset), dir);
                }
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
                        for i in 0..reduction_group_size.ilog2() {
                            block_amax =
                                block_amax.max(shuffle_xor(LaneMask::FULL, block_amax, 1 << i));
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

#[derive(Copy, Clone)]
pub enum CopyDir {
    SmemToReg,
    RegToSmem,
}

/// Copies a single value between an aligned SMEM location and a (possibly unaligned) register location.
#[inline(always)]
unsafe fn copy_unit<U: Copy>(regs: *mut U, smem: *mut U, dir: CopyDir) {
    unsafe {
        match dir {
            CopyDir::SmemToReg => regs.write_unaligned(smem.read()),
            CopyDir::RegToSmem => smem.write(regs.read_unaligned()),
        }
    }
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
