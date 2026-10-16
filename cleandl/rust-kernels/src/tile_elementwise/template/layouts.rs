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
//! tiled using Mx256 tiles to align with the TMA box size
//! this access pattern has zero bank conflicts. By contrast,
//! the naive partitioning (where thread `i` gets a contiguous range of elements `i*(N/128)..(i+1)*(N/128)`)
//! would produce bank conflicts when N>1024. Note that currently, the last tile in SMEM is always padded to 256 columns
//! except when TN<256.
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

use crate::tile_elementwise::template::{NUM_WORKER_THREADS, Params};
use core::{hint::select_unpredictable, marker::PhantomData};
use ptx_std::{bf16, bf16x8, f32x4, f32x8, lane_id};

pub struct Layouts<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> Layouts<P> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_8x_vectorized(&self) -> bool {
        P::THREAD_N > 0 && P::THREAD_N.is_multiple_of(8)
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
    /// with SMEM in bf16 format and registers in f32.
    /// `smem` points to the slot base; `regs` points to the row in registers.
    #[inline(always)]
    pub unsafe fn copy_bf16_row(
        &mut self,
        regs: *mut f32,
        smem: *mut bf16,
        row: usize,
        thread_id: u32,
        dir: CopyDir,
    ) {
        unsafe {
            if self.is_8x_vectorized() {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N_DIV_8 {
                    let n = thread_id as usize * 8 + i * 8 * NUM_WORKER_THREADS;

                    let smem_tile =
                        smem.add((n / P::TMA_PANEL_COLS) * P::CTA_TM * P::TMA_PANEL_COLS);
                    let smem_row = smem_tile.add(P::TMA_PANEL_COLS * row);

                    let offset = (n % P::TMA_PANEL_COLS) / 8;

                    match dir {
                        CopyDir::SmemToReg => {
                            let val = smem_row
                                .cast::<bf16x8>()
                                .add(offset)
                                .read()
                                .to_f32()
                                .to_array();
                            #[rustc_unroll(full)]
                            for j in 0..8 {
                                regs.add(i * 8 + j).write(val[j]);
                            }
                        }
                        CopyDir::RegToSmem => {
                            let val = regs.cast::<f32x8>().add(i).read_unaligned();
                            let val = bf16x8::from_f32(val);
                            smem_row.cast::<bf16x8>().add(offset).write(val);
                        }
                    }
                }
            } else {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N {
                    let n = thread_id as usize + i * NUM_WORKER_THREADS;

                    let smem_tile =
                        smem.add((n / P::TMA_PANEL_COLS) * P::CTA_TM * P::TMA_PANEL_COLS);
                    let smem_row = smem_tile.add(P::TMA_PANEL_COLS * row);

                    let offset = n % P::TMA_PANEL_COLS;

                    match dir {
                        CopyDir::SmemToReg => {
                            let val = smem_row.add(offset).read().to_f32();
                            regs.add(i).write(val);
                        }
                        CopyDir::RegToSmem => {
                            let val = bf16::from_f32(regs.add(i).read());
                            smem_row.add(offset).write(val);
                        }
                    }
                }
            }
        }
    }

    /// Copies one row-slice of elements between SMEM and this thread's registers,
    /// with SMEM and registers in f32.
    /// `smem` points to the slot base; `regs` points to the row in registers.
    #[inline(always)]
    pub unsafe fn copy_f32_row(
        &mut self,
        regs: *mut f32,
        smem: *mut f32,
        row: usize,
        thread_id: u32,
        dir: CopyDir,
    ) {
        unsafe {
            if self.is_8x_vectorized() {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N_DIV_8 {
                    let n = thread_id as usize * 8 + i * 8 * NUM_WORKER_THREADS;

                    let smem_tile = smem
                        .add((n / P::TMA_PANEL_COLS) * P::CTA_TM * P::CTA_TM * P::TMA_PANEL_COLS);
                    let smem_row = smem_tile.add(P::TMA_PANEL_COLS * row);

                    let offset = (n % P::TMA_PANEL_COLS) / 4;

                    let smem0 = smem_row.cast::<f32x4>().add(offset);
                    let smem1 = smem_row.cast::<f32x4>().add(offset + 1);

                    let lane_parity = ((lane_id() % 8) / 4) == 1;
                    let (first_smem, second_smem) =
                        select_unpredictable(lane_parity, (smem1, smem0), (smem0, smem1));

                    match dir {
                        CopyDir::SmemToReg => {
                            let first = first_smem.read();
                            let second = second_smem.read();

                            let (val0, val1) =
                                select_unpredictable(lane_parity, (second, first), (first, second));

                            regs.cast::<f32x4>().add(i * 2).write_unaligned(val0);
                            regs.cast::<f32x4>().add(i * 2 + 1).write_unaligned(val1);
                        }
                        CopyDir::RegToSmem => {
                            let val0 = regs.cast::<f32x4>().add(i * 2).read_unaligned();
                            let val1 = regs.cast::<f32x4>().add(i * 2 + 1).read_unaligned();

                            let (first, second) =
                                select_unpredictable(lane_parity, (val1, val0), (val0, val1));

                            first_smem.write(first);
                            second_smem.write(second);
                        }
                    }
                }
            } else {
                todo!()
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum CopyDir {
    SmemToReg,
    RegToSmem,
}

/// Iterates over each TMA panel, its offset along the N dimension,
/// and its offset (in elements) within an SMEM slot.
#[inline(always)]
pub fn iter_tma_panels<P: Params>() -> impl Iterator<Item = (usize, usize)> {
    (0..P::NUM_TMA_PANELS).map(|idx| (idx * P::TMA_PANEL_COLS, idx * P::TMA_PANEL_COLS * P::CTA_TM))
}
