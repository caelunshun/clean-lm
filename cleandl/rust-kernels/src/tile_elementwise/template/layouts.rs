//! Helpers for loading from and storing to the shared memory layouts
//! used by the tile_elementwise template in various situations, to/from
//! the corresponding optimized thread partitioning layout.
//!
//! There are two concepts:
//! - Thread partitioning layout: assignment of columns within the tile to worker threads. Determined based on TN.
//! - SMEM layout: placement of elements for loads/stores in SMEM. Depends on element type size in memory.
//!
//! Currently, the kernel statically selects one of two layout classes based on the value of TM:
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
//! For bf16x8 loads and stores in SMEM, the SMEM layout is unswizzled and row-major;
//! this access pattern has zero bank conflicts. By contrast,
//! the naive partitioning (where thread `i` gets a contiguous range of elements `i*(N/128)..(i+1)*(N/128)`)
//! would produce bank conflicts when N>1024.
//!
//! For 8-bit elements, we use the same SMEM layout as bf16, accessing it with 64-bit loads/stores.
//!
//! For f32, we cannot use f32x8 as the load/store type, because 256-bit SMEM operations do not exist.
//! We must use two back-to-back f32x4 operations instead.
//! However, doing such f32x4 loads/stores on the unswizzled SMEM layout would
//! cause a two-way bank conflict on every access, due to the stride across lanes.
//! The solution is to use the 32B TMA swizzling pattern in SMEM:
//!
//! ```
//! Logical N-index   | 0..4 | 4..8 | 8..12  | 12..16 | 16..20 | 20..24 | 24..28 | 28..32 | 32..36 | 36..40 | 40..44 | 44..48 | 48..52 | 52..56 | 56..60 | 60..64 | etc.
//! SMEM N-index      | 0..4 | 4..8 | 8..12  | 12..16 | 16..20 | 20..24 | 24..28 | 28..32 | 36..40 | 32..36 | 44..48 | 40..44 | 52..56 | 48..52 | 60..64 | 56..60 | etc.
//! f32x4-in-pair     | 0    | 1    | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | etc.
//! Accessing lane    | 0    | 0    | 1      | 1      | 2      | 2      | 3      | 3      | 4      | 4      | 5      | 5      | 6      | 6      | 7      | 7      | etc.
//! ```
//!
//! Each warp issues two operations: one for each lane's f32x4-in-pair-0, then one for each lane's f32x4-in-pair-1.
//! The hardware splits our 32x128-bit memory operation for each of the two f32x4s
//! into four 8x128-bit SMEM wavefronts.
//! The swizzle pattern ensures that each wavefront accesses all 32 banks.
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
//! so that host-side TMA encoding code is unaffected. Rows are padded to 16 bytes for TMA compatability.
//! bf16 operations are done
//! using 16-bit accesses, f32 with 32-bit accesses, and f8 with 8-bit accesses,
//! i.e. all SMEM operations are unvectorized. For consistency, f32 keeps 32B swizzling as in the vectorized
//! layout, although the swizzle is not technically necessary.

use crate::tile_elementwise::template::{NUM_WORKER_THREADS, Params};
use core::marker::PhantomData;
use ptx_std::{bf16, bf16x8, f32x4, f32x8};

pub struct Layouts<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> Layouts<P> {
    #[inline(always)]
    pub const fn new() -> Self {
        const {
            assert!(P::TN >= 128);
            assert!(P::TN.is_power_of_two());
            assert!(P::TM.is_power_of_two());
        }

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
    /// `smem` and `regs` point to the start of the row in the corresponding
    /// address spaces.
    #[inline(always)]
    pub unsafe fn copy_bf16_row(
        &mut self,
        regs: *mut f32,
        smem: *mut bf16,
        thread_id: u32,
        dir: CopyDir,
    ) {
        unsafe {
            if self.is_8x_vectorized() {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N_DIV_8 {
                    match dir {
                        CopyDir::SmemToReg => {
                            let val = smem
                                .cast::<bf16x8>()
                                .add(thread_id as usize + i * NUM_WORKER_THREADS)
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
                            smem.cast::<bf16x8>()
                                .add(thread_id as usize + i * NUM_WORKER_THREADS)
                                .write(val);
                        }
                    }
                }
            } else {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N {
                    match dir {
                        CopyDir::SmemToReg => {
                            let val = smem
                                .add(thread_id as usize + i * NUM_WORKER_THREADS)
                                .read()
                                .to_f32();
                            regs.add(i).write(val);
                        }
                        CopyDir::RegToSmem => {
                            let val = bf16::from_f32(regs.add(i).read());
                            smem.add(thread_id as usize + i * NUM_WORKER_THREADS)
                                .write(val);
                        }
                    }
                }
            }
        }
    }

    /// Copies one row-slice of elements between SMEM and this thread's registers,
    /// with SMEM and registers in f32.
    /// `smem` and `regs` point to the start of the row in the corresponding
    /// address spaces.
    #[inline(always)]
    pub unsafe fn copy_f32_row(
        &mut self,
        regs: *mut f32,
        smem: *mut f32,
        thread_id: u32,
        dir: CopyDir,
    ) {
        unsafe {
            if self.is_8x_vectorized() {
                #[rustc_unroll(full)]
                for i in 0..P::THREAD_N_DIV_8 {
                    #[rustc_unroll(full)]
                    for part in 0..2 {
                        let offset = thread_id as usize * 2 + i * 2 * NUM_WORKER_THREADS + part;
                        let offset_swizzled = offset ^ ((offset % 16) / 8);
                        let smem_addr = smem.cast::<f32x4>().add(offset_swizzled);
                        match dir {
                            CopyDir::SmemToReg => {
                                let val = smem_addr.read().to_array();
                                #[rustc_unroll(full)]
                                for j in 0..4 {
                                    regs.add(i * 8 + part * 4 + j).write(val[j]);
                                }
                            }
                            CopyDir::RegToSmem => {
                                let val = regs.cast::<f32x4>().add(i * 2).read_unaligned();
                                smem_addr.write(val);
                            }
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
