//! Forward RMSNorm for row sizes that fit in shared memory.
//! Input is bf16. Output is either bf16 or MXFP8. In MXFP8 mode,
//! a rowwise block scaled output is always written; an optional additional columnwise
//! block scaled output can also be selected. The latter requires a cluster
//! size such that microblock_size * cluster_size == 32. If cluster_size > 1
//! then a global workspace of (num_clusters * cluster_size * row_size) bf16 elements
//! is required.
//!
//! CLC is used to form a persistent kernel.
//!
//! Each block of rows (corresponding to a thread block in the launch)
//! is split into microblocks which are streamed through the SM.
//!
//! The kernel uses 4 worker warps for doing math on the row elements. They keep `gamma`
//! stationary in registers across microblocks (and blocks if using CLC). The warps stream
//! input rows from SMEM, compute statistics, apply normalization, and then execute the configured
//! epilogue. Epilogues stage stores through SMEM to allow TMA.
//!
//! The remaining 4 warps (warps 4-7) are assigned as follows:
//! * warp 4: TMA loads
//! * warp 5: epilogue TMA stores
//! * warp 6: CLC scheduler loop
//! * remainder: no role

use crate::utils::clc::{ClcScheduler, ClcSmem};
use ptx_std::{MBarrier, WARP_SIZE, bf16, bf16x2, cluster_size, tma::TmaDescriptor};

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpRole {
    Worker,
    TmaLoad,
    TmaStore,
    Clc,
}

const NUM_WORKER_WARPS: usize = 4;

struct SmemData<const LOAD_SLOTS: usize> {
    load_slot_produced: [MBarrier; LOAD_SLOTS],
    load_slot_consumed: [MBarrier; LOAD_SLOTS],
    store_slot_produced: MBarrier,
    store_slot_consumed: MBarrier,
    cluster_col_amax_produced: MBarrier,
    cluster_col_amax_consumed: MBarrier,
    clc: ClcSmem,
}

struct Smem<
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const STORE_BITS_PER_ELEMENT: usize,
    const SF_BLOCK_SIZE: usize,
> {
    base: *mut u8,
}

impl<
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const STORE_BITS_PER_ELEMENT: usize,
    const SF_BLOCK_SIZE: usize,
> Smem<ROW_SIZE, MICROBLOCK_SIZE, LOAD_SLOTS, STORE_BITS_PER_ELEMENT, SF_BLOCK_SIZE>
{
    const LOAD_SLOT_BYTES: usize = size_of::<bf16>() * ROW_SIZE * MICROBLOCK_SIZE;
    const STORE_SLOT_BYTES: usize = ROW_SIZE * MICROBLOCK_SIZE * STORE_BITS_PER_ELEMENT / 8;
    const STORE_SLOT_SF_BYTES: usize = ROW_SIZE * MICROBLOCK_SIZE / SF_BLOCK_SIZE;
    const REDUCTION_SCRATCH_BYTES: usize = size_of::<f32>() * NUM_WORKER_WARPS * MICROBLOCK_SIZE;

    #[inline(always)]
    pub unsafe fn get_load_slot(&self, slot_idx: usize) -> *mut u8 {
        unsafe { self.base.add(slot_idx * LOAD_SLOTS) }
    }

    #[inline(always)]
    pub unsafe fn get_store_slot(&self) -> *mut u8 {
        unsafe { self.get_load_slot(LOAD_SLOTS) }
    }

    #[inline(always)]
    pub unsafe fn get_store_slot_sf(&self) -> *mut u8 {
        unsafe { self.get_store_slot().add(Self::STORE_SLOT_BYTES) }
    }

    #[inline(always)]
    pub unsafe fn get_reduction_scratch(&self) -> *mut u8 {
        unsafe { self.get_store_slot_sf().add(Self::STORE_SLOT_SF_BYTES) }
    }

    #[inline(always)]
    pub unsafe fn data(&self) -> &SmemData<LOAD_SLOTS> {
        unsafe {
            &*self
                .get_reduction_scratch()
                .add(Self::REDUCTION_SCRATCH_BYTES)
                .cast()
        }
    }
}

struct RmsnormFwd<
    'a,
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const BLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const STORE_BITS_PER_ELEMENT: usize,
    const SF_BLOCK_SIZE: usize,
    const ELEMENTS_PER_THREAD: usize,
    const BF16X2_PER_THREAD: usize,
> {
    smem: Smem<ROW_SIZE, MICROBLOCK_SIZE, LOAD_SLOTS, STORE_BITS_PER_ELEMENT, SF_BLOCK_SIZE>,
    cluster_reduction_scratch: *mut bf16,
    gamma: *const bf16,
    tma_x: &'a TmaDescriptor,
    tma_y: &'a TmaDescriptor,
    tma_y_sf: Option<&'a TmaDescriptor>,
    tma_y_colmajor: Option<&'a TmaDescriptor>,
    tma_y_colmajor_sf: Option<&'a TmaDescriptor>,
    num_rows: u32,
    actual_row_size: u32,
    actual_row_size_recip: f32,

    current_block: u32,
    current_microblock: u32,

    clc: ClcScheduler,
}

impl<
    'a,
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const BLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const STORE_BITS_PER_ELEMENT: usize,
    const SF_BLOCK_SIZE: usize,
    const ELEMENTS_PER_THREAD: usize,
    const BF16X2_PER_THREAD: usize,
>
    RmsnormFwd<
        'a,
        ROW_SIZE,
        MICROBLOCK_SIZE,
        BLOCK_SIZE,
        LOAD_SLOTS,
        STORE_BITS_PER_ELEMENT,
        SF_BLOCK_SIZE,
        ELEMENTS_PER_THREAD,
        BF16X2_PER_THREAD,
    >
{
    #[inline(always)]
    fn current_block_first_row(&self) -> u32 {
        self.current_block * BLOCK_SIZE as u32
    }

    #[inline(always)]
    fn current_microblock_first_row(&self) -> u32 {
        self.current_block_first_row() + self.current_microblock * MICROBLOCK_SIZE as u32
    }

    #[inline(always)]
    pub fn init_barriers(&self) {
        unsafe {
            #[rustc_unroll(full)]
            for slot_idx in 0..LOAD_SLOTS {
                self.smem.data().load_slot_produced[slot_idx].init(1);
                self.smem.data().load_slot_consumed[slot_idx].init(NUM_WORKER_WARPS as u32);
            }
            self.smem
                .data()
                .store_slot_produced
                .init(NUM_WORKER_WARPS as u32);
            self.smem.data().store_slot_consumed.init(1);
            self.smem
                .data()
                .cluster_col_amax_produced
                .init(cluster_size() * NUM_WORKER_WARPS as u32);
            self.smem
                .data()
                .cluster_col_amax_consumed
                .init(cluster_size() * NUM_WORKER_WARPS as u32);

            MBarrier::fence_init();
        }
    }

    #[inline(always)]
    fn next_block(&mut self) -> bool {
        // Go to next block via CLC, else done
        unsafe {
            match self.clc.get_next_block() {
                Some(block) => {
                    self.current_block = block.x * BLOCK_SIZE as u32;
                    self.current_microblock = 0;
                    true
                }
                None => false,
            }
        }
    }

    #[inline(always)]
    fn next_microblock(&mut self) -> bool {
        const {
            assert!(BLOCK_SIZE.is_multiple_of(MICROBLOCK_SIZE));
        }
        let num_microblocks = BLOCK_SIZE as u32 / MICROBLOCK_SIZE as u32;
        if self.current_microblock + 1 == num_microblocks {
            self.next_block()
        } else {
            self.current_microblock += 1;
            true
        }
    }

    #[inline(always)]
    pub fn run_worker_warps(&mut self, thr_gamma: &[bf16x2; BF16X2_PER_THREAD]) {
        unsafe {
            loop {
                if !self.next_microblock() {
                    break;
                }
            }
        }
    }
}
