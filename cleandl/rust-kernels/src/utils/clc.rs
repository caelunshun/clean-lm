use core::{cell::UnsafeCell, mem::MaybeUninit};
use ptx_std::{
    Dim3, MBarrier, MemoryOrdering, PhaseParity, Scope, block_id, block_id_in_cluster,
    block_rank_in_cluster,
    cluster::map_to_cta,
    cluster_size,
    fence::fence_proxy_async_generic_sync_restrict,
    launch::{CancelDelivery, CancelResponse, try_cancel_launch},
    num_warps,
    warp_collectives::{LaneMask, elect, sync_warp, vote_any},
    warp_id,
};

/// Utility for implementing pipelined Cluster Launch Control (CLC) for a kernel.
///
/// One warp per CTA is assigned to CLC scheduling duty and must call `Self::run_scheduler()`.
/// All other warps in the cluster must call `get_next_block()` to get the current block index
/// in a loop, until it returns `None`.
#[derive(Debug)]
pub struct ClcScheduler {
    has_returned_initial_block: bool,
    smem: *mut ClcSmem,
    idx: u32,
}

const STAGES: usize = 3;

/// Shared memory data required by ClcScheduler.
#[derive(Debug)]
#[repr(C)]
pub struct ClcSmem {
    produced: [MBarrier; STAGES],
    consumed: [MBarrier; STAGES],
    responses: [UnsafeCell<MaybeUninit<CancelResponse>>; STAGES],
}

impl ClcScheduler {
    /// # Safety
    /// Must be called by all threads in every CTA.
    /// Additionally, threads in the cluster must synchronize
    /// before calling any methods on this object.
    #[inline(always)]
    pub unsafe fn new(smem: *mut ClcSmem) -> Self {
        unsafe {
            if warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                #[rustc_unroll(full)]
                for i in 0..STAGES {
                    (&*smem).produced[i].init(1);
                    (&*smem).consumed[i].init((num_warps() - 1) * cluster_size()); // scheduler warps don't arrive on this barrier
                }

                MBarrier::fence_init();
            }
        }

        Self {
            has_returned_initial_block: false,
            smem,
            idx: 0,
        }
    }

    #[inline(always)]
    pub unsafe fn run_scheduler(&mut self) {
        if block_rank_in_cluster() != 0 {
            return;
        }
        unsafe {
            loop {
                // Wait for slot to be consumed
                if self.idx as usize >= STAGES {
                    let consumed = &(&*self.smem).consumed[self.idx as usize % STAGES];
                    if elect(LaneMask::FULL).is_leader {
                        while !consumed
                            .try_wait_scoped::<{ Scope::Cta }, { MemoryOrdering::Acquire }>(
                                PhaseParity::of_phase(self.idx / STAGES as u32 - 1),
                            )
                        {}
                    }
                }

                let response_ptr = (&*self.smem).responses[self.idx as usize % STAGES].get();

                let mut done = false;
                if elect(LaneMask::FULL).is_leader {
                    let ready = &(&*self.smem).produced[self.idx as usize % STAGES];
                    ready
                        .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                            size_of::<CancelDelivery>() as u32,
                        );
                    try_cancel_launch::<{ CancelDelivery::Cluster }>(response_ptr.cast(), ready);

                    // Observe for ourselves whether we should continue the loop.
                    while !ready.try_wait(PhaseParity::of_phase(self.idx / STAGES as u32)) {}

                    let response = response_ptr.read().assume_init();
                    if !response.is_canceled() {
                        done = true;
                    }
                }

                if vote_any(LaneMask::FULL, done) {
                    break;
                }

                self.idx += 1;
            }
        }
    }

    /// Must be called by every warp with all lanes active.
    /// Must not be called again by the same warp after returning `None`.
    #[inline(always)]
    pub unsafe fn get_next_block(&mut self) -> Option<Dim3> {
        if !self.has_returned_initial_block {
            self.has_returned_initial_block = true;
            return Some(block_id());
        }

        unsafe {
            let ready = &(&*self.smem).produced[self.idx as usize % STAGES];
            let parity = PhaseParity::of_phase(self.idx / STAGES as u32);
            if elect(LaneMask::FULL).is_leader {
                while !ready
                    .try_wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Acquire }>(parity)
                {
                }
            }
            sync_warp(LaneMask::FULL);

            let canceled_first_cta_id = (&*self.smem).responses[self.idx as usize % STAGES]
                .get()
                .read()
                .assume_init()
                .first_cta_id();

            fence_proxy_async_generic_sync_restrict::<{ MemoryOrdering::Release }>();
            if elect(LaneMask::FULL).is_leader {
                let consumed = &(&*self.smem).consumed[self.idx as usize % STAGES];
                let consumed = &*map_to_cta(consumed as *const MBarrier as *mut MBarrier, 0);
                consumed.arrive_scoped::<{ Scope::Cluster }, { MemoryOrdering::Release }>(1);
            }

            self.idx += 1;
            canceled_first_cta_id.map(|d3| Dim3 {
                x: d3.x + block_id_in_cluster().x,
                y: d3.y + block_id_in_cluster().y,
                z: d3.z + block_id_in_cluster().z,
            })
        }
    }
}
