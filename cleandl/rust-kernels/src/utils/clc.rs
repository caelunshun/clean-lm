use core::{cell::UnsafeCell, mem::MaybeUninit};
use ptx_std::{
    Dim3, MBarrier, MemoryOrdering, PhaseParity, Scope, WARP_SIZE, block_id, block_id_in_cluster,
    block_rank_in_cluster, block_size,
    cluster::map_to_cta,
    cluster_size,
    fence::{ProxyKind, fence_proxy, fence_sync_restrict},
    launch::{CancelDelivery, CancelResponse, try_cancel_launch},
    warp_collectives::{LaneMask, elect, sync_warp, vote_any},
    warp_id,
};

/// Utility for implementing pipelined Cluster Launch Control (CLC) for a kernel.
///
/// One warp per CTA is assigned to CLC scheduling duty and must call `Self::run_scheduler()`.
/// All other warps in the cluster must call `get_next_block()` to get the current block index
/// in a loop, until it returns `None`.
#[derive(Debug)]
pub struct ClcScheduler<const CLUSTER_ENABLED: bool> {
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

impl<const CLUSTER_ENABLED: bool> ClcScheduler<CLUSTER_ENABLED> {
    const SCOPE: Scope = if CLUSTER_ENABLED {
        Scope::Cluster
    } else {
        Scope::Cta
    };
    const DELIVERY: CancelDelivery = if CLUSTER_ENABLED {
        CancelDelivery::Cluster
    } else {
        CancelDelivery::Block
    };

    /// # Safety
    /// Must be called by all threads in every CTA.
    /// Additionally, threads in the cluster (if CLUSTER_ENABLED) or CTA (if not CLUSTER_ENABLED) must synchronize
    /// before calling any methods on this object.
    #[inline(always)]
    pub unsafe fn new(smem: *mut ClcSmem) -> Self {
        unsafe {
            let num_ctas = if CLUSTER_ENABLED { cluster_size() } else { 1 };
            if warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                #[rustc_unroll(full)]
                for i in 0..STAGES {
                    (&*smem).produced[i].init(1);
                    (&*smem).consumed[i].init((block_size() - WARP_SIZE) * num_ctas); // scheduler warps don't arrive on this barrier
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
        let is_leader_cta = block_rank_in_cluster() == 0;
        unsafe {
            loop {
                if is_leader_cta {
                    // Wait for slot to be consumed
                    if self.idx as usize >= STAGES {
                        let consumed = &(&*self.smem).consumed[self.idx as usize % STAGES];
                        if elect(LaneMask::FULL).is_leader {
                            while !consumed
                                .try_wait_scoped::<{ Self::SCOPE }, { MemoryOrdering::Relaxed }>(
                                    PhaseParity::of_phase(self.idx / STAGES as u32 - 1),
                                )
                            {}
                            fence_sync_restrict::<{ MemoryOrdering::Acquire }>();
                        }
                    }
                }

                let response_ptr = (&*self.smem).responses[self.idx as usize % STAGES].get();

                let mut done = false;
                if elect(LaneMask::FULL).is_leader {
                    let ready = &(&*self.smem).produced[self.idx as usize % STAGES];
                    ready.arrive_expect_tx_scoped::<{ Self::SCOPE }, { MemoryOrdering::Relaxed }>(
                        size_of::<CancelResponse>() as u32,
                    );
                    fence_proxy::<{ ProxyKind::AsyncSharedCta }>();
                    if is_leader_cta {
                        try_cancel_launch::<{ Self::DELIVERY }>(response_ptr.cast(), ready);
                    }

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

            while !ready.try_wait_scoped::<{ Self::SCOPE }, { MemoryOrdering::Relaxed }>(parity) {}
            fence_sync_restrict::<{ MemoryOrdering::Acquire }>();

            sync_warp(LaneMask::FULL);

            let canceled_first_cta_id = (&*self.smem).responses[self.idx as usize % STAGES]
                .get()
                .read()
                .assume_init()
                .first_cta_id();

            fence_proxy::<{ ProxyKind::AsyncSharedCta }>();

            let consumed = &(&*self.smem).consumed[self.idx as usize % STAGES];
            let consumed = &*map_to_cta(consumed as *const MBarrier as *mut MBarrier, 0);

            fence_sync_restrict::<{ MemoryOrdering::Release }>();
            consumed.arrive_scoped::<{ Self::SCOPE }, { MemoryOrdering::Relaxed }>(1);

            self.idx += 1;
            canceled_first_cta_id.map(|d3| Dim3 {
                x: d3.x + block_id_in_cluster().x,
                y: d3.y + block_id_in_cluster().y,
                z: d3.z + block_id_in_cluster().z,
            })
        }
    }
}
