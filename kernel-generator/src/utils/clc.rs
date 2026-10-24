use crate::utils::parity::parity_from_idx;
use nvvm_builder::{
    Ptr, Shared, SharedCarve, Val, Var, block_dim_x, block_dim_y, block_dim_z, clc,
    cluster_block_idx_x, cluster_block_idx_y, cluster_block_idx_z, cluster_block_rank,
    cluster_dim_x, cluster_dim_y, cluster_dim_z, fence_proxy_async_generic_acquire,
    fence_proxy_async_generic_release, fence_sync_restrict_acquire, if_, if_else, loop_,
    mbarrier::{self, MbarrierArray},
    select, shared, with_leader, with_warp,
};

/// Utility for running a Cluster Launch Control (CLC) work-stealing
/// pipeline. One warp per CTA is assigned to CLC scheduling duty.
/// Every other thread must loop on `Self::next_block_idx` until
/// it indicates there is no more work.
pub struct ClcScheduler {
    response_bufs: Val<Ptr<u32, Shared>>,
    response_produced_bars: MbarrierArray,
    response_consumed_bars: MbarrierArray,
    num_stages: usize,
    leader_warp_id: u32,
    i: Var<u32>,
}

const CLC_RESPONSE_WORDS: usize = 4;

impl ClcScheduler {
    pub fn init(num_stages: usize, leader_warp_id: u32, _smem: &mut SharedCarve) -> Self {
        let response_bufs = shared::<u32>(
            "clc_response",
            num_stages * CLC_RESPONSE_WORDS,
            CLC_RESPONSE_WORDS * size_of::<u32>(),
        );
        let response_produced_bars = MbarrierArray::shared("clc_produced", num_stages);
        let response_consumed_bars = MbarrierArray::shared("clc_consumed", num_stages);

        with_warp(leader_warp_id, || {
            with_leader(|| {
                response_produced_bars.init_all(1);
                response_consumed_bars.init_all(
                    cluster_dim_x()
                        * cluster_dim_y()
                        * cluster_dim_z()
                        * (block_dim_x() * block_dim_y() * block_dim_z()),
                );
                mbarrier::fence_init();
            });
        });

        Self {
            response_bufs,
            response_produced_bars,
            response_consumed_bars,
            leader_warp_id,
            num_stages,
            i: Var::new(0),
        }
    }

    pub fn emit_scheduler_loop_on_leader_warp(&self) {
        with_warp(self.leader_warp_id, || {
            let is_leader_cta = cluster_block_rank().eq(0);

            loop_(|l| {
                let slot_idx = self.i % self.num_stages as u32;

                let needs_wait = is_leader_cta & self.i.ge(self.num_stages as u32);
                if_(needs_wait, || {
                    let consumed_bar = self.response_consumed_bars.get(slot_idx);
                    consumed_bar
                        .cluster()
                        .relaxed()
                        .wait_parity(parity_from_idx(
                            self.i - self.num_stages as u32,
                            self.num_stages as u32,
                        ));
                    fence_sync_restrict_acquire();
                });

                let response_buf = self
                    .response_bufs
                    .offset(CLC_RESPONSE_WORDS as u32 * slot_idx);
                let produced_bar = self.response_produced_bars.get(slot_idx);

                if_(is_leader_cta, || {
                    with_leader(|| {
                        produced_bar
                            .cluster()
                            .relaxed()
                            .arrive_expect_tx((CLC_RESPONSE_WORDS * size_of::<u32>()) as u32);
                        clc::ResponseBuf::at(response_buf).try_cancel_multicast(produced_bar);
                    });
                });

                produced_bar
                    .cluster()
                    .relaxed()
                    .wait_parity(parity_from_idx(self.i, self.num_stages as u32));
                fence_proxy_async_generic_acquire();

                let response = clc::ResponseBuf::at(response_buf).load();
                fence_proxy_async_generic_release();
                let consumed_bar = self.response_consumed_bars.get(slot_idx).remote(0);
                consumed_bar.arrive(); // TODO should be relaxed

                if_else(
                    response.is_canceled(),
                    || {
                        self.i.set(self.i + 1);
                    },
                    || {
                        // Failed to cancel; done.
                        l.break_();
                    },
                );
            });
        });
    }

    /// Returns (is_some, x, y, z)
    pub fn get_next_block_idx(&self) -> (Val<bool>, Val<u32>, Val<u32>, Val<u32>) {
        let slot_idx = self.i % self.num_stages as u32;
        let bar = self.response_produced_bars.get(slot_idx);

        bar.cluster()
            .relaxed()
            .wait_parity(parity_from_idx(self.i, self.num_stages as u32));
        fence_proxy_async_generic_acquire();

        let response = clc::ResponseBuf::at(
            self.response_bufs
                .offset(slot_idx * CLC_RESPONSE_WORDS as u32),
        )
        .load();

        fence_proxy_async_generic_release();
        let bar = self.response_consumed_bars.get(slot_idx).remote(0);
        bar.arrive(); // TODO should be relaxed

        let success = response.is_canceled();
        let x = select(
            success,
            response.first_block_idx_x() + cluster_block_idx_x(),
            0,
        );
        let y = select(
            success,
            response.first_block_idx_y() + cluster_block_idx_y(),
            0,
        );
        let z = select(
            success,
            response.first_block_idx_z() + cluster_block_idx_z(),
            0,
        );

        if_(success, || self.i.set(self.i + 1));

        (success, x, y, z)
    }
}
