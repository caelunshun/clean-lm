use crate::{
    tile_elementwise::template::{
        BlockScaledStoreDescriptor, NUM_WORKER_THREADS, Params, Smem, layouts::Layouts,
    },
    utils::{
        block_scale::BlockScaledFormat,
        compiler_magic::{array_map_unrolled, array_zip_unrolled, uniform_warp_id},
    },
};
use core::simd::u32x4;
use ptx_std::{
    F32Ext, MBarrier, MemoryOrdering, PhaseParity, Scope, Vector, bf16, bf16x8, bf16xN,
    block_id_in_cluster,
    cluster::{map_to_cta, st_async},
    f32x8, f32xN,
    fence::fence_sync_restrict,
    thread_id,
    warp_collectives::{LaneMask, elect},
};

pub unsafe fn worker_warp_blockscaled_colwise_store<P: Params, BS: BlockScaledFormat>(
    smem: &Smem<P>,
    thr_vals: [[f32; P::THREAD_N]; P::THREAD_M],
    desc: &BlockScaledStoreDescriptor,
    cluster_resharding_idx: &mut u32,
) {
    unsafe {
        // Every column resides in a single warp within a CTA, so our thr_amax
        // reduction is easier. However, when CTA_TM < BLOCK_SIZE, we need to re-shard
        // our data so each CTA can hold full BLOCK_SIZE-length columns, requiring a one-way trip through DSMEM.

        assert!(P::DOES_STORE_COLWISE_BLOCK_SCALED_DATA);
        assert!(P::CGA_TM.is_multiple_of(BS::BLOCK_SIZE));

        if P::CTA_TM >= BS::BLOCK_SIZE {
            // No need for a cluster-wide communication; can quantize and store directly from this CTA's
            // column chunks.

            // Transpose our register view such that each element of the outer array contains a column of elements.
            const BLOCK_SIZE_PER_THREAD<BS: BlockScaledFormat, P: Params>: usize =
                const { BS::BLOCK_SIZE.min(P::THREAD_M) };
            const BLOCKS_PER_THREAD_PER_COL<BS: BlockScaledFormat, P: Params>: usize =
                const { P::THREAD_M / BLOCK_SIZE_PER_THREAD::<BS, P> };

            let mut thr_vals_chunked = [[[0.0f32; BLOCK_SIZE_PER_THREAD::<BS, P>];
                BLOCKS_PER_THREAD_PER_COL::<BS, P>];
                P::THREAD_N];
            #[rustc_unroll(full)]
            for i in 0..P::THREAD_M {
                #[rustc_unroll(full)]
                for j in 0..P::THREAD_N {
                    thr_vals_chunked[j][i / BLOCK_SIZE_PER_THREAD::<BS, P>]
                        [i % BLOCK_SIZE_PER_THREAD::<BS, P>] = thr_vals[i][j];
                }
            }

            let thr_amaxs = array_map_unrolled(
                thr_vals_chunked,
                #[inline(always)]
                |chunks| {
                    array_map_unrolled(
                        chunks,
                        #[inline(always)]
                        |chunk| f32xN::from_array(chunk).reduce_max(),
                    )
                },
            );

            // TODO(claude) quantize, store elements and scale factors to GMEM in column-major format
            // (for the elements on the unpredicated path: use 256-bit stores, with each thread sending a column-chunk of 32 FP8 elements)
        } else {
            // We don't have full column blocks within one CTA; re-shard the data.

            const RESHARDED_CTA_TN<P: Params, BS: BlockScaledFormat>: usize =
                P::CTA_TN / P::CLUSTER_RESHARD_GROUP_WIDTH;
            const RESHARDED_CTA_THREAD_N<P: Params, BS: BlockScaledFormat>: usize =
                RESHARDED_CTA_TN::<P, BS> / NUM_WORKER_THREADS;

            // Ensure resharded data can be evenly partitioned across worker threads.
            // Runtime assert, not const, because this condition only needs
            // to hold when this path is actually used.
            assert!(RESHARDED_CTA_TN::<P, BS>.is_multiple_of(NUM_WORKER_THREADS));

            let slot_idx = *cluster_resharding_idx as usize % 2;

            let reshard_group_offset = block_id_in_cluster().x
                / P::CLUSTER_RESHARD_GROUP_WIDTH as u32
                * P::CLUSTER_RESHARD_GROUP_WIDTH as u32;

            let bar = &smem.bars.cluster_reshard_buffer_produced[slot_idx];
            #[rustc_unroll(full)]
            for i in 0..P::THREAD_M {
                if Layouts::<P>::new().is_8x_vectorized() {
                    // Vectorized DSMEM stores
                    #[rustc_unroll(full)]
                    for chunk_idx in 0..P::THREAD_N / 8 {
                        let j = thread_id().x as usize * 8 + chunk_idx * NUM_WORKER_THREADS * 8;
                        let recipient_idx =
                            (j / RESHARDED_CTA_TN::<P, BS>) as u32 + reshard_group_offset;
                        let recipient_rank =
                            recipient_idx + block_id_in_cluster().y * P::CLUSTER_SIZE_M as u32;
                        let dst = smem.cluster_reshard_buffers[slot_idx]
                            .as_ptr()
                            .cast::<bf16>()
                            .add(i * RESHARDED_CTA_TN::<P, BS> + j % RESHARDED_CTA_TN::<P, BS>)
                            .cast::<bf16x8>()
                            .cast_mut();
                        let dst = map_to_cta(dst, recipient_rank);

                        let chunk = bf16x8::from_f32(f32x8::from_array(
                            thr_vals[i][chunk_idx * 8..][..8].try_into().unwrap(),
                        ));
                        st_async(
                            dst.cast::<u32x4>(),
                            bytemuck::must_cast(chunk),
                            map_to_cta(bar as *const MBarrier as *mut MBarrier, recipient_rank),
                        );
                    }
                } else {
                    // Unvectorized DSMEM stores.
                    todo!()
                }
            }

            // Wait for data to arrive from peer CTAs.
            bar.wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                PhaseParity::of_phase(*cluster_resharding_idx / 2),
            );
            fence_sync_restrict::<{ MemoryOrdering::Acquire }>();

            // Re-arm the cluster receive barrier.
            if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                bar.arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                    (P::CLUSTER_RESHARD_BUFFER_ELEMENTS * size_of::<bf16>()) as u32,
                );
            }

            // Load the received data into worker threads' registers.
            // Each worker thread holds a BLOCK_SIZE (M) x RESHARDED_CTA_THREAD_N (N) chunk of data.
            // We then compute amax values (no synchronization needed), quantize, and send
            // to GMEM in column-major format.
            //
            // Note that the received data in SMEM is flat row-major.
            // We transpose within a lane's registers.
            let mut thr_vals_resharded =
                [[bf16::ZERO; BS::BLOCK_SIZE]; RESHARDED_CTA_THREAD_N::<P, BS>];
            #[rustc_unroll(full)]
            for i in 0..BS::BLOCK_SIZE {
                if RESHARDED_CTA_THREAD_N::<P, BS>.is_multiple_of(8) {
                    // Vectorized SMEM loads
                    #[rustc_unroll(full)]
                    for chunk_idx in 0..RESHARDED_CTA_THREAD_N::<P, BS> / 8 {
                        let src = smem.cluster_reshard_buffers[slot_idx]
                            .as_ptr()
                            .cast::<bf16x8>()
                            .add(
                                i * RESHARDED_CTA_TN::<P, BS> / 8
                                    + thread_id().x as usize
                                    + chunk_idx * NUM_WORKER_THREADS,
                            );

                        let vals = src.read().to_array();

                        #[rustc_unroll(full)]
                        for k in 0..8 {
                            thr_vals_resharded[chunk_idx * 8 + k][i] = vals[k];
                        }
                    }
                } else {
                    // Unvectorized SMEM loads
                    #[rustc_unroll(full)]
                    for element_idx in 0..RESHARDED_CTA_THREAD_N::<P, BS> {
                        let src = smem.cluster_reshard_buffers[slot_idx]
                            .as_ptr()
                            .cast::<bf16>()
                            .add(
                                i * RESHARDED_CTA_TN::<P, BS>
                                    + thread_id().x as usize
                                    + element_idx * NUM_WORKER_THREADS,
                            );

                        let val = src.read();
                        thr_vals_resharded[element_idx][i] = val;
                    }
                }
            }

            let thr_amaxes = array_map_unrolled(
                thr_vals_resharded,
                #[inline(always)]
                |col| bf16xN::from_array(col).abs().reduce_max().to_f32(),
            );
            let thr_sfs = BS::compute_scale_factor(f32xN::from_array(thr_amaxes)).to_array();

            let thr_vals_quantized = array_zip_unrolled(thr_vals_resharded, thr_sfs, |col, sf| {
                BS::quantize_bf16(Vector::from_array(col), BS::splat_scale_factor(sf))
            });

            // TODO store to GMEM elementwise with bounds checks if any lane needs them,
            // otherwise 256-bit stores. Also store scale factors using scalar (8-bit) stores.

            *cluster_resharding_idx += 1;
        }
    }
}
