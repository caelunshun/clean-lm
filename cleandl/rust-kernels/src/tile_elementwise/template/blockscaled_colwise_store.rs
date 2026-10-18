use crate::{
    tile_elementwise::template::{
        BlockScaledStoreDescriptor, GmemStore, NUM_WORKER_THREADS, Params, Smem,
        layouts::{Layouts, store_unit_to_gmem},
    },
    utils::{
        block_scale::{BlockScaledFormat, SF_TILE_NCOLS, SF_TILE_NROWS, scale_factor_pos},
        compiler_magic::{array_map_unrolled, array_zip_unrolled, uniform_warp_id},
    },
};
use core::simd::{u32x4, u32x8};
use ptx_std::{
    MBarrier, MemoryOrdering, PhaseParity, Scope, Vector, bf16, bf16x8, bf16xN,
    block_id_in_cluster,
    cluster::{map_to_cta, st_async},
    f32x8, f32xN,
    fence::fence_sync_restrict,
    memory::store_global,
    thread_id,
    warp_collectives::{LaneMask, elect, vote_any},
};

#[inline(always)]
pub unsafe fn worker_warp_blockscaled_colwise_store<P: Params, BS: BlockScaledFormat>(
    smem: &Smem<P>,
    thr_vals: [[f32; P::THREAD_N]; P::THREAD_M],
    desc: &BlockScaledStoreDescriptor,
    cluster_resharding_idx: &mut u32,
) {
    unsafe {
        // Every column resides in a single thread within a CTA, so our thr_amax
        // reduction is easier. However, when CTA_TM < BLOCK_SIZE, we need to re-shard
        // our data so each CTA can hold full BLOCK_SIZE-length columns, requiring a one-way trip through DSMEM.
        //
        // In both cases, values are converted to bf16 before computing amaxes and quantizing.

        assert!(P::DOES_STORE_COLWISE_BLOCK_SCALED_DATA);

        if P::CTA_TM >= BS::BLOCK_SIZE {
            // No need for a cluster-wide communication; can quantize and store directly from this CTA's
            // column chunks.

            // Transpose our register view such that each element of the outer array contains a column of elements.
            const BLOCK_SIZE_PER_THREAD<BS: BlockScaledFormat, P: Params>: usize =
                const { BS::BLOCK_SIZE.min(P::THREAD_M) };
            const BLOCKS_PER_THREAD_PER_COL<BS: BlockScaledFormat, P: Params>: usize =
                const { P::THREAD_M / BLOCK_SIZE_PER_THREAD::<BS, P> };

            let thr_vals_bf16 = array_map_unrolled(
                thr_vals,
                #[inline(always)]
                |row| bf16xN::from_f32(f32xN::from_array(row)).to_array(),
            );

            let mut thr_vals_chunked = [[[bf16::ZERO; BLOCK_SIZE_PER_THREAD::<BS, P>];
                BLOCKS_PER_THREAD_PER_COL::<BS, P>];
                P::THREAD_N];
            #[rustc_unroll(full)]
            for i in 0..P::THREAD_M {
                #[rustc_unroll(full)]
                for j in 0..P::THREAD_N {
                    thr_vals_chunked[j][i / BLOCK_SIZE_PER_THREAD::<BS, P>]
                        [i % BLOCK_SIZE_PER_THREAD::<BS, P>] = thr_vals_bf16[i][j];
                }
            }

            let thr_amaxs = array_map_unrolled(
                thr_vals_chunked,
                #[inline(always)]
                |chunks| {
                    array_map_unrolled(
                        chunks,
                        #[inline(always)]
                        |chunk| bf16xN::from_array(chunk).abs().reduce_max().to_f32(),
                    )
                },
            );

            // Scale factors are computed one block index at a time, so that the conversions are vectorized across columns.
            let mut thr_sfs =
                [[Default::default(); BLOCKS_PER_THREAD_PER_COL::<BS, P>]; P::THREAD_N];
            #[rustc_unroll(full)]
            for b in 0..BLOCKS_PER_THREAD_PER_COL::<BS, P> {
                let block_sfs = BS::compute_scale_factor(f32xN::from_array(array_map_unrolled(
                    thr_amaxs,
                    #[inline(always)]
                    |amaxs| amaxs[b],
                )))
                .to_array();
                #[rustc_unroll(full)]
                for j in 0..P::THREAD_N {
                    thr_sfs[j][b] = block_sfs[j];
                }
            }

            let thr_vals_quantized = array_zip_unrolled(
                thr_vals_chunked,
                thr_sfs,
                #[inline(always)]
                |chunks, sfs| {
                    array_zip_unrolled(
                        chunks,
                        sfs,
                        #[inline(always)]
                        |chunk, sf| {
                            BS::quantize_bf16(bf16xN::from_array(chunk), BS::splat_scale_factor(sf))
                                .to_array()
                        },
                    )
                },
            );

            store_quantized_cols::<BS, _, _, _, _, _>(
                desc,
                thr_vals_quantized,
                thr_sfs,
                [
                    desc.coord_mn[0] + block_id_in_cluster().x * P::CTA_TM as u32,
                    desc.coord_mn[1] + block_id_in_cluster().y * P::CTA_TN as u32,
                ],
                #[inline(always)]
                |j| Layouts::<P>::new().thread_element_n_index(thread_id().x, j),
            );
        } else {
            // We don't have full column blocks within one CTA; re-shard the data.

            const RESHARDED_CTA_THREAD_N<P: Params>: usize =
                P::CLUSTER_RESHARDED_CTA_TN / NUM_WORKER_THREADS;
            // Widest vector width for the SMEM loads that evenly divides each thread's columns.
            const RESHARD_LOAD_VECTOR_WIDTH<P: Params>: usize = const {
                if RESHARDED_CTA_THREAD_N::<P>.is_multiple_of(8) {
                    8
                } else if RESHARDED_CTA_THREAD_N::<P>.is_multiple_of(4) {
                    4
                } else if RESHARDED_CTA_THREAD_N::<P>.is_multiple_of(2) {
                    2
                } else {
                    1
                }
            };

            let slot_idx = *cluster_resharding_idx as usize % 2;

            let reshard_group_offset = block_id_in_cluster().x
                / P::CLUSTER_RESHARD_GROUP_WIDTH as u32
                * P::CLUSTER_RESHARD_GROUP_WIDTH as u32;
            let idx_in_group = block_id_in_cluster().x - reshard_group_offset;
            // Row of the recipients' buffers at which this CTA's rows begin.
            let row_offset_in_group = idx_in_group as usize * P::CTA_TM;

            // Before sending, wait for all worker threads of this CTA to finish reading
            // the previous reshard's data, so we avoid a data race.
            if *cluster_resharding_idx > 0 {
                let prev_idx = *cluster_resharding_idx - 1;
                smem.bars.cluster_reshard_buffer_consumed[prev_idx as usize % 2]
                    .wait(PhaseParity::of_phase(prev_idx / 2));
            }

            let produced_bar = &smem.bars.cluster_reshard_buffer_produced[slot_idx];
            #[rustc_unroll(full)]
            for i in 0..P::THREAD_M {
                if Layouts::<P>::new().is_8x_vectorized() {
                    // Vectorized DSMEM stores
                    #[rustc_unroll(full)]
                    for chunk_idx in 0..P::THREAD_N / 8 {
                        let j = thread_id().x as usize * 8 + chunk_idx * NUM_WORKER_THREADS * 8;
                        let recipient_idx =
                            (j / P::CLUSTER_RESHARDED_CTA_TN) as u32 + reshard_group_offset;
                        let recipient_rank =
                            recipient_idx + block_id_in_cluster().y * P::CLUSTER_SIZE_M as u32;
                        let dst = smem.cluster_reshard_buffers[slot_idx]
                            .as_ptr()
                            .cast::<bf16>()
                            .add(
                                (row_offset_in_group + i) * P::CLUSTER_RESHARDED_CTA_TN
                                    + j % P::CLUSTER_RESHARDED_CTA_TN,
                            )
                            .cast::<bf16x8>()
                            .cast_mut();
                        let dst = map_to_cta(dst, recipient_rank);

                        let chunk = bf16x8::from_f32(f32x8::from_array(
                            thr_vals[i][chunk_idx * 8..][..8].try_into().unwrap(),
                        ));
                        st_async(
                            dst.cast::<u32x4>(),
                            bytemuck::must_cast(chunk),
                            map_to_cta(
                                produced_bar as *const MBarrier as *mut MBarrier,
                                recipient_rank,
                            ),
                        );
                    }
                } else {
                    // Unvectorized DSMEM stores.
                    todo!()
                }
            }

            // Wait for data to arrive from peer CTAs.
            produced_bar.wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
                PhaseParity::of_phase(*cluster_resharding_idx / 2),
            );
            fence_sync_restrict::<{ MemoryOrdering::Acquire }>();

            // Re-arm the cluster receive barrier.
            if uniform_warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                produced_bar
                    .arrive_expect_tx_scoped::<{ Scope::Cluster }, { MemoryOrdering::Relaxed }>(
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
            let thr_vals_resharded: [[bf16; BS::BLOCK_SIZE]; RESHARDED_CTA_THREAD_N::<P>] =
                match RESHARD_LOAD_VECTOR_WIDTH::<P> {
                    8 => load_resharded_cols::<P, 8, _, _>(smem, slot_idx),
                    4 => load_resharded_cols::<P, 4, _, _>(smem, slot_idx),
                    2 => load_resharded_cols::<P, 2, _, _>(smem, slot_idx),
                    _ => load_resharded_cols::<P, 1, _, _>(smem, slot_idx),
                };

            // Release this slot for reuse (see the wait before sending above).
            smem.bars.cluster_reshard_buffer_consumed[slot_idx].arrive();

            let thr_amaxes = array_map_unrolled(
                thr_vals_resharded,
                #[inline(always)]
                |col| bf16xN::from_array(col).abs().reduce_max().to_f32(),
            );
            let thr_sfs = BS::compute_scale_factor(f32xN::from_array(thr_amaxes)).to_array();

            let thr_vals_quantized = array_zip_unrolled(thr_vals_resharded, thr_sfs, |col, sf| {
                [
                    BS::quantize_bf16(Vector::from_array(col), BS::splat_scale_factor(sf))
                        .to_array(),
                ]
            });

            // This CTA holds the group's BLOCK_SIZE rows, for its shard of the columns.
            let tid = thread_id().x as usize;
            store_quantized_cols::<BS, _, _, _, _, _>(
                desc,
                thr_vals_quantized,
                array_map_unrolled(
                    thr_sfs,
                    #[inline(always)]
                    |sf| [sf],
                ),
                [
                    desc.coord_mn[0] + reshard_group_offset * P::CTA_TM as u32,
                    desc.coord_mn[1]
                        + block_id_in_cluster().y * P::CTA_TN as u32
                        + idx_in_group * P::CLUSTER_RESHARDED_CTA_TN as u32,
                ],
                // Same column assignment as the SMEM loads above.
                #[inline(always)]
                |e| resharded_col_index(tid, e, RESHARD_LOAD_VECTOR_WIDTH::<P>),
            );

            *cluster_resharding_idx += 1;
        }
    }
}

/// Loads this thread's columns of the received reshard buffer (flat row-major) from SMEM,
/// transposed so that each element of the outer array holds one column, using `V`-element vector loads.
#[inline(always)]
unsafe fn load_resharded_cols<P: Params, const V: usize, const M: usize, const N: usize>(
    smem: &Smem<P>,
    slot_idx: usize,
) -> [[bf16; M]; N] {
    unsafe {
        let mut cols = [[bf16::ZERO; M]; N];
        #[rustc_unroll(full)]
        for i in 0..M {
            #[rustc_unroll(full)]
            for chunk_idx in 0..N / V {
                let src = smem.cluster_reshard_buffers[slot_idx]
                    .as_ptr()
                    .cast::<bf16>()
                    .add(
                        i * P::CLUSTER_RESHARDED_CTA_TN
                            + resharded_col_index(thread_id().x as usize, chunk_idx * V, V),
                    )
                    .cast::<bf16xN<V>>();

                let vals = src.read().to_array();

                #[rustc_unroll(full)]
                for k in 0..V {
                    cols[chunk_idx * V + k][i] = vals[k];
                }
            }
        }
        cols
    }
}

/// Column of the reshard buffer holding thread `tid`'s `e`th column, when loaded with `v`-element vectors.
#[inline(always)]
fn resharded_col_index(tid: usize, e: usize, v: usize) -> usize {
    (e / v) * NUM_WORKER_THREADS * v + tid * v + e % v
}

/// Stores this thread's quantized column chunks to the column-major output tensor,
/// along with their scale factors.
#[inline(always)]
unsafe fn store_quantized_cols<
    BS: BlockScaledFormat,
    Q: Copy,
    SF: Copy,
    const C: usize,
    const B: usize,
    const L: usize,
>(
    desc: &BlockScaledStoreDescriptor,
    vals: [[[Q; L]; B]; C],
    sfs: [[SF; B]; C],
    [m0, n0]: [u32; 2],
    col_offset: impl Fn(usize) -> usize,
) {
    unsafe {
        // The column-major tensor is a row-major tensor with M and N swapped.
        let dst = GmemStore {
            tma_desc: desc.tma_desc,
            coord_mn: [n0, m0],
            cache_policy: desc.cache_policy,
            reduction: None,
        }
        .mxn_tile::<Q>();

        let aligned = (L * size_of::<Q>()).is_multiple_of(size_of::<u32x8>())
            && dst.start.addr().is_multiple_of(size_of::<u32x8>())
            && (dst.row_stride * size_of::<Q>()).is_multiple_of(size_of::<u32x8>());
        let in_bounds = col_offset(C - 1) < dst.valid_rows && B * L <= dst.valid_cols;
        let predicated = vote_any(LaneMask::FULL, !(aligned && in_bounds));

        let store = #[inline(always)]
        |predicated: bool| {
            #[rustc_unroll(full)]
            for j in 0..C {
                let col = dst.row(col_offset(j));
                #[rustc_unroll(full)]
                for b in 0..B {
                    if predicated {
                        #[rustc_unroll(full)]
                        for k in 0..L {
                            let m = b * L + k;
                            if m < col.valid_cols {
                                store_unit_to_gmem(
                                    vals[j][b].as_ptr().add(k),
                                    col.start.add(m),
                                    col.cache_policy,
                                    None,
                                );
                            }
                        }
                    } else {
                        #[rustc_unroll(full)]
                        for k in 0..L * size_of::<Q>() / size_of::<u32x8>() {
                            store_global(
                                col.start.add(b * L).cast::<u32x8>().add(k),
                                vals[j][b].as_ptr().cast::<u32x8>().add(k).read_unaligned(),
                                col.cache_policy,
                            );
                        }
                    }
                }
            }
        };
        if predicated {
            store(true)
        } else {
            store(false)
        }

        let sf_cols = desc.tma_desc.global_size[0]
            .div_ceil(BS::BLOCK_SIZE as u32)
            .next_multiple_of(SF_TILE_NCOLS);
        let sf_rows = desc.tma_desc.global_size[1].next_multiple_of(SF_TILE_NROWS);

        #[rustc_unroll(full)]
        for j in 0..C {
            let sf_row = n0 + col_offset(j) as u32;
            #[rustc_unroll(full)]
            for b in 0..B {
                let sf_col = (m0 + (b * L) as u32) / BS::BLOCK_SIZE as u32;
                if sf_row < sf_rows && sf_col < sf_cols {
                    desc.sf_ptr
                        .cast::<SF>()
                        .add(scale_factor_pos(sf_row, sf_col, sf_cols))
                        .write(sfs[j][b]);
                }
            }
        }
    }
}
