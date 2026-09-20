use core::{hint::assert_unchecked, slice};

use cuda_device::{
    DynamicSharedArray, async_copy,
    float::rsqrt_approx_ftz_f32,
    sync_threads,
    uniform::grid_dim_x,
    warp::{self, warp_id},
};
use cuda_device_ext::bf16;
use cuda_macros::{kernel, launch_bounds};

use crate::{
    PrefetchMode, WARP_SIZE, lane_id, make_register_view, num_warps,
    smem_ring::StagingBufferRing,
    view::{
        AlignedView, AlignedViewMut, PartitionU128Chunks, PartitionWithinWarpCoalesced,
        elements_per_u128,
    },
    zip2_for_each,
};

/// Template for fast RMSNorm where the vector size
/// is small enough to fit within a single warp's registers, enabling
/// execution with zero CTA-level synchronization.
/// A warp has capacity for up to 32 * 256 * 2 = 16384 bf16 elements,
/// but it also needs intermediates, so likely around 4096
/// is the peak without spilling (need to test).
///
/// Persistent kernel. Designed for occupancy of 1. Instantiate
/// with maximum number of warps that fits in the register file.
///
/// Uses dynamic shared memory. Gamma is cached at the beginning of dynamic shared memory.
/// The rest holds prefetch buffers.
fn rmsnorm_bf16_warp_persistent<
    const INNER_SIZE: usize,
    const ELEMENTS_PER_THREAD: usize,
    const PREFETCH_STAGES: usize,
    const PREFETCH_MODE: PrefetchMode,
>(
    gmem_x: *const bf16,
    gmem_gamma: *const bf16,
    gmem_y: *mut bf16,
    gmem_r_inv: *mut f32,
    num_vectors: usize,
) {
    unsafe {
        const ELEMENTS_PER_LOAD_PER_THREAD: usize = 8;
        const ELEMENTS_PER_LOAD_PER_WARP: usize = ELEMENTS_PER_LOAD_PER_THREAD * WARP_SIZE;

        const {
            // Required for fully vectorized loads.
            assert!(PREFETCH_STAGES > 0);
            assert!(INNER_SIZE.is_multiple_of(ELEMENTS_PER_LOAD_PER_WARP));
            assert!(ELEMENTS_PER_THREAD == INNER_SIZE / WARP_SIZE);
        }

        let stride = INNER_SIZE;

        let gmem_x = AlignedView::from_ptr(gmem_x, num_vectors * stride);
        let gmem_gamma = AlignedView::from_ptr(gmem_gamma, INNER_SIZE);
        let mut gmem_y = AlignedViewMut::from_ptr(gmem_y, num_vectors * stride);
        let gmem_r_inv = slice::from_raw_parts_mut(gmem_r_inv, num_vectors);

        let smem = slice::from_raw_parts_mut(
            DynamicSharedArray::<u128>::get(),
            INNER_SIZE / elements_per_u128::<bf16>() * (PREFETCH_STAGES + 1),
        );
        let (smem_gamma, smem_staging) =
            smem.split_at_mut(INNER_SIZE / elements_per_u128::<bf16>());
        let mut smem_gamma = AlignedViewMut::<bf16>::new(smem_gamma);
        let mut smem_ring =
            StagingBufferRing::<bf16>::new(smem_staging, INNER_SIZE, PREFETCH_STAGES);

        // Fetch gamma into SMEM using one warp.
        if warp_id() == 0 {
            zip2_for_each(
                smem_gamma.partition_iter(PartitionWithinWarpCoalesced),
                gmem_gamma.partition_iter(PartitionWithinWarpCoalesced),
                |mut dst, src| {
                    dst.async_copy_from_cg(src);
                },
            );
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_all();
        }

        let cta_stride = grid_dim_x().get() as usize * num_warps();
        let row_iter = (warp_id() as usize..num_vectors).step_by(cta_stride);

        // Initial load pipeline fill for each warp.
        for (slot, row) in row_iter.clone().take(PREFETCH_STAGES).enumerate() {
            let gmem_row = gmem_x.slice(row * stride..).slice(..INNER_SIZE);

            zip2_for_each(
                smem_ring
                    .slot_mut(slot)
                    .partition_iter(PartitionWithinWarpCoalesced),
                gmem_row.partition_iter(PartitionWithinWarpCoalesced),
                |mut dst, src| {
                    dst.async_copy_from_cg(src);
                },
            );

            async_copy::cp_async_commit_group();
        }

        sync_threads();

        for (iteration, row_idx) in row_iter.clone().enumerate() {
            make_register_view!(x, bf16, ELEMENTS_PER_THREAD);

            let prefetch_stages_minus_one = const { PREFETCH_STAGES as u32 - 1 };
            async_copy::cp_async_wait_group(prefetch_stages_minus_one);

            let slot = iteration % PREFETCH_STAGES;
            zip2_for_each(
                x.partition_iter(PartitionU128Chunks),
                smem_ring
                    .slot(slot)
                    .partition_iter(PartitionWithinWarpCoalesced),
                |mut dst, src| {
                    dst.copy_from(src);
                },
            );

            // Prefetch next row for this slot.
            let next_row_idx = row_idx + cta_stride * PREFETCH_STAGES;
            if next_row_idx < num_vectors {
                let gmem_next_row =
                    gmem_x.slice(stride * next_row_idx..stride * next_row_idx + INNER_SIZE);
                zip2_for_each(
                    smem_ring
                        .slot_mut(slot)
                        .partition_iter(PartitionWithinWarpCoalesced),
                    gmem_next_row.partition_iter(PartitionWithinWarpCoalesced),
                    |mut dst, src| {
                        dst.async_copy_from_cg(src);
                    },
                );
            }

            let mut thr_sum2 = 0.0f32;
            for val in &x {
                let val = val.to_f32();
                thr_sum2 += val * val;
            }

            // Intra-warp reduction.
            for i in 0..WARP_SIZE.ilog2() {
                thr_sum2 += warp::shuffle_xor_f32_sync(u32::MAX, thr_sum2, 1 << i);
            }

            let r_inv = rsqrt_approx_ftz_f32(thr_sum2 + 1e-6);

            make_register_view!(gamma, bf16, ELEMENTS_PER_THREAD);
            zip2_for_each(
                gamma.partition_iter(PartitionU128Chunks),
                smem_gamma.partition_iter(PartitionWithinWarpCoalesced),
                |mut dst, src| {
                    dst.copy_from(src.as_view());
                },
            );

            zip2_for_each(&mut x, &gamma, |x, gamma| {
                *x = bf16::from_f32(x.to_f32() * gamma.to_f32() * r_inv);
            });

            // Store to GMEM.
            let mut gmem_row = gmem_y.slice_mut(row_idx * stride..row_idx * stride + INNER_SIZE);
            zip2_for_each(
                x.partition_iter(PartitionU128Chunks),
                gmem_row.partition_iter(PartitionWithinWarpCoalesced),
                |src, mut dst| {
                    dst.copy_from(src.as_view());
                },
            );

            if lane_id() == 0 {
                *gmem_r_inv.get_unchecked_mut(row_idx) = r_inv;
            }
        }
    }
}

#[kernel]
#[launch_bounds(256, 1)]
pub fn rmsnorm_bf16_warp_persistent_inner2048_sm90_sm10x(
    gmem_x: *const bf16,
    gmem_gamma: *const bf16,
    gmem_y: *mut bf16,
    gmem_r_inv: *mut f32,
    num_vectors: usize,
) {
    rmsnorm_bf16_warp_persistent::<2048, 64, 2, { PrefetchMode::CpAsyncBulk }>(
        gmem_x,
        gmem_gamma,
        gmem_y,
        gmem_r_inv,
        num_vectors,
    );
}
