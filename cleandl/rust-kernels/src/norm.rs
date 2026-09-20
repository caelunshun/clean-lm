use core::slice;

use cuda_device::{
    DynamicSharedArray, async_copy,
    float::rsqrt_approx_ftz_f32,
    sync_threads,
    uniform::grid_dim_x,
    warp::{self, lane_id, warp_id},
};
use cuda_device_ext::{bf16, bf16x2};
use cuda_macros::{kernel, launch_bounds};

use crate::{
    PrefetchMode, WARP_SIZE, as_aligned_slice, as_aligned_slice_mut, as_bf16x2_slice,
    as_bf16x2_slice_mut, num_warps,
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
    x: &[bf16],
    gamma: &[bf16],
    y: &mut [bf16],
    r_inv_buf: &mut [bf16],
    num_vectors: usize,
    stride: usize,
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

        let smem = DynamicSharedArray::<bf16>::get();
        let smem_gamma = slice::from_raw_parts_mut(smem, INNER_SIZE);
        let smem_buffer = smem.add(INNER_SIZE);

        let get_buffer_slot = |slot: usize| {
            let ptr = smem_buffer.add(INNER_SIZE * (warp_id() as usize + slot * num_warps()));
            slice::from_raw_parts_mut(ptr, INNER_SIZE)
        };

        // Fetch gamma into SMEM using one warp.
        if warp_id() == 0 {
            for i in 0..INNER_SIZE / ELEMENTS_PER_LOAD_PER_WARP {
                let offset = lane_id() as usize * ELEMENTS_PER_LOAD_PER_THREAD
                    + i * ELEMENTS_PER_LOAD_PER_WARP;
                async_copy::cp_async_cg_16(
                    smem_gamma.as_mut_ptr().add(offset).cast(),
                    gamma.as_ptr().add(offset).cast(),
                );
            }
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_all();
        }

        let cta_stride = grid_dim_x().get() as usize * num_warps();
        let row_iter = (warp_id() as usize..num_vectors).step_by(cta_stride);

        // Initial load pipeline fill for each warp.
        for (slot, row) in row_iter.clone().take(PREFETCH_STAGES).enumerate() {
            let dst = get_buffer_slot(slot);
            for i in 0..INNER_SIZE / ELEMENTS_PER_LOAD_PER_WARP {
                let offset = lane_id() as usize * ELEMENTS_PER_LOAD_PER_THREAD
                    + i * ELEMENTS_PER_LOAD_PER_WARP;
                async_copy::cp_async_cg_16(
                    dst.as_mut_ptr().add(offset).cast(),
                    x.as_ptr().add(stride * row + offset).cast(),
                );
            }
            async_copy::cp_async_commit_group();
        }

        sync_threads();

        for (iter, row_idx) in row_iter.clone().enumerate() {
            let mut thr_row: [bf16; ELEMENTS_PER_THREAD] = [bf16::ZERO; _];

            let prefetch_stages_minus_one = const { (PREFETCH_STAGES as u32).saturating_sub(1) };
            async_copy::cp_async_wait_group(prefetch_stages_minus_one);

            let slot = iter % PREFETCH_STAGES;
            let slot_ptr = get_buffer_slot(slot);

            for i in 0..ELEMENTS_PER_THREAD / ELEMENTS_PER_LOAD_PER_THREAD {
                let idx = lane_id() as usize + i * ELEMENTS_PER_LOAD_PER_WARP;
                as_aligned_slice_mut(&mut thr_row[idx..])[0] =
                    as_aligned_slice(&slot_ptr[idx..])[0];
            }

            // Done with buffer slot; prefetch next row.
            if let Some(next_row_idx) = row_iter.clone().nth(iter + PREFETCH_STAGES) {
                let dst = get_buffer_slot(slot);
                warp::sync_mask(u32::MAX);
                for i in 0..INNER_SIZE / ELEMENTS_PER_LOAD_PER_WARP {
                    let offset = lane_id() as usize * ELEMENTS_PER_LOAD_PER_THREAD
                        + i * ELEMENTS_PER_LOAD_PER_WARP;
                    async_copy::cp_async_cg_16(
                        dst.as_mut_ptr().add(offset).cast(),
                        x.as_ptr().add(stride * next_row_idx + offset).cast(),
                    );
                }
                async_copy::cp_async_commit_group();
            }

            let mut thr_sum2 = 0.0f32;
            for bf16_pair in as_bf16x2_slice(&thr_row) {
                let (a, b) = bf16_pair.to_f32_pair();
                thr_sum2 += a * a + b * b;
            }

            // Intra-warp reduction
            for i in 0..WARP_SIZE.ilog2() {
                thr_sum2 += warp::shuffle_xor_f32(thr_sum2, 1 << i);
            }

            let r_inv = rsqrt_approx_ftz_f32(thr_sum2 + 1e-6);

            let mut thr_gamma = [bf16::ZERO; ELEMENTS_PER_THREAD];
            for i in 0..ELEMENTS_PER_THREAD / ELEMENTS_PER_LOAD_PER_THREAD {
                let idx = lane_id() as usize + i * ELEMENTS_PER_LOAD_PER_WARP;
                as_aligned_slice_mut(&mut thr_gamma[idx..])[0] =
                    as_aligned_slice(&smem_gamma[idx..])[0];
            }

            for (val, &gamma) in as_bf16x2_slice_mut(&mut thr_row)
                .iter_mut()
                .zip(as_bf16x2_slice(&thr_gamma))
            {
                let (v1, v2) = val.to_f32_pair();
                let (g1, g2) = gamma.to_f32_pair();

                let y1 = v1 * r_inv * g1;
                let y2 = v2 * r_inv * g2;
                *val = bf16x2::from_f32_pair(y1, y2);
            }

            // Write to GMEM.
            for i in 0..ELEMENTS_PER_THREAD / ELEMENTS_PER_LOAD_PER_THREAD {
                let idx = lane_id() as usize + i * ELEMENTS_PER_LOAD_PER_WARP;
                as_aligned_slice_mut(&mut y[idx..])[0] = as_aligned_slice(&thr_row[idx..])[0];
            }

            if lane_id() == 0 {
                r_inv_buf[row_idx] = bf16::from_f32(r_inv);
            }
        }
    }
}

#[kernel]
#[launch_bounds(256, 1)]
pub fn rmsnorm_bf16_warp_persistent_inner2048_sm90_sm10x(
    x: &[bf16],
    gamma: &[bf16],
    y: &mut [bf16],
    r_inv: &mut [bf16],
    num_vectors: usize,
    stride: usize,
) {
    rmsnorm_bf16_warp_persistent::<2048, 64, 2, { PrefetchMode::CpAsyncBulk }>(
        x,
        gamma,
        y,
        r_inv,
        num_vectors,
        stride,
    );
}
