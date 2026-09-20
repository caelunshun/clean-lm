use cuda_device::{
    DynamicSharedArray, async_copy, blockIdx_x,
    cooperative_groups::{ThreadGroup, this_thread_block},
    float::rsqrt_approx_ftz_f32,
    uniform::grid_dim_x,
    warp::{self, warp_id},
};
use cuda_device_ext::{bf16, bf16x2};
use cuda_macros::{device, kernel, launch_bounds};

use crate::{WARP_SIZE, collective_async_copy_g2s_128bit, elements_per_u128, lane_id};

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
#[device]
fn rmsnorm_bf16_warp_persistent<
    const INNER_SIZE: usize,
    const ELEMENTS_PER_THREAD: usize,
    const PREFETCH_STAGES: usize,
    const NUM_WARPS: usize,
>(
    gmem_x: *const bf16,
    gmem_gamma: *const bf16,
    gmem_y: *mut bf16,
    gmem_r_inv: *mut f32,
    num_vectors: u32,
) {
    unsafe {
        let smem_base = DynamicSharedArray::<bf16, 128>::get();
        let smem_gamma = smem_base;
        let get_smem_slot = |i| smem_base.add((i + 1) * INNER_SIZE * NUM_WARPS);

        // Warp 0 copies gamma into SMEM.
        if warp_id() == 0 {
            collective_async_copy_g2s_128bit(
                gmem_gamma,
                smem_gamma,
                INNER_SIZE as u32,
                this_thread_block().tiled_partition::<32>(),
            );
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_all();
        }
        this_thread_block().sync();

        let do_prefetch = |row_idx: u32, slot_index: u32| {
            let dst =
                get_smem_slot(slot_index as usize).add((INNER_SIZE as u32 * warp_id()) as usize);
            let src = gmem_x.add(row_idx as usize * INNER_SIZE);
            collective_async_copy_g2s_128bit(
                src,
                dst,
                INNER_SIZE as u32,
                this_thread_block().tiled_partition::<32>(),
            );
            async_copy::cp_async_commit_group();
        };

        let cta_stride = NUM_WARPS as u32 * grid_dim_x().get();
        let first_row = warp_id() + blockIdx_x() * NUM_WARPS as u32;

        // Initial pipeline fill.
        #[llvm_unroll]
        for i in 0..PREFETCH_STAGES as u32 {
            let row_idx = first_row + i * cta_stride;
            if row_idx < num_vectors {
                do_prefetch(row_idx, i);
            }
        }

        let mut row_idx = first_row;
        let mut i = 0;
        while row_idx < num_vectors {
            let slot_idx = i as u32 % PREFETCH_STAGES as u32;

            // Wait for our SMEM slot to be ready.
            let pipeline_depth_minus_one = const { PREFETCH_STAGES as u32 - 1 };
            async_copy::cp_async_wait_group(pipeline_depth_minus_one);

            // Copy row and gamma from SMEM into registers.
            let mut thr_x = [bf16::ZERO; { INNER_SIZE / WARP_SIZE as usize }];
            let mut thr_gamma = [bf16::ZERO; { INNER_SIZE / WARP_SIZE as usize }];

            let chunks_per_thread = INNER_SIZE as u32 / WARP_SIZE / elements_per_u128::<bf16>();
            #[llvm_unroll]
            for j in 0..chunks_per_thread {
                let src = get_smem_slot(slot_idx as usize)
                    .add((warp_id() * INNER_SIZE as u32) as usize)
                    .add(
                        (lane_id() * elements_per_u128::<bf16>()
                            + j * elements_per_u128::<bf16>() * WARP_SIZE)
                            as usize,
                    );
                let dst = thr_x
                    .as_mut_ptr()
                    .add((j * elements_per_u128::<bf16>()) as usize);
                dst.cast::<u128>().write(src.cast::<u128>().read());

                let src = smem_gamma.add(
                    (lane_id() * elements_per_u128::<bf16>()
                        + j * elements_per_u128::<bf16>() * WARP_SIZE) as usize,
                );
                let dst = thr_gamma
                    .as_mut_ptr()
                    .add((j * elements_per_u128::<bf16>()) as usize);
                dst.cast::<u128>().write(src.cast::<u128>().read());
            }

            // With the SMEM data consumed, we can prefetch the next stage.
            warp::sync_mask(u32::MAX);
            let next_row_idx = row_idx + cta_stride * PREFETCH_STAGES as u32;
            if next_row_idx < num_vectors {
                do_prefetch(next_row_idx, slot_idx);
            }

            // Compute sum(x^2).
            let mut thr_sum2 = 0.0f32;
            #[llvm_unroll]
            for pair in thr_x.chunks_exact(2) {
                let pair = bf16x2::new(pair[0], pair[1]);
                let (v1, v2) = pair.to_f32_pair();
                thr_sum2 += v1 * v1 + v2 * v2;
            }

            // Intra-warp reduction.
            #[llvm_unroll]
            for i in 0..WARP_SIZE.ilog2() {
                thr_sum2 += warp::shuffle_xor_f32_sync(u32::MAX, thr_sum2, 1 << i);
            }

            // Compute reciprocal divisor.
            let r_inv = rsqrt_approx_ftz_f32(thr_sum2 * (1.0 / INNER_SIZE as f32) + 1e-6);

            // Apply normalization to the data.
            #[llvm_unroll]
            for i in 0..thr_x.len() / 2 {
                let x_pair = &mut thr_x[i * 2..][..2];
                let gamma_pair = &mut thr_gamma[i * 2..][..2];
                let (x1, x2) = bf16x2::new(x_pair[0], x_pair[1]).to_f32_pair();
                let (g1, g2) = bf16x2::new(gamma_pair[0], gamma_pair[1]).to_f32_pair();
                let (y1, y2) = (x1 * g1 * r_inv, x2 * g2 * r_inv);
                (x_pair[0], x_pair[1]) = bf16x2::from_f32_pair(y1, y2).to_bf16_pair();
            }

            let gmem_y = gmem_y.add(row_idx as usize * INNER_SIZE);

            // Write result to GMEM.
            #[llvm_unroll]
            for j in 0..chunks_per_thread {
                let src = thr_x
                    .as_ptr()
                    .add((j * elements_per_u128::<bf16>()) as usize);
                let dst = gmem_y.add(
                    (lane_id() * elements_per_u128::<bf16>()
                        + j * elements_per_u128::<bf16>() * WARP_SIZE) as usize,
                );
                dst.cast::<u128>().write(src.cast::<u128>().read());
            }

            let gmem_r_inv = gmem_r_inv.add(row_idx as usize);
            if lane_id() == 0 {
                gmem_r_inv.write(r_inv);
            }

            i += 1;
            row_idx += cta_stride;
        }
    }
}

#[kernel]
#[launch_bounds(384, 1)]
pub fn rmsnorm_bf16_warp_persistent_inner2048(
    gmem_x: *const bf16,
    gmem_gamma: *const bf16,
    gmem_y: *mut bf16,
    gmem_r_inv: *mut f32,
    num_vectors: u32,
) {
    cfg_select! {
        any(
            cuda_arch_specific = "90",
            cuda_arch_specific = "100",
            cuda_arch_specific = "103",
            cuda_arch_specific = "107"
        ) => {
            const PREFETCH_STAGES: usize = 4;
        }
        _ => {
            const PREFETCH_STAGES: usize = 1;
        }
    }

    rmsnorm_bf16_warp_persistent::<2048, 64, PREFETCH_STAGES, 12>(
        gmem_x,
        gmem_gamma,
        gmem_y,
        gmem_r_inv,
        num_vectors,
    );
}
