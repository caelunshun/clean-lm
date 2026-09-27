#[cfg(has_tma)]
use cuda_device::{
    Barrier,
    barrier::{
        fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
        mbarrier_try_wait_parity,
    },
    tma,
};
use cuda_device::{
    DynamicSharedArray, async_copy,
    atomic::DeviceAtomicF32,
    blockIdx_x,
    cooperative_groups::{ThreadGroup, this_thread_block},
    float::rsqrt_approx_ftz_f32,
    shared::dynamic_smem_size,
    uniform::grid_dim_x,
    warp::{self, warp_id},
};
use cuda_device_ext::{bf16, bf16x2};
use cuda_macros::{cuda_annotate, kernel, launch_bounds};

use crate::{
    WARP_SIZE, collective_async_copy_g2s_128bit, elements_per_u128, lane_id, red_add_global_f32,
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
#[cuda_annotate]
#[allow(clippy::assertions_on_constants)] // const assert would trigger bug with generic_const_exprs
fn rmsnorm_bf16_warp_persistent<
    const VECTOR_SIZE: usize,
    const PREFETCH_STAGES: usize,
    const NUM_WARPS: usize,
>(
    gmem_x: *const bf16,
    gmem_gamma: *const bf16,
    gmem_y: *mut bf16,
    gmem_r_inv: *mut f32,
    num_rows: u32,
) {
    unsafe {
        #[cfg(has_tma)]
        let num_mbarriers = PREFETCH_STAGES * NUM_WARPS;
        #[cfg(not(has_tma))]
        let num_mbarriers = 0;

        if cfg!(has_227kb_smem) {
            assert!(cfg!(has_tma));
        }

        // Not enough SMEM on consumer TMA GPUs for store staging,
        // so gate on has_227kb_smem insteaed of has_tma.
        let store_staging_space = if cfg!(has_227kb_smem) {
            size_of::<bf16>() * VECTOR_SIZE * NUM_WARPS
        } else {
            0
        };

        let required_smem = size_of::<bf16>() * VECTOR_SIZE
            + size_of::<bf16>() * (VECTOR_SIZE * PREFETCH_STAGES * NUM_WARPS)
            + num_mbarriers * size_of::<cuda_device::Barrier>()
            + store_staging_space;
        assert!(dynamic_smem_size() >= required_smem as u32);

        let smem_base = DynamicSharedArray::<bf16, 128>::get();
        let smem_gamma = smem_base;
        let get_smem_slot = |i| smem_base.add(VECTOR_SIZE + i * VECTOR_SIZE * NUM_WARPS);

        #[cfg(has_tma)]
        let mbarriers = smem_base
            .add(VECTOR_SIZE + PREFETCH_STAGES * VECTOR_SIZE * NUM_WARPS)
            .cast::<Barrier>();

        #[cfg(has_227kb_smem)]
        let store_staging_smem = smem_base.add(
            VECTOR_SIZE
                + PREFETCH_STAGES * NUM_WARPS * VECTOR_SIZE
                + size_of::<Barrier>() * num_mbarriers / size_of::<bf16>(),
        );

        #[cfg(has_tma)]
        let get_mbarrier =
            #[inline(always)]
            |slot: u32| mbarriers.add((warp_id() * PREFETCH_STAGES as u32 + slot) as usize);

        // Each warp initializes its mbarriers.
        #[cfg(has_tma)]
        if lane_id() == 0 {
            #[llvm_unroll]
            for i in 0..PREFETCH_STAGES {
                mbarrier_init(get_mbarrier(i as u32), 1);
            }
        }
        #[cfg(has_tma)]
        fence_proxy_async_shared_cta();

        // Warp 0 copies gamma into SMEM.
        if warp_id() == 0 {
            collective_async_copy_g2s_128bit(
                gmem_gamma,
                smem_gamma,
                VECTOR_SIZE as u32,
                this_thread_block().tiled_partition::<32>(),
                32,
            );
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_all();
        }

        this_thread_block().sync();

        let do_prefetch = |row_idx: u32, slot_index: u32| {
            if row_idx < num_rows {
                let dst = get_smem_slot(slot_index as usize)
                    .add((VECTOR_SIZE as u32 * warp_id()) as usize);
                let src = gmem_x.add(row_idx as usize * VECTOR_SIZE);

                #[cfg(has_tma)]
                if lane_id() == 0 {
                    let bytes = (VECTOR_SIZE * size_of::<bf16>()) as u32;
                    mbarrier_arrive_expect_tx(get_mbarrier(slot_index), 1, bytes);
                    tma::cp_async_bulk_g2s_cta(
                        dst.cast(),
                        src.cast(),
                        bytes,
                        get_mbarrier(slot_index),
                    );
                }

                #[cfg(not(has_tma))]
                collective_async_copy_g2s_128bit(
                    src,
                    dst,
                    VECTOR_SIZE as u32,
                    this_thread_block().tiled_partition::<32>(),
                    32,
                );
            }
            #[cfg(not(has_tma))]
            async_copy::cp_async_commit_group();
        };

        let cta_stride = NUM_WARPS as u32 * grid_dim_x().get();
        let first_row = warp_id() + blockIdx_x() * NUM_WARPS as u32;

        // Initial pipeline fill.
        #[llvm_unroll]
        for i in 0..PREFETCH_STAGES as u32 {
            do_prefetch(first_row + i * cta_stride, i);
        }

        // Replicate gamma into registers.
        let mut thr_gamma = [bf16::ZERO; { VECTOR_SIZE / WARP_SIZE as usize }];
        let chunks_per_thread = VECTOR_SIZE as u32 / WARP_SIZE / elements_per_u128::<bf16>();
        #[llvm_unroll]
        for j in 0..chunks_per_thread {
            let src = smem_gamma.add(
                (lane_id() * elements_per_u128::<bf16>()
                    + j * elements_per_u128::<bf16>() * WARP_SIZE) as usize,
            );
            let dst = thr_gamma
                .as_mut_ptr()
                .add((j * elements_per_u128::<bf16>()) as usize);
            dst.cast::<u128>().write(src.cast::<u128>().read());
        }

        let mut row_idx = first_row;
        let mut i = 0;
        while row_idx < num_rows {
            let slot_idx = i as u32 % PREFETCH_STAGES as u32;

            // Wait for our SMEM slot to be ready.
            #[cfg(has_tma)]
            {
                let parity = i as u32 / PREFETCH_STAGES as u32 % 2;
                while !mbarrier_try_wait_parity(get_mbarrier(slot_idx), parity) {}
            }
            #[cfg(not(has_tma))]
            {
                let pipeline_depth_minus_one = const { PREFETCH_STAGES as u32 - 1 };
                async_copy::cp_async_wait_group(pipeline_depth_minus_one);
            }

            // Copy row from SMEM into registers.
            let mut thr_x = [bf16::ZERO; { VECTOR_SIZE / WARP_SIZE as usize }];

            let chunks_per_thread = VECTOR_SIZE as u32 / WARP_SIZE / elements_per_u128::<bf16>();
            #[llvm_unroll]
            for j in 0..chunks_per_thread {
                let src = get_smem_slot(slot_idx as usize)
                    .add((warp_id() * VECTOR_SIZE as u32) as usize)
                    .add(
                        (lane_id() * elements_per_u128::<bf16>()
                            + j * elements_per_u128::<bf16>() * WARP_SIZE)
                            as usize,
                    );
                let dst = thr_x
                    .as_mut_ptr()
                    .add((j * elements_per_u128::<bf16>()) as usize);
                dst.cast::<u128>().write(src.cast::<u128>().read());
            }

            // With the SMEM data consumed, we can prefetch the next stage.
            #[cfg(has_tma)]
            warp::sync_mask(u32::MAX);
            do_prefetch(row_idx + cta_stride * PREFETCH_STAGES as u32, slot_idx);

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
            let r_inv = rsqrt_approx_ftz_f32(thr_sum2 * (1.0 / VECTOR_SIZE as f32) + 1e-6);

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

            let gmem_y = gmem_y.add(row_idx as usize * VECTOR_SIZE);

            // Write result to GMEM.
            #[cfg(has_227kb_smem)]
            {
                // Stage through SMEM, then bulk copy.
                if lane_id() == 0 {
                    tma::cp_async_bulk_wait_group_read(0);
                }
                warp::sync_mask(u32::MAX);
                #[llvm_unroll]
                for j in 0..chunks_per_thread {
                    let src = thr_x
                        .as_ptr()
                        .add((j * elements_per_u128::<bf16>()) as usize);
                    let dst = store_staging_smem.add(
                        (warp_id() * VECTOR_SIZE as u32
                            + lane_id() * elements_per_u128::<bf16>()
                            + j * elements_per_u128::<bf16>() * WARP_SIZE)
                            as usize,
                    );
                    dst.cast::<u128>().write(src.cast::<u128>().read());
                }

                fence_proxy_async_shared_cta();
                warp::sync_mask(u32::MAX);

                if lane_id() == 0 {
                    let bytes = (VECTOR_SIZE * size_of::<bf16>()) as u32;
                    tma::cp_async_bulk_s2g(
                        gmem_y.cast(),
                        store_staging_smem
                            .add((warp_id() * VECTOR_SIZE as u32) as usize)
                            .cast(),
                        bytes,
                    );
                    tma::cp_async_bulk_commit_group();
                }
            }
            #[cfg(not(has_227kb_smem))]
            {
                // Direct store to GMEM.
                #[llvm_unroll]
                for j in 0..chunks_per_thread {
                    let src = thr_x
                        .as_ptr()
                        .add((j * elements_per_u128::<bf16>()) as usize);
                    let dst = gmem_y.add(
                        (lane_id() * elements_per_u128::<bf16>()
                            + j * elements_per_u128::<bf16>() * WARP_SIZE)
                            as usize,
                    );
                    dst.cast::<u128>().write(src.cast::<u128>().read());
                }
            }

            let gmem_r_inv = gmem_r_inv.add(row_idx as usize);
            if lane_id() == 0 {
                gmem_r_inv.write(r_inv);
            }

            i += 1;
            row_idx += cta_stride;
        }

        #[cfg(has_227kb_smem)]
        if lane_id() == 0 && i > 0 {
            tma::cp_async_bulk_wait_group_read(0);
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
    num_rows: u32,
) {
    cfg_select! {
        has_227kb_smem => {
            const PREFETCH_STAGES: usize = 3;
        }
        _ => {
            const PREFETCH_STAGES: usize = 1;
        }
    }

    rmsnorm_bf16_warp_persistent::<2048, PREFETCH_STAGES, 12>(
        gmem_x, gmem_gamma, gmem_y, gmem_r_inv, num_rows,
    );
}

/// Template for fast backward RMSNorm performed cooperatively by a CTA.
/// Each CTA handles `ROWS_PER_CTA` rows per persistent loop iteration;
/// the elements of each row are distributed across all warps.
/// Optionally adds a residual gradient stream to `dx` (if residual_gradient_stream
/// is not null).
///
/// Uses dynamic shared memory for prefetching
/// and inter-warp reduction.
/// `gamma` is kept stationary in registers.
#[cuda_annotate]
fn rmsnorm_bwd_bf16_cta_persistent<
    const VECTOR_SIZE: usize,
    const ROWS_PER_ITER: usize,
    const PREFETCH_STAGES: usize,
    const NUM_WARPS: usize,
>(
    gmem_x: *const bf16,
    gmem_dy: *const bf16,
    gmem_residual_gradient_stream: *const bf16,
    gmem_gamma: *const bf16,
    gmem_r_inv: *const f32,
    gmem_dx: *mut bf16,
    gmem_dgamma: *mut DeviceAtomicF32,
    num_rows: u32,
) {
    assert!(
        VECTOR_SIZE
            .is_multiple_of(NUM_WARPS * WARP_SIZE as usize * elements_per_u128::<bf16>() as usize)
    );
    assert!(ROWS_PER_ITER <= WARP_SIZE as usize);

    let thread_id = this_thread_block().thread_rank();
    let cta_threads = WARP_SIZE * NUM_WARPS as u32;

    let elements_per_row_per_thread = VECTOR_SIZE / NUM_WARPS / WARP_SIZE as usize;

    let required_smem = (4 * size_of::<bf16>() * VECTOR_SIZE * ROWS_PER_ITER * PREFETCH_STAGES)
        + (size_of::<f32>() * ROWS_PER_ITER * PREFETCH_STAGES)
        + (size_of::<f32>() * ROWS_PER_ITER * NUM_WARPS);
    assert!(dynamic_smem_size() >= required_smem as u32);

    unsafe {
        let smem_base = DynamicSharedArray::<bf16, 256>::get();

        let get_x_prefetch_slot =
            move |slot: u32| smem_base.add(slot as usize * VECTOR_SIZE * ROWS_PER_ITER);
        let get_dy_prefetch_slot = move |slot: u32| {
            smem_base.add(
                VECTOR_SIZE * ROWS_PER_ITER * PREFETCH_STAGES
                    + slot as usize * VECTOR_SIZE * ROWS_PER_ITER,
            )
        };
        let get_residual_prefetch_slot = move |slot: u32| {
            smem_base.add(
                2 * VECTOR_SIZE * ROWS_PER_ITER * PREFETCH_STAGES
                    + slot as usize * VECTOR_SIZE * ROWS_PER_ITER,
            )
        };
        // one f32 per row.
        let get_r_inv_prefetch_slot = move |slot: u32| {
            smem_base
                .add(
                    3 * VECTOR_SIZE * PREFETCH_STAGES * ROWS_PER_ITER
                        + slot as usize * ROWS_PER_ITER * (size_of::<f32>() / size_of::<bf16>()),
                )
                .cast::<f32>()
        };

        // one f32 value per row, per warp in SMEM for inter-warp reductions.
        let reduction_scratchpad = smem_base
            .add(
                3 * VECTOR_SIZE * PREFETCH_STAGES * ROWS_PER_ITER
                    + PREFETCH_STAGES * ROWS_PER_ITER * (size_of::<f32>() / size_of::<bf16>()),
            )
            .cast::<f32>();

        // Synchronously load gamma to keep in registers.
        let mut thr_gamma = [bf16::ZERO; { VECTOR_SIZE / NUM_WARPS / WARP_SIZE as usize }];

        let gamma_loads_per_thread = thr_gamma.len() as u32 / elements_per_u128::<bf16>();
        #[llvm_unroll]
        for i in 0..gamma_loads_per_thread {
            let src = gmem_gamma
                .cast::<u128>()
                .add((thread_id + i * cta_threads) as usize);
            let dst = thr_gamma.as_mut_ptr().cast::<u128>().add(i as usize);
            dst.write(src.read());
        }

        let do_prefetch = #[inline(always)]
        |first_row_idx: u32, slot_idx: u32| {
            #[llvm_unroll]
            for row_idx in first_row_idx..(first_row_idx + ROWS_PER_ITER as u32) {
                if row_idx < num_rows {
                    // `x`
                    let dst = get_x_prefetch_slot(slot_idx)
                        .add(VECTOR_SIZE * (row_idx - first_row_idx) as usize);
                    let src = gmem_x.add(row_idx as usize * VECTOR_SIZE);
                    collective_async_copy_g2s_128bit(
                        src,
                        dst,
                        VECTOR_SIZE as u32,
                        this_thread_block(),
                        cta_threads,
                    );

                    // `dy`
                    let dst = get_dy_prefetch_slot(slot_idx)
                        .add(VECTOR_SIZE * (row_idx - first_row_idx) as usize);
                    let src = gmem_dy.add(row_idx as usize * VECTOR_SIZE);
                    collective_async_copy_g2s_128bit(
                        src,
                        dst,
                        VECTOR_SIZE as u32,
                        this_thread_block(),
                        cta_threads,
                    );

                    // residual
                    if !gmem_residual_gradient_stream.is_null() {
                        let dst = get_residual_prefetch_slot(slot_idx)
                            .add(VECTOR_SIZE * (row_idx - first_row_idx) as usize);
                        let src = gmem_residual_gradient_stream.add(row_idx as usize * VECTOR_SIZE);
                        collective_async_copy_g2s_128bit(
                            src,
                            dst,
                            VECTOR_SIZE as u32,
                            this_thread_block(),
                            cta_threads,
                        );
                    }

                    // `r_inv` (all fetches issued by thread 0)
                    if thread_id == 0 {
                        let dst = get_r_inv_prefetch_slot(slot_idx)
                            .add((row_idx - first_row_idx) as usize);
                        let src = gmem_r_inv.add(row_idx as usize);
                        async_copy::cp_async_ca_4(dst.cast(), src.cast());
                    }
                }
            }
            async_copy::cp_async_commit_group();
        };

        if gmem_residual_gradient_stream.is_null() {
            // Fill residual's SMEM slots with zeroes so we effectively don't add a residual
            #[llvm_unroll]
            for slot_idx in 0..PREFETCH_STAGES as u32 {
                #[llvm_unroll]
                for i in 0..VECTOR_SIZE * ROWS_PER_ITER
                    / NUM_WARPS
                    / WARP_SIZE as usize
                    / elements_per_u128::<bf16>() as usize
                {
                    get_residual_prefetch_slot(slot_idx)
                        .cast::<u128>()
                        .add((thread_id + i as u32 * cta_threads) as usize)
                        .write(0);
                }
            }
        }

        let cta_stride = ROWS_PER_ITER as u32 * grid_dim_x().get();

        // initial pipeline fill
        #[llvm_unroll]
        for i in 0..PREFETCH_STAGES {
            let first_row_idx = blockIdx_x() * ROWS_PER_ITER as u32 + i as u32 * cta_stride;
            do_prefetch(first_row_idx, i as u32);
        }

        // dgamma partial sums accumulated per thread across the persistent loop,
        // then final sum via GMEM atomics
        let mut thr_dgamma_partials = [0.0f32; { VECTOR_SIZE / NUM_WARPS / WARP_SIZE as usize }];

        let mut first_row_idx = blockIdx_x() * ROWS_PER_ITER as u32;
        let mut slot_idx = 0;

        while first_row_idx < num_rows {
            let prefetch_stages_minus_one = const { PREFETCH_STAGES as u32 - 1 };
            async_copy::cp_async_wait_group(prefetch_stages_minus_one);

            // Consume data from SMEM
            let mut thr_x =
                [bf16::ZERO; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            let mut thr_dy =
                [bf16::ZERO; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            let mut thr_residual =
                [bf16::ZERO; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            let mut thr_r_inv = [0.0f32; ROWS_PER_ITER];

            let num_loads = thr_x.len() as u32 / elements_per_u128::<bf16>();
            #[llvm_unroll]
            for i in 0..num_loads {
                // `x`
                let src = get_x_prefetch_slot(slot_idx)
                    .cast::<u128>()
                    .add((thread_id + i * cta_threads) as usize);
                let dst = thr_x.as_mut_ptr().cast::<u128>().add(i as usize);
                dst.write(src.read());

                // `dy`
                let src = get_dy_prefetch_slot(slot_idx)
                    .cast::<u128>()
                    .add((thread_id + i * cta_threads) as usize);
                let dst = thr_dy.as_mut_ptr().cast::<u128>().add(i as usize);
                dst.write(src.read());

                // `residual`
                let src = get_residual_prefetch_slot(slot_idx)
                    .cast::<u128>()
                    .add((thread_id + i * cta_threads) as usize);
                let dst = thr_residual.as_mut_ptr().cast::<u128>().add(i as usize);
                dst.write(src.read());
            }

            // `r_inv`
            // need to wait for the leader warp who issued the load
            this_thread_block().sync();
            #[llvm_unroll]
            for i in 0..ROWS_PER_ITER {
                thr_r_inv[i] = get_r_inv_prefetch_slot(slot_idx).add(i).read();
            }
            this_thread_block().sync();

            // Once we're done consuming SMEM slots, can prefetch next row.
            do_prefetch(
                first_row_idx + cta_stride * PREFETCH_STAGES as u32,
                slot_idx,
            );

            // Compute `x_hat = x * r_inv` and `g = dy * gamma`,
            // and accumulate per-thread per-row partial sum `c = sum(g * x_hat)`
            let mut thr_x_hat =
                [0.0f32; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            let mut thr_g =
                [0.0f32; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            let mut thr_c = [0.0f32; ROWS_PER_ITER];

            #[llvm_unroll]
            for i in 0..ROWS_PER_ITER {
                if first_row_idx + i as u32 >= num_rows {
                    continue;
                }
                let offs = i * elements_per_row_per_thread;
                let r_inv = thr_r_inv[i];
                #[llvm_unroll]
                for j in 0..elements_per_row_per_thread / 2 {
                    let (x1, x2) =
                        bf16x2::new(thr_x[offs + j * 2], thr_x[offs + j * 2 + 1]).to_f32_pair();
                    let (dy1, dy2) =
                        bf16x2::new(thr_dy[offs + j * 2], thr_dy[offs + j * 2 + 1]).to_f32_pair();
                    let (gamma1, gamma2) =
                        bf16x2::new(thr_gamma[j * 2], thr_gamma[j * 2 + 1]).to_f32_pair();

                    let x_hat1 = x1 * r_inv;
                    let x_hat2 = x2 * r_inv;

                    let g1 = dy1 * gamma1;
                    let g2 = dy2 * gamma2;

                    thr_x_hat[offs + j * 2] = x_hat1;
                    thr_x_hat[offs + j * 2 + 1] = x_hat2;
                    thr_g[offs + j * 2] = g1;
                    thr_g[offs + j * 2 + 1] = g2;

                    thr_c[i] += (g1 * x_hat1 + g2 * x_hat2) * (1.0 / VECTOR_SIZE as f32);
                }
            }

            // Intra-warp reduction of `c`
            #[llvm_unroll]
            for c in &mut thr_c {
                *c = warp::reduce_sum_f32(*c);
            }

            // Inter-warp reduction of `c` by sending each warp's per-row contributions
            // to SMEM, then having each warp load all warps' values and independently compute
            // the CTA-wide sum.
            if lane_id() == 0 {
                #[llvm_unroll]
                for i in 0..ROWS_PER_ITER {
                    reduction_scratchpad
                        .add(i + warp_id() as usize * ROWS_PER_ITER)
                        .write(thr_c[i]);
                }
            }
            this_thread_block().sync();

            #[llvm_unroll]
            for i in 0..ROWS_PER_ITER {
                thr_c[i] = 0.0;
                #[llvm_unroll]
                for warp in 0..NUM_WARPS {
                    thr_c[i] += reduction_scratchpad.add(i + warp * ROWS_PER_ITER).read();
                }
            }

            // compute dx = r_inv * (g - x_hat * c) + residual
            // and add contribution dgamma += dy * x_hat
            let mut thr_dx =
                [bf16::ZERO; { VECTOR_SIZE * ROWS_PER_ITER / NUM_WARPS / WARP_SIZE as usize }];
            #[llvm_unroll]
            for i in 0..ROWS_PER_ITER {
                if first_row_idx + i as u32 >= num_rows {
                    continue;
                }
                let r_inv = thr_r_inv[i];
                let c = thr_c[i];
                let offs = i * elements_per_row_per_thread;
                #[llvm_unroll]
                for j in 0..elements_per_row_per_thread / 2 {
                    let x_hat1 = thr_x_hat[offs + j * 2];
                    let x_hat2 = thr_x_hat[offs + j * 2 + 1];
                    let (residual1, residual2) =
                        bf16x2::new(thr_residual[offs + j * 2], thr_residual[offs + j * 2 + 1])
                            .to_f32_pair();
                    let g1 = thr_g[offs + j * 2];
                    let g2 = thr_g[offs + j * 2 + 1];
                    let (dy1, dy2) =
                        bf16x2::new(thr_dy[offs + j * 2], thr_dy[offs + j * 2 + 1]).to_f32_pair();

                    let dx1 = r_inv * (g1 - x_hat1 * c) + residual1;
                    let dx2 = r_inv * (g2 - x_hat2 * c) + residual2;
                    (thr_dx[offs + j * 2], thr_dx[offs + j * 2 + 1]) =
                        bf16x2::from_f32_pair(dx1, dx2).to_bf16_pair();

                    thr_dgamma_partials[j * 2] += dy1 * x_hat1;
                    thr_dgamma_partials[j * 2 + 1] += dy2 * x_hat2;
                }
            }

            // store dx to GMEM
            #[llvm_unroll]
            for i in 0..ROWS_PER_ITER {
                if first_row_idx + i as u32 >= num_rows {
                    continue;
                }
                #[llvm_unroll]
                for j in 0..elements_per_row_per_thread / elements_per_u128::<bf16>() as usize {
                    let src = thr_dx
                        .as_ptr()
                        .add(i * elements_per_row_per_thread)
                        .cast::<u128>()
                        .add(j);
                    let dst = gmem_dx
                        .add(VECTOR_SIZE * (first_row_idx as usize + i))
                        .cast::<u128>()
                        .add(thread_id as usize + j * cta_threads as usize);
                    dst.write(src.read());
                }
            }

            first_row_idx += cta_stride;
            slot_idx = (slot_idx + 1) % PREFETCH_STAGES as u32;
        }

        // Add our contribution to dgamma to the global sums.
        #[llvm_unroll]
        for i in 0..thr_dgamma_partials.len() / elements_per_u128::<bf16>() as usize {
            #[llvm_unroll]
            for j in 0..elements_per_u128::<bf16>() as usize {
                let src = thr_dgamma_partials[i * elements_per_u128::<bf16>() as usize + j];
                let dst = gmem_dgamma.add(
                    (i * cta_threads as usize + thread_id as usize)
                        * elements_per_u128::<bf16>() as usize
                        + j,
                );
                red_add_global_f32(dst.cast(), src);
            }
        }
    }
}

#[kernel]
#[launch_bounds(256, 1)]
pub fn rmsnorm_bwd_bf16_cta_persistent_inner2048(
    gmem_x: *const bf16,
    gmem_dy: *const bf16,
    gmem_residual_gradient_stream: *const bf16,
    gmem_gamma: *const bf16,
    gmem_r_inv: *const f32,
    gmem_dx: *mut bf16,
    gmem_dgamma: *mut DeviceAtomicF32,
    num_rows: u32,
) {
    cfg_select! {
        has_227kb_smem => {
            const PREFETCH_STAGES: usize = 3;
        }
        _ => {
            const PREFETCH_STAGES: usize = 1;
        }
    }

    rmsnorm_bwd_bf16_cta_persistent::<2048, 4, PREFETCH_STAGES, 8>(
        gmem_x,
        gmem_dy,
        gmem_residual_gradient_stream,
        gmem_gamma,
        gmem_r_inv,
        gmem_dx,
        gmem_dgamma,
        num_rows,
    );
}
