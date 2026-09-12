#include "utils.cuh"

// Computes RMSNorm(x) * gamma.
//
// Outputs the RMSNorm scale (1/r) for use in backward pass.
template <int INNER_DIM, int BLOCK_SIZE>
__tile__ void rmsnorm_fwd_impl(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gamma,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ out_r_inv,
    int batch_size)
{
    ASSUME_ALIGNED(x);
    ASSUME_ALIGNED(gamma);
    ASSUME_ALIGNED(out);
    ASSUME_ALIGNED(out_r_inv);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto gamma_view = ct::partition_view{
        ct::tensor_span{gamma, ct::extents{1, INNER_DIM}},
        ct::shape<1, INNER_DIM>{}};
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto out_r_inv_view = ct::partition_view{
        ct::tensor_span{out_r_inv, ct::extents{batch_size}},
        ct::shape<BLOCK_SIZE>{}};

    int num_tiles = ct::ceildiv(batch_size, BLOCK_SIZE);

    for (int tile_idx : ct::irange((int)ct::bid().x, num_tiles, (int)ct::num_blocks().x))
    {
        auto x_tile = ct::element_cast<float>(x_view.load_masked(tile_idx, 0));

        auto x2 = x_tile * x_tile;
        auto x2_mean = ct::sum(x2, 1_ic) * (1.0f / static_cast<float>(INNER_DIM));
        auto r_inv = ct::rsqrt(x2_mean + 1e-6f, ct::round_subnormals_to_zero_t{});

        auto x_norm = x_tile * r_inv * gamma_view.load_masked(0, 0);

        out_view.store_masked(ct::element_cast<__nv_bfloat16>(x_norm), tile_idx, 0);
        out_r_inv_view.store_masked(ct::reshape(r_inv, ct::shape<BLOCK_SIZE>{}), tile_idx);
    }
}

extern "C" __tile_global__ void rmsnorm_fwd_inner2048_persistent(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gamma,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ out_r_inv,
    int batch_size)
{
    rmsnorm_fwd_impl<2048, 4>(x, gamma, out, out_r_inv, batch_size);
}

// Backward pass for rmsnorm computing the gradient w.r.t.
// the input, optionally adding a residual gradient stream
// to the gradient before writing.
//
// dgamma must be pre-zeroed to enable atomic accumulation.
template <int INNER_DIM, int BLOCK_SIZE_BATCH, int BLOCK_SIZE_INNER>
__tile__ void rmsnorm_bwd_impl(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ dy,
    const __nv_bfloat16 *__restrict__ residual_gradient_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    const float *__restrict__ r_inv,
    __nv_bfloat16 *__restrict__ dx,
    float *__restrict__ dgamma,
    int batch_size)
{
    ASSUME_ALIGNED(x);
    ASSUME_ALIGNED(dy);
    ASSUME_ALIGNED(residual_gradient_stream);
    ASSUME_ALIGNED(gamma);
    ASSUME_ALIGNED(r_inv);
    ASSUME_ALIGNED(dx);
    ASSUME_ALIGNED(dgamma);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER>{}};
    auto dy_view = ct::partition_view{
        ct::tensor_span{dy, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER>{}};
    auto residual_gradient_stream_view = ct::partition_view{
        ct::tensor_span{residual_gradient_stream, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER>{}};
    auto dx_view = ct::partition_view{
        ct::tensor_span{dx, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER>{}};
    auto dgamma_view = ct::partition_view{
        ct::tensor_span{dgamma, ct::extents{1, INNER_DIM}},
        ct::shape<1, BLOCK_SIZE_INNER>{}};
    auto gamma_view = ct::partition_view{
        ct::tensor_span{gamma, ct::extents{1, INNER_DIM}},
        ct::shape<1, BLOCK_SIZE_INNER>{}};
    auto r_inv_view = ct::partition_view{
        ct::tensor_span{r_inv, ct::extents{batch_size}},
        ct::shape<BLOCK_SIZE_BATCH>{}};

    int num_tiles_batch = ct::ceildiv(batch_size, BLOCK_SIZE_BATCH);
    int num_tiles_inner = ct::ceildiv(INNER_DIM, BLOCK_SIZE_INNER);
    int num_tiles = num_tiles_batch * num_tiles_inner;

    for (int tile_idx : ct::irange((int)ct::bid().x, num_tiles, (int)ct::num_blocks().x))
    {
        int tile_idx_batch = tile_idx / num_tiles_inner;
        int tile_idx_inner = tile_idx % num_tiles_inner;

        auto x_tile = ct::element_cast<float>(x_view.load_masked(tile_idx_batch, tile_idx_inner));

        auto r_inv_tile = ct::reshape(r_inv_view.load_masked(tile_idx_batch), ct::shape<BLOCK_SIZE_BATCH, 1>{});

        auto x_hat = x_tile * r_inv_tile;

        auto dy_tile = ct::element_cast<float>(dy_view.load_masked(tile_idx_batch, tile_idx_inner));
        auto gamma_tile = ct::element_cast<float>(gamma_view.load_masked(0, tile_idx_inner));

        auto g = dy_tile * gamma_tile;

        auto c = ct::sum(g * x_hat, 1_ic) * (1.0f / static_cast<float>(INNER_DIM));

        auto residual_gradient_stream_tile = ct::zeros<ct::tile<float, ct::shape<BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER>>>();
        if (residual_gradient_stream != nullptr)
        {
            residual_gradient_stream_tile = ct::element_cast<float>(residual_gradient_stream_view.load_masked(tile_idx_batch, tile_idx_inner));
        }

        auto dx_tile = r_inv_tile * (g - x_hat * c) + residual_gradient_stream_tile;

        dx_view.store_masked(ct::element_cast<__nv_bfloat16>(dx_tile), tile_idx_batch, tile_idx_inner);

        auto dgamma_contrib = ct::sum(dy_tile * x_hat, 0_ic);
        dgamma_view.atomic_add(dgamma_contrib, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, 0, tile_idx_inner);
    }
}

extern "C" __tile_global__ void rmsnorm_bwd_inner2048_persistent(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ dy,
    const __nv_bfloat16 *__restrict__ residual_gradient_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    const float *__restrict__ r_inv,
    __nv_bfloat16 *__restrict__ dx,
    float *__restrict__ dgamma,
    int batch_size)
{
    rmsnorm_bwd_impl<2048, 64, 64>(x, dy, residual_gradient_stream,
                                   gamma, r_inv, dx, dgamma, batch_size);
}