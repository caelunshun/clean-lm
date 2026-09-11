#include "utils.cuh"

// Computes RMSNorm(prev_layer_output + residual_stream)
// (residual_stream not added if set to null). The residual stream
// comes from the start of the _previous_ layer while
// prev_layer_output is the value before adding the residual back.
//
// Outputs the RMSNorm scale (1/r) for use in backward pass.
template <int INNER_DIM, int BLOCK_SIZE>
__tile__ void rms_prenorm_fwd_impl(
    const __nv_bfloat16 *__restrict__ prev_layer_output,
    const __nv_bfloat16 *__restrict__ residual_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ out_r_inv,
    int batch_size)
{
    ASSUME_ALIGNED(prev_layer_output);
    ASSUME_ALIGNED(residual_stream);
    ASSUME_ALIGNED(gamma);
    ASSUME_ALIGNED(out);
    ASSUME_ALIGNED(out_r_inv);

    auto prev_layer_output_view = ct::partition_view{
        ct::tensor_span{prev_layer_output, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto residual_stream_view = ct::partition_view{
        ct::tensor_span{residual_stream, ct::extents{batch_size, INNER_DIM}},
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

    auto prev_layer_output_tile = prev_layer_output_view.load_masked(ct::bid().x, 0);
    auto residual_stream_tile = ct::zeros<ct::tile<__nv_bfloat16, ct::shape<BLOCK_SIZE, INNER_DIM>>>();
    if (residual_stream != nullptr)
    {
        residual_stream_tile = residual_stream_view.load_masked(ct::bid().x, 0);
    }

    auto x = ct::element_cast<float>(prev_layer_output_tile) + ct::element_cast<float>(residual_stream_tile);

    auto x2 = x * x;
    auto x2_mean = ct::sum(x2, 1_ic) * (1.0f / static_cast<float>(INNER_DIM));
    auto r_inv = ct::rsqrt(x2_mean + 1e-6f, ct::round_subnormals_to_zero_t{});

    auto x_norm = x * r_inv * gamma_view.load_masked(0, 0);

    out_view.store_masked(ct::element_cast<__nv_bfloat16>(x_norm), ct::bid().x, 0);
    out_r_inv_view.store_masked(ct::reshape(r_inv, ct::shape<BLOCK_SIZE>{}), ct::bid().x);
}

extern "C" __tile_global__ void rms_prenorm_fwd_inner2048_4x2048(
    const __nv_bfloat16 *__restrict__ prev_layer_output,
    const __nv_bfloat16 *__restrict__ residual_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ out_r_inv,
    int batch_size)
{
    rms_prenorm_fwd_impl<2048, 4>(prev_layer_output, residual_stream, gamma, out, out_r_inv, batch_size);
}

// Backward pass for rms_prenorm_fwd_impl computing the gradient w.r.t.
// the input, optionally incorporating the residual gradient stream from
// the output of this layer (assuming pre-norm).
//
// dgamma must be pre-zeroed to enable atomic accumulation.
template <int INNER_DIM, int BLOCK_SIZE>
__tile__ void rms_prenorm_bwd_impl(
    const __nv_bfloat16 *__restrict__ prev_layer_output,
    const __nv_bfloat16 *__restrict__ prev_layer_residual_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ dy,
    const __nv_bfloat16 *__restrict__ residual_gradient_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    const float *__restrict__ r_inv,
    __nv_bfloat16 *__restrict__ dx,
    float *__restrict__ dgamma,
    int batch_size)
{
    ASSUME_ALIGNED(prev_layer_output);
    ASSUME_ALIGNED(prev_layer_residual_stream);
    ASSUME_ALIGNED(dy);
    ASSUME_ALIGNED(residual_gradient_stream);
    ASSUME_ALIGNED(gamma);
    ASSUME_ALIGNED(r_inv);
    ASSUME_ALIGNED(dx);
    ASSUME_ALIGNED(dgamma);

    auto prev_layer_output_view = ct::partition_view{
        ct::tensor_span{prev_layer_output, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto prev_layer_residual_stream_view = ct::partition_view{
        ct::tensor_span{prev_layer_residual_stream, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto dy_view = ct::partition_view{
        ct::tensor_span{dy, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto residual_gradient_stream_view = ct::partition_view{
        ct::tensor_span{residual_gradient_stream, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto dx_view = ct::partition_view{
        ct::tensor_span{dx, ct::extents{batch_size, INNER_DIM}},
        ct::shape<BLOCK_SIZE, INNER_DIM>{}};
    auto dgamma_view = ct::partition_view{
        ct::tensor_span{dgamma, ct::extents{1, INNER_DIM}},
        ct::shape<1, INNER_DIM>{}};
    auto gamma_view = ct::partition_view{
        ct::tensor_span{gamma, ct::extents{1, INNER_DIM}},
        ct::shape<1, INNER_DIM>{}};
    auto r_inv_view = ct::partition_view{
        ct::tensor_span{r_inv, ct::extents{batch_size}},
        ct::shape<BLOCK_SIZE>{}};

    auto prev_layer_residual_stream_tile = ct::zeros<ct::tile<__nv_bfloat16, ct::shape<BLOCK_SIZE, INNER_DIM>>>();
    if (prev_layer_residual_stream != nullptr)
    {
        prev_layer_residual_stream_tile = prev_layer_residual_stream_view.load_masked(ct::bid().x, 0);
    }

    auto x = ct::element_cast<float>(prev_layer_residual_stream_tile) + ct::element_cast<float>(prev_layer_output_view.load_masked(ct::bid().x, 0));

    auto r_inv_tile = ct::reshape(r_inv_view.load_masked(ct::bid().x), ct::shape<BLOCK_SIZE, 1>{});

    auto x_hat = x * r_inv_tile;

    auto dy_tile = ct::element_cast<float>(dy_view.load_masked(ct::bid().x, 0));
    auto gamma_tile = ct::element_cast<float>(gamma_view.load_masked(0, 0));

    auto g = dy_tile * gamma_tile;

    auto c = ct::sum(g * x_hat, 1_ic) * (1.0f / static_cast<float>(INNER_DIM));

    auto residual_gradient_stream_tile = ct::zeros<ct::tile<float, ct::shape<BLOCK_SIZE, INNER_DIM>>>();
    if (residual_gradient_stream != nullptr)
    {
        residual_gradient_stream_tile = ct::element_cast<float>(residual_gradient_stream_view.load_masked(ct::bid().x, 0));
    }

    auto dx_tile = r_inv_tile * (g - x_hat * c) + residual_gradient_stream_tile;

    dx_view.store_masked(ct::element_cast<__nv_bfloat16>(dx_tile), ct::bid().x, 0);

    auto dgamma_contrib = ct::sum(dy_tile * x_hat, 0_ic);
    dgamma_view.atomic_add(dgamma_contrib, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, 0, 0);
}

extern "C" __tile_global__ void rms_prenorm_bwd_inner2048_4x2048(
    const __nv_bfloat16 *__restrict__ prev_layer_output,
    const __nv_bfloat16 *__restrict__ prev_layer_residual_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ dy,
    const __nv_bfloat16 *__restrict__ residual_gradient_stream, // optional, can be null
    const __nv_bfloat16 *__restrict__ gamma,
    const float *__restrict__ r_inv,
    __nv_bfloat16 *__restrict__ dx,
    float *__restrict__ dgamma,
    int batch_size)
{
    rms_prenorm_bwd_impl<2048, 4>(prev_layer_output, prev_layer_residual_stream, dy, residual_gradient_stream,
                                  gamma, r_inv, dx, dgamma, batch_size);
}