// Fallback unfused epilogues for SkyMatmulExecutor.

#include "utils.cuh"

extern "C" __tile_global__ void swiglu_gate_train_fwd_bf16_4096(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ y,
    uint32_t n)
{
    ASSUME_ALIGNED(x);
    ASSUME_ALIGNED(gate);
    ASSUME_ALIGNED(y);

    ct::partition_view x_view{
        ct::tensor_span{x, ct::extents{n}},
        ct::shape<4096>{}};
    ct::partition_view gate_view{
        ct::tensor_span{gate, ct::extents{n}},
        ct::shape<4096>{}};
    ct::partition_view y_view{
        ct::tensor_span{y, ct::extents{n}},
        ct::shape<4096>{}};

    auto x_tile = ct::element_cast<float>(x_view.load_masked(ct::bid().x));
    auto gate_tile = ct::element_cast<float>(gate_view.load_masked(ct::bid().x));

    auto silu = ct::div(
        gate_tile,
        1.0f + ct::exp(-gate_tile, ct::round_approximate_t{}),
        ct::round_approximate_t{},
        ct::round_subnormals_to_zero_t{});
    auto y_tile = x_tile * silu;

    y_view.store_masked(ct::element_cast<__nv_bfloat16>(y_tile), ct::bid().x);
}

extern "C" __tile_global__ void swiglu_down_train_dgrad_bf16_4096(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ grad_x,
    __nv_bfloat16 *__restrict__ grad_gate,
    uint32_t n)
{
    ASSUME_ALIGNED(gate);
    ASSUME_ALIGNED(x);
    ASSUME_ALIGNED(grad_x);
    ASSUME_ALIGNED(grad_gate);

    ct::partition_view gate_view{
        ct::tensor_span{gate, ct::extents{n}},
        ct::shape<4096>{}};
    ct::partition_view x_view{
        ct::tensor_span{x, ct::extents{n}},
        ct::shape<4096>{}};
    ct::partition_view grad_x_view{
        ct::tensor_span{grad_x, ct::extents{n}},
        ct::shape<4096>{}};
    ct::partition_view grad_gate_view{
        ct::tensor_span{grad_gate, ct::extents{n}},
        ct::shape<4096>{}};

    auto gate_tile = ct::element_cast<float>(gate_view.load_masked(ct::bid().x));
    auto x_tile = ct::element_cast<float>(x_view.load_masked(ct::bid().x));
    auto grad_activation_tile = ct::element_cast<float>(grad_x_view.load_masked(ct::bid().x));

    auto sigmoid = ct::div(
        1.0f,
        1.0f + ct::exp(-gate_tile, ct::round_approximate_t{}),
        ct::round_approximate_t{},
        ct::round_subnormals_to_zero_t{});
    auto grad_x_tile = grad_activation_tile * gate_tile * sigmoid;
    auto dsilu = sigmoid + gate_tile * sigmoid * (1.0f - sigmoid);
    auto grad_gate_tile = grad_activation_tile * x_tile * dsilu;

    grad_x_view.store_masked(ct::element_cast<__nv_bfloat16>(grad_x_tile), ct::bid().x);
    grad_gate_view.store_masked(ct::element_cast<__nv_bfloat16>(grad_gate_tile), ct::bid().x);
}
