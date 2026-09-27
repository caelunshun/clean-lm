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
        x_tile,
        1.0f + ct::exp(-gate_tile, ct::round_approximate_t{}),
        ct::round_approximate_t{},
        ct::round_subnormals_to_zero_t{});
    auto y_tile = x_tile * silu;

    y_view.store_masked(ct::element_cast<__nv_bfloat16>(y_tile), ct::bid().x);
}
