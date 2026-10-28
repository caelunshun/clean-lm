#include "utils.cuh"

extern "C" __tile_global__ void accumulate_gradient_f32grad_bf16contrib_persistent(
    const __nv_bfloat16 *__restrict__ contribution,
    float *__restrict__ gradient,
    int n)
{
    ASSUME_ALIGNED(contribution);
    ASSUME_ALIGNED(gradient);
    n = ct::assume_divisible(n, 16_ic);

    constexpr int TILE_SIZE = 4096;

    auto contribution_view = ct::partition_view{
        ct::tensor_span{contribution, ct::extents{n}},
        ct::shape<TILE_SIZE>{}};
    auto gradient_view = ct::partition_view{
        ct::tensor_span{gradient, ct::extents{n}},
        ct::shape<TILE_SIZE>{}};

    int num_tiles = ct::ceildiv(n, TILE_SIZE);
    for (int tile_idx : ct::irange((int)ct::bid().x, num_tiles, (int)ct::num_blocks().x))
    {
        auto contribution_tile = contribution_view.load_masked(tile_idx);
        auto gradient_tile = gradient_view.load_masked(tile_idx);
        auto result = ct::element_cast<float>(contribution_tile) + gradient_tile;
        gradient_view.store_masked(result, tile_idx);
    }
}
