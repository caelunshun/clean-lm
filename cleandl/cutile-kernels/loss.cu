#include "utils.cuh"

// Online-softmax first-stage kernel computing a per-row maximum
// and denominator.
extern "C" __tile_global__ void online_softmax_stats_128(const __nv_bfloat16 *__restrict__ x, float *__restrict__ out_maxes, float *__restrict__ out_denoms, int batch_size, int hidden_size)
{
    x = ct::assume_aligned(x, 16_ic);
    out_maxes = ct::assume_aligned(out_maxes, 16_ic);
    out_denoms = ct::assume_aligned(out_denoms, 16_ic);
    hidden_size = ct::assume_divisible(hidden_size, 16_ic);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::extents{batch_size, hidden_size}},
        ct::shape<128, 128>{}};
    auto out_denoms_view = ct::partition_view{
        ct::tensor_span{out_denoms, ct::extents{batch_size}},
        ct::shape<128>{}};
    auto out_maxes_view = ct::partition_view{
        ct::tensor_span{out_maxes, ct::extents{batch_size}},
        ct::shape<128>{}};

    auto num_tiles = ct::ceildiv(hidden_size, 128);

    auto denoms = ct::zeros<ct::tile<float, ct::shape<128, 1>>>();
    auto maxes = ct::full<ct::tile<float, ct::shape<128, 1>>>(-std::numeric_limits<float>::infinity());

    constexpr float exp_scale = 1.4426950408f;

    for (int tile_idx : ct::irange(0, num_tiles))
    {
        auto x_tile = ct::element_cast<float>(x_view.load_masked(ct::bid().x, tile_idx)) * exp_scale;

        // Explicitly fill OOB with -inf along hidden dimension, but not along batch dimension,
        // for efficiency (allows still using TMA).
        if (tile_idx == num_tiles - 1)
        {
            auto idx = ct::iota<ct::tile<int, ct::shape<1, 128>>>() + tile_idx * 128;
            auto mask = idx < hidden_size;
            x_tile = ct::select(mask, x_tile, ct::full<ct::tile<float, ct::shape<128, 128>>>(-std::numeric_limits<float>::infinity()));
        }

        auto x_maxes = ct::reduce_max(x_tile, 1_ic);
        auto new_maxes = ct::max(maxes, x_maxes);

        auto scales = ct::exp2(maxes - new_maxes, ct::round_subnormals_to_zero_t{});
        auto sums = ct::sum(ct::exp2(x_tile - new_maxes, ct::round_subnormals_to_zero_t{}), 1_ic);

        denoms = denoms * scales + sums;
        maxes = new_maxes;
    }

    out_denoms_view.store_masked(ct::reshape(denoms, ct::shape<128>{}), ct::bid().x);
    out_maxes_view.store_masked(ct::reshape(maxes, ct::shape<128>{}), ct::bid().x);
}
