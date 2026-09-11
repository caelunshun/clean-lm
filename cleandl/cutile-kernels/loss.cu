#include "utils.cuh"

constexpr float log2_e = std::numbers::log2e_v<float>;

// Online-softmax first-stage kernel computing a per-row maximum `m`
// and softmax denominator `d` and emitting `m + ln(d)`.
extern "C" __tile_global__ void online_softmax_stats_128(const __nv_bfloat16 *__restrict__ x, float *__restrict__ out_stats, int batch_size, int hidden_size)
{
    ASSUME_ALIGNED(x);
    ASSUME_ALIGNED(out_stats);
    hidden_size = ct::assume_divisible(hidden_size, 16_ic);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::extents{batch_size, hidden_size}},
        ct::shape<128, 128>{}};
    auto out_stats_view = ct::partition_view{
        ct::tensor_span{out_stats, ct::extents{batch_size}},
        ct::shape<128>{}};

    auto num_tiles = ct::ceildiv(hidden_size, 128);

    auto denoms = ct::zeros<ct::tile<float, ct::shape<128, 1>>>();
    auto maxes = ct::full<ct::tile<float, ct::shape<128, 1>>>(-std::numeric_limits<float>::infinity());

    for (int tile_idx : ct::irange(0, num_tiles))
    {
        auto x_tile = ct::element_cast<float>(x_view.load_masked(ct::bid().x, tile_idx));

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

        auto scales = ct::exp2((maxes - new_maxes) * log2_e, ct::round_subnormals_to_zero_t{});
        auto sums = ct::sum(ct::exp2((x_tile - new_maxes) * log2_e, ct::round_subnormals_to_zero_t{}), 1_ic);

        denoms = denoms * scales + sums;
        maxes = new_maxes;
    }

    out_stats_view.store_masked(ct::reshape(ct::log(denoms) + maxes, ct::shape<128>{}), ct::bid().x);
}

// Forward+backward cross-entropy loss on logits w.r.t.
// a single target label per row.
// Requires logits pre-processed with online_softmax_stats kernel.
// Loss must be pre-zeroed. This kernel will compute the mean across batch items.
extern "C" __tile_global__ void cross_entropy_loss_fwd_128(const __nv_bfloat16 *__restrict__ logits,
                                                           const float *__restrict__ stats, const int *__restrict__ labels,
                                                           float *__restrict__ out_loss, __nv_bfloat16 *__restrict__ out_logits_grad,
                                                           int batch_size, float recip_batch_size, int num_classes)
{
    ASSUME_ALIGNED(logits);
    ASSUME_ALIGNED(stats);
    ASSUME_ALIGNED(labels);
    ASSUME_ALIGNED(out_logits_grad);
    ASSUME_ALIGNED(out_loss);
    num_classes = ct::assume_divisible(num_classes, 16_ic);

    auto logits_view = ct::partition_view{
        ct::tensor_span{logits, ct::extents{batch_size, num_classes}},
        ct::shape<128, 128>{}};
    auto stats_view = ct::partition_view{
        ct::tensor_span{stats, ct::extents{batch_size}},
        ct::shape<128>{}};
    auto labels_view = ct::partition_view{
        ct::tensor_span{labels, ct::extents{batch_size}},
        ct::shape<128>{}};
    auto out_logits_grad_view = ct::partition_view{
        ct::tensor_span{out_logits_grad, ct::extents{batch_size, num_classes}},
        ct::shape<128, 128>{}};

    auto stats_tile = ct::reshape(stats_view.load_masked(ct::bid().x), ct::shape<128, 1>{});
    auto labels_tile = ct::reshape(labels_view.load_masked(ct::bid().x), ct::shape<128, 1>{});

    auto per_row_loss = ct::zeros<ct::tile<float, ct::shape<128, 1>>>();

    int num_tiles = ct::ceildiv(num_classes, 128);
    for (int tile_idx : ct::irange(0, num_tiles))
    {
        auto logits_tile = ct::element_cast<float>(logits_view.load_masked(ct::bid().x, tile_idx));
        auto log_probs = logits_tile - stats_tile;

        auto class_idx = ct::iota<ct::tile<int, ct::shape<1, 128>>>() + 128 * tile_idx;
        auto class_matches = class_idx == labels_tile;

        auto loss_contrib = ct::select(class_matches, -log_probs, ct::zeros<ct::tile<float, ct::shape<128, 128>>>());
        per_row_loss = per_row_loss + ct::sum(loss_contrib, 1_ic);

        auto target_probs = ct::select(class_matches, ct::ones<ct::tile<float, ct::shape<128, 128>>>(),
                                       ct::zeros<ct::tile<float, ct::shape<128, 128>>>());
        auto probs = ct::exp2(log_probs * log2_e, ct::round_subnormals_to_zero_t{});

        auto grad = ct::element_cast<__nv_bfloat16>((probs - target_probs) * recip_batch_size);
        out_logits_grad_view.store_masked(grad, ct::bid().x, tile_idx);
    }

    auto loss_contrib = ct::reshape(ct::sum(per_row_loss, 0_ic) * recip_batch_size, ct::shape<1>{});
    ct::atomic_add(
        ct::full<ct::tile<float *, ct::shape<1>>>(out_loss),
        loss_contrib,
        ct::memory_order_relaxed_t{},
        ct::thread_scope_device_t{});
}
