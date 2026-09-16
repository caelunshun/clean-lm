#include "utils.cuh"

extern "C" __tile_global__ void gradient_l2_norm_reduction_f32_16384(
    const float *__restrict__ gradient, float *__restrict__ global_sum,
    int n)
{
    ASSUME_ALIGNED(gradient);
    ASSUME_ALIGNED(global_sum);
    n = ct::assume_divisible(n, 16_ic);

    auto gradient_view = ct::partition_view{
        ct::tensor_span{gradient, ct::extents{n}}, ct::shape<16384>{}};

    auto gradient_tile = gradient_view.load_masked(ct::bid().x);
    auto square = gradient_tile * gradient_tile;
    auto sum = ct::sum(square, 0_ic);

    ct::atomic_add(ct::full<ct::tile<float *, ct::shape<1>>>(global_sum), sum,
                   ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{});
}

extern "C" __tile_global__ void
adamw_weight_update_clipped_f32grad_f32master_bf16replica_4096_persistent(
    const float *__restrict__ gradient,
    const float *__restrict__ gradient_l2_global_sum,
    float *__restrict__ weight, __nv_bfloat16 *__restrict__ weight_bf16_replica,
    float *__restrict__ adam_m, float *__restrict__ adam_v, float beta1,
    float beta2, float weight_decay, float max_gradient_norm, float lr,
    float train_step, int n)
{

    float gradient_l2_global_sum_val = *gradient_l2_global_sum;
    float gradient_norm = ct::sqrt(gradient_l2_global_sum_val);
    float clip_scale = ct::min(1.0f, max_gradient_norm / (gradient_norm + 1e-6f));

    n = ct::assume_divisible(n, 16_ic);

    ASSUME_ALIGNED(gradient);
    auto gradient_view = ct::partition_view{
        ct::tensor_span{gradient, ct::extents{n}}, ct::shape<4096>{}};

    ASSUME_ALIGNED(weight);
    auto weight_view = ct::partition_view{ct::tensor_span{weight, ct::extents{n}},
                                          ct::shape<4096>{}};

    ASSUME_ALIGNED(weight_bf16_replica);
    auto weight_bf16_replica_view = ct::partition_view{
        ct::tensor_span{weight_bf16_replica, ct::extents{n}}, ct::shape<4096>{}};

    ASSUME_ALIGNED(adam_m);
    auto adam_m_view = ct::partition_view{ct::tensor_span{adam_m, ct::extents{n}},
                                          ct::shape<4096>{}};

    ASSUME_ALIGNED(adam_v);
    auto adam_v_view = ct::partition_view{ct::tensor_span{adam_v, ct::extents{n}},
                                          ct::shape<4096>{}};

    int num_tiles = (n + 4095) / 4096;
    for (int i :
         ct::irange((int)ct::bid().x, num_tiles, (int)ct::num_blocks().x))
    {
        auto gradient_tile = gradient_view.load_masked(i) * clip_scale;

        auto weight_tile = weight_view.load_masked(i);

        auto m_tile = adam_m_view.load_masked(i);
        m_tile = beta1 * m_tile + (1.0f - beta1) * gradient_tile;
        auto m_hat = ct::div(m_tile, (1.0f - ct::pow(beta1, train_step)),
                             ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

        auto v_tile = adam_v_view.load_masked(i);
        v_tile = beta2 * v_tile + (1.0f - beta2) * (gradient_tile * gradient_tile);

        auto v_hat = ct::div(v_tile, (1.0f - ct::pow(beta2, train_step)),
                             ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

        weight_tile = weight_tile -
                      lr * (ct::div(m_hat,
                                    (ct::sqrt(v_hat, ct::round_approximate_t{},
                                              ct::round_subnormals_to_zero_t{}) +
                                     1e-6f),
                                    ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{}) +
                            weight_decay * weight_tile);

        weight_view.store_masked(weight_tile, i);
        weight_bf16_replica_view.store_masked(
            ct::element_cast<__nv_bfloat16>(weight_tile), i);

        adam_m_view.store_masked(m_tile, i);
        adam_v_view.store_masked(v_tile, i);
    }
}
