#include "utils.cuh"

extern "C" __tile_global__ void rope_hd64_64x1x64(
    const __nv_bfloat16 *__restrict__ x,
    const int64_t *__restrict__ cu_seqlen,
    const __half *__restrict__ cos_table,
    const __half *__restrict__ sin_table,
    const uint64_t *__restrict__ cta_work_items,
    __nv_bfloat16 *__restrict__ out,
    int total_num_tokens,
    int num_heads,
    int max_seqlen)
{
    constexpr int BLOCK_SIZE = 64;
    constexpr int HEAD_DIM = 64;
    constexpr int HALF_HEAD_DIM = HEAD_DIM / 2;

    max_seqlen = ct::assume_divisible(max_seqlen, ct::integral_constant<BLOCK_SIZE>{});

    x = ct::assume_aligned(x, 16_ic);
    cu_seqlen = ct::assume_aligned(cu_seqlen, 16_ic);
    cos_table = ct::assume_aligned(cos_table, 16_ic);
    sin_table = ct::assume_aligned(sin_table, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto cos_table_view = ct::partition_view{
        ct::tensor_span{cos_table, ct::extents{max_seqlen, HALF_HEAD_DIM}},
        ct::shape<BLOCK_SIZE, HALF_HEAD_DIM>{}};
    auto sin_table_view = ct::partition_view{
        ct::tensor_span{sin_table, ct::extents{max_seqlen, HALF_HEAD_DIM}},
        ct::shape<BLOCK_SIZE, HALF_HEAD_DIM>{}};

    uint64_t encoded_work_item;
    encoded_work_item = cta_work_items[ct::bid().x];
    int sequence_idx = static_cast<int>(static_cast<uint32_t>(encoded_work_item));
    int token_offset = BLOCK_SIZE * static_cast<int>(static_cast<uint32_t>(encoded_work_item >> 32));

    auto cos_theta = ct::reshape(ct::element_cast<float>(cos_table_view.load(token_offset / BLOCK_SIZE, 0)),
                                 ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>{});
    auto sin_theta = ct::reshape(ct::element_cast<float>(sin_table_view.load(token_offset / BLOCK_SIZE, 0)),
                                 ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>{});

    int64_t sequence_start = cu_seqlen[sequence_idx];
    int64_t sequence_end = cu_seqlen[sequence_idx + 1];
    int seqlen = static_cast<int>(sequence_end - sequence_start);

    int64_t elements_offset = (sequence_start + token_offset) * num_heads * HEAD_DIM;

    auto x_view = ct::partition_view{
        ct::tensor_span{x + elements_offset,
                        ct::extents{seqlen - token_offset, num_heads, HEAD_DIM}},
        ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>{}};
    auto out_view = ct::partition_view{
        ct::tensor_span{out + elements_offset,
                        ct::extents{seqlen - token_offset, num_heads, HEAD_DIM}},
        ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>{}};

    for (int head_idx : ct::irange(0, num_heads))
    {
        ct::tile<__nv_bfloat16, ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>> first_half;
        [[cutile::hint(0, allow_tma = false)]] first_half = x_view.load_masked(0, head_idx, 0);
        ct::tile<__nv_bfloat16, ct::shape<BLOCK_SIZE, 1, HALF_HEAD_DIM>> second_half;
        [[cutile::hint(0, allow_tma = false)]] second_half = x_view.load_masked(0, head_idx, 1);

        auto rot_first_half = ct::element_cast<__nv_bfloat16>(first_half * cos_theta - second_half * sin_theta);
        auto rot_second_half = ct::element_cast<__nv_bfloat16>(first_half * sin_theta + second_half * cos_theta);

        [[cutile::hint(0, allow_tma = false)]] out_view.store_masked(rot_first_half, 0, head_idx, 0);
        [[cutile::hint(0, allow_tma = false)]] out_view.store_masked(rot_second_half, 0, head_idx, 1);
    }
}
