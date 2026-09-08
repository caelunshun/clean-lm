// WIP, broken

#include "utils.cuh"

struct WorkSpan
{
    // In units of sequence-position tiles (128 for rope_128)
    int sequence_pos_tile_idx;
    // In units of sequences. Note cu_seqlen and ragged tensor
    // storage.
    int begin_sequence_idx;
    int end_sequence_idx;
};

// Rotary positional embedding in either forward or backward
// pass (set dir = -1.0 for backward, 1.0 for forward).
// Requires head dimension = second tile size in kernel name.
// Does not rely on precomputed cos/sin cache. Instead computes
// those values on-the-fly inside the SM. Each CTA is assigned
// a work-span by the host. The work-span specifies
// which sequence-position tile to process and for which range
// of sequences. Using fewer work-spans per sequence-position tile
// amortizes the sin/cos overhead but may cause load balancing issues.
extern "C" __tile_global__ void rope_128x64(
    const __nv_bfloat16 *__restrict__ x,
    const int64_t *__restrict__ cu_seqlen,
    __nv_bfloat16 *__restrict__ out,
    float base_freq,
    float dir,
    int num_heads,
    const WorkSpan *__restrict__ work_spans)
{
    constexpr int BLOCK = 128;
    constexpr int HEAD_DIM = 64;
    constexpr int HALF_HEAD_DIM = HEAD_DIM / 2;

    x = ct::assume_aligned(x, 16_ic);
    cu_seqlen = ct::assume_aligned(cu_seqlen, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto work_span = work_spans[ct::bid().x];

    auto frequencies = ct::pow(
        base_freq,
        -2.0f * ct::element_cast<float>(ct::iota<ct::tile<int, ct::shape<1, HALF_HEAD_DIM>>>()) * (1.0f / static_cast<float>(HEAD_DIM)));
    auto sequence_positions = ct::iota<ct::tile<int, ct::shape<BLOCK, 1>>>() + BLOCK * work_span.begin_sequence_idx;
    auto angles = dir * ct::element_cast<float>(sequence_positions) * frequencies;

    auto cos_theta = ct::cos(angles);
    auto sin_theta = ct::sin(angles);

    for (int sequence_idx : ct::irange(work_span.begin_sequence_idx, work_span.end_sequence_idx))
    {
        int64_t seq_start = cu_seqlen[sequence_idx];
        int64_t seq_end = cu_seqlen[sequence_idx + 1];

        auto x_view = ct::partition_view{
            ct::tensor_span{
                x + sequence_idx * num_heads * HEAD_DIM,
                ct::extents{static_cast<int>(seq_end - seq_start), num_heads * HEAD_DIM}},
            ct::shape<BLOCK, HEAD_DIM>{}};
        auto out_view = ct::partition_view{
            ct::tensor_span{
                out + sequence_idx * num_heads * HEAD_DIM,
                ct::extents{static_cast<int>(seq_end - seq_start), num_heads * HEAD_DIM}},
            ct::shape<BLOCK, HEAD_DIM>{}};

        for (int head_idx : ct::irange(0, num_heads))
        {
            ct::tile<__nv_bfloat16, ct::shape<BLOCK, HEAD_DIM>> x_tile;
            [[cutile::hint(0, allow_tma = false)]] x_tile = x_view.load_masked(0, head_idx);

            auto first_half = ct::element_cast<float>(ct::extract(x_tile, ct::shape<BLOCK, HALF_HEAD_DIM>{}, 0, 0));
            auto second_half = ct::element_cast<float>(ct::extract(x_tile, ct::shape<BLOCK, HALF_HEAD_DIM>{}, 0, 1));

            auto new_first_half = first_half * cos_theta - second_half * sin_theta;
            auto new_second_half = first_half * sin_theta + second_half * cos_theta;

            auto out_tile = ct::element_cast<__nv_bfloat16>(ct::cat(new_first_half, new_second_half, 1_ic));
            out_view.store_masked(out_tile, 0, head_idx);
        }
    }
}
