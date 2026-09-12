#include "utils.cuh"

extern "C" __tile_global__ void embedding_lookup_f32in_bf16out_32x128(
    const int *__restrict__ indices, const float *__restrict__ embeddings,
    __nv_bfloat16 *__restrict__ out, int vocab_size, int num_tokens,
    int embed_dim)
{
    ASSUME_ALIGNED(indices);
    ASSUME_ALIGNED(embeddings);
    ASSUME_ALIGNED(out);
    embed_dim = ct::assume_divisible(embed_dim, 128_ic);

    auto indices_view = ct::partition_view{
        ct::tensor_span{indices, ct::shape{num_tokens, 1}}, ct::shape<32, 1>{}};
    auto out_view =
        ct::partition_view{ct::tensor_span{out, ct::shape{num_tokens, embed_dim}},
                           ct::shape<32, 128>{}};

    auto indices_tile = indices_view.load_masked(ct::bid().y, 0);

    auto offsets_tile = ct::element_cast<int64_t>(indices_tile) * ct::element_cast<int64_t>(embed_dim) +
                        ct::iota<ct::tile<int64_t, ct::shape<1, 128>>>() + (int)ct::bid().x * 128;

    ct::tile<float, ct::shape<32, 128>> embeddings_tile =
        ct::load(embeddings + offsets_tile);

    out_view.store_masked(ct::element_cast<__nv_bfloat16>(embeddings_tile),
                          ct::bid().y, ct::bid().x);
}

extern "C" __tile_global__ void embedding_scatter_add_bf16in_f32out_32x128(
    const int *__restrict__ indices, const __nv_bfloat16 *__restrict__ dL_dout,
    float *__restrict__ dL_din, int vocab_size, int num_tokens, int embed_dim)
{
    ASSUME_ALIGNED(indices);
    ASSUME_ALIGNED(dL_dout);
    ASSUME_ALIGNED(dL_din);
    embed_dim = ct::assume_divisible(embed_dim, 128_ic);

    auto indices_view = ct::partition_view{
        ct::tensor_span{indices, ct::shape{num_tokens, 1}}, ct::shape<32, 1>{}};
    auto dL_dout_view = ct::partition_view{
        ct::tensor_span{dL_dout, ct::shape{num_tokens, embed_dim}},
        ct::shape<32, 128>{}};

    auto indices_tile = indices_view.load_masked(ct::bid().y, 0);
    auto dL_dout_tile = dL_dout_view.load_masked(ct::bid().y, ct::bid().x);

    auto offsets_tile =
        ct::element_cast<int64_t>(indices_tile) * embed_dim + ct::iota<ct::tile<int64_t, ct::shape<1, 128>>>() + (int)ct::bid().x * 128;
    auto ptrs_tile = dL_din + offsets_tile;

    ct::atomic_add(ptrs_tile, ct::element_cast<float>(dL_dout_tile),
                   ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{});
}
