#include "utils.cuh"

extern "C" __tile_global__ void
f32_to_bf16_16384(const float *__restrict__ in, __nv_bfloat16 *__restrict__ out,
                  int n) {
  in = ct::assume_aligned(in, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  n = ct::assume_divisible(n, 16_ic);

  auto in_view = ct::partition_view{ct::tensor_span{in, ct::extents{n}},
                                    ct::shape<16384>{}};
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}},
                                     ct::shape<16384>{}};

  auto tile = in_view.load_masked(ct::bid().x);
  out_view.store_masked(ct::element_cast<__nv_bfloat16>(tile), ct::bid().x);
}

extern "C" 
__tile_global__ void bf16_to_f32_16384(const __nv_bfloat16 *__restrict__ in,
                                       float *__restrict__ out, int n) {
  in = ct::assume_aligned(in, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  n = ct::assume_divisible(n, 16_ic);

  auto in_view = ct::partition_view{ct::tensor_span{in, ct::extents{n}},
                                    ct::shape<16384>{}};
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}},
                                     ct::shape<16384>{}};

  auto tile = in_view.load_masked(ct::bid().x);
  out_view.store_masked(ct::element_cast<float>(tile), ct::bid().x);
}
