#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_tile.h>
#include <numbers>
#include <cmath>

namespace ct = cuda::tiles;
using namespace ct::literals;

#define ASSUME_ALIGNED(x) x = ct::assume_aligned(x, 16_ic);
