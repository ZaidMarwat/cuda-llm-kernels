// Row-wise ops on x[rows, cols] (row-major fp32, device pointers):
//   softmax_rows_naive   one thread per row, three passes (max, sum, write). Baseline.
//   softmax_rows         one warp per row, online softmax: max and sum in one pass
//                        (Milakov & Gimelshein 2018, "Online normalizer calculation for
//                        softmax"), merged across lanes with warp shuffles
//   layernorm_rows       one warp per row, Welford mean/variance per lane merged across
//                        lanes with Chan et al.'s parallel update, then normalize
// One warp per row suits the decode path, where rows are short (d_model = 768, or T scores).
#pragma once
#include <cuda_runtime.h>

namespace ck::cuda {

void softmax_rows_naive(const float* x, float* y, int rows, int cols, cudaStream_t s = nullptr);
void softmax_rows(const float* x, float* y, int rows, int cols, cudaStream_t s = nullptr);
void layernorm_rows(const float* x, float* y, const float* gamma, const float* beta, int rows,
                    int cols, float eps, cudaStream_t s = nullptr);

}  // namespace ck::cuda
