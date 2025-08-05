#include <cmath>

#include "common.cuh"
#include "rowops.cuh"
#include "rowops_math.h"

namespace ck::cuda {
namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr unsigned FULL = 0xffffffffu;

__global__ void softmax_naive_kernel(const float* x, float* y, int rows, int cols) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;
  float m = -INFINITY;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float s = 0.0f;
  for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
  for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) / s;
}

__global__ void softmax_warp_kernel(const float* x, float* y, int rows, int cols) {
  const int lane = threadIdx.x % WARP;
  const int r = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP;
  if (r >= rows) return;  // uniform per warp, so the shuffles below see all 32 lanes
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;

  // Pass 1: each lane walks columns lane, lane+32, ... (coalesced across the warp).
  float m = -INFINITY, d = 0.0f;
  for (int c = lane; c < cols; c += WARP) online_push(m, d, xr[c]);
  // Butterfly reduction: after log2(32) = 5 steps every lane holds the row's (m, d).
  for (int off = WARP / 2; off > 0; off /= 2) {
    const float m2 = __shfl_xor_sync(FULL, m, off);
    const float d2 = __shfl_xor_sync(FULL, d, off);
    online_merge(m, d, m2, d2);
  }
  // Pass 2: write. Total traffic is 2 reads + 1 write per element instead of 3 + 1.
  const float inv = 1.0f / d;
  for (int c = lane; c < cols; c += WARP) yr[c] = expf(xr[c] - m) * inv;
}

__global__ void layernorm_warp_kernel(const float* x, float* y, const float* gamma,
                                      const float* beta, int rows, int cols, float eps) {
  const int lane = threadIdx.x % WARP;
  const int r = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;

  float n = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (int c = lane; c < cols; c += WARP) welford_push(n, mean, m2, xr[c]);
  for (int off = WARP / 2; off > 0; off /= 2) {
    const float nb = __shfl_xor_sync(FULL, n, off);
    const float meanb = __shfl_xor_sync(FULL, mean, off);
    const float m2b = __shfl_xor_sync(FULL, m2, off);
    welford_merge(n, mean, m2, nb, meanb, m2b);
  }
  const float rstd = rsqrtf(m2 / float(cols) + eps);  // population variance, like PyTorch
  for (int c = lane; c < cols; c += WARP) yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

}  // namespace

void softmax_rows_naive(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  softmax_naive_kernel<<<ceil_div(rows, 128), 128, 0, s>>>(x, y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void softmax_rows(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  softmax_warp_kernel<<<ceil_div(rows, WARPS_PER_BLOCK), WARPS_PER_BLOCK * WARP, 0, s>>>(
      x, y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void layernorm_rows(const float* x, float* y, const float* gamma, const float* beta, int rows,
                    int cols, float eps, cudaStream_t s) {
  layernorm_warp_kernel<<<ceil_div(rows, WARPS_PER_BLOCK), WARPS_PER_BLOCK * WARP, 0, s>>>(
      x, y, gamma, beta, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ck::cuda
