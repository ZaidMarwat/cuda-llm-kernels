#include <algorithm>
#include <cmath>

#include "attention.cuh"
#include "attention_math.h"
#include "common.cuh"

namespace ck::cuda {
namespace {

constexpr int WARP = 32, NW = 4;  // 4 warps per block
constexpr int PER_LANE = kAttnMaxHeadDim / WARP;
constexpr unsigned FULL = 0xffffffffu;

// Block (h, split) computes the partial (m, l, acc[D]) for head h over its chunk of positions.
// Each warp takes every NW-th position; lanes split the head dimension (dim = lane + 32*i),
// so loads of a K or V row are coalesced across the warp.
__global__ void __launch_bounds__(NW * WARP)
attn_partial(const float* q, const float* k, const float* v, float* ws, int T, int H, int D,
             int splits, float scale) {
  const int h = blockIdx.x, split = blockIdx.y;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const int chunk = ceil_div(T, splits);
  const int t0 = split * chunk, t1 = (t0 + chunk < T) ? t0 + chunk : T;  // no min(): portable to the clang check

  float qr[PER_LANE], acc[PER_LANE];
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) {
    const int dim = lane + WARP * i;
    qr[i] = dim < D ? q[h * D + dim] * scale : 0.0f;  // fold 1/sqrt(D) into q once
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = t0 + warp; t < t1; t += NW) {
    const float* kt = k + (size_t(t) * H + h) * D;
    const float* vt = v + (size_t(t) * H + h) * D;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
      const int dim = lane + WARP * i;
      if (dim < D) dot += qr[i] * kt[dim];
    }
    for (int off = WARP / 2; off > 0; off /= 2) dot += __shfl_xor_sync(FULL, dot, off);
    float sc, p;
    attn_push(m, l, dot, sc, p);  // every lane holds the same dot, so m and l stay in sync
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
      const int dim = lane + WARP * i;
      if (dim < D) acc[i] = acc[i] * sc + p * vt[dim];
    }
  }

  // Merge the NW warps' partials through shared memory; warp 0 writes the block's result.
  __shared__ float sm[NW], sl[NW], sacc[NW][kAttnMaxHeadDim];
  if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) {
    const int dim = lane + WARP * i;
    if (dim < D) sacc[warp][dim] = acc[i];
  }
  __syncthreads();
  if (warp != 0) return;
  for (int w = 1; w < NW; ++w) {
    float s1, s2;
    attn_merge(m, l, sm[w], sl[w], s1, s2);
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
      const int dim = lane + WARP * i;
      if (dim < D) acc[i] = acc[i] * s1 + sacc[w][dim] * s2;
    }
  }
  float* out = ws + (size_t(h) * splits + split) * (D + 2);
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) {
    const int dim = lane + WARP * i;
    if (dim < D) out[dim] = acc[i];  // unnormalized: normalized once, after all splits
  }
  if (lane == 0) { out[D] = m; out[D + 1] = l; }
}

// One block per head; thread d produces out[h, d] from the `splits` partials.
__global__ void attn_combine(const float* ws, float* out, int H, int D, int splits) {
  const int h = blockIdx.x;
  const float* base = ws + size_t(h) * splits * (D + 2);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (D + 2) + D]);
    float L = 0.0f, a = 0.0f;
    for (int s = 0; s < splits; ++s) {
      const float ms = base[s * (D + 2) + D];
      if (ms == -INFINITY) continue;  // empty chunk (more splits than positions)
      const float w = expf(ms - M);
      L += base[s * (D + 2) + D + 1] * w;
      a += base[s * (D + 2) + d] * w;
    }
    out[h * D + d] = a / L;
  }
}

}  // namespace

int attention_choose_splits(int T, int H, int num_sms) {
  const int want = ceil_div(2 * num_sms, H);
  const int max_useful = std::max(1, ceil_div(T, 64));
  return std::max(1, std::min(want, max_useful));
}

void attention_decode(const float* q, const float* k, const float* v, float* out, int T, int H,
                      int D, int splits, float* workspace, cudaStream_t s) {
  if (D > kAttnMaxHeadDim || T < 1 || splits < 1) {
    std::fprintf(stderr, "attention_decode: unsupported T=%d D=%d splits=%d\n", T, D, splits);
    std::exit(1);
  }
  const float scale = 1.0f / std::sqrt(float(D));
  attn_partial<<<dim3(H, splits), NW * WARP, 0, s>>>(q, k, v, workspace, T, H, D, splits, scale);
  CUDA_CHECK(cudaGetLastError());
  attn_combine<<<H, 128, 0, s>>>(workspace, out, H, D, splits);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ck::cuda
