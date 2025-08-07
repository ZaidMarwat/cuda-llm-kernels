// Single-query (decode-step) multi-head attention on the GPU, split-KV.
//   q [H, D], k/v [T, H, D] (same layout as ck::attention_decode_ref), out [H, D]
//
// Why split: with one thread block per head, a 12-head model launches 12 blocks, which
// leaves most of a GPU's SMs idle while each block streams T*D*2 floats. Splitting the T
// positions into `splits` chunks launches H*splits blocks that read the cache in parallel;
// a small second kernel merges the per-chunk partial softmax results.
#pragma once
#include <cuda_runtime.h>

namespace ck::cuda {

constexpr int kAttnMaxHeadDim = 128;  // 4 dims per lane

// Floats of scratch space needed for H heads and `splits` chunks: (D + 2) per (head, chunk).
inline size_t attention_workspace_floats(int H, int D, int splits) {
  return size_t(H) * splits * (D + 2);
}

// Heuristic: enough blocks for about 2 per SM, but no chunk shorter than 64 positions.
int attention_choose_splits(int T, int H, int num_sms);

void attention_decode(const float* q, const float* k, const float* v, float* out, int T, int H,
                      int D, int splits, float* workspace, cudaStream_t s = nullptr);

}  // namespace ck::cuda
