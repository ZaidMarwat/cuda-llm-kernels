// Reference (oracle) implementations of the four ops on the LLM decode path.
//
// These are deliberately simple and accumulate in double precision. Every optimized
// kernel in this repo, CPU or GPU, is tested against them. Layouts are row-major.
#pragma once
#include <cstddef>

namespace ck {

// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
void sgemm_ref(int M, int N, int K, float alpha, const float* A, const float* B,
               float beta, float* C);

// Numerically stable softmax over each row of x[rows, cols], written to y.
void softmax_rows_ref(const float* x, float* y, int rows, int cols);

// y = (x - mean) / sqrt(var + eps) * gamma + beta, per row. var is the biased
// (population) variance, which is what PyTorch's LayerNorm uses.
void layernorm_rows_ref(const float* x, float* y, const float* gamma, const float* beta,
                        int rows, int cols, float eps);

// Single-query multi-head attention, the decode step:
//   q   [H, D]      query for the new token
//   k,v [T, H, D]   cached keys/values for the T context positions
//   out [H, D]      softmax(q . k * scale) @ v per head, with scale = 1/sqrt(D)
// No causal mask is needed: during decode the new token may attend to all T.
void attention_decode_ref(const float* q, const float* k, const float* v, float* out,
                          int T, int H, int D);

}  // namespace ck
