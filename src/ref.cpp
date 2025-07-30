#include "ck/ref.h"

#include <cmath>
#include <vector>

namespace ck {

void sgemm_ref(int M, int N, int K, float alpha, const float* A, const float* B,
               float beta, float* C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;  // double so the oracle's own rounding error is negligible
      for (int p = 0; p < K; ++p) acc += double(A[i * K + p]) * double(B[p * N + j]);
      // beta == 0 must ignore C entirely, even if C holds NaN (BLAS semantics).
      double prev = beta == 0.0f ? 0.0 : double(beta) * C[i * N + j];
      C[i * N + j] = float(alpha * acc + prev);
    }
  }
}

void softmax_rows_ref(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    // Subtracting the row max keeps exp() from overflowing; the result is unchanged
    // because softmax(x) == softmax(x - c) for any constant c.
    double m = xr[0];
    for (int c = 1; c < cols; ++c) m = std::fmax(m, double(xr[c]));
    double s = 0.0;
    for (int c = 0; c < cols; ++c) s += std::exp(double(xr[c]) - m);
    for (int c = 0; c < cols; ++c) yr[c] = float(std::exp(double(xr[c]) - m) / s);
  }
}

void layernorm_rows_ref(const float* x, float* y, const float* gamma, const float* beta,
                        int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    double var = 0.0;
    for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) yr[c] = float((xr[c] - mean) * inv * gamma[c] + beta[c]);
  }
}

void attention_decode_ref(const float* q, const float* k, const float* v, float* out,
                          int T, int H, int D) {
  const double scale = 1.0 / std::sqrt(double(D));
  std::vector<double> score(T);
  for (int h = 0; h < H; ++h) {
    const float* qh = q + size_t(h) * D;
    double m = -INFINITY;
    for (int t = 0; t < T; ++t) {
      const float* kt = k + (size_t(t) * H + h) * D;
      double dot = 0.0;
      for (int d = 0; d < D; ++d) dot += double(qh[d]) * kt[d];
      score[t] = dot * scale;
      m = std::fmax(m, score[t]);
    }
    double s = 0.0;
    for (int t = 0; t < T; ++t) s += (score[t] = std::exp(score[t] - m));
    for (int d = 0; d < D; ++d) {
      double acc = 0.0;
      for (int t = 0; t < T; ++t) acc += score[t] * v[(size_t(t) * H + h) * D + d];
      out[size_t(h) * D + d] = float(acc / s);
    }
  }
}

}  // namespace ck
