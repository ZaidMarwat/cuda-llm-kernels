// Tests for the reference ops. These pin the oracle itself with hand-computed cases
// and mathematical identities, so later kernels are compared against something trusted.
#include <cmath>
#include <limits>

#include "check.h"
#include "ck/ref.h"

TEST(gemm_2x2_hand_computed) {
  float A[] = {1, 2, 3, 4}, B[] = {5, 6, 7, 8}, C[] = {1, 1, 1, 1};
  ck::sgemm_ref(2, 2, 2, 1.0f, A, B, 0.0f, C);
  // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  EXPECT(C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);
}

TEST(gemm_alpha_beta) {
  float A[] = {1, 2, 3}, B[] = {4, 5, 6}, C[] = {10};
  ck::sgemm_ref(1, 1, 3, 2.0f, A, B, 0.5f, C);  // 2*32 + 0.5*10
  EXPECT(C[0] == 69);
}

TEST(gemm_beta_zero_ignores_nan_in_c) {
  float A[] = {1}, B[] = {1}, C[] = {std::numeric_limits<float>::quiet_NaN()};
  ck::sgemm_ref(1, 1, 1, 1.0f, A, B, 0.0f, C);
  EXPECT(C[0] == 1);
}

TEST(softmax_rows_sum_to_one_and_shift_invariant) {
  auto x = check::randn(4 * 37, 1, 3.0f);
  std::vector<float> y(x.size()), x2(x), y2(x.size());
  for (auto& v : x2) v += 100.0f;  // softmax(x + c) == softmax(x)
  ck::softmax_rows_ref(x.data(), y.data(), 4, 37);
  ck::softmax_rows_ref(x2.data(), y2.data(), 4, 37);
  for (int r = 0; r < 4; ++r) {
    double s = 0;
    for (int c = 0; c < 37; ++c) s += y[r * 37 + c];
    EXPECT(std::fabs(s - 1.0) < 1e-6);
  }
  EXPECT(check::close_ratio(y2.data(), y.data(), y.size(), 1e-6, 1e-4) <= 1);
}

TEST(softmax_large_logits_do_not_overflow) {
  float x[] = {1000.0f, 1000.0f, -1000.0f}, y[3];
  ck::softmax_rows_ref(x, y, 1, 3);
  EXPECT(std::fabs(y[0] - 0.5f) < 1e-6 && std::fabs(y[1] - 0.5f) < 1e-6 && y[2] == 0.0f);
}

TEST(layernorm_output_is_normalized) {
  const int rows = 3, cols = 64;
  auto x = check::randn(rows * cols, 2, 5.0f);
  for (int i = 0; i < rows * cols; ++i) x[i] += 7.0f;  // nonzero mean on purpose
  std::vector<float> y(x.size()), g(cols, 1.0f), b(cols, 0.0f);
  ck::layernorm_rows_ref(x.data(), y.data(), g.data(), b.data(), rows, cols, 1e-5f);
  for (int r = 0; r < rows; ++r) {
    double m = 0, v = 0;
    for (int c = 0; c < cols; ++c) m += y[r * cols + c];
    m /= cols;
    for (int c = 0; c < cols; ++c) v += (y[r * cols + c] - m) * (y[r * cols + c] - m);
    v /= cols;
    EXPECT(std::fabs(m) < 1e-5 && std::fabs(v - 1.0) < 1e-3);
  }
}

TEST(layernorm_applies_gamma_beta) {
  float x[] = {1, 3}, y[2], g[] = {2, 2}, b[] = {10, 10};
  ck::layernorm_rows_ref(x, y, g, b, 1, 2, 0.0f);  // normalized = [-1, 1]
  EXPECT(std::fabs(y[0] - 8.0f) < 1e-6 && std::fabs(y[1] - 12.0f) < 1e-6);
}

TEST(attention_single_position_returns_v) {
  const int H = 2, D = 4;
  auto q = check::randn(H * D, 3), k = check::randn(H * D, 4), v = check::randn(H * D, 5);
  std::vector<float> out(H * D);
  ck::attention_decode_ref(q.data(), k.data(), v.data(), out.data(), 1, H, D);
  EXPECT(check::close_ratio(out.data(), v.data(), v.size(), 1e-6, 1e-6) <= 1);
}

TEST(attention_zero_query_averages_v) {
  const int T = 5, H = 1, D = 3;
  std::vector<float> q(D, 0.0f);
  auto k = check::randn(T * D, 6), v = check::randn(T * D, 7);
  std::vector<float> out(D), mean(D, 0.0f);
  ck::attention_decode_ref(q.data(), k.data(), v.data(), out.data(), T, H, D);
  for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) mean[d] += v[t * D + d] / T;
  EXPECT(check::close_ratio(out.data(), mean.data(), D, 1e-6, 1e-5) <= 1);
}

TEST(attention_matches_softmax_composition) {
  // Recompute attention for one head from the other reference ops: scores = q K^T / sqrt(D),
  // p = softmax(scores), out = p V. Uses H=1 so K is a plain [T, D] matrix.
  const int T = 9, D = 8;
  auto q = check::randn(D, 8), k = check::randn(T * D, 9), v = check::randn(T * D, 10);
  std::vector<float> kt(D * T), s(T), p(T), want(D), got(D);
  for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) kt[d * T + t] = k[t * D + d];
  ck::sgemm_ref(1, T, D, 1.0f / std::sqrt(float(D)), q.data(), kt.data(), 0.0f, s.data());
  ck::softmax_rows_ref(s.data(), p.data(), 1, T);
  ck::sgemm_ref(1, D, T, 1.0f, p.data(), v.data(), 0.0f, want.data());
  ck::attention_decode_ref(q.data(), k.data(), v.data(), got.data(), T, 1, D);
  EXPECT(check::close_ratio(got.data(), want.data(), D, 1e-6, 1e-5) <= 1);
}

int main() { return check::run_all(); }
