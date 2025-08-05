// CPU emulation of the warp-level algorithms in cuda/rowops.cu: 32 "lanes" each process
// columns lane, lane+32, ..., then a butterfly of xor-shuffles merges their states exactly
// as __shfl_xor_sync would (every lane reads its partner's value from before the step).
// This tests the math and the reduction order; the kernels themselves still need a GPU.
#include <array>
#include <cmath>

#include "../cuda/rowops_math.h"
#include "check.h"
#include "ck/ref.h"

namespace {
constexpr int W = 32;

std::vector<float> warp_softmax(const std::vector<float>& x) {
  const int cols = int(x.size());
  std::array<float, W> m, d;
  m.fill(-INFINITY);
  d.fill(0.0f);
  for (int l = 0; l < W; ++l)
    for (int c = l; c < cols; c += W) ck::online_push(m[l], d[l], x[c]);
  for (int off = W / 2; off > 0; off /= 2) {
    auto m0 = m, d0 = d;  // values before this step, as a shuffle sees them
    for (int l = 0; l < W; ++l) ck::online_merge(m[l], d[l], m0[l ^ off], d0[l ^ off]);
  }
  // Every lane ends with the max exactly, but with d only approximately: each lane sums the
  // 32 partial results in a different grouping ((0+16)+(8+24)... vs (1+17)+(9+25)...), so
  // the fp32 rounding differs by lane. The kernel lets each lane use its own d. That is
  // harmless (relative differences ~1e-7) but means output columns are not normalized by a
  // bit-identical sum. This test caught that assumption; an earlier version asserted equality.
  for (int l = 1; l < W; ++l) EXPECT(m[l] == m[0] && std::fabs(d[l] - d[0]) <= 1e-6f * d[0]);
  std::vector<float> y(cols);
  for (int c = 0; c < cols; ++c) y[c] = expf(x[c] - m[c % W]) / d[c % W];
  return y;
}

void warp_moments(const std::vector<float>& x, float& mean, float& var) {
  const int cols = int(x.size());
  std::array<float, W> n{}, mu{}, m2{};
  for (int l = 0; l < W; ++l)
    for (int c = l; c < cols; c += W) ck::welford_push(n[l], mu[l], m2[l], x[c]);
  for (int off = W / 2; off > 0; off /= 2) {
    auto n0 = n, mu0 = mu, m20 = m2;
    for (int l = 0; l < W; ++l) ck::welford_merge(n[l], mu[l], m2[l], n0[l ^ off], mu0[l ^ off], m20[l ^ off]);
  }
  mean = mu[0];
  var = m2[0] / cols;
}
}  // namespace

TEST(warp_softmax_matches_ref_for_many_widths) {
  for (int cols : {1, 5, 31, 32, 33, 64, 100, 768, 4097}) {
    auto x = check::randn(cols, 30 + cols, 4.0f);
    std::vector<float> want(cols);
    ck::softmax_rows_ref(x.data(), want.data(), 1, cols);
    auto got = warp_softmax(x);
    EXPECT(check::close_ratio(got.data(), want.data(), cols, 1e-7, 1e-5) <= 1);
  }
}

TEST(warp_softmax_extreme_logits_and_masking) {
  std::vector<float> x = {1e4f, -1e4f, 1e4f - 1.0f, -INFINITY, 0.0f};
  std::vector<float> want(x.size());
  ck::softmax_rows_ref(x.data(), want.data(), 1, int(x.size()));
  auto got = warp_softmax(x);
  for (float v : got) EXPECT(std::isfinite(v));
  EXPECT(got[3] == 0.0f);
  EXPECT(check::close_ratio(got.data(), want.data(), x.size(), 1e-7, 1e-5) <= 1);
}

TEST(warp_welford_matches_ref) {
  for (int cols : {1, 7, 32, 33, 768, 5000}) {
    auto x = check::randn(cols, 50 + cols, 3.0f);
    double m = 0, v = 0;
    for (float e : x) m += e;
    m /= cols;
    for (float e : x) v += (e - m) * (e - m);
    v /= cols;
    float mean, var;
    warp_moments(x, mean, var);
    EXPECT(std::fabs(mean - m) < 1e-5 * (1 + std::fabs(m)));
    EXPECT(std::fabs(var - v) < 1e-4 * (1 + v));
  }
}

TEST(welford_survives_large_mean_where_sum_of_squares_fails) {
  // mean 1e4, std 1: the one-pass E[x^2]-E[x]^2 formula in fp32 loses all precision.
  auto x = check::randn(768, 77, 1.0f);
  for (auto& e : x) e += 1e4f;
  float mean, var;
  warp_moments(x, mean, var);
  float s = 0, s2 = 0;
  for (float e : x) { s += e; s2 += e * e; }
  const float naive_var = s2 / 768 - (s / 768) * (s / 768);
  double m = 0, v = 0;
  for (float e : x) m += e;
  m /= 768;
  for (float e : x) v += (e - m) * (e - m);
  v /= 768;
  std::printf("    true var %.4f, welford %.4f, naive fp32 sum-of-squares %.4f\n", v, var, naive_var);
  EXPECT(std::fabs(var - v) < 1e-2 * v);
  EXPECT(std::fabs(naive_var - v) > 0.1 * v);  // documents why the kernel uses Welford
}

int main() { return check::run_all(); }
