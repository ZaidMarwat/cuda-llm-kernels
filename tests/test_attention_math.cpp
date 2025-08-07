// CPU emulation of the split-KV decode attention in cuda/attention.cu, using the same
// attn_push / attn_merge helpers and the same work split: `splits` chunks of positions,
// NW=4 "warps" per chunk taking every 4th position, a warp merge, then a cross-chunk combine.
#include <cmath>
#include <vector>

#include "../cuda/attention_math.h"
#include "check.h"
#include "ck/ref.h"

namespace {

std::vector<float> split_kv_attention(const std::vector<float>& q, const std::vector<float>& k,
                                      const std::vector<float>& v, int T, int H, int D, int splits) {
  const int NW = 4;
  const float scale = 1.0f / std::sqrt(float(D));
  std::vector<float> out(size_t(H) * D);
  const int chunk = (T + splits - 1) / splits;
  for (int h = 0; h < H; ++h) {
    std::vector<float> sm(splits), sl(splits), sacc(size_t(splits) * D);
    for (int sp = 0; sp < splits; ++sp) {
      const int t0 = sp * chunk, t1 = std::min(T, t0 + chunk);
      float bm = -INFINITY, bl = 0.0f;
      std::vector<float> bacc(D, 0.0f);
      for (int w = 0; w < NW; ++w) {
        float m = -INFINITY, l = 0.0f;
        std::vector<float> acc(D, 0.0f);
        for (int t = t0 + w; t < t1; t += NW) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += q[h * D + d] * scale * k[(size_t(t) * H + h) * D + d];
          float sc, p;
          ck::attn_push(m, l, dot, sc, p);
          for (int d = 0; d < D; ++d) acc[d] = acc[d] * sc + p * v[(size_t(t) * H + h) * D + d];
        }
        float s1, s2;
        ck::attn_merge(bm, bl, m, l, s1, s2);
        for (int d = 0; d < D; ++d) bacc[d] = bacc[d] * s1 + acc[d] * s2;
      }
      sm[sp] = bm;
      sl[sp] = bl;
      for (int d = 0; d < D; ++d) sacc[size_t(sp) * D + d] = bacc[d];
    }
    // combine, as attn_combine does
    float M = -INFINITY;
    for (int sp = 0; sp < splits; ++sp) M = std::fmax(M, sm[sp]);
    for (int d = 0; d < D; ++d) {
      float L = 0, a = 0;
      for (int sp = 0; sp < splits; ++sp) {
        if (sm[sp] == -INFINITY) continue;
        const float w = std::exp(sm[sp] - M);
        L += sl[sp] * w;
        a += sacc[size_t(sp) * D + d] * w;
      }
      out[size_t(h) * D + d] = a / L;
    }
  }
  return out;
}

}  // namespace

TEST(split_kv_matches_ref_across_split_counts) {
  struct C { int T, H, D; };
  for (C c : {C{1, 1, 8}, C{5, 2, 64}, C{100, 12, 64}, C{1024, 4, 128}, C{37, 3, 80}}) {
    auto q = check::randn(size_t(c.H) * c.D, 1, 2.0f);
    auto k = check::randn(size_t(c.T) * c.H * c.D, 2, 2.0f);
    auto v = check::randn(size_t(c.T) * c.H * c.D, 3);
    std::vector<float> want(size_t(c.H) * c.D);
    ck::attention_decode_ref(q.data(), k.data(), v.data(), want.data(), c.T, c.H, c.D);
    for (int splits : {1, 2, 3, 7, 16, c.T + 5}) {  // T+5: some chunks are empty
      auto got = split_kv_attention(q, k, v, c.T, c.H, c.D, splits);
      double r = check::close_ratio(got.data(), want.data(), want.size(), 1e-6, 1e-4);
      if (r > 1) std::printf("    T=%d H=%d D=%d splits=%d ratio %.2f\n", c.T, c.H, c.D, splits, r);
      EXPECT(r <= 1);
    }
  }
}

TEST(merge_handles_empty_sides) {
  float m = -INFINITY, l = 0, s1, s2;
  ck::attn_merge(m, l, -INFINITY, 0, s1, s2);
  EXPECT(m == -INFINITY && l == 0 && s1 == 0 && s2 == 0);
  ck::attn_merge(m, l, 2.0f, 3.0f, s1, s2);
  EXPECT(m == 2.0f && l == 3.0f && s1 == 0 && s2 == 1.0f);
}

TEST(push_is_stable_for_large_scores) {
  float m = -INFINITY, l = 0, sc, p;
  ck::attn_push(m, l, 1e4f, sc, p);
  ck::attn_push(m, l, 1e4f + 1.0f, sc, p);
  EXPECT(std::isfinite(l) && std::fabs(l - (1.0f + std::exp(-1.0f))) < 1e-6f);
}

int main() { return check::run_all(); }
