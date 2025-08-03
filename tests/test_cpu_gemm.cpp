// Every CPU GEMM variant must match the double-precision reference on shapes chosen to
// hit edge cases: 1x1x1, sizes that are not multiples of the 6x16 micro-tile, and sizes
// that cross the KC=256 / MC=120 packing block boundaries.
#include <cmath>
#include <cstdio>

#include "check.h"
#include "ck/cpu_gemm.h"
#include "ck/ref.h"

namespace {

struct Shape { int M, N, K; };
const Shape kShapes[] = {{1, 1, 1}, {7, 13, 5}, {6, 16, 1}, {33, 65, 129},
                         {121, 17, 257}, {130, 50, 600}, {200, 150, 300}};

using Fn = void (*)(int, int, int, const float*, const float*, float*);

void check_variant(const char* name, Fn fn) {
  for (const auto& s : kShapes) {
    auto A = check::randn(size_t(s.M) * s.K, 11), B = check::randn(size_t(s.K) * s.N, 12);
    std::vector<float> want(size_t(s.M) * s.N), got(size_t(s.M) * s.N, NAN);
    ck::sgemm_ref(s.M, s.N, s.K, 1.0f, A.data(), B.data(), 0.0f, want.data());
    fn(s.M, s.N, s.K, A.data(), B.data(), got.data());
    // fp32 accumulation error grows roughly like sqrt(K) * eps * |terms|.
    double atol = 1e-5 * std::sqrt(double(s.K)) * 4;
    double r = check::close_ratio(got.data(), want.data(), got.size(), atol, 1e-4);
    if (r > 1) std::printf("    %s %dx%dx%d: worst error ratio %.2f\n", name, s.M, s.N, s.K, r);
    EXPECT(r <= 1);
  }
}

void packed_mt_4(int M, int N, int K, const float* A, const float* B, float* C) {
  ck::sgemm_packed_mt(M, N, K, A, B, C, 4);
}
void packed_mt_64(int M, int N, int K, const float* A, const float* B, float* C) {
  ck::sgemm_packed_mt(M, N, K, A, B, C, 64);  // more threads than row strips
}

}  // namespace

TEST(naive_matches_ref) { check_variant("naive", ck::sgemm_naive); }
TEST(ikj_matches_ref) { check_variant("ikj", ck::sgemm_ikj); }
TEST(blocked_matches_ref) { check_variant("blocked", ck::sgemm_blocked); }
TEST(packed_matches_ref) { check_variant("packed", ck::sgemm_packed); }
TEST(packed_mt4_matches_ref) { check_variant("packed_mt4", packed_mt_4); }
TEST(packed_mt64_matches_ref) { check_variant("packed_mt64", packed_mt_64); }

int main() { return check::run_all(); }
