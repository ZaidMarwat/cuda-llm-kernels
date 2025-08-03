// Measures GFLOP/s of each CPU GEMM rung on square matrices. Reports the best of `reps`
// runs (best-of is the usual choice for compute kernels: noise only ever slows a run down).
//   ./build/bench_cpu_gemm [threads]     prints a markdown table
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "../tests/check.h"
#include "ck/cpu_gemm.h"
#ifdef CK_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

static double best_seconds(const std::function<void()>& f, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (s < best) best = s;
  }
  return best;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? std::atoi(argv[1]) : int(std::thread::hardware_concurrency());
  const int sizes[] = {256, 512, 1024, 2048};
  std::printf("| variant | 256 | 512 | 1024 | 2048 |\n|---|---|---|---|---|\n");
  struct V { std::string name; std::function<void(int, const float*, const float*, float*)> f; int max_n; };
  std::vector<V> vs = {
      {"naive", [](int n, auto a, auto b, auto c) { ck::sgemm_naive(n, n, n, a, b, c); }, 1024},
      {"ikj", [](int n, auto a, auto b, auto c) { ck::sgemm_ikj(n, n, n, a, b, c); }, 2048},
      {"blocked", [](int n, auto a, auto b, auto c) { ck::sgemm_blocked(n, n, n, a, b, c); }, 2048},
      {"packed", [](int n, auto a, auto b, auto c) { ck::sgemm_packed(n, n, n, a, b, c); }, 2048},
      {"packed_mt (" + std::to_string(threads) + " thr)",
       [threads](int n, auto a, auto b, auto c) { ck::sgemm_packed_mt(n, n, n, a, b, c, threads); }, 2048},
#ifdef CK_HAVE_ACCELERATE
      {"Apple Accelerate cblas_sgemm (reference)",
       [](int n, auto a, auto b, auto c) {
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
       }, 2048},
#endif
  };
  for (auto& v : vs) {
    std::printf("| %s |", v.name.c_str());
    for (int n : sizes) {
      if (n > v.max_n) { std::printf(" (skipped) |"); continue; }
      auto A = check::randn(size_t(n) * n, 1), B = check::randn(size_t(n) * n, 2);
      std::vector<float> C(size_t(n) * n);
      int reps = n <= 512 ? 5 : 3;
      double s = best_seconds([&] { v.f(n, A.data(), B.data(), C.data()); }, reps);
      std::printf(" %.1f |", 2.0 * n * n * n / s / 1e9);
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  std::printf("\nGFLOP/s = 2*n^3 / best-of-reps wall time. naive skipped above 1024 (too slow).\n");
}
