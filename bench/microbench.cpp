// Measures the two numbers a roofline needs on the host CPU, instead of trusting a spec sheet:
//   peak fp32 FMA throughput (GFLOP/s), single thread and all threads
//   sustained memory bandwidth (GB/s) with a STREAM-style triad, single thread and all threads
// Output is `key value` lines so scripts can parse it.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

typedef float v4 __attribute__((vector_size(16)));

// CHAINS independent accumulators hide FMA latency: with latency L cycles and P FMA pipes,
// at least L*P chains must be in flight to keep every pipe busy (M1: about 4 x 4 = 16).
constexpr int CHAINS = 16;

#if defined(__aarch64__)
#include <arm_neon.h>
static inline v4 fma4(v4 acc, v4 x, v4 y) {  // acc + x*y, one `fmla` instruction
  return (v4)vfmaq_f32((float32x4_t)acc, (float32x4_t)x, (float32x4_t)y);
}
#elif defined(__FMA__)
#include <immintrin.h>
static inline v4 fma4(v4 acc, v4 x, v4 y) { return (v4)_mm_fmadd_ps((__m128)x, (__m128)y, (__m128)acc); }
#else
#error "microbench needs AArch64 or x86 with FMA (build with -march=native)"
#endif

static float fma_loop(long iters) {
  v4 acc[CHAINS];
  for (int i = 0; i < CHAINS; ++i) acc[i] = v4{1.0f, 1.0f, 1.0f, 1.0f} * float(i + 1) * 1e-3f;
  const v4 m = {0.999999f, 0.999999f, 0.999999f, 0.999999f};
  const v4 a = {1e-7f, 1e-7f, 1e-7f, 1e-7f};
  for (long it = 0; it < iters; ++it)
    // An explicit fused multiply-add. Plain `acc += a * m` is not safe here: a*m is loop
    // invariant, so the compiler hoists it and the loop becomes adds (verified in the
    // disassembly: 64 fadd, 4 fmla). The intrinsic is a single FMA the compiler will not split.
    for (int i = 0; i < CHAINS; ++i) acc[i] = fma4(acc[i], a, m);
  float s = 0;
  for (int i = 0; i < CHAINS; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  return s;  // returned so the compiler cannot delete the loop
}

template <class F>
static double best_seconds(F&& f, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  return best;
}

static double peak_gflops(int threads) {
  const long iters = 50'000'000;
  volatile float sink = 0;
  double s = best_seconds([&] {
    std::vector<std::thread> pool;
    std::vector<float> out(threads);
    for (int t = 0; t < threads; ++t) pool.emplace_back([&, t] { out[t] = fma_loop(iters); });
    for (auto& th : pool) th.join();
    for (float v : out) sink = sink + v;
  }, 3);
  return double(threads) * iters * CHAINS * 4 /*lanes*/ * 2 /*flops per FMA*/ / s / 1e9;
}

static double triad_gbs(int threads) {
  const size_t n = size_t(1) << 25;  // 32M floats = 128 MiB per array, far larger than caches
  std::vector<float> a(n, 0.0f), b(n, 1.0f), c(n, 2.0f);
  const float k = 3.0f;
  double s = best_seconds([&] {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        size_t lo = n * t / threads, hi = n * (t + 1) / threads;
        for (size_t i = lo; i < hi; ++i) a[i] = b[i] + k * c[i];
      });
    for (auto& th : pool) th.join();
  }, 5);
  // STREAM convention: 3 arrays x 4 bytes counted (reads of b, c and write of a). Write-allocate
  // traffic is not counted, so this is the bandwidth a kernel can actually use.
  if (a[n / 2] != 7.0f) std::fprintf(stderr, "triad check failed\n");
  return 3.0 * 4.0 * n / s / 1e9;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? std::atoi(argv[1]) : int(std::thread::hardware_concurrency());
  std::printf("threads %d\n", threads);
  std::printf("peak_gflops_1t %.1f\n", peak_gflops(1));
  std::printf("peak_gflops_mt %.1f\n", peak_gflops(threads));
  std::printf("triad_gbs_1t %.1f\n", triad_gbs(1));
  std::printf("triad_gbs_mt %.1f\n", triad_gbs(threads));
}
