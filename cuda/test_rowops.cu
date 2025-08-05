// Checks the CUDA softmax and LayerNorm kernels against the CPU references.
//   make cuda && ./build/test_rowops
#include <cmath>
#include <cstdio>
#include <vector>

#include "../tests/check.h"
#include "ck/ref.h"
#include "common.cuh"
#include "rowops.cuh"

namespace {

struct Dev {  // tiny RAII device buffer
  float* p = nullptr;
  explicit Dev(const std::vector<float>& h) {
    CUDA_CHECK(cudaMalloc(&p, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  explicit Dev(size_t n) {
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(p, 0xFF, n * sizeof(float)));  // NaN: unwritten outputs fail
  }
  ~Dev() { cudaFree(p); }
  std::vector<float> get(size_t n) const {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
  }
};

int checks = 0, failures = 0;
void report(const char* what, int rows, int cols, double r) {
  ++checks;
  failures += !(r <= 1);
  std::printf("[%s] %-16s %5d x %6d  worst err/tol = %.3f\n", r <= 1 ? "  OK  " : " FAIL ", what,
              rows, cols, r);
}

}  // namespace

int main() {
  struct Shape { int rows, cols; float scale; };
  const Shape shapes[] = {{1, 1, 1}, {3, 31, 1}, {7, 32, 1}, {5, 33, 1}, {64, 768, 3},
                          {16, 4096, 3}, {2, 100000, 3}, {1000, 129, 50}};
  for (const auto& s : shapes) {
    const size_t n = size_t(s.rows) * s.cols;
    auto x = check::randn(n, 40 + s.cols, s.scale);
    auto g = check::randn(s.cols, 41), b = check::randn(s.cols, 42);
    std::vector<float> want(n);
    Dev dx(x), dg(g), db(b);

    ck::softmax_rows_ref(x.data(), want.data(), s.rows, s.cols);
    {
      Dev dy(n);
      ck::cuda::softmax_rows_naive(dx.p, dy.p, s.rows, s.cols);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto got = dy.get(n);
      report("softmax_naive", s.rows, s.cols, check::close_ratio(got.data(), want.data(), n, 1e-7, 1e-5));
    }
    {
      Dev dy(n);
      ck::cuda::softmax_rows(dx.p, dy.p, s.rows, s.cols);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto got = dy.get(n);
      report("softmax_warp", s.rows, s.cols, check::close_ratio(got.data(), want.data(), n, 1e-7, 1e-5));
    }
    ck::layernorm_rows_ref(x.data(), want.data(), g.data(), b.data(), s.rows, s.cols, 1e-5f);
    {
      Dev dy(n);
      ck::cuda::layernorm_rows(dx.p, dy.p, dg.p, db.p, s.rows, s.cols, 1e-5f);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto got = dy.get(n);
      report("layernorm_warp", s.rows, s.cols, check::close_ratio(got.data(), want.data(), n, 1e-5, 1e-4));
    }
  }
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
