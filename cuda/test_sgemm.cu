// Checks every CUDA SGEMM variant against the CPU double-precision reference (src/ref.cpp)
// on shapes that exercise partial tiles in every dimension. Exit code 0 only if all pass.
//   make cuda && ./build/test_sgemm
#include <cmath>
#include <cstdio>
#include <vector>

#include "../tests/check.h"
#include "ck/ref.h"
#include "common.cuh"
#include "sgemm.cuh"

int main() {
  struct Shape { int M, N, K; };
  const Shape shapes[] = {{1, 1, 1}, {1, 768, 768}, {7, 13, 5}, {33, 65, 129},
                          {64, 64, 16}, {100, 300, 17}, {257, 129, 513}, {512, 512, 512}};
  const ck::cuda::Sgemm variants[] = {ck::cuda::Sgemm::naive, ck::cuda::Sgemm::coalesced,
                                      ck::cuda::Sgemm::smem, ck::cuda::Sgemm::regtile};
  int failures = 0, checks = 0;
  for (const auto& s : shapes) {
    auto A = check::randn(size_t(s.M) * s.K, 21), B = check::randn(size_t(s.K) * s.N, 22);
    std::vector<float> want(size_t(s.M) * s.N), got(want.size());
    ck::sgemm_ref(s.M, s.N, s.K, 1.0f, A.data(), B.data(), 0.0f, want.data());
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, got.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (auto v : variants) {
      CUDA_CHECK(cudaMemset(dC, 0xFF, got.size() * sizeof(float)));  // NaN pattern: catches unwritten outputs
      ck::cuda::sgemm(v, s.M, s.N, s.K, dA, dB, dC);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
      double atol = 4e-5 * std::sqrt(double(s.K));
      double r = check::close_ratio(got.data(), want.data(), got.size(), atol, 1e-4);
      bool ok = r <= 1;
      ++checks;
      failures += !ok;
      std::printf("[%s] %-18s %4dx%4dx%4d  worst err/tol = %.3f\n", ok ? "  OK  " : " FAIL ",
                  ck::cuda::name(v), s.M, s.N, s.K, r);
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
