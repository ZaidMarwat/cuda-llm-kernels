// Checks the split-KV CUDA decode attention against the CPU reference, for several split
// counts including the heuristic choice and more splits than positions.
//   make cuda && ./build/test_attention
#include <cstdio>
#include <vector>

#include "../tests/check.h"
#include "attention.cuh"
#include "ck/ref.h"
#include "common.cuh"

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
  struct C { int T, H, D; };
  const C cases[] = {{1, 1, 8}, {5, 2, 64}, {100, 12, 64}, {1024, 12, 64}, {4096, 32, 128}, {37, 3, 80}};
  int checks = 0, failures = 0;
  for (const C& c : cases) {
    auto q = check::randn(size_t(c.H) * c.D, 1, 2.0f);
    auto k = check::randn(size_t(c.T) * c.H * c.D, 2, 2.0f);
    auto v = check::randn(size_t(c.T) * c.H * c.D, 3);
    std::vector<float> want(size_t(c.H) * c.D), got(want.size());
    ck::attention_decode_ref(q.data(), k.data(), v.data(), want.data(), c.T, c.H, c.D);
    float *dq, *dk, *dv, *dout, *dws;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 4));
    CUDA_CHECK(cudaMalloc(&dk, k.size() * 4));
    CUDA_CHECK(cudaMalloc(&dv, v.size() * 4));
    CUDA_CHECK(cudaMalloc(&dout, got.size() * 4));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    const int heuristic = ck::cuda::attention_choose_splits(c.T, c.H, prop.multiProcessorCount);
    for (int splits : {1, 3, heuristic, c.T + 5}) {
      CUDA_CHECK(cudaMalloc(&dws, ck::cuda::attention_workspace_floats(c.H, c.D, splits) * 4));
      CUDA_CHECK(cudaMemset(dout, 0xFF, got.size() * 4));
      ck::cuda::attention_decode(dq, dk, dv, dout, c.T, c.H, c.D, splits, dws);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(dws));
      double r = check::close_ratio(got.data(), want.data(), got.size(), 1e-6, 1e-4);
      ++checks;
      failures += !(r <= 1);
      std::printf("[%s] T=%5d H=%2d D=%3d splits=%5d  worst err/tol = %.3f\n",
                  r <= 1 ? "  OK  " : " FAIL ", c.T, c.H, c.D, splits, r);
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
  }
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
