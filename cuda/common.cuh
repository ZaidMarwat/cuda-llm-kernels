#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                                \
  do {                                                                                  \
    cudaError_t err_ = (expr);                                                          \
    if (err_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,          \
                   cudaGetErrorString(err_));                                           \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

namespace ck {
inline __host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }
}  // namespace ck
