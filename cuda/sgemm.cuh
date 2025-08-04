// CUDA SGEMM ladder: C[M,N] = A[M,K] @ B[K,N], row-major fp32, device pointers.
//
// The progression follows the same ideas as the CPU ladder in src/cpu_gemm.cpp:
//   naive       one thread per C element; threadIdx.x walks rows, so a warp's loads of A
//               and stores of C are strided by K and N (uncoalesced)
//   coalesced   same work, threadIdx.x walks columns: a warp reads 32 consecutive floats of
//               B and writes 32 consecutive floats of C (one or two memory transactions)
//   smem        32x32 tiles of A and B staged in shared memory; each global value is loaded
//               once per tile instead of once per thread that needs it (32x fewer loads)
//   regtile     64x64 block tile, each thread computes a 4x4 micro-tile from registers,
//               so each shared-memory load feeds 4 FMAs (like the CPU micro-kernel)
// The structure is the well-known one from the CUDA C++ Programming Guide's shared-memory
// matmul and Simon Boehm's 2022 article "How to Optimize a CUDA Matmul Kernel for
// cuBLAS-like Performance". The code here is my own implementation of those ideas.
#pragma once
#include <cuda_runtime.h>

namespace ck::cuda {

enum class Sgemm { naive, coalesced, smem, regtile };
const char* name(Sgemm v);

// Launches the kernel on `stream`. Does not synchronize.
void sgemm(Sgemm v, int M, int N, int K, const float* A, const float* B, float* C,
           cudaStream_t stream = nullptr);

}  // namespace ck::cuda
