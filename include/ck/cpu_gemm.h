// CPU SGEMM optimization ladder: C[M,N] = A[M,K] @ B[K,N], row-major, fp32.
//
// Each rung changes one thing, so the speedup of each step can be attributed:
//   naive    i-j-k loops; the inner loop walks B down a column (stride N), so most loads miss cache
//   ikj      loop interchange; the inner loop walks rows of B and C contiguously and vectorizes
//   blocked  ikj inside cache tiles, so a tile of B is reused while it is still in cache
//   packed   GotoBLAS-style: pack A and B panels into the order the micro-kernel reads them,
//            then a 6x16 register-blocked micro-kernel does 24 vector FMAs per 4 vector loads
//   packed_mt  packed, with rows of C split across threads
// The GPU kernels in later milestones follow the same ideas (coalescing, tiling into fast
// memory, register blocking), which is why the CPU ladder is built first.
#pragma once

namespace ck {

void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C);
void sgemm_ikj(int M, int N, int K, const float* A, const float* B, float* C);
void sgemm_blocked(int M, int N, int K, const float* A, const float* B, float* C);
void sgemm_packed(int M, int N, int K, const float* A, const float* B, float* C);
void sgemm_packed_mt(int M, int N, int K, const float* A, const float* B, float* C,
                     int threads);

}  // namespace ck
