#include "ck/cpu_gemm.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace ck {

void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int p = 0; p < K; ++p) acc += A[i * K + p] * B[p * N + j];  // B stride N: cache-hostile
      C[i * N + j] = acc;
    }
}

void sgemm_ikj(int M, int N, int K, const float* A, const float* B, float* C) {
  std::memset(C, 0, sizeof(float) * size_t(M) * N);
  for (int i = 0; i < M; ++i)
    for (int p = 0; p < K; ++p) {
      const float a = A[i * K + p];
      const float* b = B + size_t(p) * N;
      float* c = C + size_t(i) * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];  // unit stride: the compiler vectorizes this
    }
}

void sgemm_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
  // 64x256 floats of B = 64 KiB, about half of the M1 performance core's 128 KiB L1d.
  constexpr int BI = 64, BK = 64, BJ = 256;
  std::memset(C, 0, sizeof(float) * size_t(M) * N);
  for (int i0 = 0; i0 < M; i0 += BI)
    for (int p0 = 0; p0 < K; p0 += BK)
      for (int j0 = 0; j0 < N; j0 += BJ) {
        const int i1 = std::min(i0 + BI, M), p1 = std::min(p0 + BK, K), j1 = std::min(j0 + BJ, N);
        for (int i = i0; i < i1; ++i)
          for (int p = p0; p < p1; ++p) {
            const float a = A[i * K + p];
            const float* b = B + size_t(p) * N;
            float* c = C + size_t(i) * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

// ---------------------------------------------------------------------------------------
// Packed GEMM. Loop structure follows Goto & van de Geijn, "Anatomy of High-Performance
// Matrix Multiplication" (ACM TOMS 2008), and the BLIS papers. This is my own
// implementation of that published structure, not copied code.
//
//   for jc in N step NC:          B panel [KC x NC] packed once, reused by every ic
//     for pc in K step KC:
//       pack B[pc:pc+KC, jc:jc+NC] into NR-wide column strips
//       for ic in M step MC:
//         pack A[ic:ic+MC, pc:pc+KC] into MR-tall row strips
//         for each (MR x NR) tile: micro-kernel over KC
// ---------------------------------------------------------------------------------------
namespace {

// GCC/Clang vector extension: 4 floats = one 128-bit NEON (or SSE) register. Portable
// across ARM and x86 without intrinsics, and the code still reads like arithmetic.
typedef float v4 __attribute__((vector_size(16)));

constexpr int MR = 6, NR = 16;            // 6 rows x 4 vectors = 24 accumulator registers
constexpr int KC = 256, MC = 120, NC = 3072;
// KC*NR*4 B = 16 KiB: one packed B strip lives in L1 for the whole micro-kernel.
// MC*KC*4 B = 120 KiB: the packed A block lives in L2 and streams through L1.

// Pack rows [0,mc) x cols [0,kc) of A (leading dim lda) into MR-tall strips, column-major
// within a strip: strip s holds A[s*MR + r][p] at out[s*MR*kc + p*MR + r]. Rows past mc
// are zero-filled so the micro-kernel never needs an edge case.
void pack_a(int mc, int kc, const float* A, int lda, float* out) {
  for (int s = 0; s < mc; s += MR)
    for (int p = 0; p < kc; ++p)
      for (int r = 0; r < MR; ++r) *out++ = (s + r < mc) ? A[size_t(s + r) * lda + p] : 0.0f;
}

// Pack rows [0,kc) x cols [0,nc) of B into NR-wide strips, row-major within a strip.
void pack_b(int kc, int nc, const float* B, int ldb, float* out) {
  for (int s = 0; s < nc; s += NR)
    for (int p = 0; p < kc; ++p)
      for (int c = 0; c < NR; ++c) *out++ = (s + c < nc) ? B[size_t(p) * ldb + s + c] : 0.0f;
}

// C_tile[MR x NR] += Apanel[MR x kc] @ Bpanel[kc x NR]. Accumulates in registers and
// writes back once. `m`,`n` are the valid rows/cols of this tile (smaller at the edges).
void micro_kernel(int kc, const float* a, const float* b, float* C, int ldc, int m, int n) {
  v4 acc[MR][NR / 4] = {};
  for (int p = 0; p < kc; ++p) {
    v4 bv[NR / 4];
    std::memcpy(bv, b + p * NR, sizeof(bv));  // memcpy: alignment-safe vector load
    for (int r = 0; r < MR; ++r) {
      const float s = a[p * MR + r];
      const v4 av = {s, s, s, s};  // broadcast
      for (int q = 0; q < NR / 4; ++q) acc[r][q] += av * bv[q];
    }
  }
  if (m == MR && n == NR) {
    for (int r = 0; r < MR; ++r) {
      float* c = C + size_t(r) * ldc;
      for (int q = 0; q < NR / 4; ++q) {
        v4 cur;
        std::memcpy(&cur, c + 4 * q, sizeof(v4));
        cur += acc[r][q];
        std::memcpy(c + 4 * q, &cur, sizeof(v4));
      }
    }
  } else {  // edge tile: spill to a buffer, add only the valid part
    float tmp[MR][NR];
    std::memcpy(tmp, acc, sizeof(tmp));
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < n; ++c) C[size_t(r) * ldc + c] += tmp[r][c];
  }
}

}  // namespace

void sgemm_packed(int M, int N, int K, const float* A, const float* B, float* C) {
  std::memset(C, 0, sizeof(float) * size_t(M) * N);
  std::vector<float> bp(size_t(KC) * ((std::min(N, NC) + NR - 1) / NR * NR));
  std::vector<float> ap(size_t(KC) * ((std::min(M, MC) + MR - 1) / MR * MR));
  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      pack_b(kc, nc, B + size_t(pc) * N + jc, N, bp.data());
      for (int ic = 0; ic < M; ic += MC) {
        const int mc = std::min(MC, M - ic);
        pack_a(mc, kc, A + size_t(ic) * K + pc, K, ap.data());
        for (int jr = 0; jr < nc; jr += NR)
          for (int ir = 0; ir < mc; ir += MR)
            micro_kernel(kc, ap.data() + size_t(ir) * kc, bp.data() + size_t(jr) * kc,
                         C + size_t(ic + ir) * N + jc + jr, N, std::min(MR, mc - ir),
                         std::min(NR, nc - jr));
      }
    }
  }
}

void sgemm_packed_mt(int M, int N, int K, const float* A, const float* B, float* C,
                     int threads) {
  // Split C by rows: each thread owns a disjoint row range, so no synchronization is
  // needed. Each thread packs its own copy of B, which costs K*N extra copies per thread;
  // that is small next to the 2*M*N*K/threads flops each thread does.
  threads = std::max(1, std::min(threads, (M + MR - 1) / MR));
  const int strips = (M + MR - 1) / MR;  // MR-row strips, rounded up
  const int rows = (strips + threads - 1) / threads * MR;  // whole strips per thread
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    const int i0 = t * rows, i1 = std::min(M, i0 + rows);
    if (i0 >= i1) break;
    pool.emplace_back([=] { sgemm_packed(i1 - i0, N, K, A + size_t(i0) * K, B, C + size_t(i0) * N); });
  }
  for (auto& th : pool) th.join();
}

}  // namespace ck
