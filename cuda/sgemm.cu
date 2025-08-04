#include "common.cuh"
#include "sgemm.cuh"

namespace ck::cuda {
namespace {

__global__ void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  // threadIdx.x -> row. Neighbouring threads in a warp touch rows r, r+1, ..., so their
  // accesses to A[r*K + p] and C[r*N + c] are K or N floats apart.
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= M || c >= N) return;
  float acc = 0.0f;
  for (int p = 0; p < K; ++p) acc += A[r * K + p] * B[p * N + c];
  C[r * N + c] = acc;
}

__global__ void sgemm_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
  // threadIdx.x -> column. A warp shares one row of A (a broadcast) and reads/writes 32
  // consecutive columns of B and C.
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= M || c >= N) return;
  float acc = 0.0f;
  for (int p = 0; p < K; ++p) acc += A[r * K + p] * B[p * N + c];
  C[r * N + c] = acc;
}

constexpr int TILE = 32;

__global__ void sgemm_smem(int M, int N, int K, const float* A, const float* B, float* C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int r = blockIdx.y * TILE + ty, c = blockIdx.x * TILE + tx;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // Each thread loads one element of each tile. Out-of-range elements are zero, so the
    // inner loop needs no bounds checks and ragged edges just add zeros.
    As[ty][tx] = (r < M && k0 + tx < K) ? A[r * K + k0 + tx] : 0.0f;
    Bs[ty][tx] = (k0 + ty < K && c < N) ? B[(k0 + ty) * N + c] : 0.0f;
    __syncthreads();  // the tile must be complete before anyone reads it
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];  // As row: broadcast; Bs row: conflict-free
    __syncthreads();  // ...and fully consumed before the next iteration overwrites it
  }
  if (r < M && c < N) C[r * N + c] = acc;
}

// Block computes a BM x BN tile of C with (BM/TM)*(BN/TN) threads; each thread owns a
// TM x TN micro-tile held in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemm_regtile(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  // A is stored transposed (As[k][m]) so the TM values a thread needs for one k are
  // contiguous. The transposing store has some bank conflicts; loads in the hot loop do not.
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float a[TM], b[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int m = i / BK, k = i % BK;  // consecutive threads read consecutive k of one A row
      const int gr = row0 + m, gk = k0 + k;
      As[k][m] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int k = i / BN, n = i % BN;  // consecutive threads read consecutive columns of B
      const int gk = k0 + k, gc = col0 + n;
      Bs[k][n] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[k][tr * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];  // TM+TN loads feed TM*TN FMAs
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = row0 + tr * TM + i, c = col0 + tc * TN + j;
      if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}

}  // namespace

const char* name(Sgemm v) {
  switch (v) {
    case Sgemm::naive: return "naive";
    case Sgemm::coalesced: return "coalesced";
    case Sgemm::smem: return "smem_tiled_32";
    case Sgemm::regtile: return "regtile_64x64_4x4";
  }
  return "?";
}

void sgemm(Sgemm v, int M, int N, int K, const float* A, const float* B, float* C,
           cudaStream_t stream) {
  switch (v) {
    case Sgemm::naive: {
      dim3 block(32, 32), grid(ceil_div(M, 32), ceil_div(N, 32));
      sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case Sgemm::coalesced: {
      dim3 block(32, 32), grid(ceil_div(N, 32), ceil_div(M, 32));
      sgemm_coalesced<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case Sgemm::smem: {
      dim3 block(TILE, TILE), grid(ceil_div(N, TILE), ceil_div(M, TILE));
      sgemm_smem<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case Sgemm::regtile: {
      constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
      dim3 block((BM / TM) * (BN / TN)), grid(ceil_div(N, BN), ceil_div(M, BM));
      sgemm_regtile<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());  // catches bad launch configurations immediately
}

}  // namespace ck::cuda
