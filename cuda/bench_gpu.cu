// Times every CUDA kernel in this repo and prints CSV for roofline/gpu_report.py:
//   make cuda-bench > results/gpu_<name>.csv
//   python -m roofline.gpu_report results/gpu_<name>.csv --device t4
//
// Method: one warm-up launch, then `iters` timed launches. Before each timed launch the L2
// cache is flushed by writing a buffer larger than L2. Otherwise a batch-1 decode GEMM
// (2-9 MB of weights) can be served from L2 and look faster than DRAM allows. Each launch is
// bracketed by its own CUDA events, and the median time is reported.
// cuBLAS SGEMM is included as a reference ceiling, not as a kernel of this repo.
#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "../tests/check.h"
#include "attention.cuh"
#include "common.cuh"
#include "rowops.cuh"
#include "sgemm.cuh"

namespace {

float* dalloc_random(size_t n, unsigned seed) {
  auto h = check::randn(n, seed);
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

struct Timer {
  char* flush = nullptr;
  size_t flush_bytes;
  int iters;
  Timer(size_t l2_bytes, int iters_) : flush_bytes(std::max<size_t>(2 * l2_bytes, 64 << 20)), iters(iters_) {
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  }
  ~Timer() { cudaFree(flush); }
  double median_us(const std::function<void()>& launch) {
    launch();  // warm-up (also triggers lazy module loading)
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
      CUDA_CHECK(cudaMemset(flush, i & 0xFF, flush_bytes));  // evict L2
      CUDA_CHECK(cudaEventRecord(a));
      launch();
      CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b));
      CUDA_CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    return 1000.0 * ms[iters / 2];
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 21;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int l2 = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  Timer timer(size_t(l2), iters);
  const std::string dev = prop.name;
  std::printf("device,kernel,variant,p1,p2,p3,p4,median_us\n");
  auto row = [&](const char* kernel, const std::string& variant, int p1, int p2, int p3, int p4, double us) {
    std::printf("%s,%s,%s,%d,%d,%d,%d,%.3f\n", dev.c_str(), kernel, variant.c_str(), p1, p2, p3, p4, us);
    std::fflush(stdout);
  };

  // ---- SGEMM: p1..p3 = M, N, K ----------------------------------------------------------
  cublasHandle_t blas;
  if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublasCreate failed\n"); return 1; }
  struct G { int M, N, K; };
  for (G g : {G{1, 2304, 768}, G{1, 768, 768}, G{1, 3072, 768}, G{1, 768, 3072}, G{64, 2304, 768},
              G{1024, 1024, 1024}, G{2048, 2048, 2048}, G{4096, 4096, 4096}}) {
    float* A = dalloc_random(size_t(g.M) * g.K, 1);
    float* B = dalloc_random(size_t(g.K) * g.N, 2);
    float* C;
    CUDA_CHECK(cudaMalloc(&C, size_t(g.M) * g.N * sizeof(float)));
    for (auto v : {ck::cuda::Sgemm::naive, ck::cuda::Sgemm::coalesced, ck::cuda::Sgemm::smem,
                   ck::cuda::Sgemm::regtile}) {
      if (v == ck::cuda::Sgemm::naive && g.M >= 4096) continue;  // minutes per launch; skip
      row("sgemm", ck::cuda::name(v), g.M, g.N, g.K, 0,
          timer.median_us([&] { ck::cuda::sgemm(v, g.M, g.N, g.K, A, B, C); }));
    }
    // cuBLAS is column-major. Row-major C = A @ B is column-major C^T = B^T @ A^T, so pass B
    // first with the dimensions swapped; no transposes are actually performed.
    const float one = 1.0f, zero = 0.0f;
    row("sgemm", "cublas_reference", g.M, g.N, g.K, 0, timer.median_us([&] {
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, g.N, g.M, g.K, &one, B, g.N, A, g.K, &zero, C, g.N);
        }));
    cudaFree(A); cudaFree(B); cudaFree(C);
  }
  cublasDestroy(blas);

  // ---- softmax / layernorm: p1, p2 = rows, cols ------------------------------------------
  struct R { int rows, cols; };
  for (R r : {R{12, 1024}, R{12, 4096}, R{1, 768}, R{64, 768}, R{4096, 768}, R{4096, 4096}}) {
    const size_t n = size_t(r.rows) * r.cols;
    float* x = dalloc_random(n, 3);
    float* g = dalloc_random(r.cols, 4);
    float* b = dalloc_random(r.cols, 5);
    float* y;
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    row("softmax", "naive", r.rows, r.cols, 0, 0, timer.median_us([&] { ck::cuda::softmax_rows_naive(x, y, r.rows, r.cols); }));
    row("softmax", "warp_online", r.rows, r.cols, 0, 0, timer.median_us([&] { ck::cuda::softmax_rows(x, y, r.rows, r.cols); }));
    row("layernorm", "warp_welford", r.rows, r.cols, 0, 0,
        timer.median_us([&] { ck::cuda::layernorm_rows(x, y, g, b, r.rows, r.cols, 1e-5f); }));
    cudaFree(x); cudaFree(g); cudaFree(b); cudaFree(y);
  }

  // ---- decode attention: p1..p4 = T, H, D, splits ----------------------------------------
  struct A3 { int T, H, D; };
  for (A3 a : {A3{1024, 12, 64}, A3{4096, 12, 64}, A3{16384, 12, 64}, A3{4096, 32, 128}}) {
    float* q = dalloc_random(size_t(a.H) * a.D, 6);
    float* k = dalloc_random(size_t(a.T) * a.H * a.D, 7);
    float* v = dalloc_random(size_t(a.T) * a.H * a.D, 8);
    float* out;
    CUDA_CHECK(cudaMalloc(&out, size_t(a.H) * a.D * sizeof(float)));
    const int heur = ck::cuda::attention_choose_splits(a.T, a.H, prop.multiProcessorCount);
    for (int splits : {1, heur}) {
      float* ws;
      CUDA_CHECK(cudaMalloc(&ws, ck::cuda::attention_workspace_floats(a.H, a.D, splits) * sizeof(float)));
      row("attention_decode", splits == 1 ? "one_block_per_head" : "split_kv", a.T, a.H, a.D, splits,
          timer.median_us([&] { ck::cuda::attention_decode(q, k, v, out, a.T, a.H, a.D, splits, ws); }));
      cudaFree(ws);
      if (heur == 1) break;
    }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(out);
  }
  return 0;
}
