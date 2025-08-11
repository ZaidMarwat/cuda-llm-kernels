// Minimal stand-in for <cuda_runtime.h> so clang can *syntax- and type-check* the .cu files
// on a machine without the CUDA toolkit (`make cuda-syntax`). It declares only what this
// repo uses. Nothing built against it can run: this is a static check, not a compile for a
// GPU, and it does not replace building with nvcc.
#pragma once
#include <cstddef>
#include <cstdint>
#include "__clang_cuda_builtin_vars.h"  // threadIdx, blockIdx, blockDim, gridDim, warpSize

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __restrict__ __restrict

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
struct cudaDeviceProp { char name[256]; int multiProcessorCount; int major, minor; size_t sharedMemPerBlock; };
constexpr cudaError_t cudaSuccess = 0;

// Kernel launch hooks clang lowers `<<<...>>>` to (which one depends on the assumed CUDA version).
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void* = nullptr);
cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = nullptr);
cudaError_t cudaMalloc(void**, size_t);
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
enum cudaDeviceAttr { cudaDevAttrL2CacheSize = 89 };
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);

__device__ void __syncthreads();
__device__ float __shfl_xor_sync(unsigned, float, int, int = 32);
__device__ float __shfl_down_sync(unsigned, float, unsigned, int = 32);
__device__ float __shfl_sync(unsigned, float, int, int = 32);
__device__ float __expf(float);
__device__ float expf(float);
__device__ float rsqrtf(float);
__device__ float fmaxf(float, float);
__device__ float __fdividef(float, float);
__device__ float __ldg(const float*);
struct float4 { float x, y, z, w; };
