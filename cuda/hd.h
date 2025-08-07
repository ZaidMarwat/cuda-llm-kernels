// CK_HD marks small math helpers that compile for both host and device.
// nvcc defines __CUDACC__; clang in CUDA mode defines __CUDA__ (and __CUDACC__ only via the
// real CUDA headers). Checking both makes the headers work with either compiler.
#pragma once
#if defined(__CUDACC__) || defined(__CUDA__)
#define CK_HD __host__ __device__ __forceinline__
#else
#define CK_HD inline
#endif
