// Stub of the few cuBLAS declarations bench_gpu.cu uses, for `make cuda-syntax` only.
#pragma once
typedef struct cublasContext* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS = 0 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
cublasStatus_t cublasCreate(cublasHandle_t*);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                           const float*, const float*, int, const float*, int, const float*, float*, int);
