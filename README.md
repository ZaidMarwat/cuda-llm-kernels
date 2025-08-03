# cuda-llm-kernels

Kernels for the decode path of a GPT-style model (GEMM, softmax, LayerNorm,
single-query attention), built up step by step and measured against a roofline model.

Work in progress. So far: CPU reference implementations, plus a CPU SGEMM
optimization ladder (naive, loop interchange, cache blocking, packed register-blocked
micro-kernel, multithreaded) with tests and a benchmark.

```bash
make test     # reference + CPU GEMM correctness
make bench    # GFLOP/s table for the CPU ladder
```
