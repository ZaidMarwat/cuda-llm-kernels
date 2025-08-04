# cuda-llm-kernels

Kernels for the decode path of a GPT-style model (GEMM, softmax, LayerNorm,
single-query attention), built up step by step and measured against a roofline model.

Work in progress. So far:
- CPU reference implementations and tests
- a CPU SGEMM optimization ladder with a benchmark
- a roofline model (`roofline/`) whose host-CPU ceilings are measured by
  `bench/microbench.cpp` rather than taken from a spec sheet

```bash
make test                     # C++ and Python tests
make bench                    # GFLOP/s table for the CPU ladder
make roofline-host HOST_KEY=host-m1 HOST_NAME="Apple M1, 4 P-cores (measured)"
make report                   # roofline bounds for decode-path shapes on each device
```
