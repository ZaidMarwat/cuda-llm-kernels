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

## CUDA kernels (UNRUN: no NVIDIA GPU on the development machine)

- `cuda/sgemm.cu`: four SGEMM kernels (naive, coalesced, shared-memory tiled,
  register-tiled). `cuda/test_sgemm.cu` checks each one against the CPU reference.
- `cuda/rowops.cu`: a naive softmax baseline, a warp-per-row online softmax, and a
  warp-per-row Welford LayerNorm. `cuda/test_rowops.cu` checks them. The per-lane math is in
  `cuda/rowops_math.h`, and `tests/test_rowops_math.cpp` runs it on the CPU through a
  32-lane butterfly emulation, so the algorithm itself is tested here even without a GPU.
They have been type-checked with clang's CUDA front end (`make cuda-syntax`), but
**not yet compiled with nvcc or run on a GPU**. To run them on a machine with CUDA:

```bash
make cuda-test            # builds with nvcc -arch=native, runs the correctness checks
```
