# cuda-llm-kernels

The four operations a GPT-style model runs on every decode step (matrix multiply,
softmax, LayerNorm, single-query attention), written as CUDA kernels and as a CPU
optimization ladder. Each version is checked against a double-precision reference and
measured against a roofline model, so "fast" always means "a stated fraction of what the
hardware allows".

> **Status of GPU results:** the CUDA kernels have been type-checked with clang's CUDA
> front end, but **not yet compiled with nvcc or run on an NVIDIA GPU**. Every number
> below comes from the CPU on an Apple M1. GPU results are marked placeholders; see
> [Running on a GPU](#running-on-a-gpu).

## Problem

During LLM decoding with batch size 1, every operation reads a whole weight matrix or the
whole KV cache to produce one token. That makes decode memory-bandwidth-bound, while
large-batch prefill is compute-bound. Writing kernels for this regime means knowing which
ceiling applies and how close a kernel gets to it. This repo builds that up from scratch:

1. a trusted reference for each op,
2. the classic GEMM optimizations, measured on a CPU where they can be run here,
3. a roofline model whose CPU ceilings are measured rather than taken from a spec sheet,
4. CUDA kernels for each op, plus the harness to verify and benchmark them on a GPU.

## Layout

| Path | What it is |
|---|---|
| `src/ref.cpp` | Reference SGEMM, softmax, LayerNorm and decode attention (double accumulation) |
| `src/cpu_gemm.cpp` | CPU SGEMM ladder: naive, ikj, cache-blocked, packed 6x16 micro-kernel, multithreaded |
| `bench/microbench.cpp` | Measured peak FMA throughput and STREAM-triad bandwidth |
| `roofline/` | Roofline model, device table, report generators |
| `cuda/sgemm.cu` | SGEMM: naive, coalesced, 32x32 shared-memory tiled, 64x64 register-tiled (4x4/thread) |
| `cuda/rowops.cu` | Softmax (naive and warp-per-row online) and Welford LayerNorm (warp-per-row) |
| `cuda/attention.cu` | Split-KV decode attention plus a merge kernel |
| `cuda/*_math.h` | Per-lane math shared by the kernels and the CPU emulation tests |
| `cuda/test_*.cu`, `cuda/bench_gpu.cu` | GPU correctness checks and the GPU benchmark (CSV) |
| `tools/cuda_syntax/` | Stub CUDA/cuBLAS headers for `make cuda-syntax` (static check, no toolkit needed) |

## Design notes

**GEMM, CPU and GPU, the same three ideas.** (1) Access memory in the order it is laid
out. That is loop interchange on the CPU and mapping `threadIdx.x` to columns on the GPU,
so a warp's loads coalesce. (2) Reuse data while it sits in fast memory: cache blocking
and packing on the CPU, shared-memory tiles on the GPU. (3) Hold a small tile of C in
registers, so each value loaded feeds several FMAs: the 6x16 NEON micro-kernel on the CPU,
4x4 per thread on the GPU. The structure follows Goto & van de Geijn (2008), the CUDA
Programming Guide's shared-memory matmul, and Simon Boehm's 2022 CUDA matmul article. The
code is my own.

**Softmax and LayerNorm: one warp per row.** Decode rows are short (768 features, or T
attention scores), so a warp per row with shuffle reductions avoids shared memory and
block-level syncs. Softmax uses the online normalizer (Milakov & Gimelshein, 2018), which
gets the max and the sum in one pass, so each element is read twice and written once.
LayerNorm uses Welford's algorithm with Chan et al.'s parallel merge, because the
one-pass `E[x^2] - E[x]^2` formula collapses in fp32 when the mean is large. A test shows
this: at mean 1e4 and variance 0.977 it returns -61.3.

**Attention: split the KV cache.** With one block per head, a 12-head model launches 12
blocks, too few to use a GPU's bandwidth. `attention_decode` splits the T positions into
chunks. Each block (head, chunk) produces a partial `(max, sum, weighted V)`, and a
second kernel merges them exactly, because partial softmaxes combine by rescaling to the
larger max. This is the idea behind FlashDecoding.

**Testing kernels without a GPU.** The per-lane math (online softmax, Welford merge,
attention partial merge) lives in small headers that compile for host and device.
CPU tests run that math through an emulated 32-lane xor-shuffle butterfly and the same
split / warp / merge structure as the kernels, and compare against the reference. That
verifies the algorithm and reduction order. It cannot verify indexing, synchronization or
races in the CUDA code itself, which is what `make cuda-test` is for.

## How to run

```bash
make test                 # 23 C++ tests (4 binaries) + 10 Python tests; CPU only
make bench                # CPU GEMM ladder, GFLOP/s
make microbench THREADS=4 # measured FMA peak and memory bandwidth
make roofline-host HOST_KEY=host-m1 HOST_NAME="Apple M1, 4 P-cores (measured)"
make report               # roofline bounds for the decode-path shapes on every device
make cuda-syntax          # clang type-check of all .cu files, no CUDA toolkit needed
```

Requirements: a C++17 compiler and Python 3 (standard library only).

### Running on a GPU

Needs an NVIDIA GPU and the CUDA toolkit (nvcc, cuBLAS):

```bash
make cuda-test                                   # all GPU correctness checks
make -s cuda-bench > results/gpu_t4.csv          # median of 21 L2-flushed launches
python3 -m roofline.gpu_report results/gpu_t4.csv --device t4 > results/gpu_t4.md
```

## Results

### CPU (measured: Apple M1, macOS 14, Apple clang 15, `-O3 -mcpu=native`)

Measured ceilings (`bench/microbench`):

| | 1 thread | 4 threads (P-cores) | 8 threads (4P + 4E) |
|---|---|---|---|
| FMA peak, GFLOP/s | 102.3 | 377.7 | 501.3 |
| Triad bandwidth, GB/s | 58.1 | 56.9 | 48.7 |

That gives a ridge point of about 6.6 flop/byte at 4 threads. 102 GFLOP/s per core matches
4 FMA pipes x 4 lanes x 2 flops at about 3.2 GHz.

SGEMM ladder, GFLOP/s, square n x n (`make bench`, best of 3-5 runs):

| variant | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| naive | 1.9 | 1.8 | 1.7 | (skipped) |
| ikj (loop interchange) | 26.2 | 26.0 | 25.7 | 23.0 |
| blocked | 31.3 | 27.8 | 20.7 | 18.7 |
| packed 6x16 micro-kernel | 84.1 | 89.0 | 89.9 | 84.5 |
| packed, 4 threads | 155.7 | 289.3 | 323.8 | 313.0 |
| packed, 8 threads | 150.3 | 288.6 | 338.2 | 377.1 |
| Apple Accelerate (reference only) | 1152.1 | 1334.1 | 1112.2 | 898.1 |

- Single-thread packed SGEMM reaches **88% of the measured 1-core FMA peak** at n=1024
  (89.9 of 102.3). With 4 threads it reaches 83% of the 4-thread peak at n=2048 (313.0 of 377.7).
- Cache blocking alone made things *worse* than plain ikj at n >= 1024. The gain came from
  packing plus register blocking, which changes how many FMAs each load feeds.
- Accelerate's best (1334 GFLOP/s) is 2.7x the measured 8-thread NEON FMA peak. That is consistent with
  Apple's separate AMX matrix unit, so it is shown as a reference, not a target.

### GPU: PLACEHOLDER (not yet measured)

| kernel | shape | median us | % of roofline |
|---|---|---|---|
| sgemm regtile | 4096^3 | `<FILL IN>` | `<FILL IN>` |
| sgemm regtile vs cuBLAS | 4096^3 | `<FILL IN>` | `<FILL IN>` |
| softmax warp_online | 4096 x 4096 | `<FILL IN>` | `<FILL IN>` |
| layernorm | 4096 x 768 | `<FILL IN>` | `<FILL IN>` |
| attention split_kv vs one block/head | T=16384, H=12, D=64 | `<FILL IN>` | `<FILL IN>` |

Fill this in from `roofline.gpu_report` output. Bounds the model predicts for comparison
come from `make report`, e.g. batch-1 decode ops on a T4 are memory-bound at about 0.5 flop/byte.

## Limitations

- **The GPU code has not been run.** Until `make cuda-test` passes on real hardware, treat the
  CUDA kernels as unverified. The CPU emulation tests check the math, not the kernels.
- FP32 only. Production LLM inference uses FP16/BF16 with tensor cores (and often int8/FP8),
  which changes every compute ceiling here.
- The SGEMM kernels stop at register tiling: no vectorized `float4` loads, no double
  buffering, no warp tiling or tensor cores. Expect them to fall well short of cuBLAS.
- Decode attention supports head dim up to 128, with no paging (the KV cache is contiguous) and no masking.
- Roofline bytes are compulsory traffic only. GPU peaks are datasheet values, not measured.
- The CPU benchmark is single-machine, best-of-N wall time, no frequency pinning.

## What I learned

- A microbenchmark is only as good as its disassembly. My first FMA-peak loop measured
  65 GFLOP/s because AArch64 `fmla` overwrites its addend and the compiler inserted moves.
  The second "fixed" version reported 102 GFLOP/s while actually running adds, because the
  compiler hoisted the loop-invariant multiply. Only the intrinsic version, checked
  instruction by instruction, can be trusted.
- In a warp shuffle reduction, lanes do not end with bit-identical sums. Each lane adds the
  32 partials in a different grouping. The max is exact; the sum differs in the last bits.
- Static checks pay off even without hardware. `clang -fsyntax-only` in CUDA mode caught a
  host/device macro that only worked under nvcc (it tested `__CUDACC__` alone), plus a
  dependence on a device `min()` that only the real CUDA headers provide.

## License

MIT. See `LICENSE`.
