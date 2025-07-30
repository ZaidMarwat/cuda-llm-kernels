# cuda-llm-kernels

Kernels for the decode path of a GPT-style model (GEMM, softmax, LayerNorm,
single-query attention), built up step by step and measured against a roofline model.

Work in progress. So far: CPU reference implementations and their tests.

```bash
make test
```
