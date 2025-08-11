# GPU results: PLACEHOLDER, not yet measured

No GPU numbers exist yet. The kernels have only been type-checked (`make cuda-syntax`).
To produce real results on a machine with an NVIDIA GPU and the CUDA toolkit:

```bash
make cuda-test                                   # all correctness checks must pass first
make -s cuda-bench > results/gpu_<gpu>.csv       # e.g. gpu_t4.csv
python3 -m roofline.gpu_report results/gpu_<gpu>.csv --device <key> > results/gpu_<gpu>.md
```

`<key>` is the matching entry in `roofline/devices.json` (t4, l4, a100-40gb, rtx4090). Add an entry
if your GPU is not listed, and cite its datasheet. Then copy the table into the README's Results
section and delete this file.
