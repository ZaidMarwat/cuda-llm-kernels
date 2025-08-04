"""Roofline model for the kernels in this repo.

The roofline (Williams, Waterman and Patterson, CACM 2009) bounds a kernel's speed by
the slower of two ceilings:

    time >= max(flops / peak_flops, bytes / bandwidth)

Arithmetic intensity (AI) = flops / bytes. Kernels with AI below the device's ridge point
(peak / bandwidth) are memory bound; above it, compute bound.

`bytes` here is the compulsory DRAM traffic: every input read once and every output
written once. Real kernels move more (cache misses, re-reads), so this is a lower bound
on time and an upper bound on achievable FLOP/s. That is the point: "% of roofline" says
how close a kernel is to the best that this memory system allows.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

F32 = 4  # bytes


@dataclass(frozen=True)
class Device:
    key: str
    name: str
    peak_gflops: float
    bw_gbs: float
    source: str

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (flop/byte) where the two ceilings meet."""
        return self.peak_gflops / self.bw_gbs


@dataclass(frozen=True)
class Work:
    kernel: str
    shape: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes


def gemm(M: int, N: int, K: int) -> Work:
    # C = A @ B: one multiply and one add per (i, j, p). Read A and B, write C.
    return Work("sgemm", f"{M}x{N}x{K}", 2.0 * M * N * K, F32 * (M * K + K * N + M * N))


def softmax(rows: int, cols: int) -> Work:
    # Per element: compare for max, subtract, exp, add to sum, divide. exp is counted as one
    # flop by convention (it is really several), so AI here is a slight underestimate.
    return Work("softmax", f"{rows}x{cols}", 5.0 * rows * cols, F32 * 2 * rows * cols)


def layernorm(rows: int, cols: int) -> Work:
    # Per element: add (mean), sub+mul+add (variance), sub, mul (scale), fma (gamma, beta).
    # Traffic: read x, write y, read gamma and beta once.
    return Work("layernorm", f"{rows}x{cols}", 8.0 * rows * cols,
                F32 * (2 * rows * cols + 2 * cols))


def attention_decode(T: int, H: int, D: int) -> Work:
    # q.k (2 flops/elt) and p*v (2 flops/elt) over T*H*D, plus a softmax over T*H scores.
    # Traffic is dominated by reading the K and V caches: 2*T*H*D floats.
    flops = 4.0 * T * H * D + 5.0 * T * H
    return Work("attention_decode", f"T={T},H={H},D={D}", flops, F32 * (2 * T * H * D + 2 * H * D))


def bound(dev: Device, w: Work) -> dict:
    t_compute = w.flops / (dev.peak_gflops * 1e9)
    t_memory = w.bytes / (dev.bw_gbs * 1e9)
    t = max(t_compute, t_memory)
    return {
        "intensity": w.intensity,
        "bound": "compute" if t_compute >= t_memory else "memory",
        "min_time_s": t,
        "attainable_gflops": w.flops / t / 1e9,
    }


def percent_of_roofline(dev: Device, w: Work, measured_s: float) -> float:
    return 100.0 * bound(dev, w)["min_time_s"] / measured_s


def load_devices(path: str | None = None) -> dict[str, Device]:
    path = path or os.path.join(os.path.dirname(__file__), "devices.json")
    with open(path) as f:
        raw = json.load(f)
    return {k: Device(k, v["name"], float(v["peak_gflops"]), float(v["bw_gbs"]), v["source"])
            for k, v in raw.items() if not k.startswith("_")}


# The decode-step workload of a GPT-2-small-sized model (d=768, 12 heads of 64), batch B.
# These are the shapes that minfer (github.com/ZaidMarwat/minfer) runs per layer.
def decode_layer_workload(B: int = 1, T: int = 1024, d: int = 768, H: int = 12) -> list[Work]:
    return [
        layernorm(B, d),
        gemm(B, 3 * d, d),                 # QKV projection
        attention_decode(T, H, d // H),    # per sequence; times B below in the report
        gemm(B, d, d),                     # attention output projection
        layernorm(B, d),
        gemm(B, 4 * d, d),                 # MLP up
        gemm(B, d, 4 * d),                 # MLP down
    ]
