"""Turn bench_gpu CSV into a roofline table: achieved GFLOP/s, GB/s and % of roofline.

    python -m roofline.gpu_report results/gpu_t4.csv --device t4 > results/gpu_t4.md

% of roofline = (roofline minimum time) / (measured time). 100% means the kernel moves
only compulsory bytes at full bandwidth (memory-bound) or runs at datasheet peak FLOP/s
(compute-bound). Datasheet peaks are boost-clock numbers, so even cuBLAS usually lands
below 100% on compute-bound GEMMs.
"""
from __future__ import annotations

import argparse
import csv
import sys

from . import model


def work_for(row: dict) -> model.Work:
    k = row["kernel"]
    p = [int(row[f"p{i}"]) for i in range(1, 5)]
    if k == "sgemm":
        return model.gemm(p[0], p[1], p[2])
    if k == "softmax":
        return model.softmax(p[0], p[1])
    if k == "layernorm":
        return model.layernorm(p[0], p[1])
    if k == "attention_decode":
        return model.attention_decode(p[0], p[1], p[2])
    raise ValueError(f"unknown kernel {k!r}")


def report(rows: list[dict], dev: model.Device) -> str:
    out = [f"Device in CSV: {rows[0]['device'] if rows else '?'}; roofline from `{dev.key}` "
           f"({dev.peak_gflops:,.0f} GFLOP/s, {dev.bw_gbs:,.0f} GB/s; {dev.source})", "",
           "| kernel | variant | shape | median us | GFLOP/s | GB/s (compulsory) | bound | % of roofline |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w = work_for(r)
        us = float(r["median_us"])
        b = model.bound(dev, w)
        shape = w.shape + (f", splits={r['p4']}" if r["kernel"] == "attention_decode" else "")
        out.append(f"| {r['kernel']} | {r['variant']} | {shape} | {us:,.1f} | "
                   f"{w.flops / us / 1e3:,.1f} | {w.bytes / us / 1e3:,.1f} | {b['bound']} | "
                   f"{model.percent_of_roofline(dev, w, us * 1e-6):.1f}% |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--device", required=True, help="key in roofline/devices.json")
    args = ap.parse_args(argv)
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    dev = model.load_devices()[args.device]
    sys.stdout.write(report(rows, dev) + "\n")


if __name__ == "__main__":
    main()
