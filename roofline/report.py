"""Print roofline bounds for the decode-path kernels on each device.

    python -m roofline.report                      # all devices, GPT-2-small decode shapes
    python -m roofline.report --device host-m1     # one device
"""
from __future__ import annotations

import argparse

from . import model


def table(dev: model.Device, works: list[model.Work]) -> str:
    lines = [f"### {dev.name}: peak {dev.peak_gflops:,.0f} GFLOP/s, {dev.bw_gbs:,.0f} GB/s, "
             f"ridge {dev.ridge:.1f} flop/B",
             "", "| kernel | shape | flop/B | bound | min time (us) | attainable GFLOP/s |",
             "|---|---|---|---|---|---|"]
    for w in works:
        b = model.bound(dev, w)
        lines.append(f"| {w.kernel} | {w.shape} | {b['intensity']:.2f} | {b['bound']} | "
                     f"{b['min_time_s'] * 1e6:,.2f} | {b['attainable_gflops']:,.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    devs = model.load_devices()
    keys = [args.device] if args.device else list(devs)
    works = model.decode_layer_workload(B=args.batch) + [model.gemm(1024, 1024, 1024),
                                                         model.gemm(2048, 2048, 2048)]
    for k in keys:
        print(table(devs[k], works))
        print()
    print("min time = max(flops/peak, compulsory bytes/bandwidth); a lower bound, not a prediction.")


if __name__ == "__main__":
    main()
