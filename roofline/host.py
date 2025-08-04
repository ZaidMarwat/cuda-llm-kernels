"""Add the host CPU to devices.json from a microbench output file.

    ./build/microbench 4 > results/host_microbench.txt
    python -m roofline.host results/host_microbench.txt host-m1 "Apple M1 (4 P-cores, measured)"

Uses the multithreaded FMA peak and the best of the 1-thread / multi-thread triad
bandwidth, because those are the ceilings a multithreaded kernel can reach.
"""
from __future__ import annotations

import json
import os
import sys


def parse(path: str) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.strip().partition(" ")
            if v:
                out[k] = float(v)
    return out


def main() -> None:
    src, key, name = sys.argv[1], sys.argv[2], sys.argv[3]
    m = parse(src)
    path = os.path.join(os.path.dirname(__file__), "devices.json")
    with open(path) as f:
        devs = json.load(f)
    devs[key] = {"name": name, "peak_gflops": m["peak_gflops_mt"],
                 "bw_gbs": max(m["triad_gbs_1t"], m["triad_gbs_mt"]),
                 "source": f"measured by bench/microbench ({int(m['threads'])} threads), see {src}"}
    with open(path, "w") as f:
        json.dump(devs, f, indent=2)
        f.write("\n")
    print(f"{key}: {devs[key]}")


if __name__ == "__main__":
    main()
