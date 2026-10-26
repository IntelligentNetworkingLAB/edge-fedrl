"""Aggregate metrics.json files across runs into a comparison table.

  python scripts/aggregate.py results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="results")
    a = p.parse_args()
    groups = defaultdict(list)
    for mpath in sorted(Path(a.root).rglob("metrics.json")):
        m = json.loads(mpath.read_text())
        key = m["method"]
        # noise level is encoded in the path (results/ns0.1/private_proto_per_seed0)
        for part in mpath.parts:
            if part.startswith("ns"):
                key = f"{m['method']}({part})"
        groups[key].append(m)

    cols = ["pre_change_return", "post_change_auc", "recovery_step_T80",
            "post_change_failures_20k", "C3_final_return", "auroc_C3_post",
            "comm_bytes"]
    header = f"{'method':<28}" + "".join(f"{c:>26}" for c in cols)
    print(header)
    print("-" * len(header))
    for key, ms in sorted(groups.items()):
        line = f"{key + f' (n={len(ms)})':<28}"
        for c in cols:
            vals = [m[c] for m in ms if m.get(c) is not None]
            if not vals:
                line += f"{'-':>26}"
            else:
                mean, sd = np.mean(vals), np.std(vals)
                line += f"{mean:>17.1f} +-{sd:>6.1f}"
        print(line)


if __name__ == "__main__":
    main()
