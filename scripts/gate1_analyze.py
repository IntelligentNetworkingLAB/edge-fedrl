"""Gate 1 verdict: paired (per-seed) analysis, since every branch of a seed
starts from the identical pre-change checkpoint.

Reports, per method:
  - early_fail  : cumulative unhealthy terminations in the first 20k post-change steps
  - auc50       : return AUC over the 50k post-change window
  - final_ret   : mean of the last 3 C3 evaluations
  - auroc       : mean AUROC of applied priority score vs tagged failures (C3, post-change)
And the paired comparisons that decide Go/Stop:
  true/private vs random/shuffled  (early failures, per-seed direction)
  true/private vs noisy_per        (AUC)
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gate1" / "branch"
METHODS = ["local_per", "noisy_per", "proto_random", "proto_shuffled",
           "proto_per", "private_proto_per"]
SEEDS = [0, 1, 2, 3, 4]


def load(method, seed):
    d = ROOT / f"{method}_seed{seed}"
    m = json.loads((d / "metrics.json").read_text())
    auroc = None
    ap = d / "auroc.csv"
    if ap.exists():
        rows = list(csv.DictReader(ap.open()))
        vals = [float(r["auroc"]) for r in rows if int(r["client"]) == 2]
        auroc = float(np.mean(vals)) if vals else None
    return m, auroc


def col(metric):
    out = {}
    for meth in METHODS:
        out[meth] = np.array([load(meth, s)[0][metric] for s in SEEDS], float)
    return out


def show(name, data, fmt="{:8.1f}", lower_better=False):
    print(f"\n### {name}  ({'lower' if lower_better else 'higher'} better)")
    print(f"{'method':<20}" + "".join(f"  s{s}" .rjust(9) for s in SEEDS) +
          "     mean     sd")
    for meth in METHODS:
        v = data[meth]
        print(f"{meth:<20}" + "".join(fmt.format(x).rjust(9) for x in v) +
              f"{v.mean():9.1f}{v.std():8.1f}")


def paired(a, b, data, lower_better=True):
    """Per-seed win count of a over b (a better than b)."""
    da, db = data[a], data[b]
    diff = db - da if lower_better else da - db  # positive = a better
    wins = int((diff > 0).sum())
    pct = 100 * diff.mean() / np.abs(db).mean()
    return wins, pct, diff


def main():
    fails = col("post_change_failures_20k")
    auc = col("post_change_auc")
    final = col("C3_final_return")
    aurocs = {m: np.array([load(m, s)[1] for s in SEEDS
                           if load(m, s)[1] is not None], float) for m in METHODS}

    show("Early failures (first 20k post-change)", fails, "{:8.0f}", lower_better=True)
    show("Post-change AUC (50k window)", auc)
    show("C3 final return", final)

    print("\n### AUROC: does the priority score predict tagged failures? (C3, post-change)")
    for m in METHODS:
        a = aurocs[m]
        if len(a):
            print(f"{m:<20} mean={a.mean():.3f}  (need >0.65 for signal)")
        else:
            print(f"{m:<20} n/a (no shared prototypes)")

    print("\n" + "=" * 70)
    print("GO / STOP CRITERIA")
    print("=" * 70)

    print("\n[1] true/private beats random & shuffled on early failures "
          "(>=10%, >=4/5 seeds)")
    for real in ("proto_per", "private_proto_per"):
        for ctrl in ("proto_random", "proto_shuffled"):
            w, pct, _ = paired(real, ctrl, fails, lower_better=True)
            print(f"    {real:<18} vs {ctrl:<15}: {w}/5 seeds better, "
                  f"mean {pct:+.1f}% fewer failures")

    print("\n[2] post-change AUC >=5% higher than noisy_per (matched priority noise)")
    for real in ("proto_per", "private_proto_per"):
        w, pct, _ = paired(real, "noisy_per", auc, lower_better=False)
        print(f"    {real:<18} vs noisy_per     : {w}/5 seeds better, "
              f"mean AUC {pct:+.1f}%")

    print("\n[3] priority-score AUROC > 0.65 (real failure prediction)")
    for real in ("proto_per", "private_proto_per"):
        a = aurocs[real]
        print(f"    {real:<18}: AUROC {a.mean():.3f}  "
              f"-> {'PASS' if a.mean() > 0.65 else 'FAIL'}")


if __name__ == "__main__":
    main()
