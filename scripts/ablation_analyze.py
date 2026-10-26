"""Stage-1 ablation verdict: refresh frequency, magnitude, reset-once, stationary.

Paired per-seed (identical checkpoints). Reference rows come from gate1/branch
(local_per, per_prng @10k refresh, lam 2.0).
"""
import json
from pathlib import Path

import numpy as np

GATE = Path(__file__).resolve().parents[1] / "gate1"
SEEDS = [0, 1, 2, 3, 4]

ROWS = [
    # (label, dir, method)
    ("local_per",        "branch",             "local_per"),
    ("prng static",      "ablation/static",    "per_prng"),
    ("prng refresh 1k",  "ablation/refresh1k", "per_prng"),
    ("prng refresh 5k",  "ablation/refresh5k", "per_prng"),
    ("prng refresh 10k", "branch",             "per_prng"),
    ("prng lam 0.5",     "ablation/lam0.5",    "per_prng"),
    ("prng lam 2.0",     "branch",             "per_prng"),
    ("prng lam 8.0",     "ablation/lam8.0",    "per_prng"),
    ("reset-once",       "ablation/reset",     "per_reset"),
]
STAT = [
    ("stationary local_per", "ablation/stationary", "local_per"),
    ("stationary per_prng",  "ablation/stationary", "per_prng"),
]


def load(sub, method):
    vals = {}
    for m in ("post_change_auc", "post_change_failures_20k", "C3_final_return"):
        v = []
        for s in SEEDS:
            p = GATE / sub / f"{method}_seed{s}" / "metrics.json"
            v.append(json.loads(p.read_text())[m] if p.exists() else np.nan)
        vals[m] = np.array(v, float)
    return vals


def main():
    ref = load("branch", "local_per")["post_change_auc"]
    print(f"{'variant':<22}{'AUC mean':>10}{'vs local_per':>14}{'wins':>7}"
          f"{'fails20k':>10}{'final':>9}")
    print("-" * 72)
    for label, sub, method in ROWS:
        d = load(sub, method)
        auc = d["post_change_auc"]
        diff = auc - ref
        wins = f"{int(np.nansum(diff > 0))}/{int(np.sum(~np.isnan(diff)))}"
        pct = 100 * np.nanmean(diff) / np.nanmean(ref)
        print(f"{label:<22}{np.nanmean(auc):>10.0f}{pct:>+13.1f}%{wins:>7}"
              f"{np.nanmean(d['post_change_failures_20k']):>10.1f}"
              f"{np.nanmean(d['C3_final_return']):>9.0f}")

    print("\nStationary control (no actual change; same windows):")
    sref = load("ablation/stationary", "local_per")["post_change_auc"]
    for label, sub, method in STAT:
        d = load(sub, method)
        auc = d["post_change_auc"]
        diff = auc - sref
        wins = f"{int(np.nansum(diff > 0))}/{int(np.sum(~np.isnan(diff)))}"
        pct = 100 * np.nanmean(diff) / np.nanmean(sref)
        print(f"{label:<22}{np.nanmean(auc):>10.0f}{pct:>+13.1f}%{wins:>7}"
              f"{np.nanmean(d['post_change_failures_20k']):>10.1f}"
              f"{np.nanmean(d['C3_final_return']):>9.0f}")


if __name__ == "__main__":
    main()
