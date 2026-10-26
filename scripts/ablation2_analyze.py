"""Stage-2 verdict: freshness vs perturbation vs epoch-holding + lock-in metrics.

Part A (mech): paired AUC vs local_per for
  per_reset_periodic (freshness), per_reset_prng (freshness+perturbation),
  per_batch_prng (perturbation without epoch-holding), vs per_prng reference.

Part B (diag): post-change lock-in metrics averaged over C3 rows, per method.
Expected if the lock-in story holds:
  local_per: high topk_jaccard / rank_persist, low pri_reliability after change
  per_prng:  higher turnover (low jaccard), same-or-better reliability, better AUC
"""
import csv
import json
from pathlib import Path

import numpy as np

GATE = Path(__file__).resolve().parents[1] / "gate1"
SEEDS = [0, 1, 2, 3, 4]
CHANGE = 150_000


def metric(sub, method, key):
    v = []
    for s in SEEDS:
        p = GATE / sub / f"{method}_seed{s}" / "metrics.json"
        v.append(json.loads(p.read_text())[key] if p.exists() else np.nan)
    return np.array(v, float)


def diag_means(sub, method, post=True):
    """Mean of each diag metric over C3 rows (client==2), post- or pre-change."""
    acc = {}
    for s in SEEDS:
        p = GATE / sub / f"{method}_seed{s}" / "diag.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            if int(r["client"]) != 2:
                continue
            step = float(r["step"])
            if post != (step > CHANGE):
                continue
            for k in ("ess", "top1_mass", "gini", "topk_jaccard",
                      "rank_persist", "pri_reliability"):
                try:
                    val = float(r[k])
                except ValueError:
                    continue
                if not np.isnan(val):
                    acc.setdefault(k, []).append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ref = metric("branch", "local_per", "post_change_auc")
    rows = [
        ("local_per (ref)",   "branch",         "local_per"),
        ("per_prng @10k",     "branch",         "per_prng"),
        ("reset_periodic",    "ablation2/mech", "per_reset_periodic"),
        ("reset+prng",        "ablation2/mech", "per_reset_prng"),
        ("per_batch_prng",    "ablation2/mech", "per_batch_prng"),
    ]
    print("=== Part A: mechanism separation (paired AUC vs local_per) ===")
    print(f"{'variant':<18}{'AUC':>8}{'delta':>9}{'wins':>7}{'fails20k':>10}{'final':>8}")
    for label, sub, m in rows:
        auc = metric(sub, m, "post_change_auc")
        d = auc - ref
        wins = f"{int(np.nansum(d > 0))}/{int(np.sum(~np.isnan(d)))}"
        print(f"{label:<18}{np.nanmean(auc):>8.0f}"
              f"{100 * np.nanmean(d) / np.nanmean(ref):>+8.1f}%{wins:>7}"
              f"{np.nanmean(metric(sub, m, 'post_change_failures_20k')):>10.1f}"
              f"{np.nanmean(metric(sub, m, 'C3_final_return')):>8.0f}")

    print("\n=== Part B: replay lock-in diagnostics (C3, post-change means) ===")
    dm = [("local_per", "ablation2/diag"), ("per_soft", "ablation2/diag"),
          ("per_eps", "ablation2/diag"), ("per_prng", "ablation2/diag"),
          ("per_reset_periodic", "ablation2/mech"),
          ("per_reset_prng", "ablation2/mech"),
          ("per_batch_prng", "ablation2/mech")]
    keys = ("ess", "top1_mass", "gini", "topk_jaccard", "rank_persist",
            "pri_reliability")
    print(f"{'method':<20}" + "".join(k.rjust(13) for k in keys))
    for m, sub in dm:
        d = diag_means(sub, m, post=True)
        print(f"{m:<20}" + "".join(
            (f"{d[k]:13.3f}" if k in d else " " * 13) for k in keys))

    print("\npre-change reference (local_per):")
    d = diag_means("ablation2/diag", "local_per", post=False)
    print(f"{'local_per (pre)':<20}" + "".join(
        (f"{d[k]:13.3f}" if k in d else " " * 13) for k in keys))


if __name__ == "__main__":
    main()
