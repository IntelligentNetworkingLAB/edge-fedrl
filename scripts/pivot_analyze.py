"""Pivot verdict: decompose the replay-priority effect on adaptation.

All branches start from the identical pre-change checkpoint, so comparisons are
paired per seed. The ladder of interest:

  local      uniform replay (no prioritization)
  local_per  greedy PER (the thing that may hurt under shift)
  per_soft   de-greedified PER (alpha 0.3)
  per_eps    PER + 50% uniform floor
  per_prng   PER x iid multiplicative noise (pure-local 'noise')
  noisy_per  prototype-shaped permuted noise (federated 'noise', content-free)

Question: is the gain 'noise', or just 'less greedy PER'? If per_soft/per_eps
(no noise, just softer) match per_prng/noisy_per, it's de-greedification.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gate1" / "branch"
LADDER = ["local", "local_per", "per_soft", "per_eps", "per_prng", "noisy_per"]
SEEDS = [0, 1, 2, 3, 4]


def col(metric):
    out = {}
    for m in LADDER:
        vals = []
        for s in SEEDS:
            p = ROOT / f"{m}_seed{s}" / "metrics.json"
            vals.append(json.loads(p.read_text())[metric] if p.exists() else np.nan)
        out[m] = np.array(vals, float)
    return out


def show(name, data, fmt="{:8.1f}", lower_better=False):
    print(f"\n### {name}  ({'lower' if lower_better else 'higher'} better)")
    print(f"{'method':<12}" + "".join(f"s{s}".rjust(9) for s in SEEDS) +
          "     mean      sd")
    for m in LADDER:
        v = data[m]
        print(f"{m:<12}" + "".join(fmt.format(x).rjust(9) for x in v) +
              f"{np.nanmean(v):9.1f}{np.nanstd(v):8.1f}")


def paired_pct(a, b, data, lower_better=False):
    da, db = data[a], data[b]
    diff = (db - da) if lower_better else (da - db)
    wins = int(np.nansum(diff > 0))
    n = int(np.sum(~np.isnan(diff)))
    pct = 100 * np.nanmean(diff) / np.abs(np.nanmean(db))
    return wins, n, pct


def main():
    fails = col("post_change_failures_20k")
    auc = col("post_change_auc")
    final = col("C3_final_return")

    show("Early failures (first 20k post-change)", fails, "{:8.0f}", lower_better=True)
    show("Post-change AUC (50k window)", auc)
    show("C3 final return", final)

    print("\n" + "=" * 66)
    print("Is greedy PER the problem? (paired vs local_per, higher AUC = better)")
    print("=" * 66)
    for m in ["local", "per_soft", "per_eps", "per_prng", "noisy_per"]:
        w, n, pct = paired_pct(m, "local_per", auc)
        tag = "  <- noise" if m in ("per_prng", "noisy_per") else "  <- softer, no noise"
        print(f"  {m:<10} vs local_per: {w}/{n} seeds better, AUC {pct:+5.1f}%{tag}")

    print("\nReading: if the 'softer, no noise' rows match the 'noise' rows,")
    print("the effect is de-greedification of PER under shift, not noise per se.")


if __name__ == "__main__":
    main()
