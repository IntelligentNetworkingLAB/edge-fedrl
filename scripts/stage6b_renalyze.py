"""Stage 6B re-analysis: phase-normalized, per-seed paired scoring.

Raw lifetime AUC summed the two regimes directly, but A is the high-reward
regime (returnA_auc ~3.5x phaseB_auc), so the sum mostly measures A competence
and cannot weigh a B-adaptation gain fairly. Here every phase is normalized by
local_per's value ON THE SAME SEED, then the two phases are weighted equally:

    S_s = 0.5 * AUC_B[s]/AUC_B_local[s] + 0.5 * AUC_A'[s]/AUC_A'_local[s]

Also reports the plasticity/retention split the raw metric hid:
  B  0-10k  adaptation speed right after the shift to B
  A' 0-10k  retention right after the regime returns
  A' first eval, and T90 (steps to reach 90% of local_per's A' plateau).

Everything is paired per seed; win counts are over seeds, not over means.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gate1"
CONDS = [("local_per", "stage6b/local_per", "local_per"),
         ("rho=1.00", "stage6b/rho1.0", "per_recent"),
         ("rho=0.75", "stage6b/rho0.75", "per_recent"),
         ("rho=0.50", "stage6b/rho0.5", "per_recent")]
SEEDS = [0, 1, 2, 3, 4]
B_LO, B_HI = 150_000, 200_000      # phase B
A_LO, A_HI = 200_000, 250_000      # regime A returns
CID = 2


def curve(sub, meth, seed):
    """{step: return} for the changed client."""
    p = ROOT / sub / f"{meth}_seed{seed}" / "eval.csv"
    out = {}
    for r in csv.DictReader(p.open()):
        if int(r["client"]) == CID:
            out[int(r["step"])] = float(r["return"])
    return out


def window_mean(c, lo, hi):
    v = [c[k] for k in c if lo < k <= hi]
    return float(np.mean(v)) if v else np.nan


def metric_all(key_fn):
    """key_fn(curve) -> scalar; returns (n_cond, n_seed) array."""
    return np.array([[key_fn(curve(sub, m, s)) for s in SEEDS]
                     for _, sub, m in CONDS], float)


def show(title, arr, base_row=0, pct=True):
    print(f"\n### {title}")
    print(f"{'condition':<12}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS) +
          f"{'mean':>10}" + ("     vs local_per" if pct else ""))
    base = arr[base_row]
    for i, (lab, _, _) in enumerate(CONDS):
        v = arr[i]
        line = f"{lab:<12}" + "".join(f"{x:9.0f}" for x in v) + f"{np.nanmean(v):10.0f}"
        if pct and i != base_row:
            d = v - base
            line += f"  {100*np.nanmean(d)/np.nanmean(base):+6.1f}%  {int((d>0).sum())}/5"
        print(line)


def main():
    aucB = metric_all(lambda c: window_mean(c, B_LO, B_HI))
    aucA = metric_all(lambda c: window_mean(c, A_LO, A_HI))
    b10 = metric_all(lambda c: window_mean(c, B_LO, B_LO + 10_000))
    a10 = metric_all(lambda c: window_mean(c, A_LO, A_LO + 10_000))
    afirst = metric_all(lambda c: window_mean(c, A_LO, A_LO + 5_000))

    show("Phase B mean return (plasticity)", aucB)
    show("Phase B first 10k (adaptation speed)", b10)
    show("Phase A' mean return (retention+relearning)", aucA)
    show("Phase A' first 10k (retention)", a10)
    show("Phase A' first eval (immediate retention)", afirst)

    print("\n" + "=" * 74)
    print("PHASE-NORMALIZED SCORE  S = 0.5*(B/B_local) + 0.5*(A'/A'_local), per seed")
    print("=" * 74)
    print(f"{'condition':<12}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS) +
          f"{'mean':>9}{'sd':>8}   wins")
    for i, (lab, _, _) in enumerate(CONDS):
        s = 0.5 * (aucB[i] / aucB[0]) + 0.5 * (aucA[i] / aucA[0])
        wins = int((s > 1.0).sum())
        print(f"{lab:<12}" + "".join(f"{x:9.3f}" for x in s) +
              f"{np.nanmean(s):9.3f}{np.nanstd(s):8.3f}   {wins}/5")

    print("\n(reference) raw lifetime AUC verdict was: no method beats local_per")
    print("Per-seed paired scores above decide whether that verdict was a")
    print("property of the methods or of the reward-scale-weighted metric.")


if __name__ == "__main__":
    main()
