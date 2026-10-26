"""Stage 6A verdict: is the adaptation gain caused by post-change sampling mass?

Two questions:
  Q1  Does forcing a higher post-change batch quota monotonically improve
      post-change AUC (full buffer, nothing else changed)?
  Q2  Does per_reset_periodic work by RAISING the post-change sampling mass?
      (compare its m_post / observed batch fraction against local_per, and
      check whether the seeds where reset helps are the seeds where it lifts
      m_post the most)

Reference points from earlier stages (same paired checkpoints):
  local_per        AUC 5695            fresh-30k buffer  AUC 7656 (+34%)
  per_reset_periodic +15.7%            per_prng          +11.7%
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gate1"
QUOTAS = (0.25, 0.50, 0.75, 1.00)
SEEDS = [0, 1, 2, 3, 4]


def auc(sub, m):
    out = []
    for s in SEEDS:
        p = ROOT / sub / f"{m}_seed{s}" / "metrics.json"
        out.append(json.loads(p.read_text())["post_change_auc"] if p.exists()
                   else np.nan)
    return np.array(out, float)


def regime(sub, m, col, cid=2, last_k=None):
    """Mean of a regime.csv column for the changed client, per seed."""
    out = []
    for s in SEEDS:
        p = ROOT / sub / f"{m}_seed{s}" / "regime.csv"
        if not p.exists():
            out.append(np.nan)
            continue
        v = [float(r[col]) for r in csv.DictReader(p.open())
             if int(r["client"]) == cid and r[col] not in ("", "nan")]
        if last_k:
            v = v[-last_k:]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out, float)


def line(label, v, base=None):
    s = f"{label:<22}" + "".join(f"{x:9.0f}" if not np.isnan(x) else "      nan"
                                 for x in v)
    if base is not None:
        d = v - base
        ok = ~np.isnan(d)
        if ok.any():
            s += f" | {100 * np.nanmean(d) / np.nanmean(base):+6.1f}%  {int((d[ok] > 0).sum())}/{int(ok.sum())}"
    return s


def main():
    lp = auc("branch", "local_per")
    print("=== Q1: forced post-change batch quota (FULL buffer) ===")
    print(f"{'condition':<22}" + "".join(f"       s{s}" for s in SEEDS) +
          " |    vs local_per")
    print(line("local_per (natural)", lp))
    for q in QUOTAS:
        print(line(f"rho={q:.2f}", auc(f"stage6a/rho{q}", "per_stratified"), lp))
    print(line("reset_periodic", auc("ablation2/mech", "per_reset_periodic"), lp))
    print("  [reference] fresh-30k buffer local_per: +34.4%  5/5")

    print("\n=== observed post-change fraction actually achieved ===")
    print(f"{'condition':<22}{'buffer share':>14}{'batch obs':>12}"
          f"{'m_post':>10}{'a_eff':>10}")
    rows = [("local_per", "stage6a/diag", "local_per"),
            ("reset_periodic", "stage6a/diag", "per_reset_periodic")]
    rows += [(f"rho={q:.2f}", f"stage6a/rho{q}", "per_stratified") for q in QUOTAS]
    for label, sub, m in rows:
        share = np.nanmean(regime(sub, m, "buffer_share_post"))
        obs = np.nanmean(regime(sub, m, "obs_post_frac"))
        mp = np.nanmean(regime(sub, m, "m_post"))
        ae = np.nanmean(regime(sub, m, "a_eff"))
        print(f"{label:<22}{share:>14.3f}{obs:>12.3f}{mp:>10.3f}{ae:>10.0f}")

    print("\n=== Q2: does reset help BY raising post-change sampling mass? ===")
    lp_d = auc("stage6a/diag", "local_per")
    rs_d = auc("stage6a/diag", "per_reset_periodic")
    d_auc = rs_d - lp_d
    d_obs = (regime("stage6a/diag", "per_reset_periodic", "obs_post_frac")
             - regime("stage6a/diag", "local_per", "obs_post_frac"))
    print("seed                  " + "".join(f"{s:>9}" for s in SEEDS))
    print("d_AUC (reset-lp)      " + "".join(f"{x:9.0f}" for x in d_auc))
    print("d_obs_post_frac       " + "".join(f"{x:9.3f}" for x in d_obs))
    ok = ~(np.isnan(d_auc) | np.isnan(d_obs))
    if ok.sum() >= 3 and np.std(d_obs[ok]) > 0:
        r = np.corrcoef(d_auc[ok], d_obs[ok])[0, 1]
        print(f"\ncorr(d_AUC, d_post_sampling) = {r:+.3f}  (n={int(ok.sum())})")
        print("  strongly positive -> reset acts through regime composition")
        print("  near zero        -> reset acts through some other channel")
    else:
        print("\n(not enough completed seeds yet for the correlation)")

    print("\n=== TD error by regime (are post-change transitions high-TD?) ===")
    for label, sub, m in rows[:2]:
        tp = np.nanmean(regime(sub, m, "td_pre"))
        ts = np.nanmean(regime(sub, m, "td_post"))
        print(f"{label:<22} td_pre {tp:7.3f}   td_post {ts:7.3f}   "
              f"ratio {ts / tp if tp else float('nan'):.3f}")


if __name__ == "__main__":
    main()
