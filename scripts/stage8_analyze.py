"""Stage 8 verdict table: is the detector-free quota distinguishable from
known recency replay (ERE, TGR) at matched expected recency?

Reports performance AND sampler geometry side by side, because a performance
difference is only interpretable if the samplers' expected recency is matched
and the remaining difference is shape/entropy.

Rows: PER (baseline) | Oracle quota (upper bound, uses the change point)
      Recent-window quota | ERE-PER | TGR (recency-matched)
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]
CID = 2
CHANGE = 150_000

CONDS = [("Hopper 0.85", "gate1", "stage7/hopper085", "stage8/hopper085"),
         ("Walker 0.65", "gate1_walker", "stage7/walker065", "stage8/walker065")]
# label, subdir kind ("s7" = stage-7 dir, "s8" = stage-8 dir), method, oracle?
ROWS = [("PER (baseline)", "s7", "local_per", False),
        ("Oracle quota", "s7", "per_stratified", True),
        ("Recent-window quota", "s8", "per_recent", False),
        ("ERE-PER (cyc)", "s8", "per_ere", False),
        ("TGR (matched)", "s8", "per_tgr", False)]
# Label notes:
#   "Oracle quota"  uses the true change point -> UPPER BOUND, not a candidate.
#   "ERE-PER (cyc)" is an ONLINE CYCLIC adaptation of ERE. Canonical ERE runs a
#                   block of K updates after collecting an episode, with no new
#                   transitions entering during the block; our loop interleaves
#                   one update per env step, so the shrink cycle runs while the
#                   buffer grows. An ERE loss here is NOT sufficient to claim
#                   superiority over canonical ERE.


def _dir(root, s7, s8, kind, meth, seed):
    return ROOT / root / (s7 if kind == "s7" else s8) / f"{meth}_seed{seed}"


def ev(d, lo, hi):
    p = d / "eval.csv"
    if not p.exists():
        return np.nan
    v = [float(r["return"]) for r in csv.DictReader(p.open())
         if int(r["client"]) == CID and CHANGE + lo < int(r["step"]) <= CHANGE + hi]
    return float(np.mean(v)) if v else np.nan


def sampler(d, key):
    p = d / "regime.csv"
    if not p.exists():
        return np.nan
    v = []
    for r in csv.DictReader(p.open()):
        if int(r["client"]) != CID or key not in r:
            continue
        try:
            v.append(float(r[key]))
        except (ValueError, TypeError):
            pass
    return float(np.mean(v)) if v else np.nan


def sampler_ends(d, key):
    """(first, last) logged value. The buffer grows 150k -> 200k during a branch,
    so samplers matched on expected recency at the start can drift apart."""
    p = d / "regime.csv"
    if not p.exists():
        return (np.nan, np.nan)
    v = []
    for r in csv.DictReader(p.open()):
        if int(r["client"]) != CID or key not in r:
            continue
        try:
            v.append(float(r[key]))
        except (ValueError, TypeError):
            pass
    return (v[0], v[-1]) if v else (np.nan, np.nan)


def series(root, s7, s8, kind, meth, fn):
    return np.array([fn(_dir(root, s7, s8, kind, meth, s)) for s in SEEDS], float)


def main():
    for lab, root, s7, s8 in CONDS:
        print("=" * 104)
        print(f"{lab}")
        print("=" * 104)
        print(f"{'method':<22}{'early AUC':>11}{'full AUC':>11}"
              f"{'vs PER early':>14}{'vs PER full':>13}"
              f"{'recency':>10}{'entropy':>9}{'rec50k':>8}")
        base_e = base_f = None
        for rlab, kind, meth, is_oracle in ROWS:
            e = series(root, s7, s8, kind, meth, lambda d: ev(d, 0, 10_000))
            f = series(root, s7, s8, kind, meth, lambda d: ev(d, 0, 50_000))
            if np.all(np.isnan(e)):
                print(f"{rlab:<22}{'(pending)':>11}")
                continue
            if base_e is None:
                base_e, base_f = e.copy(), f.copy()
            mu = series(root, s7, s8, kind, meth, lambda d: sampler(d, "exp_recency"))
            en = series(root, s7, s8, kind, meth, lambda d: sampler(d, "entropy_norm"))
            rm = series(root, s7, s8, kind, meth, lambda d: sampler(d, "recent50k_mass"))
            de = 100 * np.nanmean(e - base_e) / np.nanmean(base_e)
            df = 100 * np.nanmean(f - base_f) / np.nanmean(base_f)
            we = int(np.nansum((e - base_e) > 0))
            wf = int(np.nansum((f - base_f) > 0))
            tag = " *oracle" if is_oracle else ""
            print(f"{rlab:<22}{np.nanmean(e):11.0f}{np.nanmean(f):11.0f}"
                  f"{de:>+9.1f}% {we}/5{df:>+8.1f}% {wf}/5"
                  f"{np.nanmean(mu):10.3f}{np.nanmean(en):9.3f}"
                  f"{np.nanmean(rm):8.3f}{tag}")
        print("  (recency / entropy columns are TIME-AVERAGED over the branch)")
        hdr, hdr2 = "recency start->end", "entropy start->end"
        print(f"  {'method':<22}{hdr:>22}{hdr2:>24}")
        for rlab, kind, meth, _ in ROWS[2:]:
            f0 = series(root, s7, s8, kind, meth,
                        lambda d: sampler_ends(d, "exp_recency")[0])
            f1 = series(root, s7, s8, kind, meth,
                        lambda d: sampler_ends(d, "exp_recency")[1])
            e0 = series(root, s7, s8, kind, meth,
                        lambda d: sampler_ends(d, "entropy_norm")[0])
            e1 = series(root, s7, s8, kind, meth,
                        lambda d: sampler_ends(d, "entropy_norm")[1])
            if np.all(np.isnan(f0)):
                continue
            print(f"  {rlab:<22}{np.nanmean(f0):10.3f} ->{np.nanmean(f1):8.3f}"
                  f"{np.nanmean(e0):13.3f} ->{np.nanmean(e1):8.3f}")
        print("  matching is valid only if quota and TGR agree at BOTH ends")
        print()

    # ---- head-to-head: quota vs each baseline, paired per seed ----
    print("=" * 104)
    print("HEAD-TO-HEAD (paired per seed): recent-window quota vs each baseline")
    print("=" * 104)
    verdict = {}
    for lab, root, s7, s8 in CONDS:
        q_e = series(root, s7, s8, "s8", "per_recent", lambda d: ev(d, 0, 10_000))
        q_f = series(root, s7, s8, "s8", "per_recent", lambda d: ev(d, 0, 50_000))
        if np.all(np.isnan(q_e)):
            continue
        for rival in ("per_ere", "per_tgr"):
            r_e = series(root, s7, s8, "s8", rival, lambda d: ev(d, 0, 10_000))
            r_f = series(root, s7, s8, "s8", rival, lambda d: ev(d, 0, 50_000))
            if np.all(np.isnan(r_e)):
                continue
            de, df = q_e - r_e, q_f - r_f
            print(f"  {lab:<13} quota vs {rival:<9} "
                  f"early {100*np.nanmean(de)/np.nanmean(r_e):+7.1f}% "
                  f"{int(np.nansum(de>0))}/5   "
                  f"full {100*np.nanmean(df)/np.nanmean(r_f):+7.1f}% "
                  f"{int(np.nansum(df>0))}/5")
            verdict[(lab, rival)] = (np.nanmean(df) > 0, int(np.nansum(df > 0)))

    if len(verdict) == 4:
        strong = all(better and w >= 4 for better, w in verdict.values())
        any_loss = any(not better for better, _ in verdict.values())
        envs = {lab for lab, _ in verdict}
        split = (any(all(verdict[(l, r)][0] for r in ("per_ere", "per_tgr"))
                     for l in envs)
                 and any(not all(verdict[(l, r)][0] for r in ("per_ere", "per_tgr"))
                         for l in envs))
        print("\nVERDICT (judged on the DEPLOYABLE per_recent, never the oracle)")
        if strong:
            print("  C. quota beats BOTH baselines in BOTH environments (>=4/5)")
            print("     -> CONDITIONAL GO. Still required before any claim:")
            print("        - time-averaged recency of quota ~= TGR (table above)")
            print("        - no single outlier seed carrying the mean")
            print("        - final ablation: recent-window UNIFORM vs per_recent")
            print("          (PER inside partitions) and TGR x PER, to separate")
            print("          'hard mass floor' from 'TD priority within partition'")
        elif split:
            print("  D. different winners in different environments")
            print("     -> a single-method paper is not supported; this is a")
            print("        replay-distribution SELECTION problem, and three")
            print("        online diagnostics have already failed to predict it")
        elif any_loss:
            print("  A. a known recency baseline matches or beats the quota")
            print("     -> STOP: the effect is the known recency effect.")
            print("        If ONLY ERE loses, recall ours is online-cyclic ERE and")
            print("        canonical block ERE would still need checking; if TGR")
            print("        ties or wins, that caveat is moot and STOP stands.")
        else:
            print("  quota ahead but under 4/5 seeds somewhere")
            print("     -> no reliable difference; treat as case A (STOP)")


if __name__ == "__main__":
    main()
