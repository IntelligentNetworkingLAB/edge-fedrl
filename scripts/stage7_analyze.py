"""Stage 7 verdict: does the PER starvation gap predict when recency helps?

Design notes enforced here:
  * G is measured ONLY on the local_per branch (never on the rho=0.75 branch),
    so the predictor cannot contain the intervention's own outcome.
  * G is averaged over the EARLY post-change window (10k and 20k reported),
    where starvation should be largest.
  * The effect is split into early (0-10k) and full (0-50k) gains; the
    mechanism predicts corr(G, D_early) > corr(G, D_full).
  * Condition-level n = 4. Correlations over pooled seeds would be
    pseudoreplication, so the primary evidence is RANK AGREEMENT, per-condition
    paired win rates, and seed-level direction consistency. Regressions are
    labelled exploratory and use within-seed centering (seed fixed effects).
  * The confound to kill: G may be a mere proxy for drift severity. Competing
    predictors (|1-c|, return drop, TD ratio, buffer share) are scored the same
    way, and the Hopper-0.65 vs Walker-0.65 pair (identical |1-c|) is the
    discriminating comparison.

  python scripts/stage7_analyze.py [--plot]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]
CID = 2
CHANGE = 150_000

# label, root, sub_local, sub_rho, drift |1-c|, marker
CONDITIONS = [
    ("Hopper 0.85", "gate1", "stage7/hopper085", "stage7/hopper085", 0.15, "o"),
    ("Hopper 0.65", "gate1", "stage6a/diag", "stage6a/rho0.75", 0.35, "s"),
    ("Hopper 0.45", "gate1", "stage7/hopper045", "stage7/hopper045", 0.55, "^"),
    ("Walker 0.65", "gate1_walker", "stage7/walker065", "stage7/walker065", 0.35, "D"),
]


def _rows(root, sub, meth, seed, fname):
    p = ROOT / root / sub / f"{meth}_seed{seed}" / fname
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def eval_window(root, sub, meth, seed, lo, hi):
    """Mean eval return of the changed client in (CHANGE+lo, CHANGE+hi]."""
    v = [float(r["return"]) for r in _rows(root, sub, meth, seed, "eval.csv")
         if int(r["client"]) == CID and CHANGE + lo < int(r["step"]) <= CHANGE + hi]
    return float(np.mean(v)) if v else np.nan


def starvation(root, sub, seed, window):
    """G = mean over the early window of (buffer share of post) - (sampled post).
    Measured on local_per only."""
    b, o = [], []
    for r in _rows(root, sub, "local_per", seed, "regime.csv"):
        if int(r["client"]) != CID or int(r["step"]) > CHANGE + window:
            continue
        try:
            b.append(float(r["buffer_share_post"]))
            o.append(float(r["obs_post_frac"]))
        except ValueError:
            pass
    return float(np.mean(b) - np.mean(o)) if b else np.nan


def td_ratio(root, sub, seed, window):
    v = []
    for r in _rows(root, sub, "local_per", seed, "regime.csv"):
        if int(r["client"]) != CID or int(r["step"]) > CHANGE + window:
            continue
        try:
            tp, ts = float(r["td_pre"]), float(r["td_post"])
            if tp > 0:
                v.append(ts / tp)
        except ValueError:
            pass
    return float(np.mean(v)) if v else np.nan


def return_drop(root, sub, seed):
    """pre-change plateau minus the first post-change eval, on local_per."""
    pre = [float(r["return"]) for r in _rows(root, sub, "local_per", seed, "eval.csv")
           if int(r["client"]) == CID and CHANGE - 20_000 < int(r["step"]) <= CHANGE]
    post = eval_window(root, sub, "local_per", seed, 0, 5_000)
    return float(np.mean(pre)) - post if pre else np.nan


def collect():
    """Per condition, per seed: predictors and paired gains."""
    out = []
    for lab, root, sub_l, sub_r, drift, mk in CONDITIONS:
        rec = {"label": lab, "drift": drift, "marker": mk, "seeds": {}}
        for s in SEEDS:
            lp_e = eval_window(root, sub_l, "local_per", s, 0, 10_000)
            rh_e = eval_window(root, sub_r, "per_stratified", s, 0, 10_000)
            lp_f = eval_window(root, sub_l, "local_per", s, 0, 50_000)
            rh_f = eval_window(root, sub_r, "per_stratified", s, 0, 50_000)
            rec["seeds"][s] = {
                "G10": starvation(root, sub_l, s, 10_000),
                "G20": starvation(root, sub_l, s, 20_000),
                "tdr": td_ratio(root, sub_l, s, 20_000),
                "drop": return_drop(root, sub_l, s),
                "d_early": rh_e - lp_e,
                "d_full": rh_f - lp_f,
                "d_early_pct": 100 * (rh_e - lp_e) / lp_e if lp_e else np.nan,
                "d_full_pct": 100 * (rh_f - lp_f) / lp_f if lp_f else np.nan,
            }
        out.append(rec)
    return out


def col(rec, key):
    return np.array([rec["seeds"][s][key] for s in SEEDS], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    data = collect()
    ready = [r for r in data if not np.all(np.isnan(col(r, "d_early")))]
    if not ready:
        print("no completed conditions yet")
        return

    print("=" * 92)
    print("PER-CONDITION SUMMARY  (G from local_per only; gains paired per seed)")
    print("=" * 92)
    print(f"{'condition':<13}{'|1-c|':>7}{'G@10k':>9}{'G@20k':>9}"
          f"{'D_early%':>11}{'win':>6}{'D_full%':>10}{'win':>6}")
    for r in ready:
        ge, g2 = np.nanmean(col(r, "G10")), np.nanmean(col(r, "G20"))
        de, df = col(r, "d_early"), col(r, "d_full")
        n = int(np.sum(~np.isnan(de)))
        print(f"{r['label']:<13}{r['drift']:>7.2f}{ge:>+9.4f}{g2:>+9.4f}"
              f"{np.nanmean(col(r,'d_early_pct')):>+11.1f}"
              f"{int(np.nansum(de>0)):>4}/{n}"
              f"{np.nanmean(col(r,'d_full_pct')):>+10.1f}"
              f"{int(np.nansum(df>0)):>4}/{n}")

    # ---- primary evidence: rank agreement, not a pooled correlation ----
    print("\n" + "=" * 92)
    print("PRIMARY: rank agreement (condition-level, n=%d)" % len(ready))
    print("=" * 92)
    order_G = [r["label"] for r in sorted(ready, key=lambda r: -np.nanmean(col(r, "G20")))]
    order_D = [r["label"] for r in sorted(ready, key=lambda r: -np.nanmean(col(r, "d_early_pct")))]
    order_c = [r["label"] for r in sorted(ready, key=lambda r: -r["drift"])]
    print(f"  by G (desc)        : {' > '.join(order_G)}")
    print(f"  by D_early (desc)  : {' > '.join(order_D)}")
    print(f"  by |1-c| (desc)    : {' > '.join(order_c)}")
    print(f"  G order == gain order : {order_G == order_D}")
    print(f"  drift order == gain order : {order_c == order_D}")

    hop = [r for r in ready if r["label"].startswith("Hopper")]
    if len(hop) == 3:
        hs = sorted(hop, key=lambda r: -r["drift"])
        gs = [np.nanmean(col(r, "G20")) for r in hs]
        ds = [np.nanmean(col(r, "d_early_pct")) for r in hs]
        print(f"\n  Hopper monotonicity (0.45 > 0.65 > 0.85 expected):")
        print(f"    G      : {gs[0]:+.4f} > {gs[1]:+.4f} > {gs[2]:+.4f}   "
              f"{'YES' if gs[0] > gs[1] > gs[2] else 'NO'}")
        print(f"    D_early: {ds[0]:+.1f}% > {ds[1]:+.1f}% > {ds[2]:+.1f}%   "
              f"{'YES' if ds[0] > ds[1] > ds[2] else 'NO'}")

    # ---- the discriminating pair: same |1-c|, different environment ----
    pair = [r for r in ready if abs(r["drift"] - 0.35) < 1e-9]
    if len(pair) == 2:
        print("\n" + "=" * 92)
        print("CONFOUND TEST: Hopper 0.65 vs Walker 0.65 (identical |1-c| = 0.35)")
        print("=" * 92)
        for r in pair:
            print(f"  {r['label']:<13} G@20k {np.nanmean(col(r,'G20')):+.4f}   "
                  f"D_early {np.nanmean(col(r,'d_early_pct')):+.1f}%   "
                  f"D_full {np.nanmean(col(r,'d_full_pct')):+.1f}%")
        dg = np.nanmean(col(pair[0], "G20")) - np.nanmean(col(pair[1], "G20"))
        dd = (np.nanmean(col(pair[0], "d_early_pct"))
              - np.nanmean(col(pair[1], "d_early_pct")))
        print(f"  drift severity is identical, so any difference is NOT severity.")
        print(f"  same sign (G tracks gain): {'YES' if dg * dd > 0 else 'NO'}")

    # ---- seed-level direction consistency (within condition) ----
    print("\n" + "=" * 92)
    print("SEED-LEVEL: does a seed with larger G get a larger gain? (within cond.)")
    print("=" * 92)
    for r in ready:
        g, d = col(r, "G20"), col(r, "d_early")
        ok = ~(np.isnan(g) | np.isnan(d))
        if ok.sum() >= 4 and np.std(g[ok]) > 1e-9:
            print(f"  {r['label']:<13} corr(G, D_early) = "
                  f"{np.corrcoef(g[ok], d[ok])[0,1]:+.3f}  (n={int(ok.sum())})")
        else:
            print(f"  {r['label']:<13} (insufficient variation)")

    # ---- exploratory: competing predictors, seed-centered ----
    print("\n" + "=" * 92)
    print("EXPLORATORY (n small; seed-centered = seed fixed effects)")
    print("=" * 92)
    rows = []
    for r in ready:
        for s in SEEDS:
            v = r["seeds"][s]
            rows.append((s, r["drift"], v["G20"], v["tdr"], v["drop"],
                         v["d_early"], v["d_full"]))
    A = np.array([x for x in rows if not np.any(np.isnan(np.array(x, float)))],
                 float)
    if len(A) >= 8:
        seed_id, drift, G, tdr, drop, de, df = A.T

        def center(v):                    # remove per-seed mean
            out = v.copy()
            for s in np.unique(seed_id):
                m = seed_id == s
                out[m] = v[m] - v[m].mean()
            return out

        def r2(X, y):
            X = np.column_stack([np.ones(len(y))] + [center(c) for c in X])
            beta, *_ = np.linalg.lstsq(X, center(y), rcond=None)
            pred = X @ beta
            ss = ((center(y) - pred) ** 2).sum()
            tot = ((center(y) - center(y).mean()) ** 2).sum()
            return 1 - ss / tot if tot > 0 else np.nan

        print(f"  target = D_early           R^2")
        print(f"    drift only               {r2([drift], de):.3f}")
        print(f"    G only                   {r2([G], de):.3f}")
        print(f"    drift + G                {r2([drift, G], de):.3f}"
              f"   (does G add over severity?)")
        print(f"    TD ratio only            {r2([tdr], de):.3f}")
        print(f"    return drop only         {r2([drop], de):.3f}")
        print(f"\n  mechanism check: G should predict EARLY better than FULL")
        ce = np.corrcoef(center(G), center(de))[0, 1]
        cf = np.corrcoef(center(G), center(df))[0, 1]
        print(f"    corr(G, D_early) = {ce:+.3f}")
        print(f"    corr(G, D_full)  = {cf:+.3f}   "
              f"{'consistent' if ce > cf else 'INCONSISTENT with mechanism'}")
    else:
        print(f"  (only {len(A)} complete rows; need >=8)")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in ready:
            ax.scatter(col(r, "G20"), col(r, "d_early_pct"), marker=r["marker"],
                       s=70, label=f"{r['label']} (|1-c|={r['drift']:.2f})",
                       alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(r"starvation gap $G$ = buffer share $-$ sampled share"
                      "\n(local_per only, first 20k after the shift)")
        ax.set_ylabel(r"early gain of $\rho=0.75$ vs PER  [%]  (0-10k)")
        ax.set_title("Does PER starvation predict when recency correction helps?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ROOT / "gate1" / "stage7_scatter.png"
        fig.savefig(out, dpi=150)
        print(f"\nscatter written to {out}")


if __name__ == "__main__":
    main()
