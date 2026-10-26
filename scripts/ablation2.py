"""Ablation stage 2: separate freshness / perturbation / epoch-holding, and
collect replay lock-in diagnostics (diag.csv) for the mechanism figure.

New branches (all from the shared Gate 1 checkpoints):

  per_reset_periodic  recompute ALL priorities to current |TD| every 10k (freshness)
  per_reset_prng      recompute + redraw noise every 10k (freshness + perturbation)
  per_batch_prng      noise redrawn every batch (perturbation without epoch-holding)

Diag reruns (identical training to the pivot runs, now logging diag.csv):

  local_per, per_prng, per_soft, per_eps  -> ablation2/diag/

The decision table (user's design):
  Fixed ~ Periodic            -> one-shot reranking / smoothing effect
  Reset-only ~ Periodic       -> priority freshness is the mechanism
  Periodic > Fixed,Reset-only -> temporal subset diversification is independent
  Per-batch < Periodic        -> holding a subset for a replay epoch is required

  python scripts/ablation2.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jobs_for(seed):
    j = [("ablation2/mech", m, []) for m in
         ("per_reset_periodic", "per_reset_prng", "per_batch_prng")]
    j += [("ablation2/diag", m, []) for m in
          ("local_per", "per_prng", "per_soft", "per_eps")]
    return [(sub, m, extra + ["--seed", str(seed)]) for sub, m, extra in j]


def run_job(sub, method, extra, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-sub", sub, *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = sub.replace("/", "_") + "_" + method + "_" + "_".join(
        a.lstrip("-") for a in extra)
    with open(log_dir / f"log_{name}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate1_run.py"),
                            *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env, creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + f" {sub} {method} {extra}",
          flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads-per-job", type=int, default=3)
    a = p.parse_args()
    logs = ROOT / "gate1" / "logs"

    todo = []
    for s in a.seeds:
        for sub, m, extra in jobs_for(s):
            if not (ROOT / "gate1" / sub / f"{m}_seed{s}" / "metrics.json").exists():
                todo.append((sub, m, extra))
    print(f"{len(todo)} stage-2 branches (skipping already-done)")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(
            lambda t: run_job(t[0], t[1], t[2], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
