"""Mechanism ablation for per_prng (the pivot headline method).

All branches reuse the Gate 1 checkpoints. Families:

  refresh   how often is the multiplicative noise re-drawn?
              static (once), 1k, 5k steps   [10k = the existing per_prng run]
  lam       noise magnitude: 0.5, 8.0       [2.0 = the existing per_prng run]
  reset     per_reset: forget stale TD priorities once at the change, no noise
  stationary  no-op change control: does per_prng help WITHOUT a shift?
              (local_per vs per_prng, change_scale kept at 1.0)

  python scripts/ablation.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jobs_for(seed):
    j = []
    # noise-refresh frequency (10k == existing gate1/branch per_prng)
    for tag, refresh in (("static", -1), ("refresh1k", 1000), ("refresh5k", 5000)):
        j.append((f"ablation/{tag}", "per_prng",
                  ["--prng-refresh", str(refresh)]))
    # noise magnitude (2.0 == existing)
    for lam in (0.5, 8.0):
        j.append((f"ablation/lam{lam}", "per_prng", ["--lam", str(lam)]))
    # forget-stale-priorities baseline
    j.append(("ablation/reset", "per_reset", []))
    # stationary control: same branch, but the change is a no-op
    for m in ("local_per", "per_prng"):
        j.append(("ablation/stationary", m, ["--no-change"]))
    return [(sub, m, extra + ["--seed", str(seed)]) for sub, m, extra in j]


def run_job(sub, method, extra, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-sub", sub, *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = sub.replace("/", "_") + "_" + "_".join(a.lstrip("-") for a in extra) + "_" + method
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
    print(f"{len(todo)} ablation branches (skipping already-done)")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(
            lambda t: run_job(t[0], t[1], t[2], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
