"""Stage 6A: does post-change SAMPLING MASS cause the adaptation gain?

The fresh-buffer result confounded three things (buffer size, transition age,
regime composition). This isolates the last one: keep the FULL buffer, change
nothing but the fraction of every mini-batch drawn from post-change data.

  per_stratified rho in {0.25, 0.50, 0.75, 1.00}   full buffer, forced quota
  local_per, per_reset_periodic                    re-run to log regime.csv

The re-runs are needed because the original branches predate the regime
diagnostics (m_post / observed batch fraction / effective replay age / TD by
regime), so the "does reset work by raising M_post?" question cannot be
answered from the existing logs.

Jobs are ordered seed-major so seed 0 finishes across all conditions first.

  python scripts/stage6a.py --seeds 0 1 2 3 4 --workers 1
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = (0.25, 0.50, 0.75, 1.00)


def jobs_for(seed):
    j = [("stage6a/diag", "local_per", []),
         ("stage6a/diag", "per_reset_periodic", [])]
    j += [(f"stage6a/rho{q}", "per_stratified", ["--post-quota", str(q)])
          for q in QUOTAS]
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
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads-per-job", type=int, default=2)
    a = p.parse_args()
    logs = ROOT / "gate1" / "logs"
    todo = []
    for s in a.seeds:
        for sub, m, extra in jobs_for(s):
            if not (ROOT / "gate1" / sub / f"{m}_seed{s}" / "metrics.json").exists():
                todo.append((sub, m, extra))
    print(f"{len(todo)} stage-6A branches (skipping already-done), "
          f"{a.workers} in parallel")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(
            lambda t: run_job(t[0], t[1], t[2], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
