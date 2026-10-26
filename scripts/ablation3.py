"""Stage 2.5: complete the per_reset_periodic picture (it beat per_prng).

  refresh curve   recompute every 5k / 20k   [10k = existing ablation2/mech run]
  stationary      no-op change control: is freshness also safe/helpful w/o drift?

  python scripts/ablation3.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jobs_for(seed):
    j = [
        ("ablation3/reset5k", "per_reset_periodic", ["--prng-refresh", "5000"]),
        ("ablation3/reset20k", "per_reset_periodic", ["--prng-refresh", "20000"]),
        ("ablation3/stationary", "per_reset_periodic", ["--no-change"]),
    ]
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
    print(("OK  " if r.returncode == 0 else "FAIL") + f" {sub} {extra}", flush=True)
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
    print(f"{len(todo)} stage-2.5 branches (skipping already-done)")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(
            lambda t: run_job(t[0], t[1], t[2], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
