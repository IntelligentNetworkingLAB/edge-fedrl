"""Gate 1 driver: bases first, then all branch variants, in parallel low-priority
processes. Skips anything already completed, so it can be re-run after interruption.

  python scripts/gate1.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH_METHODS = ("local_per", "proto_per", "private_proto_per",
                  "proto_random", "proto_shuffled", "noisy_per")


def run_job(args, threads, log_dir):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(a.lstrip("-") for a in args)
    with open(log_dir / f"log_{name}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate1_run.py"),
                            *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env, creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + " " + " ".join(args), flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads-per-job", type=int, default=3)
    p.add_argument("--out-root", default="gate1")
    a = p.parse_args()
    root = ROOT / a.out_root
    logs = root / "logs"

    bases = [["--phase", "base", "--seed", str(s)] for s in a.seeds
             if not (root / "checkpoints" / f"base_seed{s}.pt").exists()]
    print(f"stage 1: {len(bases)} base runs")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda j: run_job(j, a.threads_per_job, logs), bases))
    if any(codes):
        sys.exit("base runs failed; not starting branches")

    branches = [["--phase", "branch", "--seed", str(s), "--method", m]
                for s in a.seeds for m in BRANCH_METHODS
                if not (root / "branch" / f"{m}_seed{s}" / "metrics.json").exists()]
    print(f"stage 2: {len(branches)} branch runs")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda j: run_job(j, a.threads_per_job, logs), branches))
    print(f"done: {codes.count(0)}/{len(codes)} branches succeeded")


if __name__ == "__main__":
    main()
