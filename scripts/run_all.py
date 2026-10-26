"""Batch driver: runs the full preliminary-experiment matrix in parallel processes.

Experiment A (utility):      local, local_per, proto_per        x seeds
Experiment B (privacy):      private_proto_per, noise in {0.05, 0.1, 0.2}  x seeds
                             (noise 0 == proto_per from Experiment A)
Experiment C (FL baseline):  fedavg                              x seeds

Each run is a separate process pinned to few torch threads so several runs
fit on one machine. Adjust --workers to your core count.

  python scripts/run_all.py --seeds 0 1 2 --workers 4
  python scripts/run_all.py --seeds 0 --total-steps 100000 --workers 4   # shorter pilot
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_jobs(seeds, total, out_dir):
    jobs = []
    for s in seeds:
        for m in ("local", "local_per", "proto_per", "fedavg"):
            jobs.append(["--method", m, "--seed", str(s)])
        for ns in (0.05, 0.1, 0.2):
            jobs.append(["--method", "private_proto_per", "--noise-sigma", str(ns),
                         "--seed", str(s)])
    common = ["--total-steps", str(total), "--change-step", str(total // 2),
              "--out-dir", out_dir]
    return [j + common for j in jobs]


def job_done(args):
    """A run is complete when its metrics.json exists (skip on restart)."""
    method = args[args.index("--method") + 1]
    seed = args[args.index("--seed") + 1]
    out = Path(args[args.index("--out-dir") + 1])
    if method == "private_proto_per":
        ns = args[args.index("--noise-sigma") + 1]
        out = out / f"ns{float(ns)}"
    return (ROOT / out / f"{method}_seed{seed}" / "metrics.json").exists()


def run_job(args, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    log = Path(args[args.index("--out-dir") + 1])
    log.mkdir(parents=True, exist_ok=True)
    name = "_".join(a.lstrip("-") for a in args[:6])
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    with open(log / f"log_{name}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), *args],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                           creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + " " + " ".join(args), flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads-per-job", type=int, default=3)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    jobs = make_jobs(a.seeds, a.total_steps, a.out_dir)
    skipped = [j for j in jobs if job_done(j)]
    jobs = [j for j in jobs if not job_done(j)]
    print(f"{len(jobs)} runs ({len(skipped)} already done), {a.workers} in parallel")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda j: run_job(j, a.threads_per_job), jobs))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
