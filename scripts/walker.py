"""Stage 4: generalization to Walker2d-v5.

Same protocol as the Hopper gate: 4 clients with action scales
{0.70, 0.85, 1.00, 1.15}, shared 150k local_per base checkpoint per seed, then
50k-step branches from the identical checkpoint with C3 damaged 1.0 -> 0.65.

Branch methods: local_per (ref), per_prng, per_reset_periodic.

Bases for missing seeds are trained first, then branches — one driver call.

  python scripts/walker.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "gate1_walker"
METHODS = ("local_per", "per_prng", "per_reset_periodic")


def run_job(args, threads, log_dir, tag):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"log_{tag}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate1_run.py"),
                            "--env-id", "Walker2d-v5", "--out-root", OUT, *args],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env, creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + f" {tag}", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads-per-job", type=int, default=3)
    a = p.parse_args()
    root = ROOT / OUT
    logs = root / "logs"

    bases = [s for s in a.seeds
             if not (root / "checkpoints" / f"base_seed{s}.pt").exists()]
    if bases:
        print(f"{len(bases)} walker bases to train")
        with ThreadPoolExecutor(a.workers) as ex:
            codes = list(ex.map(lambda s: run_job(
                ["--phase", "base", "--seed", str(s)],
                a.threads_per_job, logs, f"base_s{s}"), bases))
        if any(codes):
            sys.exit("base training failed; not launching branches")

    todo = [(m, s) for s in a.seeds for m in METHODS
            if not (root / "branch" / f"{m}_seed{s}" / "metrics.json").exists()]
    print(f"{len(todo)} walker branches")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda t: run_job(
            ["--phase", "branch", "--method", t[0], "--seed", str(t[1])],
            a.threads_per_job, logs, f"branch_{t[0]}_s{t[1]}"), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
