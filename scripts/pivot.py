"""Pivot experiment: is the replay-priority benefit specifically 'noise', or just
de-greedification of PER under distribution shift?

Reuses the Gate 1 pre-change checkpoints (base_seed*.pt), so no base training.
Each branch loads the identical checkpoint, applies the C3 actuator change, and
trains 50k steps with one local priority scheme (NO sharing):

  local        uniform replay (no prioritization at all)
  local_per    standard greedy PER (alpha=0.6)          [from Gate 1]
  per_soft     de-greedified PER (alpha=0.3)
  per_eps      PER with 50% uniform batch floor
  per_prng     PER x iid multiplicative noise (noisy_per without any prototypes)
  noisy_per    prototype-shaped permuted priority noise  [from Gate 1]

If per_soft / per_eps / per_prng all match noisy_per, the effect is
'greedy PER hurts adaptation'. If only the noise variants help, it is
specifically stochastic priority perturbation.

  python scripts/pivot.py --seeds 0 1 2 3 4 --workers 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIVOT_METHODS = ("local", "local_per", "per_soft", "per_eps", "per_prng", "noisy_per")


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

    missing = [s for s in a.seeds
               if not (root / "checkpoints" / f"base_seed{s}.pt").exists()]
    if missing:
        sys.exit(f"missing base checkpoints for seeds {missing}; run gate1.py first")

    jobs = [["--phase", "branch", "--seed", str(s), "--method", m]
            for s in a.seeds for m in PIVOT_METHODS
            if not (root / "branch" / f"{m}_seed{s}" / "metrics.json").exists()]
    print(f"{len(jobs)} pivot branches (skipping already-done)")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda j: run_job(j, a.threads_per_job, logs), jobs))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
