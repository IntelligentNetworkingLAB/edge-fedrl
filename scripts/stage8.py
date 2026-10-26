"""Stage 8 -- TERMINAL NOVELTY GATE.

Everything that survives this project reduces to one claim: forcing more
current-regime data into the batch speeds adaptation. ERE and Truncated
Geometric Replay already exploit recency, so unless a DETECTOR-FREE quota beats
them at MATCHED expected recency, there is no method contribution here.

Important framing correction: `per_stratified` (stage 6A/7) partitions on the
true change point, so it is an ORACLE upper bound, not a deployable method.
The deployable version is `per_recent`, a sliding recency window that never
sees the change point -- the same information ERE and TGR use.

Methods (all detector-free except the oracle, which is already measured):
  per_recent  rho=0.75, W=50k    deployable version of the quota
  per_ere     eta=0.996          ERE-PER (prioritized inside the recent range)
  per_tgr     alpha calibrated   truncated geometric, expected recency matched
                                 to per_recent -- differs only in SHAPE
                                 (smooth full support vs hard partition)

Conditions: Hopper 1.0->0.85 (largest recency effect) and Walker 1.0->0.65
(second environment). local_per and the oracle quota already exist.

Verdict:
  ERE/TGR >= quota            -> STOP, the effect is known recency
  oracle >> all detector-free -> STOP, the gain was change-point information
  quota > ERE and TGR in both -> conditional GO

  python scripts/stage8.py --seeds 0 1 2 3 4 --workers 1
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = [
    ("hopper085", "gate1", "stage8/hopper085", "Hopper-v5", ["--change-scale", "0.85"]),
    ("walker065", "gate1_walker", "stage8/walker065", "Walker2d-v5", ["--change-scale", "0.65"]),
]
METHODS = [
    ("per_recent", ["--post-quota", "0.75", "--recent-window", "50000"]),
    ("per_ere", ["--ere-eta", "0.996"]),
    ("per_tgr", ["--tgr-alpha", "-1",            # <0 = calibrate
                 "--tgr-match-window", "50000", "--tgr-match-quota", "0.75"]),
]


def run_job(root, sub, env_id, method, extra, seed, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-root", root,
            "--out-sub", sub, "--env-id", env_id, "--seed", str(seed), *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"log_{sub.replace('/', '_')}_{method}_s{seed}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate1_run.py"),
                            *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env, creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + f" {sub} {method} s{seed}",
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
    for s in a.seeds:                      # seed-major
        for _, root, sub, env_id, cextra in CONDITIONS:
            for m, mextra in METHODS:
                if not (ROOT / root / sub / f"{m}_seed{s}" / "metrics.json").exists():
                    todo.append((root, sub, env_id, m, cextra + mextra, s))
    print(f"{len(todo)} stage-8 branches (skipping already-done), "
          f"{a.workers} in parallel")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda t: run_job(*t, a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
