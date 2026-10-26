"""Stage 7 — the life-or-death gate: does the PER starvation gap PREDICT when
recency correction helps?

Starvation gap for the changed client, measured on plain local_per:

    G = buffer_share_post - sampled_post_mass       (G > 0 = new regime starved)

Intervention gain, same seeds and checkpoints:

    D_rho = AUC(per_stratified rho=0.75) - AUC(local_per)

The claim survives only if corr(G, D_rho) > 0 ACROSS environments and drift
strengths -- i.e. starvation is what decides whether recency correction is
needed, rather than "recency always helps on Hopper".

Conditions (Hopper 0.65 is already done in gate1/stage6a and is reused):
    Hopper  1.0 -> 0.85   mild drift
    Hopper  1.0 -> 0.45   severe drift
    Walker  1.0 -> 0.65   different environment (reset/prng failed here)

Each condition runs local_per (to measure G) and per_stratified rho=0.75
(to measure D_rho), 5 seeds, from the existing 150k checkpoints.

  python scripts/stage7.py --seeds 0 1 2 3 4 --workers 1
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, out_root, out_sub, env_id, extra args)
CONDITIONS = [
    ("hopper_0.85", "gate1", "stage7/hopper085", "Hopper-v5", ["--change-scale", "0.85"]),
    ("hopper_0.45", "gate1", "stage7/hopper045", "Hopper-v5", ["--change-scale", "0.45"]),
    ("walker_0.65", "gate1_walker", "stage7/walker065", "Walker2d-v5", ["--change-scale", "0.65"]),
]
METHODS = [("local_per", []), ("per_stratified", ["--post-quota", "0.75"])]


def run_job(root, sub, env_id, method, extra, seed, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-root", root,
            "--out-sub", sub, "--env-id", env_id, "--seed", str(seed), *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{sub}_{method}_s{seed}".replace("/", "_")
    with open(log_dir / f"log_{name}.txt", "w") as f:
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
    for s in a.seeds:                      # seed-major: seed 0 completes first
        for _, root, sub, env_id, extra in CONDITIONS:
            for m, mextra in METHODS:
                done = (ROOT / root / sub / f"{m}_seed{s}" / "metrics.json").exists()
                if not done:
                    todo.append((root, sub, env_id, m, extra + mextra, s))
    print(f"{len(todo)} stage-7 branches (skipping already-done), "
          f"{a.workers} in parallel")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda t: run_job(*t, a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
