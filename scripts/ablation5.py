"""Stage 5a (staleness manipulation) + 5b (adaptive gating).

5a — causal test of 'staleness level predicts the benefit' on Hopper:
     truncate the replay to the most recent 30k at branch start (fresh buffer).
     Prediction: local_per reliability rises, and the reset/prng advantage
     shrinks vs the full-buffer branches (gate1/branch).
       ablation5/fresh30k/{local_per, per_prng, per_reset_periodic}

5b — per_adaptive (re-anchor only when measured reliability < tau=0.8):
     Hopper branches (expect gate ON -> ~reset_periodic) and Walker branches
     (expect gate OFF -> ~local_per, rescuing per_prng's harm).
       gate1/ablation5/adaptive/  and  gate1_walker/adaptive/

  python scripts/ablation5.py --seeds 0 1 2 3 4 --workers 2
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
    for m in ("local_per", "per_prng", "per_reset_periodic"):
        j.append(("gate1", "ablation5/fresh30k", m, ["--keep-recent", "30000"], "Hopper-v5"))
    j.append(("gate1", "ablation5/adaptive", "per_adaptive", [], "Hopper-v5"))
    j.append(("gate1_walker", "adaptive", "per_adaptive", [], "Walker2d-v5"))
    return [(root, sub, m, extra + ["--seed", str(seed)], env)
            for root, sub, m, extra, env in j]


def run_job(root, sub, method, extra, env_id, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-root", root,
            "--out-sub", sub, "--env-id", env_id, *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{root}_{sub}".replace("/", "_") + "_" + method + "_" + "_".join(
        a.lstrip("-") for a in extra)
    with open(log_dir / f"log_{name}.txt", "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate1_run.py"),
                            *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env, creationflags=flags)
    print(("OK  " if r.returncode == 0 else "FAIL") + f" {root}/{sub} {method} {extra}",
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
        for root, sub, m, extra, env_id in jobs_for(s):
            if not (ROOT / root / sub / f"{m}_seed{s}" / "metrics.json").exists():
                todo.append((root, sub, m, extra, env_id))
    print(f"{len(todo)} stage-5 branches (skipping already-done)")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda t: run_job(
            t[0], t[1], t[2], t[3], t[4], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
