"""Stage 6B: repeated regimes A -> B -> A (plasticity vs retention).

Stage 6A showed that in a SINGLE change, training only on post-change data wins
by a mile (rho=1.00: +93.7% AUC, +195% final return). That is exactly why the
single-change setting cannot carry a claim: "throw the old data away" is an
overwhelming baseline and is well covered by prior work.

The real question is what happens when a previous regime comes back:

    [0, 150k)      A   actuator scale 1.00   (reused base checkpoints)
    [150k, 200k)   B   actuator scale 0.65
    [200k, 250k)   A   actuator scale 1.00   (regime returns)

Recency-only sampling should adapt to B fastest and then have to relearn A from
scratch; a mixture that retains the archive should recover A faster. The claim
lives or dies on lifetime AUC, not on either phase alone.

Methods (all full buffer, nothing deleted; quota is over a SLIDING 50k window):
  local_per     natural PER                       (no forced composition)
  per_recent    rho=1.00   recent-only            (expect: fast B, forgets A)
  per_recent    rho=0.75
  per_recent    rho=0.50   balanced mixture       (expect: best lifetime)

  python scripts/stage6b.py --seeds 0 1 2 3 4 --workers 1
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = (1.00, 0.75, 0.50)
PHASE_B = 50_000          # steps spent in regime B before A returns
BRANCH = 100_000          # 50k in B + 50k back in A
WINDOW = 50_000           # sliding recency window


def jobs_for(seed):
    common = ["--branch-steps", str(BRANCH), "--repeat-regime", str(PHASE_B)]
    j = [("stage6b/local_per", "local_per", [])]
    j += [(f"stage6b/rho{q}", "per_recent",
           ["--post-quota", str(q), "--recent-window", str(WINDOW)])
          for q in QUOTAS]
    return [(sub, m, extra + common + ["--seed", str(seed)]) for sub, m, extra in j]


def run_job(sub, method, extra, threads, log_dir):
    args = ["--phase", "branch", "--method", method, "--out-sub", sub, *extra]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    name = sub.replace("/", "_") + "_" + method + "_seed" + extra[-1]
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
    print(f"{len(todo)} stage-6B branches (skipping already-done), "
          f"{a.workers} in parallel")
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(
            lambda t: run_job(t[0], t[1], t[2], a.threads_per_job, logs), todo))
    print(f"done: {codes.count(0)}/{len(codes)} succeeded")


if __name__ == "__main__":
    main()
