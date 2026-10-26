"""Run one training configuration.

Examples:
  python scripts/run.py --method proto_per --seed 0
  python scripts/run.py --method private_proto_per --noise-sigma 0.2 --seed 1
  python scripts/run.py --method local_per --total-steps 50000   # quick run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedrl.train import Config, run, METHODS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--change-step", type=int, default=150_000)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--share-interval", type=int, default=20_000)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--noise-sigma", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    out_dir = a.out_dir
    if a.method == "private_proto_per":  # keep noise level visible in the path
        out_dir = str(Path(a.out_dir) / f"ns{a.noise_sigma}")
    cfg = Config(method=a.method, seed=a.seed, total_steps=a.total_steps,
                 change_step=a.change_step, warmup_steps=a.warmup_steps,
                 share_interval=a.share_interval, eval_interval=a.eval_interval,
                 eval_episodes=a.eval_episodes, train_every=a.train_every,
                 noise_sigma=a.noise_sigma, lam=a.lam, out_dir=out_dir)
    run(cfg)


if __name__ == "__main__":
    main()
