"""Evaluate privacy attacks on the prototypes saved by a run.

Example:
  python scripts/attack_eval.py results/proto_per_seed0/prototypes.npz
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedrl.attacks import run_attacks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--train-frac", type=float, default=0.6)
    a = p.parse_args()
    for path in a.npz:
        res = run_attacks(path, a.train_frac)
        print(path)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
