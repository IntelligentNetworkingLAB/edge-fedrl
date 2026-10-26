"""Gate 1: is prototype sharing real experience transfer, or just noise?

Phase 'base':   train all 4 clients with local_per for 150k steps (no sharing,
                no actuator change) and save a full checkpoint.
Phase 'branch': load the shared checkpoint, apply the actuator change
                (C3: 1.0 -> 0.65) immediately, train 50k more steps with one
                sharing method. Every branch of a seed starts from the exact
                same networks / replay / normalizers / eval seeds.

Branch methods:
  local_per          local TD priority only
  proto_per          true prototypes from other clients
  private_proto_per  true prototypes + Gaussian noise (sigma 0.1)
  proto_random       random centroids, same count/mean/variance
  proto_shuffled     real centroids, failure labels shuffled
  noisy_per          prototype-shaped priority noise, decorrelated from content

Examples:
  python scripts/gate1_run.py --phase base   --seed 0
  python scripts/gate1_run.py --phase branch --seed 0 --method proto_random
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedrl.train import Config, run

BRANCH_METHODS = ("local_per", "proto_per", "private_proto_per",
                  "proto_random", "proto_shuffled", "noisy_per",
                  "local", "per_soft", "per_eps", "per_prng", "per_reset",
                  "per_reset_periodic", "per_reset_prng", "per_batch_prng",
                  "per_adaptive", "per_stratified", "per_recent",
                  "per_ere", "per_ere_uniform", "per_tgr")
NEVER = 10 ** 9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["base", "branch"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", choices=BRANCH_METHODS, default="proto_per")
    p.add_argument("--base-steps", type=int, default=150_000)
    p.add_argument("--branch-steps", type=int, default=50_000)
    p.add_argument("--out-root", default="gate1")
    p.add_argument("--out-sub", default="branch",
                   help="subdir under out-root for this family of branches")
    p.add_argument("--noise-sigma", type=float, default=0.1)
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--prng-refresh", type=int, default=0,
                   help="per_prng noise refresh: 0=share_interval, N steps, -1=static")
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--alpha-soft", type=float, default=0.3,
                   help="PER exponent used by per_soft (ESS-matching control)")
    p.add_argument("--keep-recent", type=int, default=0,
                   help="staleness manipulation: keep only most recent K at branch")
    p.add_argument("--gate-tau", type=float, default=0.8)
    p.add_argument("--post-quota", type=float, default=-1.0,
                   help="per_stratified: forced post-change fraction per batch")
    p.add_argument("--recent-window", type=int, default=0,
                   help="per_recent: sliding recency window width")
    p.add_argument("--repeat-regime", type=int, default=0,
                   help="steps into the branch at which the regime returns to A")
    p.add_argument("--ere-eta", type=float, default=0.996)
    p.add_argument("--tgr-alpha", type=float, default=-1.0,
                   help="<0 calibrates alpha to match recent-window recency")
    p.add_argument("--tgr-match-window", type=int, default=50_000)
    p.add_argument("--tgr-match-quota", type=float, default=0.75)
    p.add_argument("--change-scale", type=float, default=0.65,
                   help="actuator scale C3 drops to (drift strength)")
    p.add_argument("--no-change", action="store_true",
                   help="stationary control: the 'change' keeps C3 at scale 1.0")
    a = p.parse_args()
    root = Path(a.out_root)
    ck = str(root / "checkpoints" / f"base_seed{a.seed}.pt")

    if a.phase == "base":
        cfg = Config(method="local_per", seed=a.seed, total_steps=a.base_steps,
                     env_id=a.env_id,
                     change_step=NEVER, checkpoint_save=ck,
                     out_dir=str(root / "base_runs"))
    else:
        cfg = Config(method=a.method, seed=a.seed, env_id=a.env_id,
                     total_steps=a.base_steps + a.branch_steps,
                     change_step=a.base_steps, checkpoint_load=ck,
                     eval_interval=5_000, share_interval=10_000,
                     noise_sigma=a.noise_sigma, lam=a.lam,
                     prng_refresh=a.prng_refresh, per_alpha_soft=a.alpha_soft,
                     keep_recent=a.keep_recent, gate_tau=a.gate_tau,
                     post_quota=a.post_quota, recent_window=a.recent_window,
                     ere_eta=a.ere_eta, tgr_alpha=a.tgr_alpha,
                     tgr_match_window=a.tgr_match_window,
                     tgr_match_quota=a.tgr_match_quota,
                     change_step2=(a.base_steps + a.repeat_regime
                                   if a.repeat_regime else 0),
                     change_scale2=1.0,
                     # no-op change: metrics windows stay anchored at base_steps
                     change_scale=1.0 if a.no_change else a.change_scale,
                     adapt_window=a.branch_steps,
                     out_dir=str(root / a.out_sub))
    run(cfg)


if __name__ == "__main__":
    main()
