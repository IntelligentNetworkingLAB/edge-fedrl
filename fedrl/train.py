import csv
import json
import os
from dataclasses import dataclass, field, asdict

import gymnasium as gym
import numpy as np
import torch

from .per import Replay
from .projection import SharedProjector
from .prototypes import PrototypePool, extract_prototypes
from .sac import SACAgent, fedavg_sync
from .utils import RunningNorm, set_seed
from .wrappers import ScaledActionEnv

METHODS = ("local", "local_per", "fedavg", "proto_per", "private_proto_per",
           "proto_random", "proto_shuffled", "noisy_per",
           # pivot: local replay-priority regularizers (no sharing)
           "per_soft", "per_eps", "per_prng",
           # ablation: reset stale TD priorities once at branch start
           "per_reset",
           # mechanism separation: freshness vs perturbation vs epoch-holding
           "per_reset_periodic",   # recompute ALL priorities to current |TD| each epoch
           "per_reset_prng",       # recompute + redraw noise each epoch
           "per_batch_prng",       # noise redrawn every batch (no epoch holding)
           # staleness-gated: re-anchor ONLY when measured reliability drops
           "per_adaptive",
           # regime-stratified: force a post-change fraction in every batch
           "per_stratified",
           # same, but over a sliding recency window (repeated regimes)
           "per_recent",
           # recency baselines from the literature (age-only, detector-free)
           "per_ere", "per_ere_uniform", "per_tgr")
# methods whose replay priority uses a (possibly control) prototype bonus
PROTO_METHODS = ("proto_per", "private_proto_per", "proto_random",
                 "proto_shuffled", "noisy_per")
# local priority-noise regularizers refreshed every share_interval, no prototypes
PRNG_METHODS = ("per_prng", "per_reset_prng")
# methods that recompute all stored priorities to current |TD| every epoch
RESET_PERIODIC_METHODS = ("per_reset_periodic", "per_reset_prng")


@dataclass
class Config:
    method: str = "proto_per"
    seed: int = 0
    env_id: str = "Hopper-v5"
    scales: tuple = (0.70, 0.85, 1.00, 1.15)
    change_client: int = 2          # C3 (0-based)
    change_step: int = 150_000      # per-client env step of actuator damage
    change_scale: float = 0.65
    total_steps: int = 300_000      # per client
    warmup_steps: int = 10_000
    train_every: int = 1
    batch_size: int = 256
    buffer_size: int = 300_000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    share_interval: int = 20_000
    # failure tagging / prototypes
    H: int = 20
    n_prototypes: int = 16
    min_cluster: int = 10
    noise_sigma: float = 0.1        # used only by private_proto_per
    z_dim: int = 8
    z_clip: float = 3.0
    proj_seed: int = 1234
    # priority
    per_alpha: float = 0.6
    per_alpha_soft: float = 0.3   # per_soft: de-greedified PER exponent
    eps_uniform: float = 0.5      # per_eps: fraction of each batch drawn uniformly
    per_eps: float = 1e-3
    # per_prng noise refresh: 0 = every share_interval, N>0 = every N steps,
    # -1 = static (drawn once at run/branch start, never re-randomized)
    prng_refresh: int = 0
    # per_adaptive: re-anchor when pri_reliability < gate_tau (checked each epoch)
    gate_tau: float = 0.8
    # per_stratified / per_recent: forced fraction of every batch drawn from the
    # post-change (or, with recent_window, the recent) partition
    post_quota: float = -1.0   # <0 = off
    recent_window: int = 0     # per_recent: sliding window width (0 = fixed boundary)
    # recency baselines
    ere_eta: float = 0.996     # per_ere: range-shrink rate within an update cycle
    ere_K: int = 1000          # per_ere: updates per cycle
    ere_cmin: int = 5000       # per_ere: smallest sampleable recent range
    tgr_alpha: float = -1.0    # per_tgr: geometric rate; <0 = calibrate to match
    tgr_match_window: int = 50_000   # ... the expected recency of per_recent with
    tgr_match_quota: float = 0.75    # ... this window and quota
    # second regime switch (repeated-regime schedule A -> B -> A)
    change_step2: int = 0      # 0 = no second switch
    change_scale2: float = 1.0
    # staleness manipulation: keep only the most recent K transitions at branch
    # start (0 = keep everything)
    keep_recent: int = 0
    lam: float = 2.0
    sigma_z: float = 1.0
    beta_age: float = 0.5
    # misc
    adapt_window: int = 50_000
    fail_count_window: int = 20_000
    out_dir: str = "results"
    hidden: tuple = (128, 128)
    # gate-1 branching: save full state at the end of a run / resume from one
    checkpoint_save: str = ""
    checkpoint_load: str = ""


def calibrate_tgr_alpha(n, window, quota, tol=1e-9):
    """Pick the truncated-geometric rate whose EXPECTED RECENCY matches the
    recent-window quota sampler, so TGR and the quota differ only in the SHAPE
    of the age distribution (smooth full support vs hard partition), not in how
    recent they are on average. Uses only the sampling distribution -- no
    returns are consulted.

    Target (uniform within each partition, which is the age-marginal of the
    quota sampler): u ~ i/(n-1), recent = last `window` transitions.
    """
    if n <= 1 or window <= 0 or window >= n:
        return 0.001
    frac = window / n
    mu_recent = 1.0 - frac / 2.0          # mean normalized position, recent part
    mu_arch = (1.0 - frac) / 2.0          # ... archive part
    target = quota * mu_recent + (1 - quota) * mu_arch

    a = np.arange(n - 1, -1, -1, dtype=np.float64)   # age; index 0 = oldest
    u = np.arange(n, dtype=np.float64) / (n - 1)

    def mu_of(alpha):
        logp = a * np.log(max(1.0 - alpha, 1e-300))
        logp -= logp.max()
        p = np.exp(logp)
        return float((p / p.sum() * u).sum())

    lo, hi = 1e-7, 0.5                    # mu is increasing in alpha
    if mu_of(hi) < target:
        return hi
    if mu_of(lo) > target:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mu_of(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


class Client:
    def __init__(self, cid, cfg: Config, projector, rng):
        self.cid = cid
        self.cfg = cfg
        self.scale = cfg.scales[cid]
        self.env = ScaledActionEnv(gym.make(cfg.env_id), self.scale)
        self.eval_env = ScaledActionEnv(gym.make(cfg.env_id), self.scale)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.agent = SACAgent(obs_dim, act_dim, hidden=cfg.hidden)
        prioritized = cfg.method not in ("local", "fedavg")
        lam = (cfg.lam if cfg.method in
               PROTO_METHODS + PRNG_METHODS + ("per_batch_prng",) else 0.0)
        alpha = cfg.per_alpha_soft if cfg.method == "per_soft" else cfg.per_alpha
        eps_uniform = cfg.eps_uniform if cfg.method == "per_eps" else 0.0
        quota = (cfg.post_quota if cfg.method in ("per_stratified", "per_recent")
                 else None)
        if quota is not None and quota < 0:
            quota = None
        self.buffer = Replay(cfg.buffer_size, obs_dim, act_dim, cfg.z_dim,
                             prioritized=prioritized, alpha=alpha,
                             eps=cfg.per_eps, lam=lam, eps_uniform=eps_uniform,
                             post_quota=quota, rng=rng)
        self.buffer.batch_noise = cfg.method == "per_batch_prng"
        if cfg.method in ("per_ere", "per_ere_uniform"):
            self.buffer.ere_eta = cfg.ere_eta
            self.buffer.ere_K = cfg.ere_K
            self.buffer.ere_cmin = cfg.ere_cmin
            self.buffer.ere_uniform = cfg.method == "per_ere_uniform"
        if cfg.method == "per_tgr":
            self.buffer.tgr_alpha = cfg.tgr_alpha
        if cfg.method == "per_recent" and cfg.recent_window > 0:
            self.buffer.recent_window = cfg.recent_window
        self.obs_norm = RunningNorm(obs_dim)
        self.x_norm = RunningNorm(2 * obs_dim + act_dim)
        self.projector = projector
        self.pool = PrototypePool(cfg.sigma_z, cfg.beta_age)
        self.rng = rng
        self.obs, _ = self.env.reset(seed=cfg.seed * 100 + cid)
        self.obs_norm.update(self.obs)
        self.ep_indices = []
        self.ep_return = 0.0
        self.ep_len = 0
        self.ep_start_step = 0
        self.episodes = []  # (start_step, end_step, return, length, terminated)

    def env_step(self, step):
        cfg = self.cfg
        if step < cfg.warmup_steps:
            a = self.env.action_space.sample()
        else:
            a = self.agent.select_action(self.obs_norm.normalize(self.obs))
        obs2, r, term, trunc, _ = self.env.step(a)
        z = self.projector.embed(self.obs, a, obs2, self.x_norm)
        idx = self.buffer.add(self.obs, a, r, obs2, float(term), z, step)
        self.ep_indices.append(idx)
        self.ep_return += r
        self.ep_len += 1
        self.obs_norm.update(obs2)
        self.obs = obs2
        if term or trunc:
            if term:  # unhealthy termination -> tag failure window
                self.buffer.mark_failure(np.array(self.ep_indices), cfg.H)
            self.episodes.append((self.ep_start_step, step, self.ep_return,
                                  self.ep_len, int(term)))
            self.obs, _ = self.env.reset()
            self.obs_norm.update(self.obs)
            self.ep_indices = []
            self.ep_return = 0.0
            self.ep_len = 0
            self.ep_start_step = step + 1

    def train_step(self, step=0):
        self.buffer.cur_step = step
        idx, w = self.buffer.sample(self.cfg.batch_size)
        batch = self.buffer.get_batch(idx, self.obs_norm)
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}
        td = self.agent.update(batch, torch.as_tensor(w))
        self.buffer.update_td(idx, td)

    def current_td(self, idx):
        """|TD error| of stored transitions under the CURRENT networks.
        Deterministic (mean action, no entropy term): consumes no RNG, so it can
        run mid-training without perturbing any sampling stream."""
        b = self.buffer
        out = np.empty(len(idx), np.float32)
        with torch.no_grad():
            for lo in range(0, len(idx), 4096):
                ii = idx[lo:lo + 4096]
                obs = torch.as_tensor(
                    self.obs_norm.normalize(b.obs[ii]).astype(np.float32))
                obs2 = torch.as_tensor(
                    self.obs_norm.normalize(b.obs2[ii]).astype(np.float32))
                act = torch.as_tensor(b.act[ii])
                rew = torch.as_tensor(b.rew[ii])
                done = torch.as_tensor(b.done[ii])
                a2 = self.agent.actor.act(obs2, deterministic=True)
                q1t, q2t = self.agent.critic_target(obs2, a2)
                target = rew + self.agent.gamma * (1 - done) * torch.min(q1t, q2t)
                q1, q2 = self.agent.critic(obs, act)
                td = 0.5 * (torch.abs(q1 - target) + torch.abs(q2 - target))
                out[lo:lo + 4096] = td.numpy()
        return out

    def recompute_priorities(self):
        """per_reset_periodic: re-anchor every stored priority to current |TD|."""
        b = self.buffer
        if not b.prioritized or b.size == 0:
            return
        td = self.current_td(np.arange(b.size))
        b.base[:b.size] = (np.abs(td) + b.eps) ** b.alpha
        b.max_base = float(b.base[:b.size].max())
        b.tree.set_many(b.base[:b.size] * b.bonus[:b.size])

    def priority_reliability(self):
        """corr(stored priority, current |TD|) on a strided subsample — the
        online staleness statistic used by the per_adaptive gate."""
        b = self.buffer
        if not b.prioritized or b.size == 0:
            return float("nan")
        sub = np.arange(0, b.size, max(1, b.size // 2048))
        q = b.base[sub] * b.bonus[sub]
        td = self.current_td(sub)
        if np.std(q) == 0 or np.std(td) == 0:
            return float("nan")
        return float(np.corrcoef(q, td)[0, 1])

    def regime_row(self, step):
        """Regime-composition diagnostics: how much of the LEARNING signal comes
        from post-change data, versus how much merely sits in the buffer."""
        b = self.buffer
        if b.size == 0 or b.regime_t is None:
            return None
        n = b.size
        post = b.t_ins[:n] >= b.regime_t
        q = (b.base[:n] * b.bonus[:n]) if b.prioritized else np.ones(n)
        tot = q.sum()
        m_post = float(q[post].sum() / tot) if tot > 0 else float("nan")
        share = float(post.mean())                       # buffer share
        a_eff = float((q / tot * (step - b.t_ins[:n])).sum()) if tot > 0 else float("nan")
        obs_frac = (b.batch_post_n / b.batch_total_n) if b.batch_total_n else float("nan")
        b.batch_post_n = b.batch_total_n = 0
        # current |TD| by regime, on a strided subsample of each partition
        def td_of(mask):
            ii = np.flatnonzero(mask)
            if len(ii) == 0:
                return float("nan")
            ii = ii[::max(1, len(ii) // 512)]
            return float(np.mean(self.current_td(ii)))
        return {"step": step, "client": self.cid, "buffer_share_post": share,
                "m_post": m_post, "obs_post_frac": obs_frac, "a_eff": a_eff,
                "td_pre": td_of(~post), "td_post": td_of(post)}

    def diag_row(self):
        b = self.buffer
        if b.size == 0:
            return None
        # fixed subsample indices so rank-persistence compares like with like
        if not hasattr(self, "_diag_sub"):
            self._diag_sub = np.arange(0, b.size, max(1, b.size // 2048))
        sub = self._diag_sub
        td_sub = self.current_td(sub) if b.prioritized else None
        d = b.diagnostics(current_td_sub=td_sub, sub=sub)
        return {"client": self.cid, **d}

    def evaluate(self):
        rets, lens, fails = [], [], 0
        self.eval_env.scale = self.scale
        for e in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.cfg.seed * 10_000 + e)
            done = trunc = False
            ret = length = 0
            while not (done or trunc):
                a = self.agent.select_action(self.obs_norm.normalize(obs), deterministic=True)
                obs, r, done, trunc, _ = self.eval_env.step(a)
                ret += r
                length += 1
            rets.append(ret)
            lens.append(length)
            fails += int(done)
        return float(np.mean(rets)), float(np.mean(lens)), fails / self.cfg.eval_episodes


def save_checkpoint(path, clients, step, round_id):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"step": step, "round_id": round_id,
            "torch_rng": torch.get_rng_state(), "clients": []}
    for c in clients:
        b = c.buffer
        data["clients"].append(dict(
            actor=c.agent.actor.state_dict(),
            critic=c.agent.critic.state_dict(),
            critic_target=c.agent.critic_target.state_dict(),
            actor_opt=c.agent.actor_opt.state_dict(),
            critic_opt=c.agent.critic_opt.state_dict(),
            alpha_opt=c.agent.alpha_opt.state_dict(),
            log_alpha=c.agent.log_alpha.detach().cpu().clone(),
            buffer={**{k: getattr(b, k)[:b.size].copy() for k in
                       ("obs", "act", "rew", "obs2", "done", "z", "fail", "near",
                        "base", "bonus", "t_ins")},
                    "ptr": b.ptr, "size": b.size, "max_base": b.max_base},
            obs_norm=dict(vars(c.obs_norm)), x_norm=dict(vars(c.x_norm)),
            rng=c.rng, scale=c.scale))
    torch.save(data, path)


def load_checkpoint(path, clients):
    data = torch.load(path, weights_only=False)
    torch.set_rng_state(data["torch_rng"])
    for c, d in zip(clients, data["clients"]):
        c.agent.actor.load_state_dict(d["actor"])
        c.agent.critic.load_state_dict(d["critic"])
        c.agent.critic_target.load_state_dict(d["critic_target"])
        c.agent.actor_opt.load_state_dict(d["actor_opt"])
        c.agent.critic_opt.load_state_dict(d["critic_opt"])
        c.agent.alpha_opt.load_state_dict(d["alpha_opt"])
        c.agent.log_alpha.data.copy_(d["log_alpha"])
        b, s = c.buffer, d["buffer"]["size"]
        for k in ("obs", "act", "rew", "obs2", "done", "z", "fail", "near",
                  "base", "bonus"):
            getattr(b, k)[:s] = d["buffer"][k]
        b.ptr, b.size, b.max_base = d["buffer"]["ptr"], s, d["buffer"]["max_base"]
        if b.prioritized:
            b.tree.set_many(b.base[:s] * b.bonus[:s])
        c.obs_norm.__dict__.update(d["obs_norm"])
        c.x_norm.__dict__.update(d["x_norm"])
        c.rng = d["rng"]
        b.rng = d["rng"]
        c.scale = d["scale"]
        c.env.scale = d["scale"]
        # branch starts on a fresh episode; identical reset seed across variants
        c.obs, _ = c.env.reset(seed=c.cfg.seed * 777 + c.cid)
        c.ep_indices, c.ep_return, c.ep_len = [], 0.0, 0
        c.ep_start_step = data["step"]
    return data["step"], data["round_id"]


def apply_variant(per_client, method, rng):
    """Transform freshly extracted prototypes into their control variants."""
    if method == "proto_random":
        # random centroids with the same count / per-dim mean / variance
        for protos in per_client:
            if not protos:
                continue
            mus = np.stack([p["mu"] for p in protos])
            m, s = mus.mean(0), mus.std(0) + 1e-6
            for p in protos:
                p["mu"] = (m + s * rng.standard_normal(m.shape)).astype(np.float32)
    elif method == "proto_shuffled":
        # real centroids, failure labels shuffled across the client's set
        for protos in per_client:
            ys = [p["y"] for p in protos]
            rng.shuffle(ys)
            for p, y in zip(protos, ys):
                p["y"] = y
    return per_client


def run(cfg: Config):
    assert cfg.method in METHODS, cfg.method
    set_seed(cfg.seed)
    out = os.path.join(cfg.out_dir, f"{cfg.method}_seed{cfg.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    n_clients = len(cfg.scales)
    probe = gym.make(cfg.env_id)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()
    projector = SharedProjector(2 * obs_dim + act_dim, cfg.z_dim, cfg.z_clip, cfg.proj_seed)
    clients = [Client(i, cfg, projector, np.random.default_rng(cfg.seed * 1000 + i))
               for i in range(n_clients)]

    eval_rows, proto_log, auroc_rows, diag_rows, gate_rows = [], [], [], [], []
    regime_rows = []
    comm_bytes = 0
    round_id = 0
    start_step = 0
    if cfg.checkpoint_load:
        start_step, round_id = load_checkpoint(cfg.checkpoint_load, clients)
        for c in clients:
            b = c.buffer
            # checkpoints written before t_ins existed: one insert per env step
            # from step 0, unwrapped, so insertion step == buffer index
            if not b.t_ins[:b.size].any():
                b.t_ins[:b.size] = np.arange(b.size)
        if cfg.keep_recent > 0:
            for c in clients:
                c.buffer.truncate_to_recent(cfg.keep_recent)
        if cfg.method == "per_reset":
            # forget stale TD priorities: every stored transition back to equal max
            for c in clients:
                b = c.buffer
                b.base[:b.size] = 1.0
                b.max_base = 1.0
                b._rebuild_trees()
    if cfg.method == "per_tgr" and cfg.tgr_alpha < 0:
        n0 = max(c.buffer.size for c in clients)
        alpha = calibrate_tgr_alpha(n0, cfg.tgr_match_window, cfg.tgr_match_quota)
        for c in clients:
            c.buffer.tgr_alpha = alpha
        print(f"[per_tgr] calibrated alpha={alpha:.3e} to match expected recency "
              f"of recent-window quota (W={cfg.tgr_match_window}, "
              f"rho={cfg.tgr_match_quota}) at n={n0}", flush=True)
    # the branch resumes exactly at the actuator change: everything already in
    # the buffer is pre-change, everything added from here on is post-change
    for c in clients:
        c.buffer.set_regime_boundary(max(start_step, cfg.change_step))
    if cfg.method in PRNG_METHODS and cfg.prng_refresh == -1:
        for c in clients:
            c.buffer.refresh_prng_bonus()  # static: drawn once, never refreshed
    uses_protos = cfg.method in PROTO_METHODS
    noise = cfg.noise_sigma if cfg.method == "private_proto_per" else 0.0
    prng_every = cfg.prng_refresh if cfg.prng_refresh > 0 else cfg.share_interval

    if cfg.checkpoint_load:
        # A branch resumes exactly at the regime switch, so it never records the
        # pre-change performance the recovery metrics are measured against.
        # Evaluate the restored checkpoint once, before any branch training.
        for c in clients:
            ret, length, frate = c.evaluate()
            eval_rows.append({"step": start_step, "client": c.cid, "return": ret,
                              "ep_len": length, "fail_rate": frate})

    for step in range(start_step, cfg.total_steps):
        cc = clients[cfg.change_client]
        if cfg.change_step2 and step >= cfg.change_step2:
            if cc.scale != cfg.change_scale2:      # regime returns (B -> A)
                cc.scale = cc.env.scale = cfg.change_scale2
        elif cc.scale != cfg.change_scale and step >= cfg.change_step:
            cc.scale = cc.env.scale = cfg.change_scale

        for c in clients:
            c.env_step(step)
            if step >= cfg.warmup_steps and step % cfg.train_every == 0:
                c.train_step(step)

        if (step + 1) % cfg.share_interval == 0:
            round_id += 1
            if uses_protos:
                per_client = []
                for c in clients:
                    protos = extract_prototypes(
                        c.buffer, round_id, k=cfg.n_prototypes,
                        min_cluster=cfg.min_cluster, noise_sigma=noise, rng=c.rng)
                    per_client.append(protos)
                    for p in protos:
                        proto_log.append({"client": c.cid, "round": round_id,
                                          "scale": c.scale, **p})
                    comm_bytes += len(protos) * (cfg.z_dim + 3) * 4
                per_client = apply_variant(per_client, cfg.method,
                                           np.random.default_rng(cfg.seed * 91 + round_id))
                for i, c in enumerate(clients):
                    others = [p for j, ps in enumerate(per_client) if j != i for p in ps]
                    c.pool.set(others)
                    q = c.buffer.refresh_bonus(c.pool, round_id,
                                               permute=(cfg.method == "noisy_per"))
                    # does the applied priority score predict tagged failures?
                    if q is not None:
                        f = c.buffer.fail[:c.buffer.size]
                        if 0 < f.sum() < len(f):
                            from sklearn.metrics import roc_auc_score
                            auroc_rows.append({"step": step + 1, "round": round_id,
                                               "client": c.cid,
                                               "auroc": float(roc_auc_score(f, q))})
            elif cfg.method == "fedavg":
                fedavg_sync([c.agent for c in clients])
                comm_bytes += n_clients * clients[0].agent.param_count() * 4 * 2

        if (step + 1) % prng_every == 0:
            if cfg.method in RESET_PERIODIC_METHODS:
                for c in clients:
                    c.recompute_priorities()   # freshness: re-anchor to current TD
            if cfg.method in PRNG_METHODS and cfg.prng_refresh != -1:
                for c in clients:
                    c.buffer.refresh_prng_bonus()  # perturbation: redraw noise
            if cfg.method == "per_adaptive":
                for c in clients:
                    r = c.priority_reliability()
                    fired = bool(r < cfg.gate_tau) if not np.isnan(r) else False
                    if fired:
                        c.recompute_priorities()
                    gate_rows.append({"step": step + 1, "client": c.cid,
                                      "reliability": r, "fired": int(fired)})

        if (step + 1) % cfg.eval_interval == 0:
            for c in clients:
                c.buffer.update_recent_partition(step + 1)
            for c in clients:
                d = c.diag_row()
                if d:
                    diag_rows.append({"step": step + 1, **d})
                r = c.regime_row(step + 1)
                if r:
                    r.update(c.buffer.sampler_stats(step + 1))
                    regime_rows.append(r)
            for c in clients:
                ret, length, frate = c.evaluate()
                eval_rows.append({"step": step + 1, "client": c.cid, "return": ret,
                                  "ep_len": length, "fail_rate": frate})
            last = [r for r in eval_rows if r["step"] == step + 1]
            print(f"[{cfg.method} s{cfg.seed}] step {step+1}: " +
                  " ".join(f"C{r['client']+1}={r['return']:.0f}" for r in last), flush=True)

    if cfg.checkpoint_save:
        save_checkpoint(cfg.checkpoint_save, clients, cfg.total_steps, round_id)

    # ---- persist logs ----
    if eval_rows:
        with open(os.path.join(out, "eval.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(eval_rows)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["client", "start_step", "end_step", "return", "length", "terminated"])
        for c in clients:
            for row in c.episodes:
                wcsv.writerow([c.cid, *row])
    if proto_log:
        np.savez(os.path.join(out, "prototypes.npz"),
                 mu=np.stack([p["mu"] for p in proto_log]),
                 y=np.array([p["y"] for p in proto_log]),
                 n=np.array([p["n"] for p in proto_log]),
                 v=np.array([p["v"] for p in proto_log]),
                 client=np.array([p["client"] for p in proto_log]),
                 scale=np.array([p["scale"] for p in proto_log]),
                 round=np.array([p["round"] for p in proto_log]))

    if auroc_rows:
        with open(os.path.join(out, "auroc.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(auroc_rows)
    if diag_rows:
        with open(os.path.join(out, "diag.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(diag_rows)
    if regime_rows:
        with open(os.path.join(out, "regime.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(regime_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(regime_rows)
    if gate_rows:
        with open(os.path.join(out, "gating.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(gate_rows)

    metrics = compute_metrics(cfg, eval_rows, clients)
    metrics["comm_bytes"] = comm_bytes
    au = [r["auroc"] for r in auroc_rows
          if r["client"] == cfg.change_client and r["step"] > cfg.change_step]
    metrics["auroc_C3_post"] = float(np.mean(au)) if au else None
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def compute_metrics(cfg, eval_rows, clients):
    cc = cfg.change_client
    rows = [r for r in eval_rows if r["client"] == cc]
    pre = [r["return"] for r in rows
           if cfg.change_step - 5 * cfg.eval_interval <= r["step"] <= cfg.change_step]
    pre_mean = float(np.mean(pre)) if pre else float("nan")
    post = [r for r in rows if cfg.change_step < r["step"] <= cfg.change_step + cfg.adapt_window]
    auc = float(sum(r["return"] for r in post))
    t80 = next((r["step"] for r in post if r["return"] >= 0.8 * pre_mean), None)
    fails = sum(1 for (s, _e, _r, _l, term) in clients[cc].episodes
                if term and cfg.change_step <= s < cfg.change_step + cfg.fail_count_window)
    final = {f"C{c.cid+1}_final_return":
             float(np.mean([r["return"] for r in eval_rows
                            if r["client"] == c.cid][-3:] or [np.nan]))
             for c in clients}
    out = {"method": cfg.method, "seed": cfg.seed,
           "pre_change_return": pre_mean, "post_change_auc": auc,
           "recovery_step_T80": t80, "post_change_failures_20k": fails, **final}
    if cfg.change_step2:
        # repeated regime A -> B -> A: phase B is [change_step, change_step2),
        # the return to A is [change_step2, end). Retention is judged against
        # pre_mean, the agent's own performance in A before it ever left.
        b = [r for r in rows if cfg.change_step < r["step"] <= cfg.change_step2]
        a2 = [r for r in rows if r["step"] > cfg.change_step2]
        w = cfg.change_step2 - cfg.change_step
        out.update({
            "phaseB_auc": float(sum(r["return"] for r in b)),
            "returnA_auc": float(sum(r["return"] for r in a2)),
            "lifetime_auc": float(sum(r["return"] for r in rows
                                      if r["step"] > cfg.change_step)),
            # how much of the original-A performance is back one window later
            "returnA_recovery_ratio": (float(np.mean([r["return"] for r in a2][-3:]))
                                       / pre_mean if a2 and pre_mean else float("nan")),
            "returnA_T80": next((r["step"] - cfg.change_step2 for r in a2
                                 if r["return"] >= 0.8 * pre_mean), None),
            # immediate cost of the switch back, first eval after returning
            "returnA_first_eval": float(a2[0]["return"]) if a2 else float("nan"),
            "phaseB_window": w})
    return out
