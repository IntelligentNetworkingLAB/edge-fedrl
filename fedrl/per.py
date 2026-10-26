import numpy as np


def _gini(x):
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * s))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def total(self):
        return self.tree[1]

    def set(self, idx: int, value: float):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def set_many(self, values: np.ndarray):
        """Rebuild leaves [0:len(values)] and all internal nodes. O(capacity)."""
        self.tree[self.capacity:self.capacity + len(values)] = values
        self.tree[self.capacity + len(values):] = 0.0
        for i in range(self.capacity - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def sample(self, value: float) -> int:
        i = 1
        while i < self.capacity:
            left = self.tree[2 * i]
            if value <= left:
                i = 2 * i
            else:
                value -= left
                i = 2 * i + 1
        return i - self.capacity


class Replay:
    """Replay buffer with optional prioritized sampling and a multiplicative
    global-prototype priority bonus.

    priority = base * bonus,  base = (|td| + eps)^alpha,  bonus = 1 + lam * q_g(z).
    Also stores the 8-dim shared embedding z and failure/near-failure flags
    used for prototype extraction.
    """

    def __init__(self, capacity, obs_dim, act_dim, z_dim, prioritized=True,
                 alpha=0.6, eps=1e-3, lam=2.0, beta_is=0.4, eps_uniform=0.0,
                 post_quota=None, rng=None):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha, self.eps, self.lam = alpha, eps, lam
        self.beta_is = beta_is
        self.eps_uniform = eps_uniform  # fraction of each batch drawn uniformly
        # regime-stratified sampling: force this fraction of every batch to come
        # from transitions collected after `regime_t` (None = ordinary sampling)
        self.post_quota = post_quota
        self.regime_t = None
        # if set, "post" means t_ins >= step - recent_window (a sliding recency
        # window) instead of a single fixed regime boundary; needed once regimes
        # repeat, where there is no one boundary that separates old from new
        self.recent_window = None
        self.tree_pre = None
        self.tree_post = None
        self.batch_post_n = 0     # observed post-regime samples since last diag
        self.batch_total_n = 0
        # recency baselines (age-only, no change point): ERE and truncated
        # geometric replay. Left as None unless the method enables them.
        self.ere_eta = None
        self.ere_K = 1000
        self.ere_cmin = 5000
        self.ere_uniform = False
        self.tgr_alpha = None
        self.cur_step = 0
        self.rng = rng or np.random.default_rng()

        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)  # 1 only on true termination
        self.z = np.zeros((capacity, z_dim), np.float32)
        self.fail = np.zeros(capacity, np.float32)   # within H steps of unhealthy termination
        self.near = np.zeros(capacity, np.float32)   # hard negative: [H, 2H) before termination

        self.base = np.ones(capacity, np.float64)
        self.bonus = np.ones(capacity, np.float64)
        self.tree = SumTree(capacity) if prioritized else None
        self.ptr = 0
        self.size = 0
        self.max_base = 1.0
        # per-batch fresh-noise reranking (per_batch_prng): candidates are drawn
        # from the PER tree, then reranked by freshly drawn multiplicative noise
        self.batch_noise = False
        self.batch_noise_pool = 1024
        # sampling-count window for lock-in diagnostics (reset by diagnostics())
        self.n_sampled = np.zeros(capacity, np.int64)
        self.t_ins = np.zeros(capacity, np.int64)   # env step at insertion
        self._diag_prev = {}

    def add(self, obs, act, rew, obs2, done, z, step=0):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.obs2[i], self.done[i], self.z[i] = obs2, done, z
        self.fail[i] = 0.0
        self.near[i] = 0.0
        self.base[i] = self.max_base
        self.bonus[i] = 1.0
        self.t_ins[i] = step
        if self.prioritized:
            self._set_priority(i)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    # ---- regime-stratified sampling -------------------------------------
    def set_regime_boundary(self, step):
        """Split the buffer into pre-regime (t_ins < step) and post-regime
        partitions and, if post_quota is set, build a sum tree per partition so
        each batch can be drawn with a forced post-regime fraction."""
        self.regime_t = step
        if self.post_quota is None or not self.prioritized:
            return
        self.tree_pre = SumTree(self.capacity)
        self.tree_post = SumTree(self.capacity)
        n = self.size
        q = self.base[:n] * self.bonus[:n]
        post = self.t_ins[:n] >= step
        self.tree_pre.set_many(np.where(post, 0.0, q))
        self.tree_post.set_many(np.where(post, q, 0.0))

    def _is_post(self, i):
        return self.regime_t is not None and self.t_ins[i] >= self.regime_t

    def update_recent_partition(self, step):
        """Slide the recency window to `step` and rebuild the partition trees.
        Cheap enough to call once per evaluation interval."""
        if self.recent_window is None or self.tree_post is None:
            return
        self.regime_t = step - self.recent_window
        self._rebuild_trees()

    def _set_priority(self, i):
        """Write priority of index i into whichever tree(s) are active."""
        p = self.base[i] * self.bonus[i]
        if self.tree_post is not None:
            if self._is_post(i):
                self.tree_post.set(int(i), p)
                self.tree_pre.set(int(i), 0.0)
            else:
                self.tree_pre.set(int(i), p)
                self.tree_post.set(int(i), 0.0)
        else:
            self.tree.set(int(i), p)

    def _rebuild_trees(self):
        n = self.size
        q = self.base[:n] * self.bonus[:n]
        if self.tree_post is not None:
            post = self.t_ins[:n] >= self.regime_t
            self.tree_pre.set_many(np.where(post, 0.0, q))
            self.tree_post.set_many(np.where(post, q, 0.0))
        else:
            self.tree.set_many(q)

    def _draw(self, tree, n_take):
        total = tree.total()
        if n_take <= 0 or total <= 0:
            return np.empty(0, np.int64), 0.0
        seg = total / n_take
        vals = (np.arange(n_take) + self.rng.random(n_take)) * seg
        idx = np.array([tree.sample(v) for v in vals], dtype=np.int64)
        return np.clip(idx, 0, self.size - 1), total

    def _sample_stratified(self, batch_size):
        """Force `post_quota` of the batch to come from the post-regime
        partition; sample by priority within each partition."""
        n_post = int(round(self.post_quota * batch_size))
        if self.tree_post.total() <= 0:      # no post data yet
            n_post = 0
        if self.tree_pre.total() <= 0:       # nothing left pre (e.g. quota 1.0)
            n_post = batch_size
        n_pre = batch_size - n_post
        i_post, tot_post = self._draw(self.tree_post, n_post)
        i_pre, tot_pre = self._draw(self.tree_pre, n_pre)
        idx = np.concatenate([i_post, i_pre]) if len(i_post) else i_pre
        # effective sampling prob = (partition share) x (within-partition prob)
        share = np.concatenate([
            np.full(len(i_post), (len(i_post) / batch_size) / max(tot_post, 1e-12)),
            np.full(len(i_pre), (len(i_pre) / batch_size) / max(tot_pre, 1e-12))])
        p_eff = (self.base[idx] * self.bonus[idx]) * share
        w = (self.size * p_eff) ** (-self.beta_is)
        return idx, (w / w.max()).astype(np.float32)

    # ---- recency baselines (detector-free, age-only) ---------------------
    def _recent_range(self, step):
        """ERE: within a cycle of `ere_K` updates, the sampleable range shrinks
        from the whole buffer down to the most recent c_min transitions:

            c_k = max(N * eta^(k * 1000 / K), c_min),   k = step mod K

        Early updates in a cycle see everything, later ones emphasise recent
        data, so a recent-data update is not overwritten by older-data updates
        (Wang & Ross 2019). Our loop does one update per env step rather than a
        per-episode burst, so the cycle position is taken modulo ere_K.
        """
        k = step % self.ere_K
        c = self.size * (self.ere_eta ** (k * 1000.0 / self.ere_K))
        return int(np.clip(c, min(self.ere_cmin, self.size), self.size))

    def _sample_ere(self, batch_size, step):
        """ERE-PER (default) or ERE-uniform: restrict to the most recent c_k
        transitions, then sample inside that range."""
        c = self._recent_range(step)
        lo = self.size - c                      # buffer is age-ordered (unwrapped)
        if self.ere_uniform or not self.prioritized:
            idx = self.rng.integers(lo, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return idx, w
        q = self.base[lo:self.size] * self.bonus[lo:self.size]
        tot = q.sum()
        if tot <= 0:
            idx = self.rng.integers(lo, self.size, batch_size)
            return idx, np.ones(batch_size, np.float32)
        p = q / tot                      # actual sampling prob inside the range
        idx = lo + self.rng.choice(c, batch_size, p=p)
        # ERE+PER corrects with the probability of the RESTRICTED range, i.e. the
        # distribution actually sampled from (c entries, prob p_i each)
        w = (c * p[idx - lo]) ** (-self.beta_is)
        return idx, (np.asarray(w, np.float64) / w.max()).astype(np.float32)

    def _tgr_probs(self):
        """Truncated geometric over age: p(a) ~ (1-alpha)^a on a = 0..N-1,
        a = 0 being the newest transition. Full support (every transition keeps
        positive probability), unlike a hard recency window."""
        n = self.size
        a = np.arange(n - 1, -1, -1, dtype=np.float64)   # index 0 = oldest
        logp = a * np.log(max(1.0 - self.tgr_alpha, 1e-300))
        logp -= logp.max()
        p = np.exp(logp)
        return p / p.sum()

    def _sample_tgr(self, batch_size):
        """Truncated geometric replay. The age-based distribution IS the method,
        so NO importance-sampling correction is applied -- adding PER-style IS
        weights would partially undo the recency bias and make this a different
        algorithm from the one being compared against."""
        p = self._tgr_probs()
        idx = self.rng.choice(self.size, batch_size, p=p)
        return idx, np.ones(batch_size, np.float32)

    def tgr_paper_alpha(self, n_max=None):
        """Our age-decay lambda expressed in the TGR paper's parameterisation
        p(i) ~ 2^(alpha*i/(N_max-1)):  alpha = -(N_max-1) * log2(1-lambda).
        The decay RATE per unit age is fixed (never rescaled by the current
        buffer size); the current size only truncates the support."""
        if self.tgr_alpha is None:
            return None
        n_max = n_max or self.capacity
        return float(-(n_max - 1) * np.log2(max(1.0 - self.tgr_alpha, 1e-300)))

    def sampler_stats(self, step=0):
        """Expected recency and sampling entropy of the CURRENT sampler, so
        different samplers can be compared at matched recency."""
        n = self.size
        if n == 0:
            return {}
        if self.tgr_alpha is not None:
            p = self._tgr_probs()
        elif self.ere_eta is not None:
            c = self._recent_range(step)
            p = np.zeros(n)
            lo = n - c
            if self.ere_uniform or not self.prioritized:
                p[lo:] = 1.0 / c
            else:
                q = self.base[lo:n] * self.bonus[lo:n]
                p[lo:] = q / q.sum() if q.sum() > 0 else 1.0 / c
        elif self.tree_post is not None:
            p = np.zeros(n)
            post = self.t_ins[:n] >= self.regime_t
            q = self.base[:n] * self.bonus[:n]
            qp, qr = q[post].sum(), q[~post].sum()
            if qp > 0:
                p[post] = self.post_quota * q[post] / qp
            if qr > 0:
                p[~post] = (1 - self.post_quota) * q[~post] / qr
            s = p.sum()
            p = p / s if s > 0 else np.full(n, 1.0 / n)
        elif self.prioritized:
            q = self.base[:n] * self.bonus[:n]
            p = q / q.sum() if q.sum() > 0 else np.full(n, 1.0 / n)
        else:
            p = np.full(n, 1.0 / n)
        u = np.arange(n) / max(n - 1, 1)         # 0 = oldest, 1 = newest
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        return {"exp_recency": float((p * u).sum()),
                "entropy_norm": float(ent / np.log(n)) if n > 1 else float("nan"),
                "recent50k_mass": float(p[max(0, n - 50_000):].sum())}

    def mark_failure(self, episode_indices, H):
        """Called on unhealthy termination with the buffer indices of the episode."""
        tail = episode_indices[-H:]
        near = episode_indices[-2 * H:-H]
        self.fail[tail] = 1.0
        self.near[near] = 1.0

    def _record_regime_counts(self, idx):
        if self.regime_t is not None:
            self.batch_post_n += int((self.t_ins[idx] >= self.regime_t).sum())
            self.batch_total_n += len(idx)

    def sample(self, batch_size):
        if not self.prioritized:
            idx = self.rng.integers(0, self.size, batch_size)
            np.add.at(self.n_sampled, idx, 1)
            w = np.ones(batch_size, np.float32)
            self._record_regime_counts(idx)
            return idx, w
        if self.tgr_alpha is not None:
            idx, w = self._sample_tgr(batch_size)
            np.add.at(self.n_sampled, idx, 1)
            self._record_regime_counts(idx)
            return idx, w
        if self.ere_eta is not None:
            idx, w = self._sample_ere(batch_size, self.cur_step)
            np.add.at(self.n_sampled, idx, 1)
            self._record_regime_counts(idx)
            return idx, w
        if self.tree_post is not None:
            idx, w = self._sample_stratified(batch_size)
            np.add.at(self.n_sampled, idx, 1)
            self._record_regime_counts(idx)
            return idx, w
        if self.batch_noise:
            idx, w = self._sample_batch_noise(batch_size)
            self._record_regime_counts(idx)
            return idx, w
        n_uni = int(round(self.eps_uniform * batch_size))
        n_pri = batch_size - n_uni
        total = self.tree.total()
        seg = total / max(n_pri, 1)
        vals = (np.arange(n_pri) + self.rng.random(n_pri)) * seg
        idx = np.array([self.tree.sample(v) for v in vals], dtype=np.int64)
        if n_uni:
            idx = np.concatenate([idx, self.rng.integers(0, self.size, n_uni)])
        idx = np.clip(idx, 0, self.size - 1)
        np.add.at(self.n_sampled, idx, 1)
        # IS weights from the effective sampling prob: mixture of PER and uniform
        pri = self.base[idx] * self.bonus[idx]
        p_per = pri / total
        p_eff = (1 - self.eps_uniform) * p_per + self.eps_uniform / self.size
        w = (self.size * p_eff) ** (-self.beta_is)
        w = (w / w.max()).astype(np.float32)
        self._record_regime_counts(idx)
        return idx, w

    def _sample_batch_noise(self, batch_size):
        """per_batch_prng: draw a candidate pool from the PER tree, rerank with
        noise redrawn THIS batch. For large buffers per-batch noise marginalizes
        out (E[p*xi] ~ p), so this should recover plain PER — included to show
        that sustained (epoch-level) perturbation is what matters."""
        m = min(self.batch_noise_pool, max(batch_size * 2, 1))
        total = self.tree.total()
        seg = total / m
        vals = (np.arange(m) + self.rng.random(m)) * seg
        cand = np.array([self.tree.sample(v) for v in vals], dtype=np.int64)
        cand = np.clip(cand, 0, self.size - 1)
        xi = 1.0 + self.lam * self.rng.random(m)
        pick = self.rng.choice(m, batch_size, replace=False, p=xi / xi.sum())
        idx = cand[pick]
        np.add.at(self.n_sampled, idx, 1)
        pri = self.base[idx] * self.bonus[idx]
        w = (self.size * pri / total) ** (-self.beta_is)  # standard-PER approx
        w = (w / w.max()).astype(np.float32)
        return idx, w

    def update_td(self, idx, td_abs):
        if not self.prioritized:
            return
        self.base[idx] = (np.abs(td_abs) + self.eps) ** self.alpha
        self.max_base = max(self.max_base, self.base[idx].max())
        for i in idx:
            self._set_priority(int(i))

    def refresh_bonus(self, pool, current_round, permute=False):
        """Recompute global-prototype bonus for the whole buffer (vectorized).

        permute=True keeps the same marginal distribution of q but destroys its
        correlation with transition content (Local Noisy PER control).
        Returns the applied q vector (or None if no pool).
        """
        if not self.prioritized:
            return None
        n = self.size
        if pool is None or len(pool.mu) == 0 or n == 0:
            self.bonus[:n] = 1.0
            self._rebuild_trees()
            return None
        q = pool.query(self.z[:n], current_round)
        if permute:
            q = self.rng.permutation(q)
        self.bonus[:n] = 1.0 + self.lam * q
        self._rebuild_trees()
        return q

    def refresh_prng_bonus(self):
        """Purely local multiplicative priority noise: bonus = 1 + lam*U(0,1),
        iid per transition, no prototypes / no sharing. Returns the noise vector.
        This is the noisy_per effect stripped of all federation machinery.
        """
        if not self.prioritized:
            return None
        n = self.size
        u = self.rng.random(n)
        self.bonus[:n] = 1.0 + self.lam * u
        self._rebuild_trees()
        return u

    def truncate_to_recent(self, keep):
        """Keep only the most recent `keep` transitions (staleness manipulation).
        Assumes the buffer has not wrapped (ptr == size), which holds when called
        right after loading a base checkpoint (150k inserts < 300k capacity)."""
        assert self.ptr == self.size, "truncate_to_recent requires an unwrapped buffer"
        if self.size <= keep:
            return
        lo = self.size - keep
        for k in ("obs", "act", "rew", "obs2", "done", "z", "fail", "near",
                  "base", "bonus"):
            arr = getattr(self, k)
            arr[:keep] = arr[lo:self.size]
        self.n_sampled[:keep] = self.n_sampled[lo:self.size]
        self.size = keep
        self.ptr = keep % self.capacity
        if self.prioritized:
            self._rebuild_trees()
        self._diag_prev = {}

    def diagnostics(self, current_td_sub=None, sub=None):
        """Replay lock-in metrics over the window since the previous call.

        ess         normalized effective sample size of the priority distribution
        top1_mass   priority mass held by the top 1% of transitions
        gini        Gini of per-transition sampling counts in the window
        topk_jaccard overlap of the top-1%-most-sampled set vs previous window
        rank_persist Spearman rho of priorities now vs previous window (subsample)
        pri_reliability Pearson corr of stored priority vs current |TD| (subsample,
                    computed by the caller and passed as current_td_sub)
        """
        n = self.size
        if n == 0:
            return {}
        q = self.base[:n] * self.bonus[:n] if self.prioritized else np.ones(n)
        tot = q.sum()
        ess = float((tot ** 2 / (q ** 2).sum()) / n)
        k = max(1, int(0.01 * n))
        top_idx = np.argpartition(q, -k)[-k:]
        top1_mass = float(q[top_idx].sum() / tot)

        counts = self.n_sampled[:n]
        gini = _gini(counts)
        topk = set(np.argpartition(counts, -k)[-k:].tolist())
        prev_topk = self._diag_prev.get("topk")
        jac = (len(topk & prev_topk) / len(topk | prev_topk)
               if prev_topk else float("nan"))

        if sub is None:
            sub = np.arange(0, n, max(1, n // 2048))
        q_sub = q[sub]
        prev = self._diag_prev.get("q_sub")
        rank_persist = (_spearman(prev, q_sub)
                        if prev is not None and len(prev) == len(q_sub)
                        else float("nan"))
        rel = (float(np.corrcoef(q_sub, current_td_sub)[0, 1])
               if current_td_sub is not None and np.std(q_sub) > 0
               and np.std(current_td_sub) > 0 else float("nan"))

        self._diag_prev = {"topk": topk, "q_sub": q_sub.copy()}
        self.n_sampled[:] = 0
        return {"ess": ess, "top1_mass": top1_mass, "gini": gini,
                "topk_jaccard": jac, "rank_persist": rank_persist,
                "pri_reliability": rel}

    def get_batch(self, idx, obs_norm):
        return {
            "obs": obs_norm.normalize(self.obs[idx]).astype(np.float32),
            "act": self.act[idx],
            "rew": self.rew[idx],
            "obs2": obs_norm.normalize(self.obs2[idx]).astype(np.float32),
            "done": self.done[idx],
        }
