import numpy as np
from sklearn.cluster import KMeans


class PrototypePool:
    """Global pool of prototypes received from other clients.

    Each prototype: (mu, y, n, v) = (centroid, failure ratio, sample count, round).
    q_g(z) = max_k y_k * exp(-||z - mu_k||^2 / (2 sigma_z^2)) * exp(-beta * age_k)
    """

    def __init__(self, sigma_z=1.0, beta_age=0.5):
        self.sigma_z = sigma_z
        self.beta_age = beta_age
        self.mu = np.zeros((0, 0))
        self.y = np.zeros(0)
        self.n = np.zeros(0)
        self.v = np.zeros(0)

    def set(self, protos):
        """protos: list of dicts with keys mu, y, n, v."""
        if not protos:
            self.mu = np.zeros((0, 0))
            return
        self.mu = np.stack([p["mu"] for p in protos])
        self.y = np.array([p["y"] for p in protos])
        self.n = np.array([p["n"] for p in protos])
        self.v = np.array([p["v"] for p in protos])

    def query(self, z, current_round):
        """z: (N, d) -> q_g: (N,) in [0, 1]."""
        if len(self.mu) == 0:
            return np.zeros(len(z))
        d2 = ((z[:, None, :] - self.mu[None, :, :]) ** 2).sum(-1)  # (N, K)
        sim = np.exp(-d2 / (2 * self.sigma_z ** 2))
        age = np.maximum(current_round - self.v, 0.0)
        score = self.y[None, :] * sim * np.exp(-self.beta_age * age)[None, :]
        return score.max(axis=1)


def extract_prototypes(buffer, round_id, k=16, min_cluster=10, max_samples=4000,
                       noise_sigma=0.0, rng=None):
    """Cluster recent embeddings into prototypes with failure ratios.

    Mixes failure-tagged, hard-negative and normal transitions so that clusters
    naturally split into high-y (failure) and low-y (hard negative) prototypes.
    Clusters smaller than `min_cluster` are dropped so no single transition maps
    to a prototype. Gaussian noise on centroids implements the private variant.
    """
    rng = rng or np.random.default_rng()
    n = buffer.size
    if n < 4 * min_cluster:
        return []
    z, fail, near = buffer.z[:n], buffer.fail[:n], buffer.near[:n]

    fail_idx = np.flatnonzero(fail > 0)
    near_idx = np.flatnonzero((near > 0) & (fail == 0))
    norm_idx = np.flatnonzero((fail == 0) & (near == 0))
    if len(fail_idx) < min_cluster:
        return []

    def pick(idx, m):
        if len(idx) <= m:
            return idx
        return rng.choice(idx, m, replace=False)

    third = max_samples // 3
    sel = np.concatenate([pick(fail_idx, third), pick(near_idx, third), pick(norm_idx, third)])
    zs, ys = z[sel], fail[sel]

    k_eff = min(k, len(sel) // min_cluster)
    if k_eff < 2:
        return []
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=int(rng.integers(1 << 30)))
    labels = km.fit_predict(zs)

    protos = []
    for c in range(k_eff):
        mask = labels == c
        cnt = int(mask.sum())
        if cnt < min_cluster:
            continue
        mu = km.cluster_centers_[c].astype(np.float32)
        if noise_sigma > 0:
            mu = mu + rng.normal(0, noise_sigma, mu.shape).astype(np.float32)
        protos.append({"mu": mu, "y": float(ys[mask].mean()), "n": cnt, "v": round_id})
    return protos
