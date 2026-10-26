import numpy as np

from .utils import RunningNorm


class SharedProjector:
    """Fixed random projection shared by all clients.

    x = [s, a, s' - s]  (25-dim for Hopper)  ->  z = clip(R @ normalize(x), -C, C)  (8-dim)

    R is drawn once from a seed common to every client. Each client keeps its own
    running normalizer for x (raw stats never leave the client).
    """

    def __init__(self, x_dim, z_dim=8, clip=3.0, shared_seed=1234):
        rng = np.random.default_rng(shared_seed)
        self.R = rng.standard_normal((z_dim, x_dim)) / np.sqrt(x_dim)
        self.clip = clip
        self.z_dim = z_dim

    def make_norm(self, x_dim):
        return RunningNorm(x_dim, clip=10.0)

    def embed(self, s, a, s2, norm: RunningNorm, update_norm=True):
        x = np.concatenate([s, a, s2 - s]).astype(np.float64)
        if update_norm:
            norm.update(x)
        xn = norm.normalize(x)
        return np.clip(self.R @ xn, -self.clip, self.clip).astype(np.float32)
