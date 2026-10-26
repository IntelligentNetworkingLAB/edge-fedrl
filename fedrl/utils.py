import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RunningNorm:
    """Running mean/std normalizer (Welford)."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-6):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.ones(dim, dtype=np.float64)
        self.count = eps
        self.clip = clip
        self.eps = eps

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x)
        n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.m2 += batch_var * n + (delta ** 2) * self.count * n / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.m2 / self.count) + self.eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)
