# Infrastructure/patchcore/coreset.py
from __future__ import annotations

from typing import Callable, Optional
import numpy as np


def coreset_random(features: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    if ratio >= 1.0:
        return features
    rng = np.random.RandomState(seed)
    n = features.shape[0]
    m = max(1, int(n * ratio))
    idx = rng.choice(n, size=m, replace=False)
    return features[idx]


def coreset_kcenter(
    features: np.ndarray,
    pool_size: int,
    ratio: float,
    seed: int,
    init: str = "farthest_from_mean",
    *,
    should_stop: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = features.shape[0]
    pool_size = min(pool_size, n)

    pool_idx = rng.choice(n, size=pool_size, replace=False)
    pool = features[pool_idx]  # [P, D]
    Pn = pool.shape[0]
    m = max(1, int(Pn * ratio))
    m = min(m, Pn)

    if init == "farthest_from_mean":
        mu = pool.mean(axis=0, keepdims=True)
        dist = np.sum((pool - mu) ** 2, axis=1)
        first = int(np.argmax(dist))
    else:
        first = int(rng.randint(0, Pn))

    centers = [first]
    dists = np.sum((pool - pool[first]) ** 2, axis=1)

    for _ in range(1, m):
        if should_stop is not None and should_stop():
            break
        nxt = int(np.argmax(dists))
        centers.append(nxt)
        new_d = np.sum((pool - pool[nxt]) ** 2, axis=1)
        dists = np.minimum(dists, new_d)

    return pool[np.array(centers, dtype=np.int64)]
