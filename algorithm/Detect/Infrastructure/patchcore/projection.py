# domain/patchcore/projection.py
from __future__ import annotations
import numpy as np
from typing import Optional


def make_random_projection(D: int, d: int, seed: int) -> np.ndarray:
    """
    生成随机投影矩阵 P: [D, d]
    """
    rng = np.random.RandomState(seed)
    P = rng.normal(0, 1, size=(D, d)).astype(np.float32)
    P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
    return P


def apply_random_projection(features: np.ndarray, P: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """
    features: [N, D]
    P: [D, d]
    return: [N, d]
    """
    N = features.shape[0]
    d = P.shape[1]
    out = np.empty((N, d), dtype=np.float32)

    if chunk <= 0:
        chunk = N

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        out[s:e] = features[s:e] @ P

    return out
