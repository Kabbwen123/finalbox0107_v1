# from __future__ import annotations
# import numpy as np

# # =========================
# # 工具函数
# # =========================
# def coreset_random(features: np.ndarray, ratio: float, seed: int) -> np.ndarray:
#     ''' 随机挑 ratio '''
#     if ratio >= 1.0:
#         return features
#     rng = np.random.RandomState(seed)
#     n = features.shape[0]
#     m = max(1, int(n * ratio))
#     idx = rng.choice(n, size=m, replace=False)
#     return features[idx]


# def coreset_kcenter(
#     features: np.ndarray,
#     pool_size: int,
#     ratio: float,
#     seed: int,
#     init: str = "farthest_from_mean",
# ) -> np.ndarray:
#     """
#     简化版 k-center greedy（在 pool 里选）
#     - 先从 features 里抽 pool_size 个点作为 pool
#     - 在 pool 上做 k-center greedy 选 m = pool_size*ratio 个中心
#     """
#     rng = np.random.RandomState(seed)
#     n = features.shape[0]
#     pool_size = min(pool_size, n)

#     pool_idx = rng.choice(n, size=pool_size, replace=False)
#     pool = features[pool_idx]  # [P, D]
#     Pn = pool.shape[0]
#     m = max(1, int(Pn * ratio))
#     m = min(m, Pn)

#     # 初始化第一个中心
#     if init == "farthest_from_mean":
#         mu = pool.mean(axis=0, keepdims=True)
#         dist = np.sum((pool - mu) ** 2, axis=1)
#         first = int(np.argmax(dist))
#     else:
#         first = int(rng.randint(0, Pn))

#     centers = [first]

#     # 维护每个点到最近中心的距离
#     dists = np.sum((pool - pool[first]) ** 2, axis=1)

#     for _ in range(1, m):
#         nxt = int(np.argmax(dists))
#         centers.append(nxt)
#         # 用新中心更新最近距离
#         new_d = np.sum((pool - pool[nxt]) ** 2, axis=1)
#         dists = np.minimum(dists, new_d)

#     return pool[np.array(centers, dtype=np.int64)]





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
