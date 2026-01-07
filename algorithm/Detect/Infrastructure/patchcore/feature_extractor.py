# from __future__ import annotations

# from typing import Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T

# # =========================
# # Transform
# # =========================
# def build_transform(input_size: Tuple[int, int]) -> T.Compose:
#     """
#     与训练/推理一致的预处理：
#     - Resize 到 (H, W)
#     - ToTensor
#     - ImageNet mean/std Normalize（与你 backbone 的 torchvision 预训练一致）

#     input_size: (H, W)
#     """
#     h, w = input_size
#     return T.Compose([
#         T.Resize((h, w)),
#         T.ToTensor(),
#         T.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])
    
# # =========================
# # Patchify (关键：训练/推理必须同一实现)
# # =========================
# def patchify(feat_maps: torch.Tensor, patch_size: int) -> torch.Tensor:
#     """
#     将特征图 [B, C, H, W] 切成滑动 patch 并展平为 2D 特征：
#       - patch_size <= 1: 等价于逐像素特征，输出维度 = C
#       - patch_size > 1 : 使用 unfold 取邻域，输出维度 = C * patch_size^2

#     返回：
#       [B*H*W, C'] 的 2D 特征（numpy 前通常会 .cpu().numpy().astype(np.float32)）
#     """
#     if feat_maps.dim() != 4:
#         raise ValueError(f"patchify expects 4D tensor [B,C,H,W], got shape={tuple(feat_maps.shape)}")

#     B, C, H, W = feat_maps.shape

#     if patch_size <= 1:
#         # [B,C,H,W] -> [B,H,W,C] -> [B*H*W,C]
#         return feat_maps.permute(0, 2, 3, 1).reshape(-1, C)

#     # padding = patch_size//2 保证输出位置数仍为 H*W
#     patches = F.unfold(
#         feat_maps,
#         kernel_size=patch_size,
#         stride=1,
#         padding=patch_size // 2,
#     )  # [B, C*ps*ps, H*W]

#     # -> [B, H*W, C*ps*ps] -> [B*H*W, C*ps*ps]
#     return patches.permute(0, 2, 1).reshape(-1, C * patch_size * patch_size)


# # =========================
# # Random Projection (训练可选，推理需对齐)
# # =========================
# def make_random_projection(D: int, d: int, seed: int) -> np.ndarray:
#     """
#     生成随机投影矩阵 P: [D, d]
#     常见做法：N(0, 1/sqrt(d))，避免数值爆炸。

#     D: 原始特征维度
#     d: 投影后维度
#     seed: 随机种子（保证可复现）
#     """
#     if d <= 0 or D <= 0:
#         raise ValueError(f"Invalid dims: D={D}, d={d}")
#     rng = np.random.RandomState(seed)
#     P = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d), size=(D, d)).astype(np.float32)
#     return P

# def apply_random_projection(
#     features: np.ndarray,
#     P: np.ndarray,
#     chunk: int = 200_000,
# ) -> np.ndarray:
#     """
#     对特征做随机投影：
#       features: [N, D]
#       P:        [D, d]
#       return:   [N, d]

#     用 chunk 分块，避免一次性矩阵乘法占用过大内存。
#     """
#     if features.ndim != 2:
#         raise ValueError(f"features must be 2D [N,D], got shape={features.shape}")
#     if P.ndim != 2:
#         raise ValueError(f"P must be 2D [D,d], got shape={P.shape}")
#     if features.shape[1] != P.shape[0]:
#         raise ValueError(f"Dim mismatch: features dim={features.shape[1]} != P.D={P.shape[0]}")

#     features = features.astype(np.float32, copy=False)
#     P = P.astype(np.float32, copy=False)

#     N = features.shape[0]
#     d = P.shape[1]
#     out = np.empty((N, d), dtype=np.float32)

#     if chunk <= 0:
#         chunk = N  # 不分块

#     for s in range(0, N, chunk):
#         e = min(N, s + chunk)
#         out[s:e] = features[s:e] @ P

#     return out




# Infrastructure/patchcore/feature_extractor.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def build_transform(input_size: Tuple[int, int]) -> T.Compose:
    h, w = input_size
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def patchify(feat_maps: torch.Tensor, patch_size: int) -> torch.Tensor:
    if feat_maps.dim() != 4:
        raise ValueError(f"patchify expects 4D [B,C,H,W], got {tuple(feat_maps.shape)}")

    B, C, H, W = feat_maps.shape
    if patch_size <= 1:
        return feat_maps.permute(0, 2, 3, 1).reshape(-1, C)

    patches = F.unfold(
        feat_maps,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )  # [B, C*ps*ps, H*W]
    return patches.permute(0, 2, 1).reshape(-1, C * patch_size * patch_size)


def make_random_projection(D: int, d: int, seed: int) -> np.ndarray:
    if d <= 0 or D <= 0:
        raise ValueError(f"Invalid dims: D={D}, d={d}")
    rng = np.random.RandomState(seed)
    P = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d), size=(D, d)).astype(np.float32)
    return P


def apply_random_projection(
    features: np.ndarray,
    P: np.ndarray,
    chunk: int = 200_000,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N,D], got {features.shape}")
    if P.ndim != 2:
        raise ValueError(f"P must be 2D [D,d], got {P.shape}")
    if features.shape[1] != P.shape[0]:
        raise ValueError(f"Dim mismatch: features dim={features.shape[1]} != P.D={P.shape[0]}")

    features = features.astype(np.float32, copy=False)
    P = P.astype(np.float32, copy=False)

    N = features.shape[0]
    d = P.shape[1]
    out = np.empty((N, d), dtype=np.float32)

    if chunk <= 0:
        chunk = N

    for s in range(0, N, chunk):
        if should_stop is not None and should_stop():
            # 不在这里 raise，留给上层决定；但一般会 raise TrainingInterruptedError
            break
        e = min(N, s + chunk)
        out[s:e] = features[s:e] @ P

    return out
