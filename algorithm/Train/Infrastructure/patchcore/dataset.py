# Infrastructure/patchcore/datasets.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class NormalImageDataset(Dataset):
    def __init__(self, img_paths: List[str], transform: T.Compose):
        if not img_paths:
            raise RuntimeError("img_paths 为空，无法训练")
        self.img_paths = [str(p) for p in img_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)


def build_sampled_train_paths(
    *,
    train_sources: List[Any],
    seed: int,
    image_pool: Dict[str, List[str]],
    shuffle: bool = True,
    min_per_class: int = 1,
) -> Tuple[List[str], Dict[str, Any]]:
    rng = random.Random(seed)
    report = {"sources": [], "seed": seed}
    all_paths: List[str] = []

    for item in train_sources:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"train_sources 元素格式必须是 [key, percent]，当前={item}")

        key, percent = item[0], float(item[1])
        if key not in image_pool:
            raise ValueError(f"image_pool 缺少 key={key}")
        pool = [str(p) for p in image_pool[key]]
        if not pool:
            raise ValueError(f"image_pool[{key}] 为空")

        if shuffle:
            rng.shuffle(pool)

        ratio = percent / 100.0 if percent > 1.0 else percent
        n = max(int(len(pool) * ratio), min_per_class)
        n = min(n, len(pool))

        picked = pool[:n]
        all_paths.extend(picked)

        report["sources"].append({
            "key": key,
            "ratio": ratio,
            "picked": n,
            "pool_size": len(pool),
        })

    if shuffle:
        rng.shuffle(all_paths)

    report["total_picked"] = len(all_paths)
    return all_paths, report
