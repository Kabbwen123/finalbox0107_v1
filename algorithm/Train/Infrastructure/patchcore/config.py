# domain/patchcore/config.py
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple


@dataclass
class PatchCoreTrainConfig:
    # preprocess
    use_preprocess: bool = True
    preprocess_blur: bool = False
    preprocess_contrast: float = 1.2
    
    # backbone / feature
    backbone_name: str = "resnet34"
    custom_weight_path: Optional[str] = None
    embedding_layers: str = "2_3"
    input_size: Tuple[int, int] = (256, 640)  # (H, W)
    patch_size: int = 3

    # dataloader
    batch_size: int = 4
    num_workers: int = 0

    # runtime
    gpu_device_id: int = 0
    seed: int = 0

    # random projection
    use_random_projection: bool = False
    projection_dim: int = 512
    projection_seed: int = 0

    # memory bank cap
    max_train_features: int = 200_000

    # knn / anomaly score
    n_neighbors: int = 5   # <-- 新增：PatchCore KNN 的 k

    # coreset
    coreset_method: str = "random"  # "random" | "kcenter"
    coreset_ratio: float = 1.0
    coreset_pool_size: int = 200_000
    kcenter_init: str = "random"

    # index
    index_type: str = "Flat"  # Flat/IVFFlat/IVFPQ/HNSW...
    ivf_nlist: int = 1024
    ivf_nprobe: int = 8
    pq_m: int = 16
    pq_nbits: int = 8

    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PatchCoreTrainConfig":
        """
        兼容两种输入：
        1) 扁平：{"backbone_name": "...", "batch_size": 16, ...}
        2) 分组：{"model": {...}, "train": {...}, "coreset": {...}, "faiss": {...}}
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be dict, got {type(cfg)}")

        # 1) 如果是分组结构：flatten
        if any(k in cfg for k in ("model", "train", "coreset", "faiss", "runtime", "projection")):
            flat: Dict[str, Any] = {}

            model = cfg.get("model") or {}
            train = cfg.get("train") or {}
            coreset = cfg.get("coreset") or {}
            faiss = cfg.get("faiss") or {}
            runtime = cfg.get("runtime") or {}
            proj = cfg.get("projection") or cfg.get("random_projection") or {}

            if not isinstance(model, dict): raise TypeError("cfg.model must be dict")
            if not isinstance(train, dict): raise TypeError("cfg.train must be dict")
            if not isinstance(coreset, dict): raise TypeError("cfg.coreset must be dict")
            if not isinstance(faiss, dict): raise TypeError("cfg.faiss must be dict")
            if not isinstance(runtime, dict): raise TypeError("cfg.runtime must be dict")
            if not isinstance(proj, dict): raise TypeError("cfg.projection must be dict")

            # ---- model ----
            flat.update(model)

            # ---- train ----
            flat.update(train)

            # ---- runtime ----
            flat.update(runtime)

            # ---- projection ----
            # 支持 projection.use_random_projection / projection.dim / projection.seed
            if proj:
                if "use_random_projection" in proj:
                    flat["use_random_projection"] = proj["use_random_projection"]
                if "projection_dim" in proj:
                    flat["projection_dim"] = proj["projection_dim"]
                if "dim" in proj:
                    flat["projection_dim"] = proj["dim"]
                if "projection_seed" in proj:
                    flat["projection_seed"] = proj["projection_seed"]
                if "seed" in proj:
                    flat["projection_seed"] = proj["seed"]

            # ---- coreset ----
            # 你的输入是 {"method": "random"}，这里映射到 coreset_method
            if "method" in coreset:
                flat["coreset_method"] = coreset["method"]
            if "coreset_method" in coreset:
                flat["coreset_method"] = coreset["coreset_method"]
            if "ratio" in coreset:
                flat["coreset_ratio"] = coreset["ratio"]
            if "pool_size" in coreset:
                flat["coreset_pool_size"] = coreset["pool_size"]
            if "kcenter_init" in coreset:
                flat["kcenter_init"] = coreset["kcenter_init"]

            # ---- faiss ----
            if "index_type" in faiss:
                flat["index_type"] = faiss["index_type"]
            # 如果你未来把 ivf/pq/hnsw 放在 faiss 子结构里，也可以在这里继续展开
            for k in ("ivf_nlist", "ivf_nprobe", "pq_m", "pq_nbits", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"):
                if k in faiss:
                    flat[k] = faiss[k]

        else:
            # 已经是扁平结构
            flat = dict(cfg)

        # 2) 类型与别名修正
        if "input_size" in flat and isinstance(flat["input_size"], list):
            if len(flat["input_size"]) != 2:
                raise ValueError(f"input_size must be length 2, got {flat['input_size']}")
            flat["input_size"] = (int(flat["input_size"][0]), int(flat["input_size"][1]))

        # 常见别名（你如果上层用别的命名，这里继续补）
        alias_map = {
            "coreset": "coreset_method",  # 兜底：有人可能写 coreset: random
        }
        for src, dst in alias_map.items():
            if src in flat and dst not in flat:
                flat[dst] = flat[src]

        # 3) 严格过滤未知字段，避免 silent fail
        allowed = {f.name for f in fields(cls)}
        unknown = set(flat.keys()) - allowed
        if unknown:
            raise TypeError(f"Unknown PatchCoreTrainConfig fields: {sorted(unknown)}")

        cfg_obj = cls(**flat)
        cfg_obj.validate()
        return cfg_obj

    def validate(self) -> None:
        if not self.embedding_layers or not isinstance(self.embedding_layers, str):
            raise ValueError("embedding_layers 不能为空")
        parts = [p for p in self.embedding_layers.split("_") if p]
        if not parts:
            raise ValueError(f"embedding_layers 格式不合法：{self.embedding_layers}")
        for p in parts:
            if p not in {"1", "2", "3", "4"}:
                raise ValueError(f"embedding_layers 格式不合法：{self.embedding_layers}（只能包含 1/2/3/4）")

        if self.patch_size <= 0:
            raise ValueError("patch_size 必须 > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        if self.max_train_features <= 0:
            raise ValueError("max_train_features 必须 > 0")
        if self.coreset_ratio <= 0:
            raise ValueError("coreset_ratio 必须 > 0")
