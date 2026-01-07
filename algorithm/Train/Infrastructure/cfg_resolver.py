


"""Utility helpers for constructing :class:`PatchCoreTrainConfig`.

The UI layer typically only supplies a handful of editable fields. The
remaining defaults come from the YAML configuration file used elsewhere in the
project (``Config/config_setting.yaml``). This module merges the two sources and
returns a validated :class:`PatchCoreTrainConfig` instance that the application
can pass down to the domain layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


from Config.config import PatchCoreTrainConfig

# Default YAML that ships with the project. The resolver will silently fall
# back to dataclass defaults if the file is missing, which keeps unit tests and
# lightweight environments happy.
_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "Config" / "config_setting.yaml"


_SECTION_KEY_MAP: Mapping[str, Mapping[str, str]] = {
    "model": {
        "backbone_name": "backbone_name",
        "custom_weight_path": "custom_weight_path",
        "embedding_layers": "embedding_layers",
        "input_size": "input_size",
        "patch_size": "patch_size",
    },
    "train": {
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "max_train_features": "max_train_features",
        "n_neighbors": "n_neighbors",
        "seed": "seed",
        "gpu_device_id": "gpu_device_id",
    },
    "projection": {
        "use_random_projection": "use_random_projection",
        "projection_dim": "projection_dim",
        "projection_seed": "projection_seed",
    },
    "coreset": {
        "method": "coreset_method",
        "ratio": "coreset_ratio",
        "pool_size": "coreset_pool_size",
        "kcenter_init": "kcenter_init",
    },
    "faiss": {
        "index_type": "index_type",
        "ivf_nlist": "ivf_nlist",
        "ivf_nprobe": "ivf_nprobe",
        "pq_m": "pq_m",
        "pq_nbits": "pq_nbits",
        "hnsw_m": "hnsw_m",
        "hnsw_ef_construction": "hnsw_ef_construction",
        "hnsw_ef_search": "hnsw_ef_search",
    },
}


def _normalize_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize UI inputs to dataclass friendly shapes.

    - ``input_size`` coming from JSON/YAML may be a list; convert it to a tuple.
    - Remove ``None`` values so dataclass defaults remain in effect.
    """

    norm: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "input_size" and isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("input_size must be a 2-element sequence")
            norm[key] = (int(value[0]), int(value[1]))
        else:
            norm[key] = value
    return norm


def _load_defaults(yaml_path: Path | str | None) -> Dict[str, Any]:
    path = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML
    if yaml is None or not path.exists():
        return {}

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat: Dict[str, Any] = {}

    for section, mapping in _SECTION_KEY_MAP.items():
        section_data = content.get(section, {}) or {}
        for src_key, dst_key in mapping.items():
            if src_key in section_data:
                flat[dst_key] = section_data[src_key]
    return flat


def build_cfg_from_ui(ui_cfg: Dict[str, Any] | None, *, yaml_path: str | Path | None = None) -> PatchCoreTrainConfig:
    """Merge UI config overrides with YAML defaults into a validated config."""

    defaults = _load_defaults(yaml_path)
    merged = {**defaults, **(ui_cfg or {})}
    normalized = _normalize_types(merged)

    cfg = PatchCoreTrainConfig(**normalized)
    cfg.validate()
    return cfg
