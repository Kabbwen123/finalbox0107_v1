# -*- coding: utf-8 -*-
"""
patchcore_inference_interface.py  —— 自定义 PatchCore 推理模块（对齐自定义 Trainer 产物）

支持两种部署：
1) 只保存 faiss_index.bin + meta.npy（推荐）：推理不需要 memory_bank.npy
2) 同时保存 memory_bank.npy：可选用作兜底/调试（但不强制）

若训练阶段启用 Random Projection：
- meta.npy 中 use_random_projection=True
- 目录下应存在 random_projection.npy（或 meta 里给了 projection_path）
推理会自动加载并对特征做投影，再送入 Faiss 搜索。

多 index_type 支持（由 meta["index_type"] 决定）：
- Flat
- IVFFlat / IVFPQ   -> 自动设置 nprobe（meta["ivf_nprobe"]）
- HNSW              -> 自动设置 efSearch（meta["hnsw_ef_search"]）
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import cv2
import time
import json

import faiss
from Infrastructure.patchcore.backbone import PatchCoreBackbone
from Infrastructure.patchcore.feature_extractor import build_transform, patchify, apply_random_projection
from Infrastructure.utils.get_gpu import get_device


# =========================
# 可视化参数（通常给 UI 调）
# =========================
ALPHA = 0.5
BETA = 0.5

HOT_ONLY = True
HOT_THRESH = 0.05

# 绝对阈值归一化（避免 OK 图也变红）
ABS_LOW_DEFAULT = 70.0
ABS_HIGH_DEFAULT = 95.0

# 如果 meta 中没有 patch_size，就用这个兜底
PATCH_SIZE_DEFAULT = 1


# =========================
# 工具函数
# =========================
def normalize_amap_absolute(amap: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    绝对阈值归一化：
        amap <= low  -> 0
        amap >= high -> 1
        中间线性
    """
    amap_norm = (amap - low) / (high - low)
    return np.clip(amap_norm, 0.0, 1.0)


def overlay_heatmap_on_image(
    orig_pil: Image.Image,
    heatmap: np.ndarray,  # Hf×Wf, [0,1]
    alpha: float = 0.5,
    beta: float = 0.5,
    hot_only: bool = True,
    hot_thresh: float = 0.0,
) -> np.ndarray:
    """
    将特征图热力值映射回原图分辨率并叠加，输出 BGR（cv2 可保存）。
    """
    w_orig, h_orig = orig_pil.size
    heat_resized = cv2.resize(heatmap, (w_orig, h_orig))
    heat_resized = np.clip(heat_resized, 0.0, 1.0)

    orig_rgb = np.array(orig_pil)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    heat_uint8 = np.uint8(255 * heat_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    if not hot_only:
        return cv2.addWeighted(orig_bgr, alpha, heat_color, beta, 0)

    mask = heat_resized > hot_thresh
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

    overlay = orig_bgr.copy()
    mixed = cv2.addWeighted(orig_bgr, alpha, heat_color, beta, 0)
    overlay[mask_3] = mixed[mask_3]
    return overlay


def _safe_get_meta(meta: Dict[str, Any], key: str, default=None):
    """meta 里可能是 numpy 标量，做一次 .item() 兼容。"""
    v = meta.get(key, default)
    if isinstance(v, np.ndarray):
        try:
            return v.item()
        except Exception:
            return v
    return v

def _normalize_index_type(index_type: str) -> str:
    """
    规范化 index_type 字符串：
    - 允许 flat / ivfflat / ivfpq / hnsw 等各种大小写
    """
    it = str(index_type).strip()
    it_map = {
        "flat": "Flat",
        "ivfflat": "IVFFlat",
        "ivfpq": "IVFPQ",
        "hnsw": "HNSW",
    }
    return it_map.get(it.lower(), it)

def _set_faiss_search_params(index, index_type: str, meta: Dict[str, Any]) -> None:
    """
    根据 index_type 设置搜索参数：
    - IVF: nprobe
    - HNSW: efSearch
    兼容 CPU/GPU；优先属性写入，失败则尝试 ParameterSpace。
    """
    index_type = _normalize_index_type(index_type)

    # ---------- IVF nprobe ----------
    if index_type in ("IVFFlat", "IVFPQ"):
        nprobe = int(_safe_get_meta(meta, "ivf_nprobe", 20))

        # 1) 常规写法（CPU IVF + 部分 GPU IVF）
        try:
            if hasattr(index, "nprobe"):
                index.nprobe = nprobe
                print(f"[PatchCore] Set IVF nprobe = {nprobe}")
                return
        except Exception as e:
            print(f"[WARN] Failed to set index.nprobe directly. Reason: {e}")

        # 2) 兜底：ParameterSpace（GPU 上更稳）
        try:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "nprobe", nprobe)
            print(f"[PatchCore] Set IVF nprobe via ParameterSpace = {nprobe}")
        except Exception as e:
            print(f"[WARN] Failed to set IVF nprobe via ParameterSpace. Reason: {e}")

    # ---------- HNSW efSearch ----------
    if index_type == "HNSW":
        ef = int(_safe_get_meta(meta, "hnsw_ef_search", 128))
        try:
            if hasattr(index, "hnsw") and hasattr(index.hnsw, "efSearch"):
                index.hnsw.efSearch = ef
                print(f"[PatchCore] Set HNSW efSearch = {ef}")
                return
        except Exception as e:
            print(f"[WARN] Failed to set HNSW efSearch directly. Reason: {e}")

        # ParameterSpace 对 HNSW 有时也能设（看 faiss build）
        try:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "efSearch", ef)
            print(f"[PatchCore] Set HNSW efSearch via ParameterSpace = {ef}")
        except Exception as e:
            print(f"[WARN] Failed to set HNSW efSearch via ParameterSpace. Reason: {e}")


# =========================
# 推理类（核心：对齐 trainer 输出）
# =========================
class PatchcoreInferencer:
    """
    自定义 PatchCore 推理接口（对齐自定义 trainer 的产物）：
    必需：
      - meta.npy
      - faiss_index.bin（默认 require_faiss_index=True）
    可选：
      - random_projection.npy（use_random_projection=True 时必需）
      - memory_bank.npy（可选：兜底/调试，不强制）
    """

    def __init__(
        self,
        model_feature_dir: str,
        use_faiss_gpu: bool = True,
        faiss_gpu_device: int = 0,
        alpha: float = ALPHA,
        beta: float = BETA,
        hot_only: bool = HOT_ONLY,
        hot_thresh: float = HOT_THRESH,
        abs_low: float = ABS_LOW_DEFAULT,
        abs_high: float = ABS_HIGH_DEFAULT,
        require_faiss_index: bool = True,
    ):
        self.model_feature_dir = model_feature_dir
        self.use_faiss_gpu = use_faiss_gpu
        self.faiss_gpu_device = faiss_gpu_device

        self.alpha = alpha
        self.beta = beta
        self.hot_only = hot_only
        self.hot_thresh = hot_thresh
        self.abs_low = abs_low
        self.abs_high = abs_high
        self.require_faiss_index = require_faiss_index

        # -------- 1) 读取 meta --------
        meta_path = os.path.join(model_feature_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found in: {model_feature_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

        # 可选：保存拆分引用，后面调试方便
        self.meta_cfg: Dict[str, Any] = self.meta.get("cfg") if isinstance(self.meta.get("cfg"), dict) else {}
        self.meta_stats: Dict[str, Any] = self.meta.get("stats") if isinstance(self.meta.get("stats"), dict) else {}

        # -------- 2) 按“新版结构”解析（兼容旧版）--------
        self.backbone_name = str(
            _meta_get(self.meta, "backbone_name", None)
            or _meta_get(self.meta, "backbone", None)
            or ""
        ).strip()
        if not self.backbone_name:
            raise ValueError("meta missing backbone_name/backbone")

        self.embedding_layers_str = _parse_embedding_layers(
            _meta_get(self.meta, "embedding_layers", None)
            or _meta_get(self.meta, "layers", None)
        )
        if not self.embedding_layers_str:
            raise ValueError("meta missing embedding_layers/layers")

        # JSON里是 [H, W]
        self.input_size = _safe_tuple2_int(_meta_get(self.meta, "input_size", None), name="input_size")

        self.patch_size = int(_meta_get(self.meta, "patch_size", PATCH_SIZE_DEFAULT))
        self.n_neighbors = int(_meta_get(self.meta, "n_neighbors", 5))
        self.custom_weight_path = _meta_get(self.meta, "custom_weight_path", None)

        # index_type（你 cfg 和 stats 里都有，cfg 优先）
        self.index_type = _normalize_index_type(_meta_get(self.meta, "index_type", "Flat"))

        # Random Projection（你 cfg 里有 use_random_projection/projection_dim 等）
        self.use_random_projection = bool(_meta_get(self.meta, "use_random_projection", False))

        # 你新的 meta.json 里没有 projection_path 字段：推理端按约定文件名拼路径即可
        # （训练端若 use_random_projection=True 通常会导出 random_projection.npy）
        proj_default_path = os.path.join(model_feature_dir, "random_projection.npy")
        self.projection_path = _meta_get(self.meta, "projection_path", proj_default_path if self.use_random_projection else None)

        # 如果开启了 RP，但文件不存在，直接报错更清晰
        if self.use_random_projection:
            if not self.projection_path or not os.path.isfile(self.projection_path):
                raise FileNotFoundError(
                    f"use_random_projection=True but random projection file not found: {self.projection_path}"
                )

        # （可选）如果你推理端会用到这些 stats 信息，也可以顺手读出来
        self.feature_dim = int(_meta_get(self.meta, "feature_dim", 0) or 0)
        self.index_d = int(_meta_get(self.meta, "index_d", 0) or 0)
        self.index_ntotal = int(_meta_get(self.meta, "index_ntotal", 0) or 0)

        # （可选）记录训练来源
        self.ok_image_folders = self.meta.get("ok_image_folders", [])
        # -------- 2) 加载 RP 矩阵（如启用）--------
        self.P = None
        if self.use_random_projection:
            if not self.projection_path:
                self.projection_path = os.path.join(model_feature_dir, "random_projection.npy")

            proj_path = self.projection_path
            if not os.path.isabs(proj_path):
                proj_path = os.path.join(model_feature_dir, proj_path)

            if not os.path.isfile(proj_path):
                raise FileNotFoundError(f"use_random_projection=True but projection file not found: {proj_path}")

            self.P = np.load(proj_path).astype(np.float32)  # [D, d]
            print(f"[PatchCore] Loaded random projection: {proj_path}, shape={self.P.shape}")

        print("[PatchCore] Loaded meta:")
        print(f"   backbone              = {self.backbone_name}")
        print(f"   embedding_layers      = {self.embedding_layers_str}")
        print(f"   input_size            = {self.input_size}")
        print(f"   patch_size            = {self.patch_size}")
        print(f"   n_neighbors           = {self.n_neighbors}")
        print(f"   index_type            = {self.index_type}")
        print(f"   custom_weight         = {self.custom_weight_path}")
        print(f"   use_random_projection = {self.use_random_projection}")

        # -------- 3) 构建 backbone --------
        layer_ids = tuple(int(s) for s in self.embedding_layers_str.split("_") if s)
        self.device = get_device()
        print(f"[PatchCore] Using device: {self.device}")

        self.model = PatchCoreBackbone(
            backbone_name=self.backbone_name,
            layers=layer_ids,
            pretrained=True,
            custom_weight_path=self.custom_weight_path,
        ).to(self.device).eval()

        self.transform = build_transform(self.input_size)

        # -------- 4) 加载 Faiss index（必需）--------
        self.faiss_index = self._load_faiss_index()
        print("[PatchCore] faiss_index type:", type(self.faiss_index))
        print("[PatchCore] faiss_index on GPU?", "Gpu" in type(self.faiss_index).__name__)
        
        # （可选）兜底：加载 memory_bank.npy（不强制）
        self.memory_bank = None
        memory_path = os.path.join(model_feature_dir, "memory_bank.npy")
        if os.path.isfile(memory_path):
            self.memory_bank = np.load(memory_path).astype(np.float32)
            print(f"[PatchCore] (Optional) Loaded memory_bank.npy: {memory_path}, shape={self.memory_bank.shape}")

    def _load_faiss_index(self):
        index_path = os.path.join(self.model_feature_dir, "faiss_index.bin")
        if not os.path.isfile(index_path):
            if self.require_faiss_index:
                raise FileNotFoundError(f"faiss_index.bin not found in: {self.model_feature_dir}")
            print("[PatchCore] faiss_index.bin not found, will fallback to naive KNN if memory_bank exists.")
            return None

        cpu_index = faiss.read_index(index_path)
        print(f"[PatchCore] Loaded Faiss CPU index: {index_path} (ntotal={cpu_index.ntotal}, d={cpu_index.d})")

        index_type = _normalize_index_type(self.index_type)

        # 先在 CPU index 上设置搜索参数（IVF nprobe / HNSW efSearch）
        _set_faiss_search_params(cpu_index, index_type, self.meta)

        if not self.use_faiss_gpu:
            print("[PatchCore] use_faiss_gpu=False, use CPU index.")
            return cpu_index

        # HNSW 通常不支持搬 GPU
        if index_type == "HNSW":
            print("[PatchCore] index_type=HNSW -> 通常不建议/不支持搬 GPU，使用 CPU index。")
            return cpu_index

        try:
            ngpu = faiss.get_num_gpus()
            print(f"[PatchCore] Faiss detected {ngpu} GPU(s).")
            if ngpu <= 0:
                print("[PatchCore] No GPU detected by Faiss, use CPU index.")
                return cpu_index

            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, self.faiss_gpu_device, cpu_index)
            print(f"[PatchCore] Faiss index moved to GPU:{self.faiss_gpu_device}")

            # 搬到 GPU 后，再尝试设置一次搜索参数
            _set_faiss_search_params(gpu_index, index_type, self.meta)
            return gpu_index

        except Exception as e:
            print(f"[PatchCore] Failed to move Faiss index to GPU, fallback to CPU. Reason: {e}")
            return cpu_index

    def infer_bgr(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_pil = Image.fromarray(img_rgb)
        return self.infer_pil(orig_pil)

    def infer_pil(self, orig_pil: Image.Image) -> Dict[str, Any]:
        """
        返回：
          - image_score_raw: float
          - amap: [Hf, Wf] 原始距离图（L2 距离平方）
          - amap_norm: [Hf, Wf] 归一化热力图
          - overlay_bgr: 叠加到原图分辨率的 BGR
        """
        orig_pil = orig_pil.convert("RGB")
        img_tensor = self.transform(orig_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_maps = self.model(img_tensor)                 # [1, C, Hf, Wf]
            _, _, Hf, Wf = feat_maps.shape
            feats = patchify(feat_maps, self.patch_size)       # [Npatch, D]
            feats_np = feats.cpu().numpy().astype(np.float32)

        # ---- 如果启用 RP，则先投影 ----
        if self.use_random_projection:
            assert self.P is not None
            if feats_np.shape[1] != self.P.shape[0]:
                raise RuntimeError(
                    f"RP 维度不匹配：features dim={feats_np.shape[1]}，P shape={self.P.shape}。"
                    "请确认训练/推理使用的 backbone/layers/patch_size 完全一致，并且推理读取的 meta 正确。"
                )
            feats_np = apply_random_projection(feats_np, self.P)

        # ---- kNN 搜索（优先 faiss index；否则兜底 memory_bank）----
        # if self.faiss_index is not None:
        #     D, _ = self.faiss_index.search(feats_np, self.n_neighbors)
        #     patch_scores = D[:, 0]  # 每个 patch 的最小距离（平方 L2）
        # else:
        #     if self.memory_bank is None:
        #         raise RuntimeError("No faiss index and no memory bank. Cannot infer.")
        #     patch_scores = self._naive_knn_min_dist(feats_np, self.memory_bank, self.n_neighbors)

        # 对齐训练端的scores得分方式
        if self.faiss_index is not None:
            D, _ = self.faiss_index.search(feats_np, self.n_neighbors)  # D: [Npatch, k]
            if D.size == 0:
                patch_scores = np.array([0.0], dtype=np.float32)
            else:
                patch_scores = D.mean(axis=1).astype(np.float32, copy=False)  # ✅ 对齐训练端
        else:
            if self.memory_bank is None:
                raise RuntimeError("No faiss index and no memory bank. Cannot infer.")
            # 原本的推理端
            # patch_scores = self._naive_knn_min_dist(feats_np, self.memory_bank, self.n_neighbors)
            # 对齐训练端
            patch_scores = self._naive_knn_mean_dist(feats_np, self.memory_bank, self.n_neighbors)


        amap = patch_scores.reshape(Hf, Wf)
        image_score_raw = float(amap.max())

        amap_norm = normalize_amap_absolute(amap, low=self.abs_low, high=self.abs_high)

        overlay_bgr = overlay_heatmap_on_image(
            orig_pil=orig_pil,
            heatmap=amap_norm,
            alpha=self.alpha,
            beta=self.beta,
            hot_only=self.hot_only,
            hot_thresh=self.hot_thresh,
        )

        return {
            "image_score_raw": image_score_raw,
            "amap": amap,
            "amap_norm": amap_norm,
            "overlay_bgr": overlay_bgr,
        }

    @staticmethod
    def _naive_knn_min_dist(features: np.ndarray, memory_bank: np.ndarray, k: int) -> np.ndarray:
        """慢速兜底（不推荐生产用）。"""
        features = features.astype(np.float32)
        memory_bank = memory_bank.astype(np.float32)
        num_patches = features.shape[0]
        patch_scores = np.empty(num_patches, dtype=np.float32)
        for i in range(num_patches):
            diff = memory_bank - features[i]
            dist_sq = np.sum(diff * diff, axis=1)
            topk = np.partition(dist_sq, k - 1)[:k]
            patch_scores[i] = np.min(topk)
        return patch_scores
    
    # 对齐训练端
    @staticmethod
    def _naive_knn_mean_dist(features: np.ndarray, memory_bank: np.ndarray, k: int) -> np.ndarray:
        """慢速兜底：返回每个 patch 到 k 近邻的距离均值（对齐 faiss.mean(axis=1)）。"""
        features = features.astype(np.float32, copy=False)
        memory_bank = memory_bank.astype(np.float32, copy=False)
        num_patches = features.shape[0]
        patch_scores = np.empty(num_patches, dtype=np.float32)

        k = int(k)
        if k <= 0:
            k = 1

        for i in range(num_patches):
            diff = memory_bank - features[i]
            dist_sq = np.sum(diff * diff, axis=1)  # [M]
            kk = min(k, dist_sq.shape[0])
            topk = np.partition(dist_sq, kk - 1)[:kk]
            patch_scores[i] = float(np.mean(topk))

        return patch_scores


def _normalize_index_type(v: Any) -> str:
    s = str(v or "Flat").strip()
    # 你内部如果还有更多别名映射，可以在这里加
    return s

def _safe_tuple2_int(v: Any, *, name: str) -> Tuple[int, int]:
    if v is None:
        raise KeyError(f"meta missing required field: {name}")
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"meta field {name} must be [H,W] or (H,W), got: {v}")
    return (int(v[0]), int(v[1]))

def _parse_embedding_layers(s: Any) -> str:
    # 你训练端存的是 "2_3"（字符串），推理端也保持字符串即可
    # 如果你后面要变成 tuple[int,...] 再扩展
    return str(s or "").strip()

def _meta_get(meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    兼容新/旧 meta.json：
    - 新版：meta["cfg"][key] / meta["stats"][key]
    - 旧版：meta[key]
    取值优先级：cfg -> stats -> top-level
    """
    cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    stats = meta.get("stats") if isinstance(meta.get("stats"), dict) else {}

    if key in cfg:
        return cfg.get(key, default)
    if key in stats:
        return stats.get(key, default)
    return meta.get(key, default)


# =========================
# 单文件测试 main
# =========================
def main():
    MODEL_FEATURE_DIR = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\mb_1204_resnet34_L2_3_in256x640_ps3_k5_M200k\training_OK"
    IMG_PATH = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\1204\NG\08190002_1_alt.jpg"
    SAVE_OVERLAY_PATH = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\result\interface_test\08190002_1_alt_overlay.jpg"

    pc = PatchcoreInferencer(
        model_feature_dir=MODEL_FEATURE_DIR,
        use_faiss_gpu=True,
        faiss_gpu_device=0,
        alpha=0.5,
        beta=0.5,
        hot_only=True,
        hot_thresh=0.05,
        abs_low=70.0,
        abs_high=95.0,
        require_faiss_index=True,
    )

    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {IMG_PATH}")
        return

    t0 = time.time()
    result = pc.infer_bgr(img_bgr)
    dt = time.time() - t0

    print(f"[INFO] image_score_raw = {result['image_score_raw']:.4f}, infer_time={dt*1000:.1f}ms")

    out_path = Path(SAVE_OVERLAY_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result["overlay_bgr"])
    print(f"[INFO] Saved overlay to: {out_path}")


if __name__ == "__main__":
    main()
