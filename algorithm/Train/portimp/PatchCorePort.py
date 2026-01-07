
from __future__ import annotations

import os
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import json
import threading
import time

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import shutil
import tempfile
from Domain.AnomalyDetector import AlgorithmPort

from Infrastructure.patchcore.backbone import PatchCoreBackbone
from Infrastructure.patchcore.config import PatchCoreTrainConfig
from Infrastructure.patchcore.dataset import NormalImageDataset
from Infrastructure.patchcore.faiss_index import sample_train_vectors_for_ivf, build_faiss_index
from Infrastructure.patchcore.feature_extractor import (
    build_transform,
    patchify,
    make_random_projection,
    apply_random_projection,
)
from Infrastructure.patchcore.coreset import coreset_random, coreset_kcenter

from Infrastructure.pindetector.IRISO_PinDetect import detect_pin, defect_count, draw_pin_overlay
from Infrastructure.align_preprocess.IRISO_Step3_Preprocess import imagepro_for_AIdetect

class PatchCorePort(AlgorithmPort):
    """
    单类版 PatchCore（去掉进度反馈与中止）：
    - train(): 输入 ok_image_folders，自行扫描图片；训练 + 落盘；训练后自动 unload()
    - predict(): 每次从磁盘 load()，推理并输出 overlay
    - load()/unload(): 管理内存与索引
    """

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self):
        # cfg.validate()
        self.cfg = None

        self.device = None
        self.transform = None

        self.model: Optional[PatchCoreBackbone] = None
        self.memory_bank: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None  # faiss.Index

        self._trained: bool = False
        self.model_feature_dir: Optional[str] = None
        self.model_id: Optional[str] = None

        self._lock = threading.Lock()

    # -----------------------------
    # Lifecycle / State
    # -----------------------------
    def is_ready(self) -> bool:
        return bool(self._trained and self.faiss_index is not None)

    def unload(self) -> None:
        self.model = None
        self.memory_bank = None
        self.projection_matrix = None
        self.faiss_index = None
        self._trained = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_backbone(self) -> None:
        if self.model is not None:
            return

        layer_ids = tuple(int(s) for s in str(self.cfg.embedding_layers).split("_") if s)
        self.model = PatchCoreBackbone(
            backbone_name=self.cfg.backbone_name,
            layers=layer_ids,
            pretrained=True,
            custom_weight_path=getattr(self.cfg, "custom_weight_path", None),
        ).to(self.device).eval()

    @staticmethod
    def get_device(device_id: int = 0) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(device_id)}")
        return torch.device("cpu")

    def _build_cfg_from_dict(self, cfg_dict: Dict[str, Any]) -> PatchCoreTrainConfig:
        print("config_dict", cfg_dict)
        cfg_obj = PatchCoreTrainConfig.from_dict(cfg_dict)
        return cfg_obj

    # -----------------------------
    # Disk IO
    # -----------------------------
    def load(self, model_feature_dir: str) -> None:
        """
        从磁盘加载：
        - memory_bank.npy
        - projection_matrix.npy（可选）
        - faiss_index.bin
        """
        d = Path(model_feature_dir)
        if not d.exists():
            raise FileNotFoundError(f"model_feature_dir not found: {model_feature_dir}")

        mb_path = d / "memory_bank.npy"
        idx_path = d / "faiss_index.bin"
        pm_path = d / "projection_matrix.npy"

        if not mb_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"missing files under: {model_feature_dir}")

        memory_bank = np.load(str(mb_path)).astype(np.float32, copy=False)
        projection_matrix = np.load(str(pm_path)).astype(np.float32, copy=False) if pm_path.exists() else None

        index = faiss.read_index(str(idx_path))

        # 可选：把 CPU index 迁到 GPU（按你的 cfg 字段名做兼容）
        use_faiss_gpu = bool(
            getattr(self.cfg, "use_faiss_gpu", False)
            or getattr(self.cfg, "faiss_on_gpu", False)
            or getattr(self.cfg, "faiss_gpu", False)
        )
        use_faiss_gpu = True
        print(use_faiss_gpu)
        if use_faiss_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                gpu_id = int(getattr(self.cfg, "gpu_device_id", 0))
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            except Exception:
                pass

        self.memory_bank = memory_bank
        self.projection_matrix = projection_matrix
        self.faiss_index = index

        self.model_feature_dir = str(d)
        self._trained = True
    
    # -----------------------------
    # Train
    # -----------------------------
    def train(
            self,
            model_id: str,
            cfg: Dict[str, Any],
            ok_image_folders: List[str],
            output_root: Optional[str] = None,
    ):
        """
        输入改为 ok_image_folders：自行从文件夹扫描图片路径（递归）。
        去掉进度与中止回调。
        - 可选：训练前对齐图做 imagepro_for_AIdetect（不resize），写入临时目录供 Dataset 读取
        - 训练结束自动删除临时预处理目录（无论成功/异常）
        """
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id is empty.")
        if not ok_image_folders:
            raise ValueError("ok_image_folders is empty.")

        pre_cache_dir: Optional[str] = None  # 用于 finally 自动删除

        try:
            with self._lock:
                try:
                    # ---- apply cfg(dict) ----
                    self.cfg = self._build_cfg_from_dict(cfg)
                    self.device = self.get_device(self.cfg.gpu_device_id)
                    self.transform = build_transform(self.cfg.input_size)
                    self.model_id = str(model_id)

                    # ---- collect images from folders ----
                    train_img_paths = self._collect_images_from_folders(ok_image_folders)
                    if not train_img_paths:
                        raise ValueError(f"No images found in ok_image_folders: {ok_image_folders}")

                    # flag：是否使用 preprocess（根据cfg确定）
                    use_preprocess = bool(getattr(self.cfg, "use_preprocess", None))

                    # ---- decide dataset paths ----
                    dataset_img_paths = train_img_paths

                    if use_preprocess:
                        # ===== 创建 dataset 之前：先对所有对齐图做 imagepro_for_AIdetect（不 resize）=====
                        base_out = output_root or getattr(self.cfg, "output_root", "./_export_models")
                        os.makedirs(base_out, exist_ok=True)

                        # 用“唯一临时目录”，避免覆盖/冲突；训练完自动删
                        pre_cache_dir = tempfile.mkdtemp(prefix=f"_pre_cache_{model_id}_", dir=base_out)

                        # 如果 ok_image_folders 只有一个根目录，尽量保留相对结构避免同名覆盖
                        keep_rel = ok_image_folders[0] if len(ok_image_folders) == 1 else None

                        pre_img_paths = preprocess_to_cache_no_resize_paths(
                            train_img_paths,
                            cache_dir=pre_cache_dir,
                            blur=bool(getattr(self.cfg, "preprocess_blur", False)),
                            contrast=float(getattr(self.cfg, "preprocess_contrast", 1.2)),
                            roi=getattr(self.cfg, "preprocess_roi", None),
                            keep_rel_to=keep_rel,
                        )
                        if not pre_img_paths:
                            raise ValueError("No preprocessed images generated.")

                        dataset_img_paths = pre_img_paths  # 用预处理后的路径训练

                    # ---- dataloader ----
                    dataset = NormalImageDataset(dataset_img_paths, self.transform)
                    loader = DataLoader(
                        dataset,
                        batch_size=int(self.cfg.batch_size),
                        shuffle=False,
                        num_workers=int(self.cfg.num_workers),
                        pin_memory=True,
                        persistent_workers=False,
                    )

                    # ---- backbone ----
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.set_device(int(self.cfg.gpu_device_id))
                        except Exception:
                            pass

                    layer_ids = tuple(int(s) for s in str(self.cfg.embedding_layers).split("_") if s)
                    model = PatchCoreBackbone(
                        backbone_name=self.cfg.backbone_name,
                        layers=layer_ids,
                        pretrained=True,
                        custom_weight_path=getattr(self.cfg, "custom_weight_path", None),
                    ).to(self.device).eval()

                    np.random.seed(int(getattr(self.cfg, "seed", 0)))

                    # ---- output dir ----
                    out_root = output_root or getattr(self.cfg, "output_root", None)
                    out_dir: Optional[Path] = None
                    if out_root:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        exp_name = (
                            f"{ts}__memory_bank__"
                            f"{self.cfg.backbone_name}_"
                            f"L{self.cfg.embedding_layers}_"
                            f"in{self.cfg.input_size[0]}x{self.cfg.input_size[1]}_"
                            f"ps{self.cfg.patch_size}_"
                            f"k{self.cfg.n_neighbors}_"
                            f"M{int(self.cfg.max_train_features) // 1000}k_"
                            f"{self.cfg.index_type}"
                        )
                        out_dir = Path(out_root) / exp_name
                        out_dir.mkdir(parents=True, exist_ok=True)

                    # ---- stage1: collect features ----
                    all_features: List[np.ndarray] = []
                    num_batches = max(1, len(loader))
                    per_batch_cap = int(self.cfg.max_train_features) // num_batches + 1
                    total_patches = 0

                    with torch.no_grad():
                        for _batch_idx, imgs in enumerate(loader):
                            imgs = imgs.to(self.device, non_blocking=True)
                            feat_maps = model(imgs)
                            feats = patchify(feat_maps, int(self.cfg.patch_size))  # [N, D]
                            feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)

                            n_i = int(feats_np.shape[0])
                            total_patches += n_i

                            if n_i > per_batch_cap:
                                idx = np.random.choice(n_i, per_batch_cap, replace=False)
                                feats_np = feats_np[idx]

                            all_features.append(feats_np)

                    features = (
                        np.concatenate(all_features, axis=0).astype(np.float32, copy=False)
                        if all_features
                        else np.zeros((0, 1), np.float32)
                    )

                    # global cap
                    if features.shape[0] > int(self.cfg.max_train_features):
                        idx = np.random.choice(features.shape[0], int(self.cfg.max_train_features), replace=False)
                        features = features[idx]

                    # ---- stage2: random projection ----
                    projection_matrix: Optional[np.ndarray] = None
                    if bool(getattr(self.cfg, "use_random_projection", False)):
                        D = int(features.shape[1]) if features.ndim == 2 else 0
                        d = int(getattr(self.cfg, "projection_dim", D))
                        projection_seed = int(getattr(self.cfg, "projection_seed", 0))
                        projection_matrix = make_random_projection(D, d, projection_seed)
                        features = apply_random_projection(features, projection_matrix)

                    # ---- stage3: coreset ----
                    if str(self.cfg.coreset_method) == "random":
                        memory_bank = coreset_random(features, float(self.cfg.coreset_ratio), int(getattr(self.cfg, "seed", 0)))
                    else:
                        memory_bank = coreset_kcenter(
                            features=features,
                            pool_size=int(self.cfg.coreset_pool_size),
                            ratio=float(self.cfg.coreset_ratio),
                            seed=int(getattr(self.cfg, "seed", 0)),
                            init=str(getattr(self.cfg, "kcenter_init", "random")),
                            should_stop=None,
                        )

                    # ---- stage4: faiss index ----
                    train_vectors = None
                    if str(self.cfg.index_type) in ("IVFFlat", "IVFPQ"):
                        train_vectors = sample_train_vectors_for_ivf(
                            features, int(self.cfg.ivf_nlist), int(getattr(self.cfg, "seed", 0))
                        )
                    faiss_index = build_faiss_index(cfg=self.cfg, memory_bank=memory_bank, train_vectors=train_vectors)

                    # ---- stats ----
                    stats: Dict[str, Any] = {
                        "model_id": self.model_id,
                        "model_feature_dir": None,
                        "device": str(self.device),
                        "train_images": int(len(dataset_img_paths)),          # 实际参与训练的张数
                        "train_image_count": int(len(dataset_img_paths)),     # 兼容上层 UI
                        "total_patches": int(total_patches),
                        "kept_features": int(features.shape[0]) if features.size > 0 else 0,
                        "feature_dim": int(memory_bank.shape[1]) if memory_bank.size > 0 else 0,
                        "index_type": str(self.cfg.index_type),
                        "index_ntotal": int(faiss_index.ntotal),
                        "index_d": int(faiss_index.d),
                        "use_preprocess": bool(use_preprocess),
                    }

                    # ---- save ----
                    if out_dir is not None:
                        stats["model_feature_dir"] = str(out_dir)

                        np.save(str(out_dir / "memory_bank.npy"), memory_bank)
                        if projection_matrix is not None:
                            np.save(str(out_dir / "projection_matrix.npy"), projection_matrix)

                        idx_to_save = faiss_index
                        try:
                            import faiss
                            if hasattr(faiss, "index_gpu_to_cpu"):
                                idx_to_save = faiss.index_gpu_to_cpu(faiss_index)
                        except Exception:
                            idx_to_save = faiss_index

                        import faiss
                        faiss.write_index(idx_to_save, str(out_dir / "faiss_index.bin"))

                        meta = {
                            "cfg": self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else {},
                            "stats": stats,
                            "ok_image_folders": list(ok_image_folders),
                        }
                        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        np.save(out_dir / "meta.npy", meta, allow_pickle=True)

                        self.model_feature_dir = str(out_dir)
                        stats["model_feature_dir"] = str(out_dir)

                    return str(out_dir)

                finally:
                    # 确保异常时也释放资源
                    try:
                        self.unload()
                    except Exception:
                        pass

        finally:
            # 自动删除预处理缓存目录（不改 Dataset 的最佳解）
            if pre_cache_dir and os.path.isdir(pre_cache_dir):
                shutil.rmtree(pre_cache_dir, ignore_errors=True)

    
    
    

    # -----------------------------
    # Predict
    # -----------------------------
    @torch.no_grad()
    def predict(
            self,
            model_id: str,
            model_path: str,
            targets,
    ) -> Iterator[Dict[str, Any]]:
        """
        新输入格式（不兼容旧格式）：
        - model_id: str
        - targets: List[(real_label, path_or_dir)]
          e.g. [('OK', r'Project\\test\\Dataset\\OK\\OK1'), ('NG', r'Project\\test\\Dataset\\NG\\NG1')]
        - model_path: 模型特征目录（包含 memory_bank.npy / faiss_index.bin / meta.json）

        生成器：每次 yield 一张图片的结果。

        输出字段：
        model_id, real_label, pre_label, score, origin_path, heatmap_path(=overlay_path), defect_count, time
        """
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id is empty.")
        if not model_path or not str(model_path).strip():
            raise ValueError("model_path is empty.")
        if not targets:
            raise ValueError("targets is empty.")
        # ---- init once (load cfg/index/backbone) ----
        with self._lock:
            self.model_id = str(model_id)
            self.model_feature_dir = str(model_path)

            # 从 meta.json 还原 cfg，并初始化 device/transform
            self.cfg = self._load_cfg_from_model_dir(str(model_path))
            self.device = self.get_device(int(getattr(self.cfg, "gpu_device_id", 0)))
            self.transform = build_transform(self.cfg.input_size)

            # load faiss + memory bank
            self.load(str(model_path))
            self._ensure_backbone()

            assert self.model is not None
            assert self.faiss_index is not None

            # score threshold
            score_thr = (
                    getattr(self.cfg, "pc_score_thr", None)
            )
            score_thr = float(score_thr) if score_thr is not None else None

            # pin 默认 ROI（沿用你旧逻辑）
            default_pin_roi_xyxy: Tuple[int, int, int, int] = (154, 20, 1364, 120)
            pin_params: Dict[str, Any] = {}

            k = int(getattr(self.cfg, "n_neighbors", 1))
            patch_size = int(getattr(self.cfg, "patch_size", 3))

        # ---- iterate & yield per image ----
        try:
            for real_label, target_path in targets:
                real_label = str(real_label).strip().upper()

                # target_path 可能是文件或文件夹（递归）
                for img_path in self._iter_images_from_target(str(target_path)):
                    t0 = time.perf_counter()

                    pre_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if pre_img is None:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        yield {
                            "model_id": self.model_id,
                            "real_label": real_label,
                            "pre_label": "NG",
                            "score": 0.0,
                            "origin_path": str(img_path),
                            "heatmap_path": "",
                            "defect_count": 0,
                            "time": float(elapsed_ms),
                        }
                        continue

                    # ---- patchcore forward ----
                    pil = self._to_pil_rgb(pre_img)
                    x = self.transform(pil).unsqueeze(0).to(self.device)

                    feat_maps = self.model(x)  # [1, C, Hf, Wf]
                    feats = patchify(feat_maps, patch_size)  # [N, D]
                    q = feats.detach().cpu().numpy().astype(np.float32, copy=False)

                    if self.projection_matrix is not None:
                        q = apply_random_projection(q, self.projection_matrix, chunk=200_000)

                    distances, _ = self.faiss_index.search(q, k)
                    patch_score = distances.mean(axis=1) if distances.size else np.array([0.0], dtype=np.float32)
                    image_score = float(patch_score.max())
                        
                    # ---- amap + image_score_raw ----
                    Hf, Wf = int(feat_maps.shape[2]), int(feat_maps.shape[3])
                    amap = patch_score.reshape(Hf, Wf) if patch_score.size == Hf * Wf else None

                    # ---- heatmap -> overlay ----
                    overlay_bgr = pre_img.copy()
                    if amap is not None:
                        amap_norm = self.normalize_amap_absolute(amap, threshold=score_thr)
                        # pre_img(BGR) -> PIL(RGB)
                        orig_pil = Image.fromarray(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))
                        overlay_bgr = self.overlay_heatmap_on_image(
                            orig_pil=orig_pil,
                            heatmap=amap_norm,
                        )
                    
                    # ---- pin detect (默认 ROI) ----
                    pin_err_boxes_pre: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
                    pin_defect_cnt = 0

                    x1, y1, x2, y2 = map(int, default_pin_roi_xyxy)
                    H, W = pre_img.shape[:2]
                    x1 = max(0, min(x1, W - 1))
                    x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H - 1))
                    y2 = max(0, min(y2, H))

                    try:
                        if x2 > x1 and y2 > y1:
                            roi_bgr = pre_img[y1:y2, x1:x2]
                            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                            _ok, boxes_roi = detect_pin(roi_gray, **pin_params)  # ROI 内坐标
                            if boxes_roi:
                                for b in boxes_roi:
                                    # 1) 兼容 ((x1,y1),(x2,y2))
                                    if (
                                        isinstance(b, (tuple, list)) and len(b) == 2
                                        and isinstance(b[0], (tuple, list)) and len(b[0]) == 2
                                        and isinstance(b[1], (tuple, list)) and len(b[1]) == 2
                                    ):
                                        (bx1, by1), (bx2, by2) = b
                                        pin_err_boxes_pre.append(
                                            ((int(bx1 + x1), int(by1 + y1)), (int(bx2 + x1), int(by2 + y1)))
                                        )
                                    # 2) 兼容 (x,y,w,h)
                                    elif isinstance(b, (tuple, list)) and len(b) == 4:
                                        bx, by, bw, bh = b
                                        pin_err_boxes_pre.append(
                                            ((int(bx + x1), int(by + y1)), (int(bx + x1 + bw), int(by + y1 + bh)))
                                        )
                                    else:
                                        raise ValueError(f"Unexpected box format from detect_pin: {b}")
                            # 这里 defect_count 也用同样的格式（pt1,pt2）
                            pin_defect_cnt = int(defect_count(pin_err_boxes_pre))
                    except Exception:
                        traceback.print_exc()

                    # overlay pin boxes（确保 draw_pin_overlay 支持 (pt1,pt2) 格式）
                    if pin_err_boxes_pre:
                        overlay_bgr = draw_pin_overlay(overlay_bgr, pin_err_boxes_pre)

                    # ---- save overlay to: <target_root>/<model_id>/... ----
                    overlay_path = self._save_overlay_near_origin(
                        overlay_bgr,
                        origin_path=str(img_path),
                        target_root=str(target_path),
                        model_id=str(model_id),
                    )
                    # ---- decision ----
                    is_ng_by_score = (image_score >= score_thr) if score_thr is not None else False
                    is_ng = bool(is_ng_by_score or (pin_defect_cnt > 0))
                    pre_label = "NG" if is_ng else "OK"

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    yield {
                        "model_id": self.model_id,
                        "real_label": real_label,
                        "pre_label": pre_label,
                        "score": float(image_score),
                        "origin_path": str(img_path),
                        "heatmap_path": overlay_path,  # 注意：这里返回 overlay 图路径
                        "defect_count": int(pin_defect_cnt),
                        "time": float(elapsed_ms),
                    }

        finally:
            # 用完就卸载，防止长期占用 GPU/内存
            with self._lock:
                self.unload()

    # -----------------------------
    # Helpers (in-class)
    # -----------------------------

    def _load_cfg_from_model_dir(self, model_feature_dir: str) -> PatchCoreTrainConfig:
        d = Path(model_feature_dir)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found under: {model_feature_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg_dict = meta.get("cfg") or {}

        # 只保留 PatchCoreTrainConfig 定义过的字段，避免 unknown fields 报错
        valid = {f.name for f in fields(PatchCoreTrainConfig)}
        cfg_clean = {k: v for k, v in cfg_dict.items() if k in valid}

        cfg_obj = PatchCoreTrainConfig.from_dict(cfg_clean)

        # 兜底：确保 input_size 之类是可用的 tuple[int,int]
        if hasattr(cfg_obj, "input_size") and cfg_obj.input_size is not None:
            try:
                cfg_obj.input_size = (int(cfg_obj.input_size[0]), int(cfg_obj.input_size[1]))
            except Exception:
                pass

        return cfg_obj

    def _iter_images_from_target(self, target_path: str) -> Iterator[str]:
        """
        target_path 可以是图片文件，也可以是文件夹（只扫描当前目录，不进入子文件夹）。
        """
        p = Path(target_path)

        # 单文件
        if p.is_file():
            if p.suffix.lower() in self._IMG_EXTS:
                yield str(p)
            return

        # 单层目录扫描（不递归）
        if p.is_dir():
            files = []
            for fp in p.iterdir():  # 只迭代一层
                if fp.is_file() and fp.suffix.lower() in self._IMG_EXTS:
                    files.append(fp)
            for fp in sorted(files):  # 稳定顺序
                yield str(fp)
            return

    def _save_overlay_near_origin(
            self,
            overlay_bgr: np.ndarray,
            *,
            origin_path: str,
            target_root: str,
            model_id: str,
    ) -> str:
        """
        保存到：<target_root>/<model_id>/[relative_parent]/<stem>_overlay.jpg
        - target_root: targets 里传入的第二个路径（文件夹或文件）
        - origin_path: 当前图片的真实路径
        """
        target_root_p = Path(target_root)
        if target_root_p.is_file():
            base_dir = target_root_p.parent
        else:
            base_dir = target_root_p

        out_root = base_dir / str(model_id)
        out_root.mkdir(parents=True, exist_ok=True)

        origin_p = Path(origin_path)
        stem = origin_p.stem or "image"

        # 尽量保留相对目录结构（避免同名冲突）
        rel_parent = Path(".")
        try:
            if base_dir.exists():
                rel = origin_p.relative_to(base_dir)
                rel_parent = rel.parent  # 可能是 "."
        except Exception:
            rel_parent = Path(".")

        out_dir = out_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay_bgr)
        return str(out_path)

    def _collect_images_from_folders(self, folders: List[str]) -> List[str]:
        """
        递归扫描 folders 下的图片；排序保证稳定；去重保证多个 folder 覆盖时不重复。
        """
        paths: List[str] = []
        for f in folders:
            if not f:
                continue
            p = Path(f)
            if p.is_file():
                if p.suffix.lower() in self._IMG_EXTS:
                    paths.append(str(p))
                continue
            if not p.exists():
                continue
            for fp in p.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in self._IMG_EXTS:
                    paths.append(str(fp))

        # 稳定排序 + 去重（保序）
        paths.sort()
        return list(dict.fromkeys(paths))

    def _save_overlay(self, overlay_bgr: np.ndarray, *, stem: str, tag: str) -> str:
        assert self.model_feature_dir is not None
        out_dir = Path(self.model_feature_dir) / "_predict" / str(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(p), overlay_bgr)
        return str(p)

    @staticmethod
    def _to_pil_rgb(src: Any) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, str):
            return Image.open(src).convert("RGB")
        arr = np.asarray(src)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3:
            raise ValueError("image ndarray must be HxWxC")
        return Image.fromarray(arr[..., ::-1].astype(np.uint8)).convert("RGB")


    @staticmethod
    def _safe_stem_from_path(path: str) -> str:
        import os
        base = os.path.basename(path) if path else "image"
        stem, _ = os.path.splitext(base)
        return stem or "image"
    
    @staticmethod
    def normalize_amap_absolute(amap: np.ndarray, threshold: float) -> np.ndarray:
        """
        单阈值二值门控：
        amap < threshold  -> 0
        amap >= threshold -> 1
        输出 float32 的 0/1（后续 overlay 可直接用）
        """
        amap = amap.astype(np.float32, copy=False)
        return (amap >= float(threshold)).astype(np.float32)
    
    
    @staticmethod
    def overlay_heatmap_on_image(
        orig_pil: Image.Image,
        heatmap: np.ndarray,  # Hf×Wf, [0,1]
        alpha: float = 0.5,
        beta: float = 0.5,
        hot_only: bool = True,
        hot_thresh: float = 0.05,
    ) -> np.ndarray:
        """
        将特征图热力值映射回原图分辨率并叠加，输出 BGR（cv2 可保存）。
        """
        w_orig, h_orig = orig_pil.size
        heat_resized = cv2.resize(heatmap.astype(np.float32), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        heat_resized = np.clip(heat_resized, 0.0, 1.0)
        orig_rgb = np.array(orig_pil)
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        heat_uint8 = np.uint8(255.0 * heat_resized)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        mixed = cv2.addWeighted(orig_bgr, alpha, heat_color, beta, 0)
        if not hot_only:
            return mixed
        mask = heat_resized > float(hot_thresh)
        mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
        overlay = orig_bgr.copy()
        overlay[mask_3] = mixed[mask_3]
        return overlay

def preprocess_to_cache_no_resize_paths(
    img_paths: List[str],
    *,
    cache_dir: str,
    blur: bool = False,
    contrast: float = 1.2,
    roi: Optional[Tuple[int, int, int, int]] = None,
    keep_rel_to: Optional[str] = None,
) -> List[str]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    out_paths: List[str] = []
    for p in img_paths:
        src = Path(p)
        if not src.is_file():
            continue

        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        gray = imagepro_for_AIdetect(
            img_bgr,
            image_size=None,   # 不 resize
            blur=blur,
            contrast=contrast,
            ROI=roi,
        )
        gray = np.clip(gray, 0, 255).astype(np.uint8, copy=False)
        out_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if keep_rel_to:
            try:
                rel = src.relative_to(Path(keep_rel_to))
                out_path = cache_root / rel.parent / f"{src.stem}_pre.png"
            except Exception:
                out_path = cache_root / f"{src.stem}_pre.png"
        else:
            out_path = cache_root / f"{src.stem}_pre.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out_bgr)
        out_paths.append(str(out_path))

    return out_paths

