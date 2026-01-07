from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

import Infrastructure.align_preprocess.GS_CV_Lib as gscv
import Infrastructure.align_preprocess.GS_CV_Match as gsmatch
# 你的工程依赖（按实际路径调整）
from Application.eventBus import EventBus
from Infrastructure.align_preprocess.IRISO_Step3_Preprocess import imagepro_for_AIdetect

# -----------------------------
# 文件 / 图像工具：更少依赖、更稳的 Unicode 路径支持
# -----------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMG_EXTS


def _iter_image_paths(folder: str) -> Iterator[Path]:
    p = Path(folder)
    if not p.is_dir():
        return
    for x in sorted(p.iterdir()):
        if _is_image_file(x):
            yield x


def _count_images(folder: str) -> int:
    return sum(1 for _ in _iter_image_paths(folder))


def _imread(path: str, flags: int) -> Optional[np.ndarray]:
    """
    尽量兼容 Windows 中文路径：
    - 优先 np.fromfile + cv2.imdecode
    - 失败再 fallback 到 cv2.imread
    """
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    return cv2.imread(p, flags)


def _imwrite(path: str, img: np.ndarray) -> bool:
    """
    尽量兼容 Windows 中文路径：
    - 优先 cv2.imencode + tofile
    - 失败再 fallback 到 cv2.imwrite
    """
    p = str(path)
    ext = Path(p).suffix.lower() or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(p)
            return True
    except Exception:
        pass
    try:
        return bool(cv2.imwrite(p, img))
    except Exception:
        return False


# -----------------------------
# Align + Preprocess：合并后的核心类
# -----------------------------
class AlignPreprocessor:
    """
    只负责：加载模板/掩码/匹配器 + 对齐 +（可选）预处理
    """

    def __init__(
            self,
            tmp_path: str,
            mask_path: str,
            scale: float = 0.5,
            match_method: str = "ORB",
            preprocess_roi: Optional[Tuple[int, int, int, int]] = None,
            preprocess_contrast: float = 1.2,
            preprocess_blur: bool = False,
            enable_preprocess: bool = True,
            preprocess_image_size: Optional[Tuple[int, int]] = None,
    ):
        self.tmp_path = str(tmp_path)
        self.mask_path = str(mask_path)
        self.scale = float(scale)
        self.match_method = str(match_method)

        # preprocess params（可被 Aligner 覆盖）
        self.preprocess_roi = preprocess_roi
        self.preprocess_contrast = float(preprocess_contrast)
        self.preprocess_blur = bool(preprocess_blur)
        self.enable_preprocess = bool(enable_preprocess)
        self.preprocess_image_size = preprocess_image_size  # (W,H) or None

        tmpl_gray = _imread(self.tmp_path, cv2.IMREAD_GRAYSCALE)
        mask_gray = _imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if tmpl_gray is None or mask_gray is None:
            raise FileNotFoundError(f"模板或掩码读取失败: {self.tmp_path}, {self.mask_path}")

        if self.scale != 1.0:
            tmpl_gray = gscv.image_resize(tmpl_gray, self.scale)
            mask_gray = gscv.image_resize(mask_gray, self.scale)

        self.template_gray = tmpl_gray
        self.mask_gray = mask_gray
        self.tmpl_h, self.tmpl_w = tmpl_gray.shape[:2]

        self.matcher = gsmatch.TemplateMatcher(method=self.match_method)

    def update_preprocess_params(
            self,
            *,
            preprocess_roi: Optional[Tuple[int, int, int, int]] = None,
            preprocess_contrast: float = 1.2,
            preprocess_blur: bool = False,
            preprocess_image_size: Optional[Tuple[int, int]] = None,
            enable_preprocess: Optional[bool] = None,
    ) -> None:
        self.preprocess_roi = preprocess_roi
        self.preprocess_contrast = float(preprocess_contrast)
        self.preprocess_blur = bool(preprocess_blur)
        self.preprocess_image_size = preprocess_image_size
        if enable_preprocess is not None:
            self.enable_preprocess = bool(enable_preprocess)

    def _resolve_do_preprocess(self, do_preprocess: Optional[bool]) -> bool:
        return self.enable_preprocess if do_preprocess is None else bool(do_preprocess)

    def align_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")

        if self.scale != 1.0:
            image_bgr_small = gscv.image_resize(image_bgr, self.scale)
        else:
            image_bgr_small = image_bgr

        img_gray = cv2.cvtColor(image_bgr_small, cv2.COLOR_BGR2GRAY)

        M, _, _ = self.matcher.match(self.template_gray, img_gray)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image_bgr_small,
            M,
            (self.tmpl_w, self.tmpl_h),
            flags=cv2.INTER_LINEAR,
        )
        aligned[self.mask_gray == 0] = (255, 255, 255)
        return aligned

    def preprocess_aligned_bgr(self, aligned_bgr: np.ndarray, do_preprocess: Optional[bool] = None) -> Dict[str, Any]:
        if aligned_bgr is None or aligned_bgr.size == 0:
            return {"preprocessed_gray": None, "preprocessed_bgr_for_model": None}

        use_preprocess = self._resolve_do_preprocess(do_preprocess)

        if use_preprocess:
            pre_gray = imagepro_for_AIdetect(
                aligned_bgr,
                blur=self.preprocess_blur,
                contrast=self.preprocess_contrast,
                ROI=self.preprocess_roi,
            )
        else:
            src = aligned_bgr
            if self.preprocess_roi is not None:
                x, y, w, h = self.preprocess_roi
                src = src[y:y + h, x:x + w]
            pre_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        if self.preprocess_image_size is not None:
            tw, th = int(self.preprocess_image_size[0]), int(self.preprocess_image_size[1])
            if tw > 0 and th > 0 and (pre_gray.shape[1], pre_gray.shape[0]) != (tw, th):
                pre_gray = cv2.resize(pre_gray, (tw, th), interpolation=cv2.INTER_AREA)

        pre_bgr = cv2.cvtColor(pre_gray, cv2.COLOR_GRAY2BGR)
        return {"preprocessed_gray": pre_gray, "preprocessed_bgr_for_model": pre_bgr}


class Aligner:
    """
    尽量少层级的“服务层”：
    - 管理 AlignPreprocessor 的生命周期（template/mask/scale/method 变化就重建）
    - 提供 align(...) 和 preprocess(...) 两个入口（与你当前 UI 交互保持一致）
    """

    def __init__(self, bus: Optional[EventBus] = None):
        self.bus = bus
        self._lock = threading.RLock()
        self._ap: Optional[AlignPreprocessor] = None
        self._sig: Optional[Tuple[str, str, float, str]] = None  # (tmp_path, mask_path, scale, method)

    def configure(self, *, tmp_path: str, mask_path: str, scale: float = 0.5, match_method: str = "ORB") -> None:
        """
        可选：先配置一次，使得你可以直接 preprocess(...)（不必先 align）
        """
        self._ensure_ap(tmp_path=tmp_path, mask_path=mask_path, scale=scale, match_method=match_method)

    def _ensure_ap(self, *, tmp_path: str, mask_path: str, scale: float, match_method: str) -> AlignPreprocessor:
        sig = (str(tmp_path), str(mask_path), float(scale), str(match_method))
        with self._lock:
            if self._ap is None or self._sig != sig:
                self._ap = AlignPreprocessor(
                    tmp_path=tmp_path,
                    mask_path=mask_path,
                    scale=scale,
                    match_method=match_method,
                )
                self._sig = sig
            return self._ap

    def _get_ap_or_raise(self) -> AlignPreprocessor:
        with self._lock:
            ap = self._ap
        if ap is None:
            raise RuntimeError(
                "preprocessor_not_initialized: call align(...) once or call configure(...) before preprocess(...)")
        return ap

    def align(
            self,
            tmp_path: str,
            mask_path: str,
            target_folder: str,
            subtag_folders: List[Tuple[str, str, str]],  # [(tag, subtag, folder)]
            scale: float = 0.5,
            match_method: str = "ORB",
            progress_every: int = 5,
    ) -> Iterator[Tuple[str, float, Optional[Dict[str, Dict[str, List[Dict[str, Optional[str]]]]]]]]:
        """
        输出目录：
          {target_folder}/{tag}/{subtag}/{filename}

        yield: (subtag_progress, file_progress, results_or_none)
          - 过程中：results_or_none = None
          - 最后一次：results_or_none = 最终 results 字典
        """

        if not target_folder:
            raise ValueError("target_folder is empty")
        if not subtag_folders:
            raise ValueError("subtag_folders is empty")

        ap = self._ensure_ap(tmp_path=tmp_path, mask_path=mask_path, scale=scale, match_method=match_method)

        total_files = sum(_count_images(folder) for _, _, folder in subtag_folders)

        # 结果直接用 dict/list，彻底去 dataclass
        results: Dict[str, Dict[str, List[Dict[str, Optional[str]]]]] = {}
        for tag, subtag, _ in subtag_folders:
            results.setdefault(tag, {}).setdefault(subtag, [])

        subtag_total = len(subtag_folders)
        out_root = Path(target_folder)
        out_root.mkdir(parents=True, exist_ok=True)

        # 没有图片：直接把空结构 yield 出去
        if total_files <= 0:
            yield f"{subtag_total}/{subtag_total}", 1.0, results
            return

        processed = 0

        for idx, (tag, subtag, src_folder) in enumerate(subtag_folders, start=1):
            subtag_progress = f"{idx}/{subtag_total}"



            aligned_folder = out_root / tag / subtag
            aligned_folder.mkdir(parents=True, exist_ok=True)

            for img_path in _iter_image_paths(src_folder):
                # time.sleep(1)
                if processed % progress_every == 0:
                    yield subtag_progress, float(processed / total_files), None

                raw_path = str(img_path.resolve())

                img = _imread(raw_path, cv2.IMREAD_COLOR)
                aligned_path: Optional[str] = None

                if img is not None:
                    try:
                        aligned = ap.align_bgr(img)
                        if isinstance(aligned, np.ndarray):
                            dst = aligned_folder / img_path.name
                            if _imwrite(str(dst), aligned):
                                aligned_path = str(dst.resolve())
                    except Exception:
                        aligned_path = None

                results[tag][subtag].append(
                    {"raw_path": raw_path, "aligned_path": aligned_path}
                )

                processed += 1


        # 最后一次：直接 yield dict（不是 json）
        yield f"{subtag_total}/{subtag_total}", 1.0, results

    # ============================================================
    # 批量预处理（保持你的事件发布语义）
    # ============================================================
    def preprocess(
            self,
            *,
            aligned_dir: str,
            out_dir: str,
            do_preprocess: bool = True,
            preprocess_roi: Optional[Tuple[int, int, int, int]] = None,
            preprocess_contrast: float = 1.2,
            preprocess_blur: bool = False,
            preprocess_image_size: Optional[Tuple[int, int]] = None,
            publish_events: bool = True,
    ) -> str:
        """
        把 aligned_dir 下所有图片 -> 预处理后写到 out_dir
        返回 out_dir（用于直接传给 train）
        """
        if publish_events and self.bus is not None:
            self.bus.publish_async(
                "PREPROCESS_STARTED",
                ts=time.time(),
                mode="folder",
                aligned_dir=aligned_dir,
                out_dir=out_dir,
                do_preprocess=bool(do_preprocess),
            )

        t0 = time.time()
        try:
            ap = self._get_ap_or_raise()

            # 直接更新 ap 的 preprocess 参数（少层级：不再搞一堆 hasattr）
            with self._lock:
                ap.update_preprocess_params(
                    preprocess_roi=preprocess_roi,
                    preprocess_contrast=preprocess_contrast,
                    preprocess_blur=preprocess_blur,
                    preprocess_image_size=preprocess_image_size,
                )

            aligned_dir_p = Path(aligned_dir)
            if not aligned_dir_p.is_dir():
                raise RuntimeError(f"aligned_dir is not a folder: {aligned_dir}")

            out_dir_p = Path(out_dir)
            out_dir_p.mkdir(parents=True, exist_ok=True)

            files = list(_iter_image_paths(aligned_dir))
            total = len(files)
            saved = 0
            skipped = 0

            for p in files:
                img = _imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    skipped += 1
                    continue

                out = ap.preprocess_aligned_bgr(img, do_preprocess=bool(do_preprocess))
                pre_bgr = out.get("preprocessed_bgr_for_model", None)

                if isinstance(pre_bgr, np.ndarray):
                    dst = out_dir_p / p.name
                    if _imwrite(str(dst), pre_bgr):
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

            if publish_events and self.bus is not None:
                self.bus.publish_async(
                    "PREPROCESS_DONE",
                    ts=time.time(),
                    mode="folder",
                    out_dir=str(out_dir_p),
                    count_total=total,
                    count_saved=saved,
                    count_skipped=skipped,
                    elapsed_sec=float(time.time() - t0),
                    do_preprocess=bool(do_preprocess),
                )

            return str(out_dir_p)

        except Exception as e:
            if publish_events and self.bus is not None:
                self.bus.publish_async(
                    "PREPROCESS_FAILED",
                    ts=time.time(),
                    mode="folder",
                    error=repr(e),
                )
            raise
