# patchcore_detector.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import yaml

from Domain.align_preprocess_interface import AlignPreprocessor
from portimp.patchcore_pin_inferencer import PatchcorePinInferencer
from portimp.aibox_service import PatchcorePinService


@dataclass
class PatchcoreInferConfig:
    # === 对齐 + 预处理相关 ===
    template_path: str = (r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\AligeTemplate.jpg")
    mask_path: str = (r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\AligeMask.bmp")
    # preprocess_template_path: str
    preprocess_image_size: Tuple[int, int] = (1520, 468)  # (W,H)
    
    scale: float = 0.5
    match_method: str = "ORB"
    preprocess_roi: Optional[Tuple[int, int, int, int]] = None
    preprocess_contrast: float = 1.2
    preprocess_blur: bool = False

    # === PatchCore + PIN 相关 ===
    # model_feature_dir: str = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\result\1225_test\20251225_161817__memory_bank__resnet34_L2_3_in256x640_ps3_k5_M20kidxFlat_\training_OK"
    # model_feature_dir: str = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\result\1225_test\20251225_162059__memory_bank__resnet34_L2_3_in256x640_ps3_k5_M20kidxFlat_\training_OK"
    model_feature_dir: str = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\result\1225_test\20251225_183915__memory_bank__resnet34_L2_3_in256x640_ps3_k5_M200kidxIVFFlat_nl447_np16_\training_OK"
    roi: Optional[Tuple[int, int, int, int]] = None
    pc_score_thr: Optional[float] = None

    pincount: int = 35
    offset_up: int = 50
    offset_down: int = 80
    abs_low: int = 95
    abs_high: int = 100
    pinsizethreshold: int = 200

    # 一次最多处理多少张（None = 不限制）
    max_images_per_call: Optional[int] = None
    # 是否进行预处理
    enable_preprocess: bool = True

    # -----------------------------
    # YAML Loader
    # -----------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PatchcoreInferConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # list → tuple（YAML 没有 tuple）
        if "preprocess_image_size" in data and data["preprocess_image_size"] is not None:
            data["preprocess_image_size"] = tuple(data["preprocess_image_size"])

        if "preprocess_roi" in data and data["preprocess_roi"] is not None:
            data["preprocess_roi"] = tuple(data["preprocess_roi"])

        if "roi" in data and data["roi"] is not None:
            data["roi"] = tuple(data["roi"])

        return cls(**data)

class PatchcorePinDetector:
    """
    内部封装：
        AlignPreprocessor + AIBoxInterface + AIBoxService

    对外接口尽量和原来的 StfpmDetector 一致：
        detect_one(name_list, image_raw_bgr=img_list)
    """

    def __init__(self, cfg: PatchcoreInferConfig):
        self.cfg = cfg
        self.enable_preprocess: bool = bool(cfg.enable_preprocess)

        # 1) 对齐 + 预处理
        self.aligner = AlignPreprocessor(
            template_path=cfg.template_path,
            mask_path=cfg.mask_path,
            # preprocess_template_path=cfg.preprocess_template_path,
            preprocess_image_size=(1520, 468),
            scale=cfg.scale,
            match_method=cfg.match_method,
            preprocess_roi=cfg.preprocess_roi,
            preprocess_contrast=cfg.preprocess_contrast,
            preprocess_blur=cfg.preprocess_blur,
        )

        # 2) PatchCore + PIN 接口
        self.inferer = PatchcorePinInferencer(
            model_feature_dir=cfg.model_feature_dir,
            pc_score_thr=cfg.pc_score_thr,
            roi=cfg.roi,
            pincount=cfg.pincount,
            offset_up=cfg.offset_up,
            offset_down=cfg.offset_down,
            abs_low=cfg.abs_low,
            abs_high=cfg.abs_high,
            pinsizethreshold=cfg.pinsizethreshold,
        )

        # 3) Service 封装（负责批量检测）
        self.service = PatchcorePinService(
            aligner=self.aligner,
            inferer=self.inferer,
            max_images_per_call=cfg.max_images_per_call,
        )

    def detect_one(
        self,
        name_list: Union[str, List[str]],
        image_raw_bgr: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        do_preprocess: bool = None,
    ) -> Tuple[int, np.ndarray, float, int]:
        """
        兼容 UI 当前的调用方式：
            - name_list 可以是字符串，或者 [str, str, ...]
            - image_raw_bgr 可以是单张 np.ndarray，或者 [np.ndarray, ...]

        内部统一转换成列表，调用 service.detect_batch_for_，
        后台对所有图片都做对齐 + PatchCore + PIN，
        但只返回“一张用于回显”的结果（当前策略：第 0 张）。
        """
        # 本次调用最终采用的预处理开关
        use_preprocess: bool = self.enable_preprocess if do_preprocess is None else bool(do_preprocess)

        # -------- 1) 规范化文件名列表 --------
        if isinstance(name_list, (list, tuple)):
            filenames: List[str] = list(name_list)
        else:
            filenames = [name_list]

        # -------- 2) 规范化图像列表 --------
        if image_raw_bgr is None:
            # 按 filename 列表自己读图
            imgs: List[np.ndarray] = []
            for f in filenames:
                img = cv2.imread(f, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"OpenCV 读图失败: {f}")
                imgs.append(img)
        else:
            if isinstance(image_raw_bgr, (list, tuple)):
                imgs = list(image_raw_bgr)
            else:
                imgs = [image_raw_bgr]

        # -------- 3) 调用批量接口，但只拿一张给 UI --------
        label, overlay, score, defect_count = self.service._detect_batch(
            filenames,
            imgs,
            do_preprocess=use_preprocess,
        )

        return label, overlay, score, defect_count