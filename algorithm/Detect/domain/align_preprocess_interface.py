# align_preprocess_interface.py
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import Infrastructure.align_preprocess.GS_CV_Lib as gscv
import Infrastructure.align_preprocess.GS_CV_Match as gsmatch
from Infrastructure.align_preprocess.IRISO_Step3_Preprocess import imagepro_for_AIdetect


class AlignPreprocessor:
    def __init__(
        self,
        template_path: str,
        mask_path: str,
        preprocess_image_size: Tuple[int, int],
        scale=0.5,
        match_method="ORB",
        preprocess_roi=None,
        preprocess_contrast=1.2,
        preprocess_blur=False,
    ):
        self.template_path = template_path
        self.mask_path = mask_path
        self.preprocess_image_size = preprocess_image_size
        self.scale = scale
        self.preprocess_roi = preprocess_roi
        self.preprocess_contrast = preprocess_contrast
        self.preprocess_blur = preprocess_blur

        # ========= 1. 读取模板 + 掩码，并缩放 =========
        tmpl_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if tmpl_gray is None or mask_gray is None:
            raise FileNotFoundError(f"模板或掩码读取失败: {template_path}, {mask_path}")

        if scale != 1.0:
            tmpl_gray = gscv.image_resize(tmpl_gray, scale)
            mask_gray = gscv.image_resize(mask_gray, scale)

        self.template_gray = tmpl_gray
        self.mask_gray = mask_gray
        self.tmpl_h, self.tmpl_w = tmpl_gray.shape[:2]

        # ========= 2. 读取 Step3 预处理模板（训练时用的 template.jpg） =========
        pw, ph = int(preprocess_image_size[0]), int(preprocess_image_size[1])
        if pw <= 0 or ph <= 0:
            raise ValueError(f"preprocess_image_size 非法: {preprocess_image_size}，应为 (W,H) 且均为正数")
        self.preprocess_image_size = (pw, ph)
        
        # ========= 3. 初始化匹配器 =========
        self.matcher = gsmatch.TemplateMatcher(method=match_method)

    # ------------------------------------------------------------------
    # 1) 对齐接口：只做 alignment
    # ------------------------------------------------------------------
    # def align_bgr(self, image_bgr):
    def align_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:

        """
        输入原始 BGR 图像，输出：
          - 对齐到模板大小的 BGR 图像（已按 mask 填充背景）
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")

        # 按同样的 scale 缩放
        if self.scale != 1.0:
            image_bgr_small = gscv.image_resize(image_bgr, self.scale)
        else:
            image_bgr_small = image_bgr.copy()

        img_gray = cv2.cvtColor(image_bgr_small, cv2.COLOR_BGR2GRAY)

        # TemplateMatcher.match 得到仿射矩阵 M
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

    # ------------------------------------------------------------------
    # 2) 预处理接口：只处理“对齐后的图”
    # ------------------------------------------------------------------
    def preprocess_aligned_bgr(
        self,
        aligned_bgr,
        do_preprocess: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        输入：aligned_bgr（已经对齐好的图）
        输出：
          {
            "preprocessed_gray": ...,
            "preprocessed_bgr_for_model": ...,
          }
        """
        if aligned_bgr is None or aligned_bgr.size == 0:
            return {
                "preprocessed_gray": None,
                "preprocessed_bgr_for_model": None,
            }

        # None -> 默认启用预处理（你也可以改成 self.enable_preprocess）
        use_preprocess = True if do_preprocess is None else bool(do_preprocess)

        if use_preprocess:
            pre_gray = imagepro_for_AIdetect(
                aligned_bgr,
                image_size=self.preprocess_image_size,
                blur=self.preprocess_blur,
                contrast=self.preprocess_contrast,
                ROI=self.preprocess_roi,
            )
        else:
            # 关闭 Step3：仍保持 ROI + resize 行为，避免切换开关导致 ROI 坐标系变化
            src = aligned_bgr
            if self.preprocess_roi is not None:
                x, y, w, h = self.preprocess_roi
                src = src[y:y + h, x:x + w]

            pre_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            target_w, target_h = self.preprocess_image_size
            if (pre_gray.shape[1], pre_gray.shape[0]) != (target_w, target_h):
                pre_gray = cv2.resize(pre_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        pre_bgr = cv2.cvtColor(pre_gray, cv2.COLOR_GRAY2BGR)

        return {
            "preprocessed_gray": pre_gray,
            "preprocessed_bgr_for_model": pre_bgr,
        }

    # ------------------------------------------------------------------
    # 3) 组合接口：对齐 +（可选）预处理
    # ------------------------------------------------------------------
    def process_bgr(self, image_bgr, do_preprocess: Optional[bool] = None):
        """
        输入原始 BGR 图像：
          - 先对齐
          - 再（可选）预处理

        返回：
          {
            "aligned_bgr": ...,
            "preprocessed_gray": ...,
            "preprocessed_bgr_for_model": ...
          }
        """
        aligned_bgr = self.align_bgr(image_bgr)
        if aligned_bgr is None:
            return {
                "aligned_bgr": None,
                "preprocessed_gray": None,
                "preprocessed_bgr_for_model": None,
            }

        pre_out = self.preprocess_aligned_bgr(aligned_bgr, do_preprocess=do_preprocess)

        return {
            "aligned_bgr": aligned_bgr,
            "preprocessed_gray": pre_out["preprocessed_gray"],
            "preprocessed_bgr_for_model": pre_out["preprocessed_bgr_for_model"],
        }

    # ------------------------------------------------------------------
    # 4) 路径接口：支持 UI 透传 do_preprocess
    # ------------------------------------------------------------------
    def process_path(self, image_path: str, do_preprocess: Optional[bool] = None):
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        return self.process_bgr(img_bgr, do_preprocess=do_preprocess)


