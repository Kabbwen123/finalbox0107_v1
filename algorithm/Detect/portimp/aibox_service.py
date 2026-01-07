# aibox_service.py
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import time
from Domain.align_preprocess_interface import AlignPreprocessor
from portimp.patchcore_pin_inferencer import PatchcorePinInferencer
import Infrastructure.align_preprocess.GS_CV_Lib as gscv


class PatchcorePinService:
    """
    封装 AlignPreprocessor + AIBoxInterface
    对外提供：
        - detect_batch(filenames, raw_images_bgr)
        - detect_one(filename, image_raw_bgr=None)
    """

    def __init__(
        self,
        aligner: AlignPreprocessor,         # 对齐+预处理
        inferer: PatchcorePinInferencer,            # PatchCore+PIN 推理
        max_images_per_call: Optional[int] = None,
    ):
        self.aligner = aligner
        self.inferer = inferer
        self.max_images_per_call = max_images_per_call

    def detect_batch(
        self,
        filenames: List[str],
        raw_images_bgr: List[np.ndarray],
        do_preprocess: bool = True,
    ) -> List[Tuple[int, np.ndarray, float, int]]:
        """
        返回列表元素：
            (label, overlay_bgr, score, defect_count)
        """
        if len(filenames) != len(raw_images_bgr):
            raise ValueError(
                f"filenames 个数({len(filenames)}) 与 raw_images_bgr 个数({len(raw_images_bgr)}) 不一致！"
            )
        if len(filenames) == 0:
            return []

        max_n = self.max_images_per_call
        if max_n is None or max_n <= 0:
            max_n = len(filenames)

        results: List[Tuple[int, np.ndarray, float, int]] = []

        for idx, (filename, raw_img) in enumerate(zip(filenames, raw_images_bgr)):
            if idx >= max_n:
                break

            if raw_img is None or raw_img.size == 0:
                raise RuntimeError(f"原图为空: {filename}")

            base_name = Path(filename).name

            # 1) 对齐 + (可选)预处理
            t0 = time.perf_counter()
            ap_out = self.aligner.process_bgr(raw_img, do_preprocess=do_preprocess)
            t_align_pre_ms = (time.perf_counter() - t0) * 1000.0

            aligned_bgr = ap_out["aligned_bgr"]
            pre_bgr = ap_out["preprocessed_bgr_for_model"]

            if aligned_bgr is None or pre_bgr is None:
                label = 1
                overlay = raw_img.copy()
                score = -1.0
                defect_count = -1
                print(f"[TIME] {base_name} align+pre={t_align_pre_ms:.2f}ms (align failed)")
                results.append((label, overlay, score, defect_count))
                continue
            
            # test
            # pre_bgr = gscv.image_resize(pre_bgr, scale=0.25)

            # 2) 推理
            t_infer0 = time.perf_counter()
            res = self.inferer.infer_with_pre_img(
                aligned_bgr=aligned_bgr,
                pre_bgr=pre_bgr,
                filename=base_name,
            )
            t_infer_ms = (time.perf_counter() - t_infer0) * 1000.0

            # 可选：打印分项耗时（inferer 内部如果有提供）
            t_pc_ms = float(res.get("t_patchcore_ms", 0.0))
            t_pin_ms = float(res.get("t_pin_ms", 0.0))
            print(
                f"[TIME] {base_name} align+pre={t_align_pre_ms:.2f}ms "
                f"pc={t_pc_ms:.2f}ms pin={t_pin_ms:.2f}ms infer_total={t_infer_ms:.2f}ms "
                f"(preprocess={'ON' if do_preprocess else 'OFF'})"
            )

            # 3) 输出统一格式
            label = 1 if res["final_status"] == "NG" else 0
            score = float(res["pc_image_score"])
            defect_count = len(res["pin_err_boxes_global"])
            overlay = res.get("final_overlay")

            results.append((label, overlay, score, defect_count))

        return results
    # ---------------- 批量输入，但只返回一张 ----------------
    def _detect_batch(
        self,
        filenames: List[str],
        raw_images_bgr: List[np.ndarray],
        do_preprocess: bool = None,
    ) -> Tuple[int, np.ndarray, float, int]:
        """
        输入：同样是多个 filename + 多张 BGR 图的列表。
        后台行为：
        内部先调用 detect_batch，对所有图片都做完推理；
        输出：就一条 (label, overlay, score, defect_count)，给 UI 用来回显。
        """
        all_results = self.detect_batch(filenames, raw_images_bgr, do_preprocess=do_preprocess)

        if not all_results:
            raise RuntimeError("detect_batch_for_ui: 输入列表为空，无法返回回显结果！")

        # 简单策略：直接取第 0 张
        return all_results[0]