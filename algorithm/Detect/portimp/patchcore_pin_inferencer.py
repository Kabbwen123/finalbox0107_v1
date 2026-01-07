import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import time
import cv2
import numpy as np

from portimp.patchcore_inference_interface import PatchcoreInferencer, overlay_heatmap_on_image
from Infrastructure.pindetector.IRISO_PinDetect import detect_pin                     
from Infrastructure.utils.crop_roi import crop_roi                     
# ==================================================


class PatchcorePinInferencer:
    '''
      - 无监督 PatchCore 整图推理（整张原图）
      - ROI 内针脚歪斜检测（在裁剪后的 ROI 上）
    '''
    def __init__(
        self,
        model_feature_dir: str,
        pc_score_thr: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
        # pin 检测参数
        pincount: int = 35,
        offset_up: int = 40,
        offset_down: int = 60,
        # PatchCore 其它参数
        use_faiss: bool = True,
        use_faiss_gpu: bool = True,
        faiss_gpu_device: int = 0,
        alpha: float = 0.5,
        beta: float = 0.5,
        hot_only: bool = True,
        hot_thresh: float = 0.05,
        abs_low: float = 80.0,
        abs_high: float = 100.0,
        pinsizethreshold: int = 500,
        errsizethreshold: int = 30,
    ):
        self.roi = roi
        self.pc_score_thr = pc_score_thr

        # pin 检测参数
        self.pincount = pincount
        self.offset_up = offset_up
        self.offset_down = offset_down
        self.pinsizethreshold = pinsizethreshold
        self.errsizethreshold = errsizethreshold

        # 记住热力图可视化的配置
        self.alpha = alpha
        self.beta = beta
        self.hot_only = hot_only
        self.hot_thresh = hot_thresh

        # 1) 初始化 PatchCore 推理器
        self.pc = PatchcoreInferencer(
            model_feature_dir=model_feature_dir,
            use_faiss_gpu=use_faiss_gpu,
            faiss_gpu_device=faiss_gpu_device,
            alpha=alpha,
            beta=beta,
            hot_only=hot_only,
            hot_thresh=hot_thresh,
            abs_low=abs_low,
            abs_high=abs_high,
        )
    # ------------------------------------------------------------------
    # 对单张 BGR 图像做两个任务
    # ------------------------------------------------------------------
    def infer_bgr(self, img_bgr: np.ndarray, filename: str = "") -> Dict[str, Any]:
        """
        输入单张 BGR 图像（原始整图），
        返回一个结果字典，包含：
            - PatchCore 结果
            - pin 检测结果（ROI 内）
        """

        h, w = img_bgr.shape[:2]

        # ========= 1) PatchCore：整图无监督推理 =========
        pc_result = self.pc.infer_bgr(img_bgr)
        pc_score = float(pc_result["image_score_raw"])
        pc_is_ng = pc_score > self.pc_score_thr
        pc_overlay = pc_result["overlay_bgr"]      # 与原图同尺寸

        # ========= 2) ROI + 针脚检测 =========
        pin_err_boxes_roi: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pin_err_boxes_global: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pin_has_error = False
        pin_bin_vis = None      # 针脚二值+框的可视化图（ROI 尺寸）
        pin_overlay_global = img_bgr.copy()  # 在整图上画 pin NG 框

        if self.roi is not None:
            x1, y1, x2, y2 = self.roi

            # 安全边界（防止 ROI 超出）
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                # 2.1 截取 ROI
                roi_bgr = crop_roi(img_bgr, (x1, y1, x2, y2))
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

                # 2.2 在 ROI 上做针脚检测（注意：detect_pin 内部还有一层上下裁剪）
                pin_bin_vis, err_boxes_roi = detect_pin(
                    roi_gray,
                    pincount=self.pincount,
                    offset_up=self.offset_up,
                    offset_down=self.offset_down,
                    pinsizethreshold=self.pinsizethreshold,
                )

                # err_boxes_roi: 在 ROI 坐标系下的矩形 [( (lx, ty), (rx, by) ), ...]
                pin_err_boxes_roi = err_boxes_roi
                pin_has_error = len(err_boxes_roi) > 0

                # 2.3 把 ROI 内的坐标，映射回整图坐标系
                for (pt1, pt2) in err_boxes_roi:
                    (lx, ty) = pt1
                    (rx, by) = pt2
                    gx1 = x1 + lx
                    gx2 = x1 + rx
                    gy1 = y1 + ty
                    gy2 = y1 + by
                    pin_err_boxes_global.append(((gx1, gy1), (gx2, gy2)))

                    # 在整图上画红框
                    cv2.rectangle(pin_overlay_global, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            else:
                print("[WARN] ROI 配置无效，跳过针脚检测。")
        else:
            print("[INFO] 未配置 ROI，不进行针脚检测。")
            
        # ========= 3) 汇总结果 =========
        # 或门：只要任一模块判定为 NG，则整张图判定为 NG
        final_is_ng = bool(pc_is_ng or pin_has_error)
        final_status = "NG" if final_is_ng else "OK"
        
        return {
            "filename": filename,

            # 总体结果（或门）
            "final_is_ng": final_is_ng,        # True/False
            "final_status": final_status,      # "NG" / "OK"

            # PatchCore 相关
            "pc_image_score": pc_score,
            "pc_is_ng": pc_is_ng,
            "pc_overlay": pc_overlay,

            # pin 相关
            "pin_has_error": pin_has_error,
            "pin_err_boxes_roi": pin_err_boxes_roi,       # ROI 内坐标
            "pin_err_boxes_global": pin_err_boxes_global, # 整图坐标
            "pin_bin_vis": pin_bin_vis,                   # ROI 二值+框可视化
            "pin_overlay_global": pin_overlay_global,     # 整图上画红框的可视化
        }

    # ------------------------------------------------------------------
    # 对单张图片路径
    # ------------------------------------------------------------------
    def infer_path(
        self,
        img_path: str,
        save_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对单张图片路径进行推理。
        如果提供 save_root，则自动保存几个可视化结果：
          - PatchCore 热力图叠加图  (pc_overlay)
          - 针脚 ROI 二值可视化     (pin_bin_vis)
          - 整图 + 针脚 NG 红框图   (pin_overlay_global)
        """
        img_path = str(img_path)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        
        filename = os.path.basename(img_path)
        stem, ext = os.path.splitext(filename)
        result = self.infer_bgr(img_bgr, filename=filename)

        if save_root is not None:
            save_root_path = Path(save_root)
            (save_root_path / "pc_overlay").mkdir(parents=True, exist_ok=True)
            (save_root_path / "pin_bin").mkdir(parents=True, exist_ok=True)
            (save_root_path / "pin_overlay_global").mkdir(parents=True, exist_ok=True)

            # 保存 PatchCore 热力图叠加
            pc_overlay_path = save_root_path / "pc_overlay" / f"{stem}_pc_overlay.jpg"
            cv2.imwrite(str(pc_overlay_path), result["pc_overlay"])   # 可选

            # 保存 ROI 内针脚检测可视化
            if result["pin_bin_vis"] is not None:
                pin_bin_path = save_root_path / "pin_bin" / f"{stem}_pin_bin.jpg"
                cv2.imwrite(str(pin_bin_path), result["pin_bin_vis"])   # 可选

            # 保存整图 + 针脚红框
            pin_overlay_global_path = save_root_path / "pin_overlay_global" / f"{stem}_pin_overlay.jpg"
            cv2.imwrite(str(pin_overlay_global_path), result["pin_overlay_global"]) # 可选

            result["pc_overlay_path"] = str(pc_overlay_path)
            if result["pin_bin_vis"] is not None:
                result["pin_bin_path"] = str(pin_bin_path)
            result["pin_overlay_global_path"] = str(pin_overlay_global_path)

        return result


    # ------------------------------------------------------------------
    # 对齐图 + 预处理图 双输入推理
    #   - pre_bgr 用于 PatchCore
    #   - aligned_bgr 用于 PIN 检测（ROI 坐标基于这张图）
    # ------------------------------------------------------------------
    def infer_with_pre_img(
        self,
        aligned_bgr: np.ndarray,
        pre_bgr: np.ndarray,
        filename: str = "",
    ) -> Dict[str, Any]:
        """
        aligned_bgr : 对齐后的整图（用于最终回显：原图+热力图+PIN框）
        pre_bgr     : 对齐+预处理后的图（用于 PatchCore 无监督推理 + PIN 检测）
        """
        # ======== 计时容器 ========
        t_pc_ms: float = 0.0
        t_pin_ms: float = 0.0
        
        # ======== 1) PatchCore：用 pre_bgr 做无监督推理 ========
        _t0 = time.perf_counter()
        pc_result = self.pc.infer_bgr(pre_bgr)
        t_pc_ms = (time.perf_counter() - _t0) * 1000.0
        pc_score = float(pc_result["image_score_raw"])
        pc_is_ng = pc_score > self.pc_score_thr

        # amap_norm：归一化后的特征图（Hf×Wf, 0~1）
        amap_norm = pc_result.get("amap_norm", None)
        pc_overlay = pc_result.get("overlay_bgr", None)  # 如果你想单独保存，也可以用
        print(pc_overlay.shape)
        
        # ======== 2) PIN 检测：在 pre_bgr 上做 ROI + 针脚检测 ========
        img_bgr_for_pin = pre_bgr
        h, w = img_bgr_for_pin.shape[:2]
        pin_err_boxes_roi: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pin_err_boxes_global: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pin_has_error = False
        pin_bin_vis = None
        pin_overlay_global = img_bgr_for_pin.copy()

        if self.roi is not None:
            x1, y1, x2, y2 = self.roi

            # 安全边界
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                # 2.1 截取 ROI
                roi_bgr = crop_roi(img_bgr_for_pin, (x1, y1, x2, y2))
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

                # 2.2 针脚检测
                _t1 = time.perf_counter()
                pin_bin_vis, err_boxes_roi = detect_pin(
                    roi_gray,
                    pincount=self.pincount,
                    offset_up=self.offset_up,
                    offset_down=self.offset_down,
                    pinsizethreshold=self.pinsizethreshold,
                )
                t_pin_ms = (time.perf_counter() - _t1) * 1000.0

                pin_err_boxes_roi = err_boxes_roi
                pin_has_error = len(err_boxes_roi) > 0

                # 2.3 ROI 坐标 → 整图坐标（注意：这里的整图坐标基于 pre_bgr）
                for (pt1, pt2) in err_boxes_roi:
                    (lx, ty) = pt1
                    (rx, by) = pt2
                    gx1 = x1 + lx
                    gx2 = x1 + rx
                    gy1 = y1 + ty
                    gy2 = y1 + by
                    pin_err_boxes_global.append(((gx1, gy1), (gx2, gy2)))

                    cv2.rectangle(pin_overlay_global, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            else:
                print("[WARN] ROI 配置无效，跳过针脚检测。")
        else:
            print("[INFO] 未配置 ROI，不进行针脚检测。")

        # ======== 3) 或门融合 ========
        final_is_ng = bool(pc_is_ng or pin_has_error)
        final_status = "NG" if final_is_ng else "OK"

        # ======== 4) 在“对齐后的原图 aligned_bgr 上”叠加热力图 ========
        # 关键改动：不再用 pc_overlay 再次 addWeighted，
        # 而是用 amap_norm + overlay_heatmap_on_image(aligned_bgr) 做“一次性叠加”
        final_overlay = aligned_bgr.copy()
        if amap_norm is not None:
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            aligned_pil = Image.fromarray(aligned_rgb)

            # 这里用的 alpha/beta/hot_only/hot_thresh 和训练时保持一致
            heat_overlay_bgr = overlay_heatmap_on_image(
                aligned_pil,
                amap_norm,
                alpha=self.alpha,
                beta=self.beta,
                hot_only=self.hot_only,
                hot_thresh=self.hot_thresh,
            )
            final_overlay = heat_overlay_bgr

        # ======== 5) 在 final_overlay 上画 PIN 框（把 pre_bgr 坐标映射到 aligned_bgr） ========
        pre_h, pre_w = pre_bgr.shape[:2]
        ali_h, ali_w = aligned_bgr.shape[:2]

        # 避免除 0
        if pre_w <= 0 or pre_h <= 0:
            print("[WARN] pre_bgr shape invalid, skip drawing pin boxes.")
        else:
            sx = ali_w / float(pre_w)
            sy = ali_h / float(pre_h)

            def clamp(v, lo, hi):
                return max(lo, min(int(v), hi))

            def map_pt_pre_to_aligned(pt):
                x, y = pt
                ax = int(round(x * sx))
                ay = int(round(y * sy))
                ax = clamp(ax, 0, ali_w - 1)
                ay = clamp(ay, 0, ali_h - 1)
                return (ax, ay)

            for (pt1, pt2) in pin_err_boxes_global:
                a1 = map_pt_pre_to_aligned(pt1)
                a2 = map_pt_pre_to_aligned(pt2)
                cv2.rectangle(final_overlay, a1, a2, (0, 0, 255), 2)
        # ======== 6) 汇总输出 ========
        return {
            "filename": filename,

            # 总体结果
            "final_is_ng": final_is_ng,
            "final_status": final_status,

            # 耗时（ms）
            "t_patchcore_ms": float(t_pc_ms),
            "t_pin_ms": float(t_pin_ms),
            
            # PatchCore
            "pc_image_score": pc_score,
            "pc_is_ng": pc_is_ng,
            "pc_overlay": pc_overlay,   # 如果想单独看 PatchCore 原始 overlay 也保留

            # PIN
            "pin_has_error": pin_has_error,
            "pin_err_boxes_roi": pin_err_boxes_roi,
            "pin_err_boxes_global": pin_err_boxes_global,
            "pin_bin_vis": pin_bin_vis,
            "pin_overlay_global": pin_overlay_global,

            # 最终：在“对齐原图”上叠加了热力图 + PIN 框的可视化
            "final_overlay": final_overlay,
        }


# ======================= 简易测试入口 =======================
if __name__ == "__main__":
    # 根据你自己的路径修改这里
    MODEL_FEATURE_DIR = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\OUTPUT\result\1225_test\20251225_145532__memory_bank__resnet34_L2_3_in256x640_ps3_k5_M20kidxFlat_\training_OK"
    TEST_IMG = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\1216_iriso_pic\NG_test\08190071_1_alt.jpg"
    SAVE_ROOT = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\1216_iriso_pic\NG_test_output111"

    ROI = (154, 20, 1364, 120)  # 用你第三个脚本里的 ROI

    aibox = PatchcorePinInferencer(
        model_feature_dir=MODEL_FEATURE_DIR,
        pc_score_thr=100.0,
        roi=ROI,
        pincount=35,
        offset_up=80,
        offset_down=100,
    )
    

    result = aibox.infer_path(TEST_IMG, save_root=SAVE_ROOT)
    print("filename:", result["filename"])
    print("FINAL status:", result["final_status"])   # 这里已经是或门后的结论
    print("    pc_is_ng:", result["pc_is_ng"], "score:", result["pc_image_score"])
    print("    pin_has_error:", result["pin_has_error"])
    print("    pin_err_boxes_global:", result["pin_err_boxes_global"])
