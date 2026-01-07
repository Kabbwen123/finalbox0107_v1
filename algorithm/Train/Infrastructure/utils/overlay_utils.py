# overlay_utils.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

Box = List[Tuple[int, int]]  # [(x1,y1),(x2,y2)]

def map_boxes_pre_to_aligned(
    boxes_pre: List[Box],
    roi_xywh: Tuple[int, int, int, int],
    pre_size: Tuple[int, int],
) -> List[Box]:
    """把 pre_img 坐标系的框，映射到 aligned 坐标系"""
    rx, ry, rw, rh = roi_xywh
    pre_w, pre_h = pre_size

    sx = rw / max(1, pre_w)
    sy = rh / max(1, pre_h)

    boxes_aligned: List[Box] = []
    for (x1, y1), (x2, y2) in boxes_pre:
        ax1 = int(rx + x1 * sx)
        ay1 = int(ry + y1 * sy)
        ax2 = int(rx + x2 * sx)
        ay2 = int(ry + y2 * sy)
        boxes_aligned.append([(ax1, ay1), (ax2, ay2)])
    return boxes_aligned


def draw_boxes(bgr: np.ndarray, boxes: List[Box], thickness: int = 2) -> np.ndarray:
    out = bgr.copy()
    for (x1, y1), (x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    return out


def blend_heatmap_on_aligned(
    aligned_bgr: np.ndarray,
    heatmap_float: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    heatmap_float: PatchCore 输出的 anomaly map（任意尺寸，float，越大越异常）
    会被 resize 到 ROI 尺寸，然后贴到 aligned_bgr 的 ROI 上
    """
    rx, ry, rw, rh = roi_xywh
    H, W = aligned_bgr.shape[:2]

    # 防越界裁剪
    rx2 = min(W, rx + rw)
    ry2 = min(H, ry + rh)
    rw2 = max(0, rx2 - rx)
    rh2 = max(0, ry2 - ry)
    if rw2 == 0 or rh2 == 0:
        return aligned_bgr

    # 归一化到 0~255
    hm = heatmap_float.astype(np.float32)
    hm = hm - hm.min()
    denom = (hm.max() + 1e-6)
    hm = hm / denom
    hm_u8 = (hm * 255.0).clip(0, 255).astype(np.uint8)

    hm_u8 = cv2.resize(hm_u8, (rw2, rh2), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, colormap)

    out = aligned_bgr.copy()
    roi = out[ry:ry2, rx:rx2]
    out[ry:ry2, rx:rx2] = cv2.addWeighted(roi, 1.0 - alpha, hm_color, alpha, 0)
    return out
