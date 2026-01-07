from __future__ import annotations

from typing import Optional

import numpy as np


class CameraSourceService:
    """
    基于 OpenCV 的相机数据源 service。
    你也可以把内部替换成你自己的相机 SDK。
    """

    def __init__(
            self,
            camera_index: int = 0,
            width: Optional[int] = None,
            height: Optional[int] = None,
    ):
        self._camera_index = int(camera_index)
        self._width = width
        self._height = height
        self._cap = None

    def open(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("CameraSourceService requires opencv-python (cv2).") from e

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={self._camera_index}")

        if self._width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._width))
        if self._height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._height))

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("CameraSourceService not opened.")
        ok, frame = self._cap.read()
        if not ok:
            return None
        # OpenCV 典型返回 BGR uint8 contiguous
        return frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
