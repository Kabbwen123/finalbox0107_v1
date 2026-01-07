from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from common.infrastructure.ShareFrame import SharedFrame


@runtime_checkable
class FrameSourceService(Protocol):
    """
    依赖倒置：图像采集只依赖这个抽象接口。
    你可以实现更多数据源（网络流、共享内存、ROS 等）而不改采集逻辑。
    """

    def open(self) -> None: ...

    def read(self) -> Optional[np.ndarray]: ...

    def close(self) -> None: ...


class ImageAcquirer:
    """
    - 注入 FrameSourceService
    - grab(): 从 service.read() 得到 np.ndarray -> 写共享内存 -> return SharedFrame
    """

    def __init__(
            self,
            service: FrameSourceService,
            num_slots: int,
            expected_shape: Optional[Tuple[int, ...]] = None,
            expected_dtype: np.dtype | str = np.uint8,
            on_full: str = "block",  # "block" | "drop"
    ):
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if on_full not in ("block", "drop"):
            raise ValueError("on_full must be 'block' or 'drop'")

        self._service = service
        self._num_slots = int(num_slots)
        self._expected_shape = tuple(expected_shape) if expected_shape is not None else None
        self._expected_dtype = np.dtype(expected_dtype)
        self._on_full = on_full

        # 共享内存池（首次拿到帧后才初始化，避免你 init 时必须给 shape）
        self._shm: Optional[SharedMemory] = None
        self._frame_bytes: Optional[int] = None
        self._slot_bytes: Optional[int] = None
        self._free_sem: Optional[mp.Semaphore] = None

        self._w = 0
        self._frame_id = 0

    def _init_pool_if_needed(self, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        if self._shm is not None:
            return

        if self._expected_shape is not None and shape != self._expected_shape:
            raise ValueError(f"shape mismatch: expected {self._expected_shape}, got {shape}")
        if dtype != self._expected_dtype:
            raise ValueError(f"dtype mismatch: expected {self._expected_dtype}, got {dtype}")

        frame_bytes = int(np.prod(shape) * dtype.itemsize)
        slot_bytes = frame_bytes
        total_bytes = slot_bytes * self._num_slots

        self._shm = SharedMemory(create=True, size=total_bytes)
        self._frame_bytes = frame_bytes
        self._slot_bytes = slot_bytes
        self._free_sem = mp.Semaphore(self._num_slots)

        # 固化期望格式（后续帧必须一致）
        self._expected_shape = shape
        self._expected_dtype = dtype

    @property
    def shm_name(self) -> str:
        if self._shm is None:
            raise RuntimeError("pool not initialized yet (no frame grabbed).")
        return self._shm.name

    def close(self) -> None:
        try:
            self._service.close()
        finally:
            if self._shm is not None:
                try:
                    self._shm.close()
                finally:
                    try:
                        self._shm.unlink()
                    except FileNotFoundError:
                        pass
                self._shm = None

    def _acquire_slot(self) -> bool:
        assert self._free_sem is not None
        if self._on_full == "block":
            self._free_sem.acquire()
            return True
        return self._free_sem.acquire(block=False)

    def grab(self) -> Optional[SharedFrame]:
        """
        调用一次 = 尝试取一帧并返回 SharedFrame。
        - service.read() 返回 None：表示暂时无帧/结束
        - on_full='drop' 且池满：返回 None
        """
        frame = self._service.read()
        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            raise TypeError("service.read() must return numpy.ndarray or None")

        if not frame.flags["C_CONTIGUOUS"]:
            # 明确：这里不自动 ascontiguousarray，避免无意 copy
            raise ValueError("frame must be C-contiguous (service should provide contiguous arrays)")

        shape = tuple(frame.shape)
        dtype = np.dtype(frame.dtype)
        self._init_pool_if_needed(shape, dtype)

        assert self._shm is not None and self._frame_bytes is not None and self._slot_bytes is not None and self._free_sem is not None

        if shape != self._expected_shape or dtype != self._expected_dtype:
            raise ValueError(
                f"frame format changed: expected {self._expected_shape}/{self._expected_dtype}, got {shape}/{dtype}")

        if not self._acquire_slot():
            return None

        slot = self._w
        offset = slot * self._slot_bytes

        # 写共享内存（一次 memcpy）
        self._shm.buf[offset: offset + self._frame_bytes] = frame.view(np.uint8).ravel()

        self._frame_id += 1
        fid = self._frame_id
        ts = time.time_ns()

        self._w = (self._w + 1) % self._num_slots

        return SharedFrame(
            shm_name=self._shm.name,
            offset=offset,
            nbytes=self._frame_bytes,
            shape=self._expected_shape,  # type: ignore[arg-type]
            dtype=str(self._expected_dtype),
            frame_id=fid,
            ts_ns=ts,
            ack_sem=self._free_sem,
        )
