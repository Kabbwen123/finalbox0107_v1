from __future__ import annotations

import time
import atexit
import threading
from typing import Optional, Tuple, Dict

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

class SharedFrame:
    """
    这是“替代 np.ndarray”的对象（你要传给算法接口的对象）。

    你在算法进程里用：
        with frame_obj as img:
            ... img 是零拷贝 ndarray 视图 ...
        # 退出 with 自动 ack，归还 slot

    注意：这里的“地址”是 (shm_name, offset)，不是进程内指针。
    """

    # 算法进程里“第一次用到某个 shm_name 才打开一次，之后复用”
    _shm_cache: Dict[str, SharedMemory] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _get_shm(cls, shm_name: str) -> SharedMemory:
        with cls._cache_lock:
            shm = cls._shm_cache.get(shm_name)
            if shm is None:
                shm = SharedMemory(name=shm_name, create=False)  # 只会在第一次发生
                cls._shm_cache[shm_name] = shm
            return shm

    @classmethod
    def _close_all_cached_shm(cls) -> None:
        with cls._cache_lock:
            for shm in cls._shm_cache.values():
                try:
                    shm.close()
                except Exception:
                    pass
            cls._shm_cache.clear()


    def __init__(
        self,
        shm_name: str,
        offset: int,
        nbytes: int,
        shape: Tuple[int, ...],
        dtype: str,
        frame_id: int,
        ts_ns: int,
        ack_sem: mp.Semaphore,
    ):
        self.shm_name = str(shm_name)
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.shape = tuple(shape)
        self.dtype = str(dtype)
        self.frame_id = int(frame_id)
        self.ts_ns = int(ts_ns)

        self._ack_sem = ack_sem
        self._acked = False
        self._view: Optional[np.ndarray] = None  # 缓存 ndarray 视图（同一对象多次用时不重复构造）

    @property
    def address(self) -> Tuple[str, int]:
        """跨进程可用的“地址/句柄”。"""
        return (self.shm_name, self.offset)

    def view(self) -> np.ndarray:
        """得到零拷贝 ndarray 视图（算法进程调用）。"""
        if self._view is not None:
            return self._view

        shm = SharedFrame._get_shm(self.shm_name)  # 第一次才会打开
        buf = shm.buf[self.offset : self.offset + self.nbytes]
        arr = np.frombuffer(buf, dtype=np.dtype(self.dtype)).reshape(self.shape)
        self._view = arr
        return arr

    def ack(self) -> None:
        """
        归还 slot（让相机侧可复用该内存区）。
        幂等：重复调用不会多释放。
        """
        if not self._acked:
            self._acked = True
            self._ack_sem.release()

    def __enter__(self) -> np.ndarray:
        return self.view()

    def __exit__(self, exc_type, exc, tb) -> None:
        # 单消费者串行：with 结束就 ack，最符合你的需求
        self.ack()


# 进程退出时，自动关闭算法进程中缓存的 shm 映射
atexit.register(SharedFrame._close_all_cached_shm)
