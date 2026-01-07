from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_EXCEPTION, ALL_COMPLETED
from typing import Any, Callable, Dict, List, Set, Optional, Tuple

Handler = Callable[..., Any]


class EventBus:
    _class_lock = threading.RLock()
    _class_subs: Dict[str, set[Handler]] = defaultdict(set)

    @classmethod
    def on(cls, topic: str):
        def deco(fn: Handler):
            with cls._class_lock:
                cls._class_subs[topic].add(fn)
            return fn
        return deco

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "evt",
        logger: Optional[Any] = None,
    ) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._logger = logger

        # 显式优先级：外部传了 error_hook 就用外部的；否则如果传了 logger，就内部生成一个
        if logger is not None:
            self._error_hook = self._make_error_hook(logger)
        else:
            self._error_hook = None

    def _make_error_hook(self, logger: Any):
        def hook(topic: str, handler: Handler, exc: BaseException) -> None:
            hname = getattr(handler, "__qualname__", getattr(handler, "__name__", repr(handler)))
            # 你的 Logger 包装器支持 .error(..., exc_info=True) 或 exc_info=(type, e, tb) 都行
            logger.error("Event handler failed | topic=%s | handler=%s", topic, hname, exc_info=exc)
        return hook

    def _safe_call(self, topic: str, h: Handler, *args, **kwargs) -> Any:
        try:
            return h(*args, **kwargs)
        except BaseException as e:
            if self._error_hook:
                try:
                    self._error_hook(topic, h, e)
                except BaseException:
                    pass
            raise

    def publish(self, topic: str, *args, **kwargs) -> None:
        with type(self)._class_lock:
            handlers = list(type(self)._class_subs.get(topic, ()))
        for h in handlers:
            self._safe_call(topic, h, *args, **kwargs)

    def publish_async(self, topic: str, *args, **kwargs) -> List[Future]:
        with type(self)._class_lock:
            handlers = list(type(self)._class_subs.get(topic, ()))
        # 关键：异步也用 _safe_call 包一层，确保异常被统一记录
        return [self._pool.submit(self._safe_call, topic, h, *args, **kwargs) for h in handlers]

    def publish_async_wait(
        self,
        topic: str,
        *args,
        timeout: Optional[float] = None,
        return_when: str = "FIRST_EXCEPTION",
        **kwargs
    ) -> Tuple[Set[Future], Set[Future]]:
        futs = self.publish_async(topic, *args, **kwargs)
        mode = FIRST_EXCEPTION if return_when == "FIRST_EXCEPTION" else ALL_COMPLETED
        return wait(futs, timeout=timeout, return_when=mode)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=False)
