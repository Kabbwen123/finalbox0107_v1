# commandbus.py
from __future__ import annotations

import logging
import os
import queue
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Literal

ExecutorKind = Literal["inline", "thread", "process"]
ExclusiveKeyFn = Callable[[Any], Optional[str]]


@dataclass(frozen=True, slots=True)
class ExecPolicy:
    """
    executor:
        - inline: 直接在调用线程中执行（handle_future 会包装到线程池返回 Future）
        - thread: 线程池执行
        - process: 进程池执行（命名命令通过 kwargs 顶层适配调用）
    timeout:
        Future.result(timeout=...) 的超时（None 表示不超时）
    exclusive_key:
        返回相同 key 的任务在父进程内互斥（thread/inline 通过 Lock；process 通过父进程侧串行化提交+等待）
        - 命名命令：exclusive_key 接收 kwargs: Dict[str, Any]
        - dataclass 命令：exclusive_key 接收 cmd 对象
    ordered:
        True 则同 key 严格 FIFO（使用 SerialExecutor）
    """
    executor: ExecutorKind = "inline"
    timeout: Optional[float] = None
    exclusive_key: Optional[ExclusiveKeyFn] = None
    ordered: bool = False


class SerialExecutor:
    """单线程 FIFO 执行器。submit(fn) -> Future；队列内任务严格按提交顺序执行。"""

    def __init__(self) -> None:
        self._q: "queue.Queue[Tuple[Callable[[], Any], Future]]" = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True, name="cmd-serial")
        self._t.start()

    def submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut

    def _run(self) -> None:
        while True:
            fn, fut = self._q.get()
            try:
                if fut.set_running_or_notify_cancel():
                    fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)


def _call_with_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """给进程池用的顶层适配：func(**kwargs)"""
    return func(**kwargs)


class CommandBus:
    """
    以“命名命令”为主，兼容 dataclass 命令。

    - 命名命令：@CommandBus.handler("name", ...)
    - 命名命令进程任务：@CommandBus.process_task("name", task_fn=top_level_func, ...)
    - dataclass 兼容：handler_dc / process_task_dc

    可选：注入 logger 后自动记录异常（不会吞异常）
      - handle(...)：捕获异常记录后继续 raise
      - handle_future(...)：对返回 Future 自动挂回调记录异常（即使调用方不 .result() 也能看到日志）
    """

    _class_lock = threading.RLock()

    _class_named_handlers: Dict[str, Tuple[Callable[..., Any], ExecPolicy]] = {}
    _class_named_process: Dict[str, Tuple[Callable[..., Any], ExecPolicy]] = {}

    _class_handlers_dc: Dict[Type[Any], Tuple[Callable[[Any], Any], ExecPolicy]] = {}
    _class_process_dc: Dict[Type[Any], Tuple[Callable[[Any], Any], ExecPolicy]] = {}

    def __init__(
        self,
        thread_workers: int = 8,
        process_workers: int = os.cpu_count() or 2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._tpool = ThreadPoolExecutor(max_workers=thread_workers, thread_name_prefix="cmd-th")
        self._ppool = ProcessPoolExecutor(max_workers=process_workers)

        with self._class_lock:
            self._named_handlers = dict(type(self)._class_named_handlers)
            self._named_process = dict(type(self)._class_named_process)
            self._handlers_dc = dict(type(self)._class_handlers_dc)
            self._process_dc = dict(type(self)._class_process_dc)

        self._key_locks: Dict[str, threading.Lock] = {}
        self._serial_execs: Dict[str, SerialExecutor] = {}
        self._logger = logger

    def __enter__(self) -> "CommandBus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # -------------------------
    # 注册：命名命令
    # -------------------------
    @classmethod
    def handler(
        cls,
        name: str,
        *,
        executor: ExecutorKind = "inline",
        key: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        ordered: bool = False,
        timeout: Optional[float] = None,
    ):
        pol = ExecPolicy(executor=executor, timeout=timeout, exclusive_key=key, ordered=ordered)

        def deco(fn: Callable[..., Any]):
            with cls._class_lock:
                cls._class_named_handlers[name] = (fn, pol)
            return fn

        return deco

    @classmethod
    def process_task(
        cls,
        name: str,
        *,
        task_fn: Callable[..., Any],
        timeout: Optional[float] = None,
        key: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        ordered: bool = False,
    ):
        pol = ExecPolicy(executor="process", timeout=timeout, exclusive_key=key, ordered=ordered)

        def deco(dummy: Callable[..., Any]):
            with cls._class_lock:
                cls._class_named_process[name] = (task_fn, pol)
            return dummy

        return deco

    # -------------------------
    # 注册：dataclass 兼容
    # -------------------------
    @classmethod
    def handler_dc(cls, cmd_type: Type[Any], policy: ExecPolicy = ExecPolicy()):
        def deco(fn: Callable[[Any], Any]):
            with cls._class_lock:
                cls._class_handlers_dc[cmd_type] = (fn, policy)
            return fn

        return deco

    @classmethod
    def process_task_dc(
        cls,
        cmd_type: Type[Any],
        *,
        task_fn: Callable[[Any], Any],
        policy: ExecPolicy = ExecPolicy(executor="process"),
    ):
        def deco(dummy: Callable[[Any], Any]):
            with cls._class_lock:
                cls._class_process_dc[cmd_type] = (task_fn, policy)
            return dummy

        return deco

    # -------------------------
    # 可选：动态导入后刷新实例快照
    # -------------------------
    def bind_class_handlers(self) -> None:
        with self._lock, self._class_lock:
            self._named_handlers.update(type(self)._class_named_handlers)
            self._named_process.update(type(self)._class_named_process)
            self._handlers_dc.update(type(self)._class_handlers_dc)
            self._process_dc.update(type(self)._class_process_dc)

    # -------------------------
    # 公共 API：命名命令
    # -------------------------
    def handle(self, name: str, /, **kwargs) -> Any:
        spec = self._resolve_named(name)
        if spec is None:
            raise KeyError(f"No named handler for '{name}'")

        kind, (fn, pol) = spec
        key = self._compute_key(pol, kwargs)

        try:
            if kind == "process":
                return self._run_process_named_blocking(fn, pol, key, kwargs)

            call = lambda: fn(**kwargs)
            return self._run_non_process_blocking(kind, pol, key, call)
        except BaseException as e:
            self._log_exception(cmd_name=name, stage="handle", executor=kind, key=key, exc=e)
            raise

    def handle_future(self, name: str, /, **kwargs) -> Future:
        spec = self._resolve_named(name)
        if spec is None:
            raise KeyError(f"No named handler for '{name}'")

        kind, (fn, pol) = spec
        key = self._compute_key(pol, kwargs)

        if kind == "process":
            fut = self._run_process_named_future(fn, pol, key, kwargs)
            self._attach_log(fut, cmd_name=name, executor=kind, key=key)
            return fut

        call = lambda: fn(**kwargs)
        fut = self._run_non_process_future(kind, pol, key, call)
        self._attach_log(fut, cmd_name=name, executor=kind, key=key)
        return fut

    # -------------------------
    # 公共 API：dataclass 兼容
    # -------------------------
    def handle_obj(self, cmd: Any) -> Any:
        spec = self._resolve_obj(cmd)
        if spec is None:
            raise KeyError(f"No handler for {type(cmd).__name__}")

        kind, (fn, pol) = spec
        key = self._compute_key(pol, cmd)
        cmd_name = type(cmd).__name__

        try:
            if kind == "process":
                return self._run_process_obj_blocking(fn, pol, key, cmd)

            call = lambda: fn(cmd)
            return self._run_non_process_blocking(kind, pol, key, call)
        except BaseException as e:
            self._log_exception(cmd_name=cmd_name, stage="handle_obj", executor=kind, key=key, exc=e)
            raise

    def handle_future_obj(self, cmd: Any) -> Future:
        spec = self._resolve_obj(cmd)
        if spec is None:
            raise KeyError(f"No handler for {type(cmd).__name__}")

        kind, (fn, pol) = spec
        key = self._compute_key(pol, cmd)
        cmd_name = type(cmd).__name__

        if kind == "process":
            fut = self._run_process_obj_future(fn, pol, key, cmd)
            self._attach_log(fut, cmd_name=cmd_name, executor=kind, key=key)
            return fut

        call = lambda: fn(cmd)
        fut = self._run_non_process_future(kind, pol, key, call)
        self._attach_log(fut, cmd_name=cmd_name, executor=kind, key=key)
        return fut

    # -------------------------
    # 资源管理
    # -------------------------
    def shutdown(self, wait: bool = True) -> None:
        self._tpool.shutdown(wait=wait, cancel_futures=False)
        self._ppool.shutdown(wait=wait, cancel_futures=False)

    # -------------------------
    # 内部：日志辅助（不会吞异常）
    # -------------------------
    def _log_exception(
        self,
        *,
        cmd_name: str,
        stage: str,
        executor: ExecutorKind,
        key: Optional[str],
        exc: BaseException,
    ) -> None:
        if self._logger is None:
            return
        # 这里在 except 块中调用，exc_info=True 会抓取当前异常堆栈
        self._logger.error(
            "Command failed | name=%s | stage=%s | executor=%s | key=%s",
            cmd_name,
            stage,
            executor,
            key,
            exc_info=True,
        )

    def _attach_log(self, fut: Future, *, cmd_name: str, executor: ExecutorKind, key: Optional[str]) -> None:
        if self._logger is None:
            return

        def _cb(f: Future) -> None:
            if f.cancelled():
                return
            exc = f.exception()
            if exc is None:
                return
            self._logger.error(
                "Command future failed | name=%s | executor=%s | key=%s",
                cmd_name,
                executor,
                key,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

        fut.add_done_callback(_cb)

    # -------------------------
    # 内部：执行辅助
    # -------------------------
    def _compute_key(self, pol: ExecPolicy, payload: Any) -> Optional[str]:
        if pol.exclusive_key is None:
            return None
        return pol.exclusive_key(payload)

    def _get_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def _get_serial(self, key: str) -> SerialExecutor:
        with self._lock:
            return self._serial_execs.setdefault(key, SerialExecutor())

    def _call_locked(self, key: Optional[str], fn: Callable[[], Any]) -> Any:
        if not key:
            return fn()
        lk = self._get_lock(key)
        with lk:
            return fn()

    def _run_non_process_blocking(
        self,
        kind: ExecutorKind,
        pol: ExecPolicy,
        key: Optional[str],
        call: Callable[[], Any],
    ) -> Any:
        if kind == "thread":
            if pol.ordered and key:
                fut = self._get_serial(key).submit(lambda: self._call_locked(key, call))
                return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

            fut = self._tpool.submit(lambda: self._call_locked(key, call))
            return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

        # inline
        return self._call_locked(key, call)

    def _run_non_process_future(
        self,
        kind: ExecutorKind,
        pol: ExecPolicy,
        key: Optional[str],
        call: Callable[[], Any],
    ) -> Future:
        if kind == "thread":
            if pol.ordered and key:
                return self._get_serial(key).submit(lambda: self._call_locked(key, call))
            return self._tpool.submit(lambda: self._call_locked(key, call))

        # inline：包装成线程池 Future
        return self._tpool.submit(lambda: self._call_locked(key, call))

    # -------- process：命名命令 --------
    def _run_process_named_blocking(
        self,
        fn: Callable[..., Any],
        pol: ExecPolicy,
        key: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Any:
        def submit_and_wait() -> Any:
            fut = self._ppool.submit(_call_with_kwargs, fn, kwargs)
            return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

        if pol.ordered and key:
            fut = self._get_serial(key).submit(lambda: self._call_locked(key, submit_and_wait))
            return fut.result()  # timeout 已在 submit_and_wait 内生效

        return self._call_locked(key, submit_and_wait)

    def _run_process_named_future(
        self,
        fn: Callable[..., Any],
        pol: ExecPolicy,
        key: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Future:
        def submit_and_wait() -> Any:
            fut = self._ppool.submit(_call_with_kwargs, fn, kwargs)
            return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

        if pol.ordered and key:
            return self._get_serial(key).submit(lambda: self._call_locked(key, submit_and_wait))

        if key:
            # process 下需要父进程侧互斥：用线程池协调（不阻塞调用方）
            return self._tpool.submit(lambda: self._call_locked(key, submit_and_wait))

        return self._ppool.submit(_call_with_kwargs, fn, kwargs)

    # -------- process：dataclass 命令 --------
    def _run_process_obj_blocking(
        self,
        fn: Callable[[Any], Any],
        pol: ExecPolicy,
        key: Optional[str],
        cmd: Any,
    ) -> Any:
        def submit_and_wait() -> Any:
            fut = self._ppool.submit(fn, cmd)
            return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

        if pol.ordered and key:
            fut = self._get_serial(key).submit(lambda: self._call_locked(key, submit_and_wait))
            return fut.result()

        return self._call_locked(key, submit_and_wait)

    def _run_process_obj_future(
        self,
        fn: Callable[[Any], Any],
        pol: ExecPolicy,
        key: Optional[str],
        cmd: Any,
    ) -> Future:
        def submit_and_wait() -> Any:
            fut = self._ppool.submit(fn, cmd)
            return fut.result(timeout=pol.timeout) if pol.timeout else fut.result()

        if pol.ordered and key:
            return self._get_serial(key).submit(lambda: self._call_locked(key, submit_and_wait))

        if key:
            return self._tpool.submit(lambda: self._call_locked(key, submit_and_wait))

        return self._ppool.submit(fn, cmd)

    # -------------------------
    # 内部：解析注册表
    # -------------------------
    def _resolve_named(self, name: str) -> Optional[Tuple[ExecutorKind, Tuple[Callable[..., Any], ExecPolicy]]]:
        with self._lock:
            if name in self._named_process:
                return "process", self._named_process[name]
            if name in self._named_handlers:
                fn, pol = self._named_handlers[name]
                return pol.executor, (fn, pol)
        return None

    def _resolve_obj(self, cmd: Any) -> Optional[Tuple[ExecutorKind, Tuple[Callable[[Any], Any], ExecPolicy]]]:
        t = type(cmd)
        with self._lock:
            if t in self._process_dc:
                return "process", self._process_dc[t]
            if t in self._handlers_dc:
                fn, pol = self._handlers_dc[t]
                return pol.executor, (fn, pol)
        return None
