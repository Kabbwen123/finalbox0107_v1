from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
from concurrent.futures import Future
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Callable, Optional, Tuple


def _exc_info(e: BaseException) -> Tuple[type, BaseException, Any]:
    return (type(e), e, e.__traceback__)


class _DailyDirTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    每天一个目录：log_root/YYYY-MM-DD/app.log
    使用 TimedRotatingFileHandler 做 midnight 轮转。
    关键点：delay=True，rollover 后下一次 emit 才会打开新文件。
    """
    def __init__(
        self,
        *,
        log_root: str,
        filename: str = "app.log",
        when: str = "midnight",
        backup_count: int = 7,
        encoding: str = "utf-8",
        utc: bool = False,
    ) -> None:
        self._log_root = log_root
        self._filename = filename
        self._current_day = _dt.date.today()

        base = self._path_for_day(self._current_day)
        os.makedirs(os.path.dirname(base), exist_ok=True)

        super().__init__(
            filename=base,
            when=when,
            interval=1,
            backupCount=backup_count,
            encoding=encoding,
            utc=utc,
            delay=True,  # 重要：rollover 后不立刻打开文件，方便我们换 baseFilename
        )

    def _path_for_day(self, day: _dt.date) -> str:
        day_dir = os.path.join(self._log_root, day.strftime("%Y-%m-%d"))
        return os.path.join(day_dir, self._filename)

    def doRollover(self) -> None:
        # 先执行父类 rollover：关闭当前文件、重命名旧文件等
        super().doRollover()

        # 切到新日期目录（通常是跨午夜）
        new_day = _dt.date.today()
        if new_day != self._current_day:
            self._current_day = new_day
            new_base = self._path_for_day(new_day)
            os.makedirs(os.path.dirname(new_base), exist_ok=True)
            self.baseFilename = os.path.abspath(new_base)
            # delay=True：不在这里打开新 stream；下一次 emit 会按新的 baseFilename 打开


class Logger:
    """
    - 同名 logger 只配置一次，避免重复 handler
    - 支持控制台 + 文件（按天目录 + midnight 轮转）
    - 提供 eventbus_error_hook 对接 EventBus
    """
    _init_lock = threading.RLock()
    _configured_names: set[str] = set()

    def __init__(
        self,
        *,
        name: str = "app",
        log_dir: str = "log",
        enable_console: bool = True,
        enable_file: bool = True,
        log_level: int = logging.INFO,
        file_log_level: int = logging.INFO,
        console_log_level: int = logging.INFO,
        filename: str = "app.log",
        backup_count: int = 7,
    ) -> None:
        self.name = name
        self.log_root = log_dir

        logger = logging.getLogger(name)

        # 让“首次构建这个 logger 的实例”记录一个全局 start_time
        now = _dt.datetime.now()
        if not hasattr(logger, "_start_time"):
            setattr(logger, "_start_time", now)
        self.start_time: _dt.datetime = getattr(logger, "_start_time")

        with Logger._init_lock:
            if name not in Logger._configured_names:
                logger.setLevel(log_level)
                logger.propagate = False

                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

                if enable_console:
                    ch = logging.StreamHandler()
                    ch.setLevel(console_log_level)
                    ch.setFormatter(fmt)
                    logger.addHandler(ch)

                if enable_file:
                    fh = _DailyDirTimedRotatingFileHandler(
                        log_root=log_dir,
                        filename=filename,
                        backup_count=backup_count,
                        encoding="utf-8",
                        utc=False,
                    )
                    fh.setLevel(file_log_level)
                    fh.setFormatter(fmt)
                    logger.addHandler(fh)

                Logger._configured_names.add(name)

        self.logger = logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self.logger, name)

    # -------------------------
    # 对接 EventBus：error_hook
    # -------------------------
    def eventbus_error_hook(self) -> Callable[[str, Callable[..., Any], BaseException], None]:
        """
        用法：bus = EventBus(..., error_hook=log.eventbus_error_hook())
        """
        def hook(topic: str, handler: Callable[..., Any], exc: BaseException) -> None:
            hname = getattr(handler, "__qualname__", getattr(handler, "__name__", repr(handler)))
            self.logger.error(
                "EventBus handler failed | topic=%s | handler=%s",
                topic,
                hname,
                exc_info=_exc_info(exc),
            )
        return hook

    # -------------------------
    # 对接异步 Future：统一记录异常
    # -------------------------
    def log_future(self, fut: Future, *, context: str = "") -> None:
        """
        给 Future 加回调：若异常，记录堆栈；不吞异常（异常仍在 fut.result() 时抛出）。
        """
        def _cb(f: Future) -> None:
            try:
                _ = f.result()
            except BaseException as e:
                self.logger.error("Future failed %s", context, exc_info=_exc_info(e))
        fut.add_done_callback(_cb)
