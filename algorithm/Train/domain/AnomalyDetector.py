# Domain/AnomalyDetector.py
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

PathItem = Union[str, Dict[str, Any]]


class TrainingState(str, Enum):
    IDLE = "IDLE"        # 未训练/未加载
    TRAINING = "TRAINING"
    READY = "READY"      # 已训练完成或已加载可推理
    FAILED = "FAILED"    # 最近一次 train 失败


class AlgorithmPort(ABC):
    """
    单一 Port 的统一接口（无进度/无中止），与 PatchCorePort 对齐：
    - train(cfg, ok_image_folders, model_id, output_root?) -> stats(dict)
    - predict(model_id, tag, model_path, path_list) -> dict
    """

    @abstractmethod
    def train(
        self,
        cfg: Dict[str, Any],
        ok_image_folders: List[str],
        model_id: str,
        output_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        model_id: str,
        model_path: str,
        targets,
    ) :
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_feature_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError


class AnomalyDetector:
    """
    仅维护“当前 Port/当前模型”的状态，不支持多 id 查询。
    - train_model(): 训练当前模型（训练完成会得到 model_feature_dir）
    - load_model(): 加载当前模型目录后进入 READY
    - detect(): 推理（model_path 不传则默认 self.model_feature_dir）
    """

    def __init__(self, port: AlgorithmPort) -> None:
        self.port = port

        # 仅记录当前 port 的状态
        self._state: TrainingState = TrainingState.IDLE
        self._last_error: Optional[Exception] = None

        # 当前模型的训练信息/可推理目录
        self.last_train_stats: Optional[Dict[str, Any]] = None

        # 可选：记录“当前模型 ID”（便于日志/UI 展示；不用于检索）
        self.model_id: Optional[str] = None

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def last_error(self) -> Optional[Exception]:
        return self._last_error

    def train_model(
        self,
        cfg: Dict[str, Any],
        ok_image_folders: List[str],
        model_id: str,
        output_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        训练当前模型：
        - 成功：state=READY，返回 stats（并写入 last_train_stats/model_feature_dir/model_id）
        - 失败：state=FAILED，抛异常
        """
        if self._state == TrainingState.TRAINING:
            raise RuntimeError("Already training.")

        self._state = TrainingState.TRAINING
        self._last_error = None
        self.last_train_stats = None
        self.model_id = str(model_id)

        print(f"Training started. model_id={self.model_id}")

        try:
            stats = self.port.train(
                cfg=cfg,
                ok_image_folders=ok_image_folders,
                model_id=self.model_id,
                output_root=output_root,
            )

            self.last_train_stats = stats
            # self.model_feature_dir = stats.get("model_feature_dir")

            self._state = TrainingState.READY
            # print(f"Training finished. model_feature_dir={self.model_feature_dir}")
            return stats

        except Exception as e:
            self._state = TrainingState.FAILED
            self._last_error = e
            self.last_train_stats = None
            print(f"Training failed: {e}")
            raise

    def load_model(self, model_feature_dir: str, *, model_id: Optional[str] = None) -> None:
        """
        UI 启动/切换模型时调用：加载指定模型目录后即可 detect。
        model_id 可选，仅用于记录当前模型标识（不做检索）。
        """
        self.port.load(model_feature_dir)
        if model_id is not None:
            self.model_id = str(model_id)
        self._state = TrainingState.READY

    def detect(
        self,
        *,
        model_id: Optional[str],
        tag: str,
        path_list: List[PathItem],
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对齐 PatchCorePort.predict 入参：
        - model_id：可传可不传；不传时使用 self.model_id；两者都空则报错
        - model_path：不传则默认 self.model_feature_dir
        """
        if self._state != TrainingState.READY:
            raise RuntimeError(f"Model is not ready. state={self._state}")

        mid = str(model_id) if model_id else (self.model_id or "")
        if not mid:
            raise RuntimeError("model_id is empty (neither argument nor cached model_id).")

        mp = model_path or self.model_feature_dir
        if not mp:
            raise RuntimeError("model_path is empty and model_feature_dir is not set.")

        # 兼容：port 未 ready 时先 load
        if not self.port.is_ready():
            self.port.load(mp)

        return self.port.predict(
            model_id=mid,
            tag=tag,
            model_path=mp,
            path_list=path_list,
        )

    def reset(self) -> None:
        self.port.unload()
        self._state = TrainingState.IDLE
        self._last_error = None
        self.last_train_stats = None
        self.model_id = None
