"""Qt bridge module to mediate between UI widgets and the application layer."""
from __future__ import annotations

from typing import Optional, List

from PySide6.QtCore import QObject, Slot, Signal

from Application.eventBus import EventBus


class QtBridge(QObject):
    progress_emitted = Signal(str, float, dict)
    train_result_emitted = Signal(int, float, str)
    assess_result_emitted = Signal(int, str, str, float, str, str, int, float)
    def __init__(self, event_bus: EventBus) -> None:
        super().__init__()
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # UI -> Application intents
    # ------------------------------------------------------------------
    @Slot()
    def transform_img(self, tmp_path: str,mask_path: str, subtag_folders: list, target_dir: str) -> None:
        """开始对齐裁剪"""
        self._event_bus.publish("UI_ALIGN_START", tmp_path,mask_path, subtag_folders, target_dir)

    @Slot()
    def start_train(self, model_id: int, config: dict, path_list: List[str],model_output_dir:str) -> None:
        """开始训练"""
        self._event_bus.publish("UI_TRAIN_START", model_id, config, path_list,model_output_dir)

    @Slot()
    def start_assess(self, model_id: int, path_list: List[(str,str)], model_path: str) -> None:
        """开始评估"""
        self._event_bus.publish("UI_VAL_START", model_id, model_path, path_list)

    # ------------------------------------------------------------------
    # Application -> UI notifications
    # ------------------------------------------------------------------
    def progress_message(self, folder_pgs: str, total_pgs: float, info: dict) -> None:
        """接收对齐裁剪进度及结果"""
        self.progress_emitted.emit(folder_pgs, total_pgs, info)

    def train_result_message(self, model_id: int, pgs: float, path: Optional[str]) -> None:
        """接收训练进度，结束时通过path传模型路径"""
        self.train_result_emitted.emit(model_id, pgs, path)

    def assess_result_message(self, model_id: int, real_label: str, pre_label: str, score: float, origin_path: str, heatmap_path: str,
                              defect_count: int, time: float) -> None:
        """接收图片评估的结果"""
        print("assess_result_message",int(model_id), real_label, pre_label, score, origin_path, heatmap_path, defect_count, time)
        self.assess_result_emitted.emit(int(model_id), real_label, pre_label, score, origin_path, heatmap_path, defect_count, time)
