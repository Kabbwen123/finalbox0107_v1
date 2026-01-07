from __future__ import annotations

from typing import List, Tuple

from dependency_injector.wiring import inject, Provide

from system import EventBus,CommandBus
from system.container import Application,Controller

'''




'''


@inject
@EventBus.on("UI_PATCHCORE_TRAIN_START")
def handle_ui_patchcore_train_start(
        *,
        task_id: str,
        cfg: dict,
        image_paths: list[str],
        controller: Controller = Provide[Application.controller],
) -> None:
    controller.command_bus.handle_future(
        "patchcore_train_task",
        task_id=task_id,
        cfg=cfg,
        image_paths=image_paths,
    )