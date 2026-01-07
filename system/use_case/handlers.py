from __future__ import annotations

from typing import List, Tuple

from dependency_injector.wiring import inject, Provide

from system import CommandBus
from system.controller import Controller
from Application.eventBus import EventBus

# 开始训练
@EventBus.on("UI_PC_TRAIN_START")
@inject
def handle_ui_pc_train_start(
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


# 对齐--------------------------------
@EventBus.on("UI_ALIGN_START")
@inject
def handle_ui_align_start(
        tmp_path: str,
        mask_path: str,
        subtag_folders: List[Tuple[str, str, str]],
        target_dir: str,
        controller: Controller = Provide[Application.controller],
) -> None:
    controller.command_bus.handle_future(
        "align_task",
        tmp_path=tmp_path,
        mask_path=mask_path,
        subtag_folders=subtag_folders,
        target_dir=target_dir
    )

@inject
@CommandBus.handler("align_task", executor="thread")
def handle_align_task(
        tmp_path: str,
        mask_path: str,
        subtag_folders: List[Tuple[str, str, str]],
        target_dir: str,
        controller: Controller = Provide[Application.controller]
):
    for folder_pgs, total_pgs, info in controller.aligner.align(tmp_path, mask_path,
                                                                target_dir,
                                                                subtag_folders):
        controller.event_bus.publish("align_status", folder_pgs, total_pgs, info)


@EventBus.on("align_status")
@inject
def handle_align_status(
        folder_pgs: str,
        total_pgs: float,
        info: dict,

) -> None:
    print(folder_pgs, total_pgs, info)
    controller.qt_bridge.progress_message(folder_pgs, total_pgs, info)


# 训练------------
@EventBus.on("UI_TRAIN_START")
@inject
def handle_ui_train_start(model_id, config, path_list, model_output_dir,
                          controller: Controller = Provide[Application.controller]):
    controller.command_bus.handle_future("train_task", model_id=model_id, config=config, path_list=path_list,
                                         model_output_dir=model_output_dir)


@CommandBus.handler("train_task", executor="thread", ordered=True)
@inject
def handle_train_task(model_id, config, path_list, model_output_dir,
                      controller: Controller = Provide[Application.controller]):
    path = controller.algorithm.train(model_id=model_id, cfg=config, ok_image_folders=path_list,
                                      output_root=model_output_dir)
    controller.qt_bridge.train_result_message(model_id, 1, path)


# 评估------------
@EventBus.on("UI_VAL_START")
@inject
def handle_ui_train_start(model_id, model_path, path_list,
                          controller: Controller = Provide[Application.controller]):
    print(model_id, model_path, path_list)
    controller.command_bus.handle_future("val_task", model_id=model_id, model_path=model_path, path_list=path_list)
    print("over")


@CommandBus.handler("val_task", executor="thread", ordered=True)
@inject
def handle_train_task(model_id, model_path, path_list,
                      controller: Controller = Provide[Application.controller]):
    try:
        for result in controller.algorithm.predict(model_id=model_id, model_path=model_path, targets=path_list):
            print("result",result)
            # model_id, real_label, pre_label, score, origin_path, path_cvt, defect_count, time = result
            controller.qt_bridge.assess_result_message(**result)
    except Exception as e:
        print(e)
