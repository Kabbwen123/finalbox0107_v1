from __future__ import annotations

from dataclasses import dataclass

from dependency_injector import containers, providers

from common import Logger
from system import CommandBus, EventBus
from ui import QtBridge


@dataclass(slots=True)
class Controller:
    logger: Logger
    event_bus: EventBus
    command_bus: CommandBus


class Application(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        packages=["system.use_case"]
    )

    config = providers.Configuration(yaml_files=["./config/config.yaml"])

    logger = providers.Singleton(
        Logger,
        enable_file=config.logger.enable_file_logging.as_(bool),
    )

    event_bus = providers.Singleton(EventBus, max_workers=8, logger=logger)

    command_bus = providers.Singleton(CommandBus)

    qt_bridge = providers.Singleton(
        QtBridge,
        event_bus=event_bus,
    )

    sql_db = providers.Singleton(
        SqlDB,
        logger=logger,
        db_path=config.sqlite.db_path.as_(str),
        table=config.sqlite.table.as_(dict),
    )

    main_ui = providers.Singleton(
        MainWindow,
        sql_db=sql_db,
        bridge=qt_bridge,
    )

    aligner = providers.Singleton(Aligner, bus=event_bus)

    algorithm = providers.Singleton(PatchCorePort)

    controller = providers.Singleton(
        Controller,
        event_bus=event_bus,
        command_bus=command_bus,
        logger=logger,
    )
