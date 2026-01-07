import sys

from PySide6.QtWidgets import QApplication

from system.container import Application


def main() -> int:
    app = QApplication(sys.argv)

    container = Application()
    main_window = container.main_ui()
    main_window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
