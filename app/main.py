"""Application entry point for GRUI (Grand Unified Imitation)."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        from dataset.cli import run as dataset_cli

        return dataset_cli(sys.argv[2:])

    app = QApplication(sys.argv)
    app.setApplicationName("GRUI")
    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
