"""Application entry point for the Imitation Recorder."""

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
    app = QApplication(sys.argv)
    app.setApplicationName("Imitation Recorder")
    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
