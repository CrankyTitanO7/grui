"""Dataset generation dialog: configure parameters and build a dataset.

Used from the player window: pick an output directory, tune the
observation window / FPS / stride / prediction horizon, and run the
builder. The build runs synchronously (like export), with a wait cursor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dataset.build import DatasetConfig, build_dataset
from storage.recording import RecordingData

logger = logging.getLogger(__name__)


class DatasetDialog(QDialog):
    """Configure and run dataset generation for one loaded recording."""

    def __init__(self, recording: RecordingData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recording = recording
        self.setWindowTitle("Build Dataset")
        self.setMinimumWidth(440)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            f"{self.recording.directory.name}\n"
            f"{len(self.recording.frame_times)} frames · {self.recording.duration:.1f}s "
            f"· {self.recording.width}×{self.recording.height}"
        )
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        default = self.recording.directory.parent / f"{self.recording.directory.name}_dataset"
        self._out_edit = QLineEdit(str(default))
        out_row.addWidget(self._out_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_out)
        out_row.addWidget(browse)
        layout.addLayout(out_row)

        form = QFormLayout()
        self._duration = QDoubleSpinBox()
        self._duration.setRange(0.1, 3600.0)
        self._duration.setDecimals(2)
        self._duration.setValue(3.0)
        self._duration.setSuffix(" s")
        self._fps = QDoubleSpinBox()
        self._fps.setRange(1.0, 240.0)
        self._fps.setDecimals(1)
        self._fps.setValue(15.0)
        self._stride = QDoubleSpinBox()
        self._stride.setRange(0.01, 3600.0)
        self._stride.setDecimals(2)
        self._stride.setValue(1.0)
        self._stride.setSuffix(" s")
        self._horizon = QDoubleSpinBox()
        self._horizon.setRange(0.0, 60.0)
        self._horizon.setDecimals(2)
        self._horizon.setValue(0.2)
        self._horizon.setSuffix(" s")
        form.addRow("Observation duration:", self._duration)
        form.addRow("Observation FPS:", self._fps)
        form.addRow("Stride:", self._stride)
        form.addRow("Prediction horizon:", self._horizon)
        layout.addLayout(form)

        hint = QLabel(
            "Samples start after one observation window; the window is the video\n"
            "history before each sample, the action is the input state at that time."
        )
        hint.setStyleSheet("color: #666666;")
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        self._build_btn = QPushButton("Build Dataset")
        self._build_btn.clicked.connect(self._on_build)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self._build_btn)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Directory", str(Path(self._out_edit.text()).parent))
        if path:
            self._out_edit.setText(path)

    def config(self) -> DatasetConfig:
        return DatasetConfig(
            observation_duration=self._duration.value(),
            fps=self._fps.value(),
            stride=self._stride.value(),
            prediction_horizon=self._horizon.value(),
        )

    def _on_build(self) -> None:
        try:
            config = self.config()
            config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Dataset", str(exc))
            return
        out = Path(self._out_edit.text().strip())
        if not out:
            QMessageBox.warning(self, "Dataset", "Choose an output directory.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = build_dataset(self.recording, config, out)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dataset build failed")
            QMessageBox.warning(self, "Dataset", f"Dataset build failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
        QMessageBox.information(
            self,
            "Dataset",
            f"Built {manifest['count']} samples ({manifest['observation_frames']} frames) →\n{result}",
        )
        self.accept()
