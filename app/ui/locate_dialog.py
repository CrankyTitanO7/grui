"""LocateAnything enrichment dialog: prompts -> locations.jsonl.

Mirrors ``grui locate`` for the GUI: pick the dataset, enter prompts, choose
the task and sampling, and run. The model costs are shown up front; the Run
button stays disabled until the user checks the acknowledgment (the GUI
equivalent of ``--iknow``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ml.locate import WARNINGS, enrich_dataset, load_locator
from storage.recording import RecordingData

logger = logging.getLogger(__name__)

_TASKS = (
    ("ground_gui", "GUI element grounding (box)"),
    ("detect", "Object detection (categories)"),
    ("point", "Point to the object"),
    ("detect_text", "Detect all text (OCR)"),
)


class LocateDialog(QDialog):
    """Text-prompted localization over a built dataset."""

    def __init__(self, recording: RecordingData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recording = recording
        self.dataset_dir = recording.directory.parent / f"{recording.directory.name}_dataset"
        self.setWindowTitle("Locate Anything (Enrichment)")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            f"{self.recording.directory.name}\n"
            f"Dataset: {self.dataset_dir}"
        )
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        if not self.dataset_dir.exists():
            layout.addWidget(
                QLabel("Dataset not found — run Build Dataset… first, then come back here.")
            )

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Dataset:"))
        self._dataset_edit = QLineEdit(str(self.dataset_dir))
        out_row.addWidget(self._dataset_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_dataset)
        out_row.addWidget(browse)
        layout.addLayout(out_row)

        layout.addWidget(QLabel("Prompts (comma separated, e.g. \"save button, search bar\"):"))
        self._prompts = QLineEdit()
        self._prompts.setPlaceholderText("the save button, the search bar")
        layout.addWidget(self._prompts)

        form = QVBoxLayout()
        form.addWidget(QLabel("Task:"))
        self._task = QComboBox()
        for value, label in _TASKS:
            self._task.addItem(label, value)
        form.addWidget(self._task)
        form.addWidget(QLabel("Process every Nth frame:"))
        self._every = QSpinBox()
        self._every.setRange(1, 1000)
        self._every.setValue(10)
        form.addWidget(self._every)
        layout.addLayout(form)

        warnings = QLabel("\n".join(f"• {w}" for w in WARNINGS))
        warnings.setWordWrap(True)
        warnings.setStyleSheet("color: #d4a017; font-size: 9pt;")
        layout.addWidget(warnings)

        self._ack = QCheckBox("I understand — proceed (equivalent of --iknow)")
        self._ack.toggled.connect(self._update_run_enabled)
        layout.addWidget(self._ack)

        buttons = QHBoxLayout()
        self._run_btn = QPushButton("Run Locate…")
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self._run_btn)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Dataset Directory", str(self.dataset_dir.parent)
        )
        if path:
            self._dataset_edit.setText(path)

    def _update_run_enabled(self, checked: bool) -> None:
        self._run_btn.setEnabled(checked)

    def _on_run(self) -> None:
        dataset_dir = Path(self._dataset_edit.text().strip())
        if not (dataset_dir / "manifest.json").exists():
            QMessageBox.warning(self, "Locate", "Not a built dataset (missing manifest.json).")
            return
        prompts = [p.strip() for p in self._prompts.text().split(",") if p.strip()]
        if not prompts:
            QMessageBox.warning(self, "Locate", "Enter at least one prompt.")
            return
        task = self._task.currentData()
        try:
            locator = load_locator()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Locate", str(exc))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = enrich_dataset(
                dataset_dir, prompts, task, every=self._every.value(), locator=locator
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("locate failed")
            QMessageBox.warning(self, "Locate", f"Locate failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        records = len(
            [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        )
        QMessageBox.information(self, "Locate", f"Wrote {records} location records →\n{out}")
        self.accept()
