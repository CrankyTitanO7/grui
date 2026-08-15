"""Perception dialog: run an optional provider over a loaded recording.

Mirrors the ``grui perception analyze`` CLI for the GUI: pick the provider,
enter prompts, choose the sampling FPS, and run. Analysis runs on a worker
thread (a 3B model can take a long time); the recording itself is never
modified. When LocateAnything is selected, the model-cost/licensing
acknowledgment gates the Run button.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from perception import get, list_providers, provider_info
from perception.runner import analyze_recording
from storage.recording import RecordingData

logger = logging.getLogger(__name__)


class _PerceptionWorker(QThread):
    """Run perception analysis off the UI thread."""

    finished = Signal(bool, str, object)  # (ok, error_message, CachedAnalysis|None)

    def __init__(self, recording: RecordingData, provider, prompts: list[str], fps: float, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._provider = provider
        self._prompts = prompts
        self._fps = fps

    def run(self) -> None:
        try:
            result = analyze_recording(
                self._recording, self._provider, self._prompts, fps=self._fps
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("perception analysis failed")
            self.finished.emit(False, str(exc), None)
        else:
            self.finished.emit(True, "", result)


class PerceptionDialog(QDialog):
    """Configure and run perception analysis for one loaded recording."""

    def __init__(self, recording: RecordingData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recording = recording
        self.setWindowTitle("Perception (optional)")
        self.setMinimumWidth(480)
        self._worker: _PerceptionWorker | None = None
        self._ack_required = False
        self._build_ui()
        self._refresh_providers()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            f"{self.recording.directory.name}\n"
            f"{len(self.recording.frame_times)} frames · {self.recording.duration:.1f}s "
            f"· {self.recording.width}×{self.recording.height}"
        )
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        note = QLabel(
            "Analysis reads the recording and writes derived results into\n"
            "<recording>/perception/ — the raw recording is never modified."
        )
        note.setStyleSheet("color: #888888;")
        layout.addWidget(note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Provider:"))
        self._provider = QComboBox()
        self._provider.currentIndexChanged.connect(self._on_provider_changed)
        row.addWidget(self._provider, 1)
        layout.addLayout(row)

        self._availability = QLabel("")
        self._availability.setWordWrap(True)
        self._availability.setStyleSheet("color: #888888;")
        layout.addWidget(self._availability)

        layout.addWidget(QLabel("Prompts (comma separated, e.g. \"boss, projectile, player\"):"))
        self._prompts = QLineEdit()
        self._prompts.setPlaceholderText("boss, projectile")
        layout.addWidget(self._prompts)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Sampling FPS:"))
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.1, 240.0)
        self._fps.setDecimals(1)
        self._fps.setValue(2.0)
        self._fps.setSuffix(" frames/s")
        fps_row.addWidget(self._fps)
        layout.addLayout(fps_row)

        self._model_row = QHBoxLayout()
        self._model_row.addWidget(QLabel("Weights file:"))
        self._model = QLineEdit()
        self._model.setPlaceholderText("yolov8n.pt (yolo only)")
        self._model_row.addWidget(self._model, 1)
        self._allow_download = QCheckBox("allow ultralytics to download missing weights")
        self._model_row.addWidget(self._allow_download)
        self._model_row_widget = QWidget()
        self._model_row_widget.setLayout(self._model_row)
        self._model_row_widget.setVisible(False)
        layout.addWidget(self._model_row_widget)

        self._ack = QCheckBox("I understand — proceed (equivalent of --iknow)")
        self._ack.setVisible(False)
        self._ack.toggled.connect(self._update_run_enabled)
        layout.addWidget(self._ack)

        buttons = QHBoxLayout()
        self._run_btn = QPushButton("Analyze…")
        self._run_btn.clicked.connect(self._on_run)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self._run_btn)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- state

    def _refresh_providers(self) -> None:
        self._provider.blockSignals(True)
        self._provider.clear()
        for provider in list_providers():
            self._provider.addItem(provider.name, provider.name)
        self._provider.blockSignals(False)
        self._on_provider_changed()

    def _on_provider_changed(self) -> None:
        name = self._provider.currentData()
        if not name:
            self._availability.setText("No perception providers installed.")
            self._run_btn.setEnabled(False)
            return
        provider = get(name)
        info = provider_info(provider)
        warnings = list(getattr(provider, "warnings", ()) or ())
        self._model_row_widget.setVisible(name == "yolo")
        if not info.available:
            self._availability.setText(
                f"Status: unavailable\n{info.install_hint or 'Install the optional dependencies.'}"
            )
            self._ack.setVisible(False)
            self._run_btn.setEnabled(False)
            return
        self._availability.setText("Status: available")
        if warnings:
            self._availability.setText("Status: available\n" + "\n".join(f"• {w}" for w in warnings))
        self._ack.setVisible(bool(warnings))
        self._ack_required = bool(warnings)
        self._update_run_enabled()

    def _update_run_enabled(self, checked: bool = False) -> None:
        if not self._provider.currentData():
            self._run_btn.setEnabled(False)
            return
        self._run_btn.setEnabled(not self._ack_required or checked)

    def _on_run(self) -> None:
        prompts = [p.strip() for p in self._prompts.text().split(",") if p.strip()]
        if not prompts:
            QMessageBox.warning(self, "Perception", "Enter at least one prompt.")
            return
        provider = get(self._provider.currentData())
        options: dict = {}
        if getattr(provider, "name", "") == "yolo":
            model = self._model.text().strip() or None
            if model:
                options["model"] = model
            options["allow_download"] = self._allow_download.isChecked()
        from perception.base import with_options

        provider = with_options(provider, **options)
        self._run_btn.setEnabled(False)
        for widget in (self._provider, self._prompts, self._fps, self._ack, self._model, self._allow_download):
            widget.setEnabled(False)
        self._worker = _PerceptionWorker(self.recording, provider, prompts, self._fps.value(), self)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self.setWindowTitle("Perception — analyzing…")

    def _on_done(self, ok: bool, error: str, result) -> None:
        for widget in (self._provider, self._prompts, self._fps, self._ack, self._model, self._allow_download):
            widget.setEnabled(True)
        self._run_btn.setEnabled(True)
        self.setWindowTitle("Perception (optional)")
        if not ok:
            QMessageBox.warning(self, "Perception", f"Analysis failed:\n{error}")
            return
        records = len(result.read_results())
        QMessageBox.information(
            self,
            "Perception",
            f"Analyzed {records} records →\n{result.results_path}",
        )
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self._worker
        if worker is not None and worker.isRunning():
            self.setWindowTitle("Perception — finishing analysis…")
            worker.wait()  # never destroy a running thread
        super().closeEvent(event)
