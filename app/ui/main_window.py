"""Main application window for the Imitation Recorder.

Controls: monitor selection, FPS, Start/Stop, quick annotations and global
hotkeys (F8 stop, F9 marker, F10 pause/resume). Session start/stop run on a
worker thread so the UI stays responsive; state changes from any thread are
forwarded to the UI thread via Qt signals.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from recorder.config import RecorderConfig, ScreenConfig
from recorder.screen import list_monitors
from recorder.session import RecordingSession, SessionState

logger = logging.getLogger(__name__)

_STATE_TEXT = {
    SessionState.IDLE: "Idle",
    SessionState.STARTING: "Starting…",
    SessionState.RECORDING: "Recording",
    SessionState.PAUSED: "Paused",
    SessionState.STOPPING: "Stopping…",
    SessionState.ERROR: "Error",
}

_STATE_COLOR = {
    SessionState.IDLE: "#666666",
    SessionState.STARTING: "#cc8800",
    SessionState.RECORDING: "#c0392b",
    SessionState.PAUSED: "#8e44ad",
    SessionState.STOPPING: "#cc8800",
    SessionState.ERROR: "#c0392b",
}


class StateBridge(QObject):
    """Forward session state changes and hotkeys (any thread) to the UI thread."""

    changed = Signal(object)
    hotkey = Signal(str)


class SessionWorker(QThread):
    """Run a blocking session operation (start/stop) off the UI thread."""

    finished = Signal(bool, str)  # (ok, error_message)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("session worker failed")
            self.finished.emit(False, str(exc))
        else:
            self.finished.emit(True, "")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Imitation Recorder")
        self.setMinimumWidth(380)
        self._session: RecordingSession | None = None
        self._worker: SessionWorker | None = None
        self._last_annotation = ""
        self._bridge = StateBridge(self)
        self._bridge.changed.connect(self._on_state)
        self._bridge.hotkey.connect(self._on_hotkey)
        self._build_ui()
        self._populate_monitors()
        self._start_hotkey_listener()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        title = QLabel("IMITATION RECORDER")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Screen:"))
        self._monitor_combo = QComboBox()
        row.addWidget(self._monitor_combo, 1)
        row.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 240)
        self._fps_spin.setValue(30)
        row.addWidget(self._fps_spin)
        layout.addLayout(row)

        self._status = QLabel(_STATE_TEXT[SessionState.IDLE])
        self._status.setStyleSheet(f"color: {_STATE_COLOR[SessionState.IDLE]}; font-weight: bold;")
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Recording")
        self._start_btn.clicked.connect(self._on_start_clicked)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._player_btn = QPushButton("Open Player")
        self._player_btn.clicked.connect(self._on_open_player)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._player_btn)
        layout.addLayout(btn_row)

        ann_row = QHBoxLayout()
        self._annotation_input = QLineEdit()
        self._annotation_input.setPlaceholderText("Annotation label (e.g. boss_start)")
        self._annotation_btn = QPushButton("Add")
        self._annotation_btn.setEnabled(False)
        self._annotation_btn.clicked.connect(self._on_add_annotation_clicked)
        ann_row.addWidget(self._annotation_input, 1)
        ann_row.addWidget(self._annotation_btn)
        layout.addLayout(ann_row)

        hints = QLabel("Hotkeys while recording:  F8 Stop    F9 Marker    F10 Pause/Resume")
        hints.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(hints)

        note = QLabel(
            "Privacy: while recording is active this application captures the selected "
            "screen, keyboard and mouse input. Recordings stay entirely local and are "
            "never transmitted."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(note)

        self.setCentralWidget(central)

        from app.ui.overlay import RecordingOverlay

        self._overlay = RecordingOverlay()

    def _populate_monitors(self) -> None:
        try:
            monitors = list_monitors()
        except Exception:  # noqa: BLE001
            logger.exception("failed to enumerate monitors")
            self._monitor_combo.addItem("Monitor 1")
            return
        if not monitors:
            self._monitor_combo.addItem("No monitors found")
            return
        self._monitor_combo.clear()
        for monitor in monitors:
            self._monitor_combo.addItem(
                f"Monitor {monitor['index'] + 1} ({monitor['width']}x{monitor['height']})",
                monitor["index"],
            )
        self._monitor_combo.addItem("All monitors (combined)", -1)

    def _start_hotkey_listener(self) -> None:
        def listen() -> None:
            try:
                from pynput import keyboard

                with keyboard.Listener(on_press=self._on_global_key) as listener:
                    listener.join()
            except Exception:  # noqa: BLE001
                logger.exception("global hotkey listener failed")

        threading.Thread(target=listen, name="ui-hotkeys", daemon=True).start()

    def _on_global_key(self, key) -> None:
        try:
            name = getattr(key, "name", None)
            if name == "f8":
                self._bridge.hotkey.emit("F8")
            elif name == "f9":
                self._bridge.hotkey.emit("F9")
            elif name == "f10":
                self._bridge.hotkey.emit("F10")
        except Exception:  # noqa: BLE001
            logger.exception("hotkey handling failed")

    # ------------------------------------------------------------- actions

    def _config_from_ui(self) -> RecorderConfig:
        monitor = self._monitor_combo.currentData()
        if monitor is None:
            monitor = 0
        return RecorderConfig(
            screen=ScreenConfig(fps=self._fps_spin.value(), monitor_index=int(monitor))
        )

    def _on_start_clicked(self) -> None:
        if self._session is not None:
            return
        session = RecordingSession(self._config_from_ui())
        session.register_observer(self._bridge.changed.emit)
        self._session = session
        self._start_btn.setEnabled(False)
        self._run_worker(session.start, self._on_started)

    def _on_started(self, ok: bool, error: str) -> None:
        if ok:
            return
        self._status.setText(f"Error: {error}")
        self._status.setStyleSheet(f"color: {_STATE_COLOR[SessionState.ERROR]}; font-weight: bold;")
        self._session = None
        self._start_btn.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        session = self._session
        if session is None:
            return
        self._stop_btn.setEnabled(False)
        self._run_worker(session.stop, self._on_stopped)

    def _on_stopped(self, ok: bool, error: str) -> None:
        if not ok:
            logger.error("stop failed: %s", error)
            QMessageBox.warning(self, "Imitation Recorder", f"Failed to stop recording cleanly:\n{error}")

    def _run_worker(self, fn, on_done) -> None:
        worker = SessionWorker(fn, self)
        worker.finished.connect(on_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_add_annotation_clicked(self) -> None:
        text = self._annotation_input.text()
        if text.strip():
            self._add_annotation(text)
            self._annotation_input.clear()

    def _add_annotation(self, label: str) -> None:
        if self._session is None:
            return
        try:
            t = self._session.add_annotation(label)
            logger.info("annotation %r recorded at t=%.3f", label, t)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to add annotation")
            QMessageBox.warning(self, "Imitation Recorder", f"Could not add annotation: {exc}")

    def _prompt_annotation(self) -> None:
        text, ok = QInputDialog.getText(
            self, "Add Annotation", "Label:", text=self._last_annotation
        )
        if ok and text.strip():
            self._last_annotation = text.strip()
            self._add_annotation(text)

    # ------------------------------------------------------------- hotkeys

    def _on_hotkey(self, key: str) -> None:
        if self._session is None:
            return
        if key == "F8":
            self._on_stop_clicked()
        elif key == "F9":
            self._prompt_annotation()
        elif key == "F10":
            self._toggle_pause()

    def _toggle_pause(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            if session.state == SessionState.RECORDING:
                session.pause()
            elif session.state == SessionState.PAUSED:
                session.resume()
        except Exception:  # noqa: BLE001
            logger.exception("pause/resume failed")

    def _on_open_player(self) -> None:
        from app.ui.player_window import PlayerWindow

        if getattr(self, "_player_window", None) is None:
            self._player_window = PlayerWindow()
            self._player_window.destroyed.connect(lambda *_: setattr(self, "_player_window", None))
        self._player_window.show()
        self._player_window.raise_()

    # ------------------------------------------------------------- state

    def _on_state(self, state) -> None:
        state = SessionState(state)
        self._status.setText(_STATE_TEXT[state])
        self._status.setStyleSheet(f"color: {_STATE_COLOR[state]}; font-weight: bold;")
        self._start_btn.setEnabled(state in (SessionState.IDLE, SessionState.ERROR))
        self._stop_btn.setEnabled(state in (SessionState.STARTING, SessionState.RECORDING, SessionState.PAUSED))
        annotatable = state in (SessionState.RECORDING, SessionState.PAUSED)
        self._annotation_btn.setEnabled(annotatable)
        self._annotation_input.setEnabled(annotatable)

        if state == SessionState.RECORDING and self._session is not None:
            self._overlay.attach_clock(self._session.clock)
            self._overlay.set_status("RECORDING")
            self._overlay.show_recording()
        elif state == SessionState.PAUSED and self._session is not None:
            self._overlay.set_status("PAUSED")
        elif state == SessionState.STOPPING:
            self._overlay.set_status("STOPPING")
        elif state in (SessionState.IDLE, SessionState.ERROR):
            self._overlay.hide()

        if state in (SessionState.IDLE, SessionState.ERROR):
            self._session = None

    # ------------------------------------------------------------ shutdown

    def shutdown(self) -> None:
        """Stop any active recording. Called on window close / app quit."""
        session = self._session
        if session is None:
            return
        if session.state in (
            SessionState.STARTING,
            SessionState.RECORDING,
            SessionState.PAUSED,
            SessionState.STOPPING,
            SessionState.ERROR,
        ):
            try:
                session.stop()
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop session during shutdown")
        self._overlay.hide()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
