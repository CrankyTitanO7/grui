"""Small unobtrusive always-on-top overlay shown while recording.

The overlay is deliberately dumb: status + elapsed time + hotkey hints. It
is frameless, translucent, draggable and lives in a corner so it interferes
with the captured application as little as possible. Note that with
full-monitor capture the overlay itself appears in the video; excluding the
overlay region from capture is future work (see README).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from recorder.clock import SessionClock


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_STATUS_STYLES = {
    "RECORDING": "color: #e74c3c; font-weight: bold; font-size: 11pt;",
    "PAUSED": "color: #8e44ad; font-weight: bold; font-size: 11pt;",
    "STOPPING": "color: #cc8800; font-weight: bold; font-size: 11pt;",
}


class RecordingOverlay(QWidget):
    """Frameless, translucent, always-on-top widget. Draggable."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(236, 80)
        self._clock: SessionClock | None = None
        self._drag_offset: QPoint | None = None

        self._panel = QWidget(self)
        self._panel.setStyleSheet("background: rgba(20, 20, 20, 210); border-radius: 8px;")
        layout = QVBoxLayout(self._panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self._status = QLabel("● RECORDING")
        self._status.setStyleSheet(_STATUS_STYLES["RECORDING"])
        self._time = QLabel("00:00:00")
        self._time.setStyleSheet("color: white; font-size: 16pt; font-weight: bold;")
        self._hints = QLabel("F8 Stop    F9 Marker    F10 Pause")
        self._hints.setStyleSheet("color: #aaaaaa; font-size: 8pt;")

        layout.addWidget(self._status)
        layout.addWidget(self._time)
        layout.addWidget(self._hints)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)

    def resizeEvent(self, event) -> None:
        self._panel.setGeometry(self.rect())
        super().resizeEvent(event)

    def attach_clock(self, clock: SessionClock) -> None:
        self._clock = clock

    def set_status(self, text: str) -> None:
        self._status.setText(f"● {text}")
        self._status.setStyleSheet(_STATUS_STYLES.get(text, _STATUS_STYLES["RECORDING"]))

    def show_recording(self) -> None:
        self._tick()
        self._timer.start()
        self.show()
        self.raise_()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        if self._clock is not None:
            self._time.setText(format_duration(self._clock.now()))

    # Dragging
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()
