"""Editable timeline bar for the player.

Draws the clip sequence, annotations (markers), keyboard events, the
selection and the playhead. Clicking seeks; dragging left-to-right (or
right-to-left) selects a region; hovering a keyboard-event dot shows the
key and its timestamp in the label at the bottom. Colors:

* clips      — dark slate with visible borders
* markers    — yellow diamonds
* keyboard events — yellow dots
* annotation ticks — cyan diamonds (perception) / green triangles (human)
* event ticks — orange squares (derived high-level events)
* episode ticks — blue bars (episode boundaries)
* selection  — translucent orange with bright edge borders + duration label
* playhead   — red line

Annotation/event ticks sit in a thin lane under the main plot; clicking
one emits ``annotationClicked(t, kind, label)`` instead of seeking
(``kind`` is ``prediction`` | ``human`` | ``event`` | ``episode``).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from editor.timeline import Timeline

logger = logging.getLogger(__name__)

_PAD = 8
_RULER_H = 16
_LABEL_H = 16
_ANNOTATION_LANE_H = 10
_HOVER_RADIUS_PX = 6.0
_CLIP_COLOR = QColor(58, 76, 92)
_CLIP_EDGE = QColor(120, 150, 170)
_SELECTION_COLOR = QColor(230, 140, 60, 110)
_SELECTION_EDGE = QColor(243, 156, 18)
_SELECTION_TEXT = QColor(40, 30, 10)
_PLAYHEAD_COLOR = QColor(231, 76, 60)
_MARKER_COLOR = QColor(241, 196, 15)
_RULER_COLOR = QColor(170, 170, 170)
_KEY_EVENT_COLOR = QColor(241, 196, 15)
_ANNOTATION_COLOR = QColor(26, 188, 156)  # cyan — human annotation
_PREDICTION_COLOR = QColor(150, 111, 214)  # purple — model proposal
_EVENT_COLOR = QColor(230, 126, 34)  # orange — derived high-level event
_EPISODE_COLOR = QColor(52, 152, 219)  # blue — episode boundary

_DRAG_THRESHOLD_PX = 4.0

_NICE_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)


def _format_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes}:{seconds:02d}"


def _nice_step(target_px: float, duration: float, width: float) -> float:
    step = target_px * duration / max(width, 1.0)
    for candidate in _NICE_STEPS:
        if candidate >= step:
            return candidate
    return _NICE_STEPS[-1]


class TimelineWidget(QWidget):
    """Interactive timeline: clips, markers, selection and playhead."""

    seeked = Signal(float)
    selectionChanged = Signal(object)
    annotationClicked = Signal(float, str, str)  # (t, kind, label); kind in prediction|human|event|episode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setMaximumHeight(128)
        self.setMouseTracking(True)
        self._timeline: Timeline | None = None
        self._duration = 0.0
        self._playhead = 0.0
        self._selection: tuple[float, float] | None = None
        self._markers: list[tuple[float, str]] = []
        self._events: list[tuple[float, str]] = []  # (t, key code)
        self._annotation_ticks: list[tuple[float, str, str]] = []  # (t, kind, label)
        self._hovered: tuple[float, str] | None = None
        self._drag_start_x: float | None = None
        self._drag_start_t: float | None = None
        self._drag_active = False

        self._hover_label = QLabel("", self)
        self._hover_label.setStyleSheet("color: #d0d0d0; font-size: 8pt;")
        self._hover_label.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------ model

    def set_model(self, timeline: Timeline | None, duration: float, markers: list[tuple[float, str]]) -> None:
        self._timeline = timeline
        self._duration = max(duration, 0.0)
        self._markers = markers
        self.update()

    def set_playhead(self, t: float) -> None:
        self._playhead = t
        self.update()

    def set_events(self, events: list[tuple[float, str]]) -> None:
        """Keyboard events (t, key code) drawn as yellow dots."""
        self._events = sorted(events)
        self.update()

    def set_annotation_ticks(self, ticks: list[tuple[float, str, str]]) -> None:
        """Annotation ticks: (t, kind, label); kind in {'prediction', 'human'}."""
        self._annotation_ticks = sorted(ticks)
        self.update()

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        self._selection = selection
        self.update()

    def clear_selection(self) -> None:
        self._selection = None
        self.update()

    # ------------------------------------------------------------ geometry

    def _plot_rect(self) -> QRectF:
        rect = QRectF(self.rect())
        return rect.adjusted(_PAD, _RULER_H + 4, -_PAD, -_LABEL_H - _ANNOTATION_LANE_H - 6)

    def _x_to_t(self, x: float) -> float:
        plot = self._plot_rect()
        if plot.width() <= 0 or self._duration <= 0:
            return 0.0
        ratio = (x - plot.left()) / plot.width()
        return min(max(ratio, 0.0), 1.0) * self._duration

    # ------------------------------------------------------------ painting

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = self._plot_rect()

        self._draw_ruler(painter, plot)
        if self._timeline is not None:
            self._draw_clips(painter, plot)
        self._draw_markers(painter, plot)
        self._draw_selection(painter, plot)
        self._draw_events(painter, plot)
        self._draw_annotation_lane(painter, plot)
        self._draw_playhead(painter, plot)
        painter.end()

    def _draw_ruler(self, painter: QPainter, plot: QRectF) -> None:
        if self._duration <= 0:
            return
        step = _nice_step(80, self._duration, plot.width())
        painter.setPen(QPen(_RULER_COLOR, 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        t = 0.0
        while t <= self._duration + 1e-9:
            x = plot.left() + (t / self._duration) * plot.width()
            painter.drawLine(int(x), int(plot.top() - 4), int(x), int(plot.top() - 1))
            label = _format_time(t)
            painter.drawText(int(x + 2), int(plot.top() - 6), label)
            t += step

    def _draw_clips(self, painter: QPainter, plot: QRectF) -> None:
        for clip in self._timeline.clips:
            left = plot.left() + (clip.start / self._duration) * plot.width() if self._duration else plot.left()
            width = (clip.length / self._duration) * plot.width() if self._duration else 0.0
            rect = QRectF(left, plot.top(), max(width - 1, 1.0), plot.height() - 2)
            painter.setBrush(_CLIP_COLOR)
            painter.setPen(QPen(_CLIP_EDGE, 1))
            painter.drawRoundedRect(rect, 3, 3)

    def _draw_markers(self, painter: QPainter, plot: QRectF) -> None:
        if self._duration <= 0:
            return
        for t, label in self._markers:
            x = plot.left() + (t / self._duration) * plot.width()
            y = plot.center().y()
            painter.setBrush(_MARKER_COLOR)
            painter.setPen(QPen(_MARKER_COLOR, 1))
            painter.drawPolygon(
                [
                    QPointF(x, y - 5),
                    QPointF(x - 4, y),
                    QPointF(x, y + 5),
                    QPointF(x + 4, y),
                ]
            )

    def _draw_selection(self, painter: QPainter, plot: QRectF) -> None:
        if self._selection is None or self._duration <= 0:
            return
        in_t, out_t = self._selection
        left = plot.left() + (in_t / self._duration) * plot.width()
        right = plot.left() + (out_t / self._duration) * plot.width()
        rect = QRectF(left, plot.top(), max(right - left, 1.0), plot.height())
        painter.fillRect(rect, _SELECTION_COLOR)
        painter.setPen(QPen(_SELECTION_EDGE, 2))
        painter.drawLine(QPointF(left, plot.top()), QPointF(left, plot.bottom()))
        painter.drawLine(QPointF(right, plot.top()), QPointF(right, plot.bottom()))
        if right - left > 28:
            painter.setPen(QPen(_SELECTION_TEXT, 1))
            painter.drawText(
                QRectF(left, plot.top(), right - left, 14),
                Qt.AlignmentFlag.AlignHCenter,
                f"{out_t - in_t:.1f}s",
            )

    def _draw_events(self, painter: QPainter, plot: QRectF) -> None:
        if not self._events or self._duration <= 0:
            return
        scale = plot.width() / self._duration
        y = plot.center().y()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_KEY_EVENT_COLOR)
        for t, _ in self._events:
            painter.drawEllipse(QPointF(plot.left() + t * scale, y), 2.5, 2.5)

    def _draw_annotation_lane(self, painter: QPainter, plot: QRectF) -> None:
        """Thin lane at the bottom: predictions (purple), human (cyan), events (orange)."""
        if not self._annotation_ticks or self._duration <= 0:
            return
        scale = plot.width() / self._duration
        lane = QRectF(plot.left(), plot.bottom() + 3, plot.width(), _ANNOTATION_LANE_H)
        painter.setPen(Qt.PenStyle.NoPen)
        for t, kind, _ in self._annotation_ticks:
            x = lane.left() + t * scale
            y = lane.center().y()
            if kind == "prediction":
                painter.setBrush(_PREDICTION_COLOR)
                painter.drawPolygon(
                    [
                        QPointF(x, y - 3), QPointF(x - 3, y),
                        QPointF(x, y + 3), QPointF(x + 3, y),
                    ]
                )
            elif kind == "event":
                painter.setBrush(_EVENT_COLOR)
                painter.drawRect(QRectF(x - 3, y - 3, 6, 6))
            elif kind == "episode":
                painter.setBrush(_EPISODE_COLOR)
                painter.drawRect(QRectF(x - 1, lane.top(), 2, lane.height()))
            else:
                painter.setBrush(_ANNOTATION_COLOR)
                painter.drawPolygon(
                    [QPointF(x, y - 3.5), QPointF(x - 3.5, y + 2), QPointF(x + 3.5, y + 2)]
                )

    def _nearest_annotation_tick(self, x: float, y: float) -> tuple[float, str, str] | None:
        """Tick nearest to (x, y) if the click landed in the annotation lane."""
        if not self._annotation_ticks or self._duration <= 0:
            return None
        plot = self._plot_rect()
        lane = QRectF(plot.left(), plot.bottom() + 3, plot.width(), _ANNOTATION_LANE_H)
        if not lane.contains(x, y):
            return None
        scale = plot.width() / self._duration
        best: tuple[float, str, str] | None = None
        best_d = _HOVER_RADIUS_PX
        for t, kind, label in self._annotation_ticks:
            distance = abs(plot.left() + t * scale - x)
            if distance < best_d:
                best_d = distance
                best = (t, kind, label)
        return best

    def _draw_playhead(self, painter: QPainter, plot: QRectF) -> None:
        if self._duration <= 0:
            return
        x = plot.left() + (self._playhead / self._duration) * plot.width()
        painter.setPen(QPen(_PLAYHEAD_COLOR, 2))
        painter.drawLine(int(x), int(plot.top() - _RULER_H), int(x), int(plot.bottom()))

    # ------------------------------------------------------------ events

    def _nearest_event(self, x: float) -> tuple[float, str] | None:
        """Event dot nearest to ``x`` (within the hover radius), else None."""
        if not self._events or self._duration <= 0:
            return None
        plot = self._plot_rect()
        scale = plot.width() / self._duration
        best: tuple[float, str] | None = None
        best_d = _HOVER_RADIUS_PX
        for t, code in self._events:
            distance = abs(plot.left() + t * scale - x)
            if distance < best_d:
                best_d = distance
                best = (t, code)
        return best

    def _update_hover(self, x: float) -> None:
        hovered = self._nearest_event(x)
        if hovered == self._hovered:
            return
        self._hovered = hovered
        if hovered is None:
            self._hover_label.clear()
        else:
            t, code = hovered
            self._hover_label.setText(f"{code} — {t:.2f}s")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            tick = self._nearest_annotation_tick(event.position().x(), event.position().y())
            if tick is not None:
                t, kind, label = tick
                self.annotationClicked.emit(t, kind, label)
                event.accept()
                return
            self._drag_start_x = event.position().x()
            self._drag_start_t = self._x_to_t(self._drag_start_x)
            self._drag_active = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start_x is not None:
            if not self._drag_active and abs(event.position().x() - self._drag_start_x) > _DRAG_THRESHOLD_PX:
                self._drag_active = True
            if self._drag_active:
                lo, hi = sorted((self._drag_start_t, self._x_to_t(event.position().x())))
                self._selection = (lo, hi)
                self.update()
                event.accept()
                return
        else:
            self._update_hover(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start_x is not None:
            if self._drag_active and self._selection is not None:
                self.selectionChanged.emit(self._selection)
            else:
                self.seeked.emit(self._x_to_t(event.position().x()))
                self._selection = None
                self.selectionChanged.emit(None)
                self.update()
            self._drag_start_x = None
            self._drag_start_t = None
            self._drag_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self._hovered = None
        self._hover_label.clear()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._hover_label.setGeometry(_PAD, self.height() - _LABEL_H, self.width() - 2 * _PAD, _LABEL_H)
