"""Transparent annotation overlay for the video frame.

Sits on top of the video label and draws one box per annotation for the
current frame (annotations use normalized 0..1 coordinates, so the overlay
works at any zoom). In edit mode the user can:

* click a box to select it (also possible without editing),
* drag inside a box to move it,
* drag a corner handle to resize it,
* drag on empty space to draw a new box (Esc cancels).

All in normalized coordinates so the window only ever sees 0..1 values;
signal deltas are already normalized too.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

_HANDLE_SIZE = 8.0
_CREATE_MIN_SIZE = 12.0  # px; smaller drags are treated as a click

# AnnotationStatus -> draw color
_STATUS_COLORS: dict[str, QColor] = {
    "predicted": QColor("#ffb74d"),
    "reviewed": QColor("#4fc3f7"),
    "verified": QColor("#81c784"),
    "corrected": QColor("#ba68c8"),
    "rejected": QColor("#e57373"),
}
_DEFAULT_COLOR = QColor("#4fc3f7")
_SELECT_COLOR = QColor("#ffd54f")
_CREATE_COLOR = QColor("#4dd0e1")


class _Box:
    """Client-side overlay item mirroring one annotation for this frame."""

    __slots__ = ("id", "label", "status", "x1", "y1", "x2", "y2")

    def __init__(self, annotation_id: str, label: str, status: str, bbox: tuple[float, float, float, float]):
        self.id = annotation_id
        self.label = label
        self.status = status
        self.x1, self.y1, self.x2, self.y2 = bbox

    def pixel_rect(self, w: float, h: float) -> QRectF:
        return QRectF(
            self.x1 * w, self.y1 * h,
            max(1.0, (self.x2 - self.x1) * w),
            max(1.0, (self.y2 - self.y1) * h),
        )


class AnnotationOverlay(QWidget):
    """Draws/handles annotation boxes over the video label."""

    annotationSelected = Signal(str)
    annotationMoved = Signal(str, float, float)  # id, dx, dy (normalized)
    annotationResized = Signal(str, float, float, float, float)  # id, x1, y1, x2, y2
    annotationCreated = Signal(float, float, float, float)  # x1, y1, x2, y2 (normalized)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._boxes: list[_Box] = []
        self._selected_id: str | None = None
        self._editing = False
        self._hover_handle: str | None = None
        self._drag: str | None = None  # "box" | "handle:<corner>" | "create"
        self._drag_start: tuple[float, float] | None = None
        self._drag_origin: tuple[float, float, float, float] | None = None
        self._create_box: tuple[float, float, float, float] | None = None
        self.setMouseTracking(True)
        self.hide()

    # ------------------------------------------------------------ state

    def set_annotations(self, boxes: list[tuple[str, str, str, tuple[float, float, float, float]]]) -> None:
        """Replace annotated boxes: (id, label, status, (x1, y1, x2, y2)) normalized."""
        self._boxes = [_Box(aid, label, status, bbox) for aid, label, status, bbox in boxes]
        self.update()

    def select_annotation(self, annotation_id: str | None) -> None:
        self._selected_id = annotation_id
        self.update()

    def set_editing(self, editing: bool) -> None:
        self._editing = editing
        self.setCursor(Qt.CursorShape.ArrowCursor if editing else Qt.CursorShape.CrossCursor)
        self._hover_handle = None
        if not editing:
            self._cancel_drag()
        self.update()

    def _cancel_drag(self) -> None:
        self._drag = None
        self._drag_start = None
        self._drag_origin = None
        self._create_box = None

    def _box_at(self, x: float, y: float) -> _Box | None:
        for box in reversed(self._boxes):
            if box.pixel_rect(self.width(), self.height()).contains(x, y):
                return box
        return None

    def _corner_at(self, box: _Box, x: float, y: float) -> str | None:
        rect = box.pixel_rect(self.width(), self.height())
        for corner, point in (
            ("tl", rect.topLeft()), ("tr", rect.topRight()),
            ("bl", rect.bottomLeft()), ("br", rect.bottomRight()),
        ):
            if (x - point.x()) ** 2 + (y - point.y()) ** 2 <= _HANDLE_SIZE ** 2:
                return corner
        return None

    # ------------------------------------------------------------ events

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        if self._editing:
            box = self._box_at(x, y)
            if box is not None:
                handle = self._corner_at(box, x, y)
                if handle is not None:
                    self._drag = f"handle:{handle}"
                    self._drag_start = (x, y)
                    self._drag_origin = (box.x1, box.y1, box.x2, box.y2)
                else:
                    self._drag = "box"
                    self._drag_start = (x, y)
                    self._drag_origin = (box.x1, box.y1, box.x2, box.y2)
            else:
                self._drag = "create"
                self._drag_start = (x, y)
                self._create_box = (x, y, x, y)
        else:
            box = self._box_at(x, y)
            if box is not None:
                self._select(box)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        if self._drag == "create":
            x1, y1, _, _ = self._create_box
            self._create_box = (x1, y1, x, y)
            self.update()
            event.accept()
            return
        if self._drag == "box":
            _box = self._selected_box()
            if _box is not None:
                dx = (x - self._drag_start[0]) / self.width()
                dy = (y - self._drag_start[1]) / self.height()
                self.annotationMoved.emit(_box.id, dx, dy)
                self._drag_start = (x, y)
            event.accept()
            return
        if self._drag is not None and self._drag.startswith("handle:"):
            _box = self._selected_box()
            if _box is not None:
                corner = self._drag.split(":", 1)[1]
                x1, y1, x2, y2 = self._drag_origin
                nx, ny = x / self.width(), y / self.height()
                if "l" in corner:
                    x1, x2 = min(nx, x2), max(nx, x2)
                else:
                    x1, x2 = min(x1, nx), max(x1, nx)
                if "t" in corner:
                    y1, y2 = min(ny, y2), max(ny, y2)
                else:
                    y1, y2 = min(y1, ny), max(y1, ny)
                self.annotationResized.emit(_box.id, x1, y1, x2, y2)
                self._drag_origin = (x1, y1, x2, y2)
            event.accept()
            return
        self._update_hover(x, y)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag == "create":
            x1, y1, x2, y2 = self._create_box
            if abs((x2 - x1) * self.width()) >= _CREATE_MIN_SIZE and abs((y2 - y1) * self.height()) >= _CREATE_MIN_SIZE:
                nx1, nx2 = sorted((x1 / self.width(), x2 / self.width()))
                ny1, ny2 = sorted((y1 / self.height(), y2 / self.height()))
                self.annotationCreated.emit(nx1, ny1, nx2, ny2)
            self._cancel_drag()
            self.update()
        else:
            self._cancel_drag()
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_drag()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_handle = None
        self.setCursor(Qt.CursorShape.ArrowCursor if self._editing else Qt.CursorShape.CrossCursor)
        super().leaveEvent(event)

    def _update_hover(self, x: float, y: float) -> None:
        handle: str | None = None
        box = self._box_at(x, y)
        if box is not None and self._editing:
            handle = self._corner_at(box, x, y)
        if not self._editing and box is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif handle is not None:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self._editing:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._hover_handle = handle

    def _selected_box(self) -> _Box | None:
        if self._selected_id is None:
            return None
        for box in self._boxes:
            if box.id == self._selected_id:
                return box
        return None

    def _select(self, box: _Box) -> None:
        self._selected_id = box.id
        self.annotationSelected.emit(box.id)
        self.update()

    # ------------------------------------------------------------ drawing

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for box in self._boxes:
            self._paint_box(painter, box)
        if self._create_box is not None:
            x1, y1, x2, y2 = self._create_box
            rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            painter.setPen(QPen(_CREATE_COLOR, 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def _paint_box(self, painter: QPainter, box: _Box) -> None:
        rect = box.pixel_rect(self.width(), self.height())
        color = _STATUS_COLORS.get(box.status, _DEFAULT_COLOR)
        selected = box.id == self._selected_id
        pen = QPen(_SELECT_COLOR if selected else color, 2.0 if selected else 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        label = f"{box.label}  [{box.status}]" if box.status != "predicted" else box.label
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(label) + 8
        flag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if rect.top() - 16 < 0:
            label_rect = QRectF(rect.left(), rect.top() + 2, tw, 14)
            painter.drawText(label_rect, flag, label)
        else:
            label_rect = QRectF(rect.left(), rect.top() - 14, tw, 14)
            painter.drawText(label_rect, flag, label)
            painter.setPen(pen)
            painter.drawLine(
                rect.left(), rect.top(),
                rect.left() + tw, rect.top(),
            )

        if selected and self._editing:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
                painter.drawRect(
                    int(point.x() - _HANDLE_SIZE / 2), int(point.y() - _HANDLE_SIZE / 2),
                    int(_HANDLE_SIZE), int(_HANDLE_SIZE),
                )