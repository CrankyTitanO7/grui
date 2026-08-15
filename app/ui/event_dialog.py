"""Manual event creation dialog for the player.

Lets the user turn a timeline selection into a hand-made high-level event
(stored in the derived ``perception/events.jsonl`` layer — the raw
recording is never touched). The span is shown for reference; ``kind`` is
a free-form name (e.g. ``manual``, ``watch``, ``boss_fight``) and the
label identifies the subject.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class EventDialog(QDialog):
    """Ask for kind + label of a manual event over a selected time span."""

    def __init__(
        self,
        span: tuple[float, float],
        parent: QWidget | None = None,
        kind_default: str = "manual",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Manual Event")
        self.setMinimumWidth(380)
        self.kind = kind_default
        self.label = ""
        start_t, end_t = span
        layout = QVBoxLayout(self)
        info = QLabel(
            f"Selected span: {start_t:.2f}s → {end_t:.2f}s "
            f"({end_t - start_t:.2f}s)\n"
            "The event is stored as derived data — the raw recording is untouched."
        )
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        form = QFormLayout()
        self._kind_edit = QLineEdit(kind_default)
        self._kind_edit.setPlaceholderText("e.g. manual, watch, boss_fight")
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("subject, e.g. boss")
        form.addRow("Kind:", self._kind_edit)
        form.addRow("Label:", self._label_edit)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        ok = QPushButton("Add Event")
        ok.clicked.connect(self._confirm)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        self._label_edit.returnPressed.connect(self._confirm)
        self._label_edit.setFocus()

    def _confirm(self) -> None:
        kind = self._kind_edit.text().strip()
        label = self._label_edit.text().strip()
        if not kind:
            self._kind_edit.setFocus()
            return
        self.kind = kind
        self.label = label
        self.accept()