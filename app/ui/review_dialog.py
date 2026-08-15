"""Review queue dialog: inspect candidates and decide their fate.

Wraps :class:`dataset.review.ReviewQueue` — the derived review layer.
Verdicts persist to ``<recording>/review/queue.jsonl``; accepting a frame
verifies its model annotations, rejecting marks them rejected. The raw
recording, perception results and video are never modified.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from dataset.review import ReviewItem, ReviewQueue

_VERDICT_SUFFIX = {"accepted": "accepted", "rejected": "rejected", "skipped": "skipped"}


def _item_text(item: ReviewItem, verdict: str | None = None) -> str:
    text = (
        f"frame {item.frame_index:>6d}  t={item.t:7.2f}s  "
        f"[{item.kind}] {item.reason}  priority={item.priority:.0f}"
    )
    if verdict:
        text += f"  → {_VERDICT_SUFFIX[verdict]}"
    return text


class ReviewDialog(QDialog):
    """Modal review queue: jump to candidates, accept/reject/skip them."""

    def __init__(
        self,
        queue: ReviewQueue,
        on_jump: Callable[[int], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.queue = queue
        self.on_jump = on_jump
        self.setWindowTitle("Review Queue")
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        self._status = QLabel("")
        layout.addWidget(self._status)

        self._list = QListWidget()
        self._list.setToolTip("Double-click an item to jump to its frame")
        self._list.itemSelectionChanged.connect(self._update_buttons)
        self._list.itemDoubleClicked.connect(lambda _item: self._on_jump())
        layout.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._refresh_btn = QPushButton("🔄 Rebuild Queue")
        self._refresh_btn.setToolTip(
            "Re-run all review strategies on this recording (keeps prior verdicts)"
        )
        self._refresh_btn.clicked.connect(self._on_rebuild)
        self._show_decided = QCheckBox("Show decided")
        self._show_decided.stateChanged.connect(lambda _state: self._refresh_list())
        self._jump_btn = QPushButton("⏭ Jump to frame")
        self._jump_btn.clicked.connect(self._on_jump)
        self._accept_btn = QPushButton("✓ Accept")
        self._accept_btn.setToolTip("Verify the frame's model annotations")
        self._accept_btn.clicked.connect(lambda: self._decide("accept"))
        self._reject_btn = QPushButton("✕ Reject")
        self._reject_btn.setToolTip("Mark the frame's model annotations as rejected")
        self._reject_btn.clicked.connect(lambda: self._decide("reject"))
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(lambda: self._decide("skip"))
        for button in (self._refresh_btn, self._jump_btn, self._accept_btn, self._reject_btn, self._skip_btn):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._show_decided)
        row.addStretch(1)
        row.addWidget(self._jump_btn)
        row.addWidget(self._accept_btn)
        row.addWidget(self._reject_btn)
        row.addWidget(self._skip_btn)
        layout.addLayout(row)

        self._refresh_list()

    # -------------------------------------------------------------- state

    def _visible_items(self) -> list[tuple[ReviewItem, str | None]]:
        """(item, verdict or None) rows currently displayed."""
        if self._show_decided.isChecked():
            return [
                (item, self.queue.verdicts.get(str(item.frame_index)))
                for item in self.queue.items
            ]
        return [(item, None) for item in self.queue.pending()]

    def _selected_item(self) -> ReviewItem | None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._visible_items()):
            return None
        return self._visible_items()[row][0]

    def _refresh_list(self) -> None:
        self._list.clear()
        for item, verdict in self._visible_items():
            self._list.addItem(QListWidgetItem(_item_text(item, verdict)))
        pending = len(self.queue.pending())
        self._status.setText(
            f"{pending} pending of {len(self.queue.items)} candidates "
            f"({self.queue.path})"
        )
        self._update_buttons()

    def _update_buttons(self) -> None:
        enabled = self._selected_item() is not None
        self._jump_btn.setEnabled(enabled)
        self._accept_btn.setEnabled(enabled)
        self._reject_btn.setEnabled(enabled)
        self._skip_btn.setEnabled(enabled)

    # ------------------------------------------------------------- actions

    def _on_rebuild(self) -> None:
        self.queue.refresh()
        self._refresh_list()
        self._status.setText(
            f"Rebuilt queue: {len(self.queue.items)} candidates, "
            f"{len(self.queue.pending())} pending"
        )

    def _on_jump(self) -> None:
        item = self._selected_item()
        if item is None or self.on_jump is None:
            return
        self.on_jump(item.frame_index)

    def _decide(self, verdict: str) -> None:
        item = self._selected_item()
        if item is None:
            return
        handler = {
            "accept": self.queue.accept,
            "reject": self.queue.reject,
            "skip": self.queue.skip,
        }[verdict]
        handler(item.frame_index)
        self._refresh_list()
        self._status.setText(f"{verdict} frame {item.frame_index}")