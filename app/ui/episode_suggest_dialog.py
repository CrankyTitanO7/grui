"""Episode suggestion dialog: choose which signals feed the segmentation.

Opened from the player window's episode row. The user checks which signals
to use, tunes the inactivity threshold, presses "Suggest" to preview the
resulting episodes, and "Apply" to write them to ``<recording>/episodes.jsonl``.
Selecting an episode and pressing "Play segment" asks the player window to
loop-play that raw-time range while the dialog stays open, so the suggested
boundaries can be eyeballed against the actual frames. Like the rest of
episode segmentation this writes *derived metadata* only — the raw recording
is never touched.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dataset.episodes import Episode, suggest_episodes, write_episodes
from storage.recording import RecordingData

logger = logging.getLogger(__name__)


class EpisodeSuggestDialog(QDialog):
    """Preview and apply episode boundaries suggested from recording signals.

    Emits ``playRequested(start, end)`` when the user picks a preview episode
    (the player window loops that raw-time range), ``stopRequested()`` to end
    the loop, and ``applied()`` after writing episodes.jsonl.
    """

    playRequested = Signal(float, float)  # (start, end) in raw recording time
    stopRequested = Signal()
    applied = Signal()

    def __init__(self, recording: RecordingData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recording = recording
        self.episodes: list[Episode] = []
        self.setWindowTitle("Suggest Episodes")
        self.setMinimumWidth(520)
        self._build_ui()
        self._on_suggest()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            f"{self.recording.directory.name} · {self.recording.duration:.1f}s"
        )
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        signals = QHBoxLayout()
        self._use_inactivity = QCheckBox("Inactivity gaps")
        self._use_inactivity.setChecked(True)
        self._use_inactivity.setToolTip("No input for N seconds → boundary")
        self._use_inactivity.toggled.connect(self._on_suggest)
        self._min_inactive = QDoubleSpinBox()
        self._min_inactive.setRange(1.0, 60.0)
        self._min_inactive.setDecimals(1)
        self._min_inactive.setValue(5.0)
        self._min_inactive.setSuffix(" s")
        self._min_inactive.valueChanged.connect(lambda _v: self._on_suggest())
        self._use_markers = QCheckBox("episode: markers")
        self._use_markers.setChecked(True)
        self._use_markers.setToolTip("User markers labelled episode:<name>")
        self._use_markers.toggled.connect(self._on_suggest)
        signals.addWidget(self._use_inactivity)
        signals.addWidget(self._min_inactive)
        signals.addWidget(self._use_markers)
        signals.addStretch(1)
        layout.addLayout(signals)

        signals2 = QHBoxLayout()
        self._use_visual = QCheckBox("Scene changes")
        self._use_visual.setToolTip("Detect visual scene changes (reads the video frames)")
        self._use_visual.toggled.connect(self._on_suggest)
        self._use_events = QCheckBox("Perception events")
        self._use_events.setToolTip("Split at derived perception event starts (perception/events.jsonl)")
        self._use_events.toggled.connect(self._on_suggest)
        self._use_input_changes = QCheckBox("Input jumps")
        self._use_input_changes.setToolTip("Split where the held keys/buttons changed sharply")
        self._use_input_changes.toggled.connect(self._on_suggest)
        signals2.addWidget(self._use_visual)
        signals2.addWidget(self._use_events)
        signals2.addWidget(self._use_input_changes)
        signals2.addStretch(1)
        layout.addLayout(signals2)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Split episodes longer than:"))
        self._max_episode = QSpinBox()
        self._max_episode.setRange(0, 3600)
        self._max_episode.setValue(0)
        self._max_episode.setSuffix(" s (0 = off)")
        self._max_episode.valueChanged.connect(lambda _v: self._on_suggest())
        split_row.addWidget(self._max_episode)
        self._preview_status = QLabel("")
        self._preview_status.setStyleSheet("color: #888888;")
        split_row.addWidget(self._preview_status)
        split_row.addStretch(1)
        layout.addLayout(split_row)

        self._list = QListWidget()
        self._list.setMinimumHeight(180)
        self._list.setToolTip("Select an episode, then press 'Play segment' to loop it in the player")
        self._list.itemDoubleClicked.connect(lambda _item: self._on_play_segment())
        layout.addWidget(self._list, 1)

        preview = QHBoxLayout()
        self._play_segment_btn = QPushButton("▶ Play segment")
        self._play_segment_btn.setToolTip(
            "Loop-play the selected episode's frame range in the player window"
        )
        self._play_segment_btn.clicked.connect(self._on_play_segment)
        self._stop_segment_btn = QPushButton("■ Stop")
        self._stop_segment_btn.setToolTip("Stop looping the segment preview")
        self._stop_segment_btn.clicked.connect(self.stopRequested.emit)
        preview_hint = QLabel("Click outside the dialog to watch the player loop the segment.")
        preview_hint.setStyleSheet("color: #666666;")
        preview.addWidget(self._play_segment_btn)
        preview.addWidget(self._stop_segment_btn)
        preview.addWidget(preview_hint)
        preview.addStretch(1)
        layout.addLayout(preview)

        buttons = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest")
        self._suggest_btn.clicked.connect(self._on_suggest)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip(
            "Write the suggested episodes to episodes.jsonl (derived data only)"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        buttons.addWidget(self._suggest_btn)
        buttons.addWidget(self._apply_btn)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _on_suggest(self, *_args: object) -> None:
        if not hasattr(self, "_list"):  # signals fire mid-UI-build; ignore
            return
        max_episode_s = (
            self._max_episode.value() if self._max_episode.value() > 0 else None
        )
        try:
            episodes = suggest_episodes(
                self.recording,
                min_inactivity=self._min_inactive.value(),
                use_inactivity=self._use_inactivity.isChecked(),
                use_markers=self._use_markers.isChecked(),
                use_visual=self._use_visual.isChecked(),
                use_events=self._use_events.isChecked(),
                use_input_changes=self._use_input_changes.isChecked(),
                max_episode_s=max_episode_s,
            )
        except Exception as exc:  # noqa: BLE001 - result shown to the user
            logger.exception("episode suggestion failed")
            self.episodes = []
            self._list.clear()
            self._preview_status.setText("suggestion failed")
            QMessageBox.warning(self, "Suggest Episodes", f"Suggestion failed:\n{exc}")
            return
        self.episodes = episodes
        self._list.clear()
        for i, episode in enumerate(episodes, 1):
            self._list.addItem(
                f"{i:>3}.  {episode.start:7.2f}s → {episode.end:7.2f}s  "
                f"({episode.reason or 'full'})"
            )
        self._preview_status.setText(f"{len(episodes)} episode(s)")
        self._play_segment_btn.setEnabled(bool(episodes))

    def _on_play_segment(self) -> None:
        """Ask the player to loop-play the raw-time range of the selected episode."""
        index = self._list.currentRow()
        if 0 <= index < len(self.episodes):
            episode = self.episodes[index]
            self.playRequested.emit(episode.start, episode.end)

    def _on_apply(self) -> None:
        if not self.episodes:
            self._on_suggest()
            if not self.episodes:
                return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            path = write_episodes(self.recording.directory, self.episodes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("writing episodes failed")
            QMessageBox.warning(self, "Suggest Episodes", f"Failed to write episodes:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        logger.info("wrote %d episodes → %s", len(self.episodes), path)
        self.applied.emit()
        self.accept()
