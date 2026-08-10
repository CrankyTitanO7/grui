"""Player + editor window: play recordings, watch live keys, edit, save.

Loading a recording opens it read-only. Drag on the timeline to select a
region (click to seek); Cut/Copy/Paste/Delete/Trim act on the selection.
Undo/redo and keyboard shortcuts are supported. Saving exports a new raw
recording (original untouched).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.keyboard_view import KeyboardView
from app.ui.timeline_widget import TimelineWidget
from editor.export import export_recording
from editor.timeline import EditSession
from player.event_state import KeyStateTimeline
from player.video_reader import VideoReader
from recorder.config import RecorderConfig
from storage.recording import RecordingData, list_recordings, load_recording

logger = logging.getLogger(__name__)


def _format_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes}:{seconds:02d}.{int((t - int(t)) * 1000):03d}"


def _frame_to_pixmap(frame: np.ndarray, size: tuple[int, int]) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(
        size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class PlayerWindow(QMainWindow):
    """Select a recording, play it with live key visualization, edit, save."""

    def __init__(self, recordings_root: Path | str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Recording Player")
        self.resize(1100, 840)
        self._root = Path(recordings_root) if recordings_root else RecorderConfig().output_dir
        self._recording: RecordingData | None = None
        self._reader: VideoReader | None = None
        self._session: EditSession | None = None
        self._keys: KeyStateTimeline | None = None
        self._clipboard: object = None
        self._playing = False
        self._at_end = False
        self._current_t = 0.0
        self._timeline_sel: tuple[float, float] | None = None

        self._select_all_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.SelectAll), self)
        self._select_all_shortcut.activated.connect(self._on_select_all)
        self._deselect_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._deselect_shortcut.activated.connect(self._on_deselect)
        self._delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self._delete_shortcut.activated.connect(self._on_delete)
        self._cut_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Cut), self)
        self._cut_shortcut.activated.connect(self._on_cut)
        self._copy_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self)
        self._copy_shortcut.activated.connect(self._on_copy)
        self._paste_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self)
        self._paste_shortcut.activated.connect(self._on_paste)
        self._undo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        self._undo_shortcut.activated.connect(self._on_undo)
        self._redo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        self._redo_shortcut.activated.connect(self._on_redo)

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._refresh_recording_list()
        self._set_enabled(False)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Recording:"))
        self._recording_combo = QComboBox()
        self._recording_combo.setMinimumWidth(340)
        self._recording_combo.currentIndexChanged.connect(self._on_combo_changed)
        top.addWidget(self._recording_combo, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_recording_list)
        top.addWidget(browse)
        top.addWidget(refresh)
        root.addLayout(top)

        self._video_label = QLabel("No recording loaded")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumHeight(320)
        self._video_label.setStyleSheet("background: #101010; color: #666; border-radius: 6px;")
        root.addWidget(self._video_label, 1)

        self._keyboard_view = KeyboardView()
        root.addWidget(self._keyboard_view)

        self._timeline = TimelineWidget()
        self._timeline.seeked.connect(self._on_seek_requested)
        self._timeline.selectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._timeline)

        transport = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._on_play_pause)
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._step_btn = QPushButton("⏭ Step")
        self._step_btn.clicked.connect(self._on_step)
        self._time_label = QLabel("0:00.000 / 0:00.000")
        transport.addWidget(self._play_btn)
        transport.addWidget(self._stop_btn)
        transport.addWidget(self._step_btn)
        transport.addWidget(self._time_label, 1)
        root.addLayout(transport)

        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Selection:"))
        self._sel_label = QLabel("—")
        self._sel_label.setStyleSheet("color: #888888;")
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._on_select_all)
        self._deselect_btn = QPushButton("Deselect")
        self._deselect_btn.clicked.connect(self._on_deselect)
        self._trim_btn = QPushButton("Trim")
        self._trim_btn.clicked.connect(self._on_trim)
        self._cut_btn = QPushButton("Cut")
        self._cut_btn.clicked.connect(self._on_cut)
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.clicked.connect(self._on_copy)
        self._paste_btn = QPushButton("Paste")
        self._paste_btn.clicked.connect(self._on_paste)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        edit_row.addWidget(self._sel_label)
        edit_row.addWidget(self._select_all_btn)
        edit_row.addWidget(self._deselect_btn)
        edit_row.addWidget(self._trim_btn)
        edit_row.addWidget(self._cut_btn)
        edit_row.addWidget(self._copy_btn)
        edit_row.addWidget(self._paste_btn)
        edit_row.addWidget(self._delete_btn)
        root.addLayout(edit_row)

        edit_row2 = QHBoxLayout()
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        self._redo_btn = QPushButton("Redo")
        self._redo_btn.clicked.connect(self._on_redo)
        self._reset_btn = QPushButton("Reset Edits")
        self._reset_btn.clicked.connect(self._on_reset)
        self._save_btn = QPushButton("Save Edits as New Recording…")
        self._save_btn.clicked.connect(self._on_save)
        edit_row2.addWidget(self._undo_btn)
        edit_row2.addWidget(self._redo_btn)
        edit_row2.addWidget(self._reset_btn)
        edit_row2.addWidget(self._save_btn, 1)
        root.addLayout(edit_row2)

        self.setCentralWidget(central)
        for button in central.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.statusBar().showMessage("")

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._play_btn, self._stop_btn, self._step_btn,
            self._select_all_btn, self._deselect_btn, self._trim_btn,
            self._cut_btn, self._copy_btn, self._paste_btn, self._delete_btn,
            self._undo_btn, self._redo_btn, self._reset_btn, self._save_btn,
        ):
            widget.setEnabled(enabled)

    def _status(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        logger.info(message)

    # ------------------------------------------------------------- loading

    def _refresh_recording_list(self) -> None:
        current = self._recording_combo.currentText()
        self._recording_combo.blockSignals(True)
        self._recording_combo.clear()
        for directory in list_recordings(self._root):
            self._recording_combo.addItem(directory.name, str(directory))
        self._recording_combo.blockSignals(False)
        if current:
            index = self._recording_combo.findText(current)
            if index >= 0:
                self._recording_combo.setCurrentIndex(index)

    def _on_combo_changed(self, index: int) -> None:
        path = self._recording_combo.itemData(index)
        if path:
            self._load(str(path))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Recording", str(self._root))
        if path:
            self._load(path)

    def _load(self, path: str | Path) -> None:
        try:
            self._teardown_reader()
            recording = load_recording(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to load recording")
            QMessageBox.warning(self, "Imitation Recorder", f"Could not load recording:\n{exc}")
            return

        if not recording.video_path.exists():
            QMessageBox.warning(self, "Imitation Recorder", "Recording has no video file.")
            return

        reader = VideoReader(recording.video_path)
        reader.start()
        if not reader.wait_ready(timeout=5.0):
            reader.stop()
            QMessageBox.warning(self, "Imitation Recorder", "Timed out opening video.")
            return
        if reader.error:
            QMessageBox.warning(self, "Imitation Recorder", f"Video error: {reader.error}")
            reader.stop()
            return
        if reader.width <= 0:
            reader.stop()
            QMessageBox.warning(self, "Imitation Recorder", "Video has no frames.")
            return

        self._reader = reader
        self._recording = recording
        self._keys = KeyStateTimeline(recording.events)
        self._clipboard = None
        self._timeline_sel = None
        self._at_end = False
        self._playing = False
        self._play_btn.setText("▶ Play")

        ranges = recording.metadata.get("edit_clips")
        self._session = EditSession(recording.duration, recording.frame_times, initial_ranges=ranges)
        self._timeline.set_model(
            self._session.timeline,
            self._session.timeline.duration or recording.duration,
            [(m["t"], str(m.get("label", ""))) for m in recording.markers if "t" in m],
        )
        self._timeline.clear_selection()
        self._update_selection_label()

        self._keyboard_view.configure(self._keys.used_codes, (recording.width, recording.height))
        self._timer.setInterval(int(1000 / max(1.0, reader.fps)))
        self._timer.start()
        self._set_enabled(True)
        self.setWindowTitle(f"Recording Player — {recording.directory.name}")
        self._seek_to(0.0)
        self._status(f"Loaded {recording.directory.name}")

    def _teardown_reader(self) -> None:
        if self._reader is not None:
            self._timer.stop()
            self._reader.stop()
            self._reader = None

    # ------------------------------------------------------------ playback

    def _tick(self) -> None:
        if self._reader is None:
            return
        frames = self._reader.drain()
        for frame_index, frame in frames:
            if frame is None:
                self._handle_eof()
                return
            self._display_frame(frame_index, frame)

    def _display_frame(self, frame_index: int, frame: np.ndarray) -> None:
        if self._recording is None:
            return
        t = self._recording.frame_time(frame_index)
        self._current_t = t
        size = self._video_label.size()
        self._video_label.setPixmap(_frame_to_pixmap(frame, (size.width(), size.height())))
        self._update_state_views(t)
        self._timeline.set_playhead(t)
        self._update_time_label()

    def _update_state_views(self, t: float) -> None:
        if self._keys is not None:
            self._keyboard_view.set_state(
                self._keys.active_keys_at(t),
                self._keys.active_buttons_at(t),
                self._keys.mouse_at(t),
            )

    def _handle_eof(self) -> None:
        self._playing = False
        self._at_end = True
        self._play_btn.setText("▶ Play")
        if self._recording is not None:
            self._timeline.set_playhead(self._session.timeline.duration)
            self._current_t = self._session.timeline.duration
            self._update_time_label()

    def _update_time_label(self) -> None:
        if self._recording is None or self._session is None:
            return
        self._time_label.setText(
            f"{_format_time(self._current_t)} / {_format_time(self._session.timeline.duration)}"
        )

    def _on_play_pause(self) -> None:
        if self._reader is None:
            return
        if self._playing:
            self._playing = False
            self._reader.set_playing(False)
            self._play_btn.setText("▶ Play")
        else:
            if self._at_end:
                self._seek_to(0.0)
            self._playing = True
            self._at_end = False
            self._reader.set_playing(True)
            self._play_btn.setText("⏸ Pause")

    def _on_stop(self) -> None:
        self._playing = False
        if self._reader is not None:
            self._reader.set_playing(False)
        self._play_btn.setText("▶ Play")
        self._seek_to(0.0)

    def _on_step(self) -> None:
        if self._reader is None:
            return
        self._playing = False
        self._reader.set_playing(False)
        self._play_btn.setText("▶ Play")
        current_index = self._recording.nearest_frame_index(self._current_t) if self._recording else 0
        self._seek_to_index(current_index + 1)

    def _seek_to_index(self, frame_index: int) -> None:
        if self._reader is None:
            return
        self._reader.seek(frame_index)

    def _seek_to(self, t: float) -> None:
        if self._recording is None or self._reader is None:
            return
        self._at_end = False
        self._seek_to_index(self._recording.nearest_frame_index(t))

    def _on_seek_requested(self, t: float) -> None:
        self._seek_to(t)
        if self._recording is not None:
            self._current_t = t
            self._update_state_views(t)
            self._timeline.set_playhead(t)
            self._update_time_label()

    # ------------------------------------------------------------ editing

    def _selection(self) -> tuple[float, float] | None:
        if self._timeline_sel is None:
            return None
        return self._timeline_sel

    def _on_selection_changed(self, selection: tuple[float, float] | None) -> None:
        if selection is None or self._recording is None:
            self._timeline_sel = None
            self._update_selection_label()
            return
        in_t, out_t = (self._recording.snap_to_frame(t) for t in selection)
        self._timeline_sel = (in_t, out_t) if in_t < out_t else None
        self._timeline.set_selection(self._timeline_sel)
        self._update_selection_label()

    def _update_selection_label(self) -> None:
        if self._timeline_sel is None:
            self._sel_label.setText("—")
        else:
            in_t, out_t = self._timeline_sel
            self._sel_label.setText(f"{_format_time(in_t)} – {_format_time(out_t)} ({out_t - in_t:.1f}s)")

    def _on_select_all(self) -> None:
        if self._session is None:
            return
        self._timeline_sel = (0.0, self._session.timeline.duration)
        self._timeline.set_selection(self._timeline_sel)
        self._update_selection_label()

    def _on_deselect(self) -> None:
        self._timeline_sel = None
        self._timeline.clear_selection()
        self._update_selection_label()

    def _refresh_timeline_view(self) -> None:
        self._timeline.set_model(
            self._session.timeline,
            self._session.timeline.duration,
            [(m["t"], str(m.get("label", ""))) for m in self._recording.markers if "t" in m],
        )
        self._update_time_label()

    def _on_trim(self) -> None:
        selection = self._selection()
        if not selection:
            self._status("Select a region first (drag on the timeline)")
            return
        self._session.trim(*selection)
        self._after_edit(f"Trimmed to {_format_time(selection[0])}–{_format_time(selection[1])}")

    def _on_cut(self) -> None:
        selection = self._selection()
        if not selection:
            self._status("Select a region first (drag on the timeline)")
            return
        self._clipboard = self._session.cut(*selection)
        self._after_edit(f"Cut {_format_time(selection[1] - selection[0])}")

    def _on_copy(self) -> None:
        selection = self._selection()
        if not selection:
            self._status("Select a region first (drag on the timeline)")
            return
        self._clipboard = self._session.copy(*selection)
        self._status(f"Copied {_format_time(selection[1] - selection[0])}")

    def _on_paste(self) -> None:
        if self._clipboard is None:
            self._status("Nothing to paste (Cut or Copy first)")
            return
        self._session.paste(self._recording.snap_to_frame(self._current_t), self._clipboard)
        self._after_edit(f"Pasted at {_format_time(self._current_t)}")

    def _on_delete(self) -> None:
        selection = self._selection()
        if not selection:
            self._status("Select a region first (drag on the timeline)")
            return
        self._session.delete(*selection)
        self._after_edit(f"Deleted {_format_time(selection[1] - selection[0])}")

    def _on_undo(self) -> None:
        if self._session.undo():
            self._after_edit("Undo")

    def _on_redo(self) -> None:
        if self._session.redo():
            self._after_edit("Redo")

    def _on_reset(self) -> None:
        self._session.reset()
        self._after_edit("Reset edits")

    def _after_edit(self, message: str) -> None:
        self._at_end = False
        self._timeline_sel = None
        self._timeline.clear_selection()
        self._update_selection_label()
        self._refresh_timeline_view()
        self._status(message)

    # -------------------------------------------------------------- saving

    def _on_save(self) -> None:
        if self._recording is None or self._session is None:
            return
        target = self._root
        answer = QMessageBox.question(
            self,
            "Save Edits",
            f"Export the edited timeline as a new recording under\n{target}\n\n"
            "The original recording is never modified.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Save:
            return
        try:
            saved = export_recording(
                self._recording,
                self._session.timeline,
                target,
                edit_history=self._session.history,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("export failed")
            QMessageBox.warning(self, "Imitation Recorder", f"Export failed:\n{exc}")
            return
        self._refresh_recording_list()
        self._status(f"Saved to {saved.directory}")

    # ------------------------------------------------------------ shutdown

    def closeEvent(self, event) -> None:  # noqa: N802
        self._teardown_reader()
        super().closeEvent(event)
