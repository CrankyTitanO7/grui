"""Player + editor window: play recordings, watch live keys, edit, save.

Loading a recording opens it read-only. Drag on the timeline to select a
region (click to seek); Cut/Copy/Paste/Delete/Trim act on the selection.
Undo/redo and keyboard shortcuts are supported. Saving exports a new raw
recording (original untouched); Build Dataset generates observation->action
training samples from the loaded recording.

Perception and annotations are optional tools layered on top: perception
results can be shown as boxes on the frame, and the annotation editor lets
the user inspect/accept/correct model proposals or draw their own boxes.
Annotations are saved back to the derived ``annotations/`` layer — the raw
recording, video and perception results are never modified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from annotation.store import (
    AnnotationStore,
    annotation_dedup_key,
    detection_dedup_key,
    load_annotations,
)
from annotation.types import AnnotationStatus
from app.ui.annotation_overlay import AnnotationOverlay
from app.ui.event_dialog import EventDialog
from app.ui.keyboard_view import KeyboardView
from app.ui.timeline_widget import TimelineWidget
from editor.export import export_recording
from editor.timeline import EditSession, remap_events
from perception.events import Event, read_events, write_events
from perception.types import BoundingBox
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
        self.setMinimumWidth(760)
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
        self._perception: dict[int, list] = {}  # frame_index -> [Detection]
        self._perception_manifest = None
        self._zoom = 1.0  # relative to fit-to-window size
        self._current_frame: np.ndarray | None = None
        self._current_frame_index = 0
        self._annotations: AnnotationStore | None = None
        self._selected_annotation_id: str | None = None
        self._annotation_mode = False
        self._events: list[Event] = []
        self._review_queue = None

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
        self._video_label.setMinimumHeight(240)
        self._video_label.setStyleSheet("background: #101010; color: #666; border-radius: 6px;")
        self._video_scroll = QScrollArea()
        self._video_scroll.setWidgetResizable(False)
        self._video_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_scroll.viewport().setStyleSheet("background: #101010;")
        self._video_scroll.setWidget(self._video_label)
        root.addWidget(self._video_scroll, 1)

        self._keyboard_view = KeyboardView()
        root.addWidget(self._keyboard_view)

        self._timeline = TimelineWidget()
        self._timeline.seeked.connect(self._on_seek_requested)
        self._timeline.selectionChanged.connect(self._on_selection_changed)
        self._timeline.annotationClicked.connect(self._on_annotation_tick_clicked)
        root.addWidget(self._timeline)

        transport = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._on_play_pause)
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._step_btn = QPushButton("⏭ Step")
        self._step_btn.clicked.connect(self._on_step)
        self._time_label = QLabel("0:00.000 / 0:00.000")
        self._show_input = QCheckBox("Show input state")
        self._show_input.setChecked(True)
        self._show_input.setToolTip("Show/hide the keyboard monitor and mouse graph")
        self._show_input.toggled.connect(self._keyboard_view.setVisible)
        transport.addWidget(self._play_btn)
        transport.addWidget(self._stop_btn)
        transport.addWidget(self._step_btn)
        transport.addWidget(self._time_label, 1)
        transport.addWidget(self._show_input)
        root.addLayout(transport)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self._zoom_out_btn = QPushButton("-")
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self._on_zoom_out)
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(25, 400)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setToolTip("Zoom relative to window fit (100% = fit)")
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self._on_zoom_in)
        self._zoom_fit_btn = QPushButton("Fit")
        self._zoom_fit_btn.setToolTip("Fit frame to window (100%)")
        self._zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self._zoom_pct_label = QLabel("100%")
        self._zoom_pct_label.setMinimumWidth(44)
        zoom_row.addWidget(self._zoom_out_btn)
        zoom_row.addWidget(self._zoom_slider, 1)
        zoom_row.addWidget(self._zoom_in_btn)
        zoom_row.addWidget(self._zoom_fit_btn)
        zoom_row.addWidget(self._zoom_pct_label)
        root.addLayout(zoom_row)

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
        self._dataset_btn = QPushButton("Build Dataset…")
        self._dataset_btn.clicked.connect(self._on_build_dataset)
        self._perception_btn = QPushButton("Perception…")
        self._perception_btn.clicked.connect(self._on_perception)
        edit_row2.addWidget(self._undo_btn)
        edit_row2.addWidget(self._redo_btn)
        edit_row2.addWidget(self._reset_btn)
        edit_row2.addWidget(self._save_btn, 1)
        edit_row2.addWidget(self._dataset_btn)
        edit_row2.addWidget(self._perception_btn)
        root.addLayout(edit_row2)

        overlay_row = QHBoxLayout()
        self._show_perception = QCheckBox("Show perception detections")
        self._show_perception.setChecked(False)
        self._show_perception.setToolTip(
            "Show all model detection boxes: dashed = not yet imported "
            "(click to import as a draft annotation), solid = already imported. "
            "Unchecks 'Show annotations' — the two views are exclusive."
        )
        self._show_perception.toggled.connect(self._on_perception_toggled)
        self._perception_status = QLabel("")
        self._perception_status.setStyleSheet("color: #888888;")
        self._next_perception_btn = QPushButton("⏭ Next Detection")
        self._next_perception_btn.setToolTip(
            "Skip to the next frame with perception detections (wraps to the first)"
        )
        self._next_perception_btn.clicked.connect(self._on_next_perception)
        overlay_row.addWidget(self._show_perception)
        overlay_row.addWidget(self._next_perception_btn)
        overlay_row.addWidget(self._perception_status, 1)
        root.addLayout(overlay_row)

        ann_row = QHBoxLayout()
        self._show_annotations = QCheckBox("Show annotations")
        self._show_annotations.setToolTip(
            "Show the human annotation layer for this frame. Unchecks "
            "'Show perception detections' — the two views are exclusive."
        )
        self._show_annotations.toggled.connect(self._on_annotations_toggled)
        self._edit_annotations_btn = QPushButton("✎ Edit Annotations")
        self._edit_annotations_btn.setCheckable(True)
        self._edit_annotations_btn.setToolTip(
            "Toggle annotation editing: click boxes to select, drag to move, "
            "drag handles to resize, drag empty space to draw a new box"
        )
        self._edit_annotations_btn.toggled.connect(self._on_annotation_mode_toggled)
        self._import_annotations_btn = QPushButton("← Import Perception")
        self._import_annotations_btn.setToolTip(
            "Import perception detections as draft annotations. Model predictions "
            "are guesses — import frame-by-frame and correct/delete by hand."
        )
        self._import_annotations_btn.clicked.connect(self._on_import_annotations)
        self._annotation_status = QLabel("")
        self._annotation_status.setStyleSheet("color: #888888;")
        ann_row.addWidget(self._show_annotations)
        ann_row.addWidget(self._edit_annotations_btn)
        ann_row.addWidget(self._import_annotations_btn)
        ann_row.addWidget(self._annotation_status, 1)
        root.addLayout(ann_row)

        edit_ann_row = QHBoxLayout()
        self._prev_ann_btn = QPushButton("◀")
        self._prev_ann_btn.setToolTip(
            "Previous annotation (navigate by time, wraps around)"
        )
        self._prev_ann_btn.clicked.connect(self._on_prev_annotation)
        self._next_ann_btn = QPushButton("▶")
        self._next_ann_btn.setToolTip(
            "Next annotation (navigate by time, wraps around)"
        )
        self._next_ann_btn.clicked.connect(self._on_next_annotation)
        self._ann_label_edit = QLineEdit()
        self._ann_label_edit.setPlaceholderText("Selected annotation label…")
        self._ann_label_edit.setMaximumWidth(220)
        self._apply_label_btn = QPushButton("Rename")
        self._apply_label_btn.clicked.connect(self._on_rename_annotation)
        self._verify_btn = QPushButton("✓ Verify")
        self._verify_btn.clicked.connect(self._on_verify_annotation)
        self._delete_ann_btn = QPushButton("✕ Delete")
        self._delete_ann_btn.clicked.connect(self._on_delete_annotation)
        self._undo_ann_btn = QPushButton("↶")
        self._undo_ann_btn.setToolTip("Undo annotation change")
        self._undo_ann_btn.clicked.connect(self._on_annotation_undo)
        self._redo_ann_btn = QPushButton("↷")
        self._redo_ann_btn.setToolTip("Redo annotation change")
        self._redo_ann_btn.clicked.connect(self._on_annotation_redo)
        self._save_annotations_btn = QPushButton("Save Annotations")
        self._save_annotations_btn.clicked.connect(self._on_save_annotations)
        edit_ann_row.addWidget(self._prev_ann_btn)
        edit_ann_row.addWidget(self._next_ann_btn)
        edit_ann_row.addWidget(self._ann_label_edit)
        edit_ann_row.addWidget(self._apply_label_btn)
        edit_ann_row.addWidget(self._verify_btn)
        edit_ann_row.addWidget(self._delete_ann_btn)
        edit_ann_row.addWidget(self._undo_ann_btn)
        edit_ann_row.addWidget(self._redo_ann_btn)
        edit_ann_row.addWidget(self._save_annotations_btn)
        edit_ann_row.addStretch(1)
        root.addLayout(edit_ann_row)

        events_row = QHBoxLayout()
        self._events_label = QLabel("⚡ Events:")
        self._events_combo = QComboBox()
        self._events_combo.setToolTip(
            "Derived high-level events (perception/events.jsonl) — select one to jump to it"
        )
        self._events_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._events_combo.currentIndexChanged.connect(self._on_event_selected)
        self._add_event_btn = QPushButton("＋ Add Event")
        self._add_event_btn.setToolTip(
            "Create a manual event from the timeline selection — drag to select "
            "a region first (Stored under perception/events.jsonl, raw data untouched)"
        )
        self._add_event_btn.clicked.connect(self._on_add_event)
        self._delete_event_btn = QPushButton("✕ Delete Event")
        self._delete_event_btn.setToolTip(
            "Remove the selected event from perception/events.jsonl "
            "(derived data only, raw data untouched)"
        )
        self._delete_event_btn.clicked.connect(self._on_delete_event)
        self._events_status = QLabel("")
        self._events_status.setStyleSheet("color: #888888;")
        events_row.addWidget(self._events_label)
        events_row.addWidget(self._events_combo)
        events_row.addWidget(self._add_event_btn)
        events_row.addWidget(self._delete_event_btn)
        events_row.addWidget(self._events_status, 1)
        root.addLayout(events_row)

        review_row = QHBoxLayout()
        self._review_btn = QPushButton("Review…")
        self._review_btn.setToolTip(
            "Open the review queue: frames flagged by review strategies "
            "(low-confidence detections, rare actions, visual novelty, "
            "unreviewed predictions) for a human look. Voting updates the "
            "review/queue.jsonl and annotation layers — raw data untouched"
        )
        self._review_btn.clicked.connect(self._on_review)
        self._review_status = QLabel("")
        self._review_status.setStyleSheet("color: #888888;")
        review_row.addWidget(self._review_btn)
        review_row.addWidget(self._review_status, 1)
        root.addLayout(review_row)

        self.setCentralWidget(central)
        for button in central.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.statusBar().showMessage("")

        self._annotation_overlay = AnnotationOverlay(self._video_label)
        self._annotation_overlay.setGeometry(self._video_label.rect())
        self._annotation_overlay.annotationSelected.connect(self._on_annotation_selected)
        self._annotation_overlay.annotationMoved.connect(self._on_annotation_moved)
        self._annotation_overlay.annotationResized.connect(self._on_annotation_resized)
        self._annotation_overlay.annotationCreated.connect(self._on_annotation_created)
        self._annotation_overlay.raise_()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._play_btn, self._stop_btn, self._step_btn,
            self._zoom_out_btn, self._zoom_slider, self._zoom_in_btn,
            self._zoom_fit_btn, self._next_perception_btn,
            self._select_all_btn, self._deselect_btn, self._trim_btn,
            self._cut_btn, self._copy_btn, self._paste_btn, self._delete_btn,
            self._undo_btn, self._redo_btn, self._reset_btn, self._save_btn,
            self._dataset_btn, self._perception_btn,
            self._show_annotations, self._edit_annotations_btn,
            self._prev_ann_btn, self._next_ann_btn,
            self._events_combo, self._events_label, self._add_event_btn,
            self._delete_event_btn,
            self._review_btn,
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
            QMessageBox.warning(self, "GRUI", f"Could not load recording:\n{exc}")
            return

        if not recording.video_path.exists():
            QMessageBox.warning(self, "GRUI", "Recording has no video file.")
            return

        reader = VideoReader(recording.video_path)
        reader.start()
        if not reader.wait_ready(timeout=5.0):
            reader.stop()
            QMessageBox.warning(self, "GRUI", "Timed out opening video.")
            return
        if reader.error:
            QMessageBox.warning(self, "GRUI", f"Video error: {reader.error}")
            reader.stop()
            return
        if reader.width <= 0:
            reader.stop()
            QMessageBox.warning(self, "GRUI", "Video has no frames.")
            return

        self._reader = reader
        self._recording = recording
        self._keys = KeyStateTimeline(recording.events)
        self._clipboard = None
        self._timeline_sel = None
        self._at_end = False
        self._playing = False
        self._play_btn.setText("▶ Play")
        self._annotations = load_annotations(recording.directory)
        self._selected_annotation_id = None
        self._annotation_mode = False
        self._edit_annotations_btn.setChecked(False)
        self._update_annotation_status_text()

        ranges = recording.metadata.get("edit_clips")
        self._session = EditSession(recording.duration, recording.frame_times, initial_ranges=ranges)
        self._timeline.set_model(
            self._session.timeline,
            self._session.timeline.duration or recording.duration,
            [(m["t"], str(m.get("label", ""))) for m in recording.markers if "t" in m],
        )
        self._timeline.set_events(self._timeline_events())
        self._timeline.clear_selection()
        self._update_selection_label()

        self._keyboard_view.configure(self._keys.used_codes, (recording.width, recording.height))
        self._timer.setInterval(int(1000 / max(1.0, reader.fps)))
        self._timer.start()
        self._set_enabled(True)
        self._load_perception(recording)
        self._refresh_annotation_view()
        self._load_events()
        self._load_review_queue(recording)
        self.setWindowTitle(f"Recording Player — {recording.directory.name}")
        self._seek_to(0.0)
        self._status(f"Loaded {recording.directory.name}")

    def _teardown_reader(self) -> None:
        if self._reader is not None:
            self._timer.stop()
            self._reader.stop()
            self._reader = None
        self._perception = {}
        self._perception_manifest = None
        self._current_frame = None
        self._show_perception.setChecked(False)
        self._next_perception_btn.setEnabled(False)
        self._annotations = None
        self._selected_annotation_id = None
        self._annotation_mode = False
        self._events = []
        self._events_combo.blockSignals(True)
        self._events_combo.clear()
        self._events_combo.blockSignals(False)
        self._events_status.setText("")
        self._delete_event_btn.setEnabled(False)
        self._edit_annotations_btn.setChecked(False)
        self._show_annotations.setChecked(False)
        self._annotation_overlay.set_editing(False)
        self._annotation_overlay.set_annotations([])
        self._annotation_overlay.select_annotation(None)
        self._annotation_overlay.hide()
        self._review_queue = None
        self._review_status.setText("")

    # --------------------------------------------------------- perception

    def _load_perception(self, recording: RecordingData) -> None:
        """Load derived perception results, if any (optional, never required)."""
        from perception.runner import CachedAnalysis

        cached = CachedAnalysis(recording.directory / "perception")
        if not cached.exists:
            self._perception_status.setText("No perception results for this recording")
            self._next_perception_btn.setEnabled(False)
            return
        manifest = cached.read_manifest()
        self._perception_manifest = manifest
        by_frame: dict[int, list] = {}
        for result in cached.read_results():
            by_frame.setdefault(result.frame_index, []).extend(result.detections)
        self._perception = by_frame
        prompts = ", ".join(manifest.prompts) if manifest else ""
        self._perception_status.setText(
            f"Perception: {manifest.provider} ({prompts}) — {len(by_frame)} frames with detections"
        )
        self._next_perception_btn.setEnabled(bool(by_frame))

    def _on_perception_toggled(self, checked: bool) -> None:
        if checked:
            if not self._perception:
                self._status("No perception detections loaded for this recording")
            self._show_annotations.setChecked(False)
        self._update_annotation_overlay()

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
        self._current_frame_index = frame_index
        self._current_frame = frame
        self._render_frame()
        self._update_annotation_overlay()
        self._update_state_views(t)
        self._timeline.set_playhead(t)
        self._update_time_label()

    def _render_size(self) -> tuple[int, int]:
        """Display size for the current frame: fit-to-viewport scaled by zoom."""
        viewport = self._video_scroll.viewport().size()
        return (
            max(1, int(viewport.width() * self._zoom)),
            max(1, int(viewport.height() * self._zoom)),
        )

    def _render_frame(self) -> None:
        """Redraw the current frame at the current zoom (no-op until one is loaded)."""
        if self._current_frame is None:
            return
        pixmap = _frame_to_pixmap(self._current_frame, self._render_size())
        self._video_label.setPixmap(pixmap)
        self._video_label.setMinimumSize(pixmap.size())
        overlay = getattr(self, "_annotation_overlay", None)
        if overlay is not None:
            overlay.setGeometry(self._video_label.rect())
            overlay.raise_()

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

    # ---------------------------------------------------------------- zoom

    def _on_zoom_changed(self, value: int) -> None:
        self._zoom = value / 100.0
        self._zoom_pct_label.setText(f"{value}%")
        self._render_frame()

    def _on_zoom_in(self) -> None:
        self._zoom_slider.setValue(min(self._zoom_slider.maximum(), self._zoom_slider.value() + 25))

    def _on_zoom_out(self) -> None:
        self._zoom_slider.setValue(max(self._zoom_slider.minimum(), self._zoom_slider.value() - 25))

    def _on_zoom_fit(self) -> None:
        self._zoom_slider.setValue(100)

    # ------------------------------------------------- next detection skip

    def _on_next_perception(self) -> None:
        """Skip to the next frame with perception detections (wraps to the first)."""
        if self._recording is None or not self._perception:
            return
        current = self._recording.nearest_frame_index(self._current_t)
        upcoming = sorted(fi for fi in self._perception if fi > current)
        target = upcoming[0] if upcoming else min(self._perception)
        self._playing = False
        self._at_end = False
        if self._reader is not None:
            self._reader.set_playing(False)
        self._play_btn.setText("▶ Play")
        self._seek_to_index(target)
        t = self._recording.frame_time(target)
        self._current_t = t
        self._update_state_views(t)
        self._timeline.set_playhead(t)
        self._update_time_label()
        if upcoming:
            self._status(f"Next detection at {_format_time(t)}")
        else:
            self._status(f"No more detections — wrapped to {_format_time(t)}")

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

    def _timeline_events(self) -> list[tuple[float, str]]:
        """Keyboard events (t, key code) mapped through the edited timeline."""
        if self._recording is None or self._session is None:
            return []
        remapped = remap_events(self._recording.events, self._session.timeline)
        return [
            (float(event["t"]), str(event.get("code") or event.get("char") or "?"))
            for event in remapped
            if event.get("device") == "keyboard"
        ]

    def _refresh_timeline_view(self) -> None:
        self._timeline.set_model(
            self._session.timeline,
            self._session.timeline.duration,
            [(m["t"], str(m.get("label", ""))) for m in self._recording.markers if "t" in m],
        )
        self._timeline.set_events(self._timeline_events())
        self._timeline.set_annotation_ticks(self._annotation_ticks())
        self._update_time_label()

    def _warn_no_selection(self) -> None:
        QMessageBox.warning(
            self,
            "No Selection",
            "Select a region first — drag on the timeline, or use Select All (Ctrl+A).",
        )

    def _on_trim(self) -> None:
        selection = self._selection()
        if not selection:
            self._warn_no_selection()
            return
        self._session.trim(*selection)
        self._after_edit(f"Trimmed to {_format_time(selection[0])}–{_format_time(selection[1])}")

    def _on_cut(self) -> None:
        selection = self._selection()
        if not selection:
            self._warn_no_selection()
            return
        self._clipboard = self._session.cut(*selection)
        self._after_edit(f"Cut {_format_time(selection[1] - selection[0])}")

    def _on_copy(self) -> None:
        selection = self._selection()
        if not selection:
            self._warn_no_selection()
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
            self._warn_no_selection()
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
            QMessageBox.warning(self, "GRUI", f"Export failed:\n{exc}")
            return
        self._refresh_recording_list()
        self._status(f"Saved to {saved.directory}")

    # ------------------------------------------------------------ shutdown

    def _on_build_dataset(self) -> None:
        if self._recording is None:
            return
        from app.ui.dataset_dialog import DatasetDialog

        DatasetDialog(self._recording, self).exec()

    def _on_perception(self) -> None:
        if self._recording is None:
            return
        from app.ui.perception_dialog import PerceptionDialog

        dialog = PerceptionDialog(self._recording, self)
        dialog.finished.connect(self._on_perception_done)
        dialog.exec()

    def _on_perception_done(self, _result: int) -> None:
        if self._recording is not None:
            self._load_perception(self._recording)
            self._refresh_annotation_view()
            self._load_events()

    # ------------------------------------------------------------- events

    def _load_events(self) -> None:
        """Load derived events (perception/events.jsonl) into the navigation combo."""
        if self._recording is None:
            return
        self._events = read_events(self._recording.directory)
        self._events_combo.blockSignals(True)
        self._events_combo.clear()
        for event in self._events:
            span = (
                f"{event.start_t:.2f}s" if event.end_t == event.start_t
                else f"{event.start_t:.2f}s-{event.end_t:.2f}s"
            )
            self._events_combo.addItem(f"{event.kind} {event.label} @ {span}", event)
        self._events_combo.blockSignals(False)
        self._delete_event_btn.setEnabled(bool(self._events))
        if self._events:
            self._events_status.setText(f"{len(self._events)} derived event(s)")
        else:
            self._events_status.setText(
                "No events — run `grui perception events` to detect them"
            )
        self._refresh_timeline_view()

    def _on_event_selected(self, index: int) -> None:
        """Jump from the Events combo to the event's start frame."""
        if index < 0 or index >= len(self._events) or self._recording is None:
            return
        event = self._events[index]
        raw_t = event.start_t
        self._seek_to(raw_t)
        self._current_t = raw_t
        self._update_state_views(raw_t)
        self._timeline.set_playhead(self._raw_to_edited(raw_t))
        self._update_time_label()
        self._status(f"Event: {event.kind} {event.label} at {event.start_t:.2f}s")

    def _on_add_event(self) -> None:
        """Turn the timeline selection into a manual event (derived data only)."""
        if self._recording is None:
            return
        selection = self._selection()
        if not selection:
            self._warn_no_selection()
            return
        edited_start, edited_end = selection
        raw_start = self._edited_to_raw(edited_start)
        raw_end = self._edited_to_raw(edited_end - 1e-9)
        if raw_start is None or raw_end is None or raw_end <= raw_start:
            QMessageBox.warning(
                self, "Add Event",
                "The selection lies entirely inside a removed region.",
            )
            return
        dialog = EventDialog((edited_start, edited_end), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        recording = load_recording(self._recording.directory)
        raw_start = recording.snap_to_frame(raw_start)
        raw_end = recording.snap_to_frame(raw_end)
        event = Event(
            kind=dialog.kind,
            label=dialog.label or dialog.kind,
            start_t=raw_start,
            end_t=raw_end,
            start_frame=recording.nearest_frame_index(raw_start),
            end_frame=recording.nearest_frame_index(raw_end),
            detail={"manual": True},
        )
        self._events = [e for e in read_events(self._recording.directory) if not (
            e.kind == event.kind and e.detail.get("manual") and (
                abs(e.start_t - event.start_t) < 0.01 and abs(e.end_t - event.end_t) < 0.01
            )
        )] + [event]
        write_events(self._recording.directory, self._events)
        self._load_events()
        self._status(f"Added {dialog.kind} event at {edited_start:.2f}s-{edited_end:.2f}s")

    def _on_delete_event(self) -> None:
        """Remove the selected event (derived data only, raw untouched)."""
        if self._recording is None:
            return
        index = self._events_combo.currentIndex()
        if index < 0 or index >= len(self._events):
            return
        event = self._events[index]
        events = [
            e for e in read_events(self._recording.directory) if e != event
        ]
        write_events(self._recording.directory, events)
        self._load_events()
        self._status(f"Deleted {event.kind} event at {event.start_t:.2f}s")

    def _edited_to_raw(self, t: float) -> float | None:
        """Map an edited-timeline time to raw source time (None if removed)."""
        if self._session is None:
            return t
        for clip in self._session.timeline.clips:
            if clip.start <= t < clip.start + clip.length:
                return clip.source_time(t)
        return None

    def _raw_to_edited(self, t: float) -> float:
        """Map a raw recording time to the edited timeline (identity when unedited)."""
        if self._session is None:
            return t
        for clip in self._session.timeline.clips:
            if clip.source_start <= t < clip.source_end:
                return clip.edited_time(t)
        return t

    # --------------------------------------------------------- review queue

    def _load_review_queue(self, recording: RecordingData) -> None:
        """Load the persisted review layer (candidates are built on demand)."""
        from dataset.review import ReviewQueue

        self._review_queue = ReviewQueue(recording)
        self._update_review_status()

    def _update_review_status(self) -> None:
        if self._review_queue is None:
            return
        pending = len(self._review_queue.pending())
        if pending:
            self._review_status.setText(f"Review: {pending} pending candidate(s)")
        else:
            self._review_status.setText(
                "Review: no pending candidates — open Review… to build the queue"
            )

    def _on_review(self) -> None:
        """Open the review queue dialog (rebuilds candidates first)."""
        if self._recording is None:
            return
        from app.ui.review_dialog import ReviewDialog
        from dataset.review import ReviewQueue

        self._review_queue = ReviewQueue(self._recording)
        self._review_queue.refresh()
        self._update_review_status()
        dialog = ReviewDialog(
            self._review_queue,
            on_jump=self._review_jump_to,
            on_edit=self._review_edit_to,
            parent=self,
        )
        dialog.finished.connect(self._on_review_done)
        dialog.exec()

    def _on_review_done(self, _result: int) -> None:
        """Verdicts may have verified/rejected annotations — reload them."""
        if self._recording is not None:
            self._annotations = load_annotations(self._recording.directory)
            self._refresh_annotation_view()
            self._update_review_status()

    def _review_jump_to(self, frame_index: int) -> None:
        """Seek the player to a review-candidate frame (raw frame index)."""
        if self._recording is None or self._reader is None:
            return
        self._playing = False
        self._at_end = False
        self._reader.set_playing(False)
        self._play_btn.setText("▶ Play")
        t = self._recording.frame_time(frame_index)
        self._seek_to_index(frame_index)
        self._current_t = t
        self._update_state_views(t)
        self._timeline.set_playhead(self._raw_to_edited(t))
        self._update_time_label()
        self._status(f"Review: frame {frame_index} at {_format_time(t)}")

    def _review_edit_to(self, frame_index: int) -> None:
        """Open a review-candidate frame in the annotation editor (§17 Edit)."""
        if self._recording is None or self._reader is None:
            return
        self._review_jump_to(frame_index)
        self._show_annotations.setChecked(True)
        self._edit_annotations_btn.setChecked(True)
        self._status(f"Annotations + edit mode on — frame {frame_index} (close Review to edit)")

    # --------------------------------------------------------- annotations

    def _on_annotations_toggled(self, checked: bool) -> None:
        if checked:
            self._show_perception.setChecked(False)
        self._update_annotation_overlay()
        self._refresh_timeline_view()

    def _on_annotation_tick_clicked(self, t: float, kind: str, label: str) -> None:
        """Annotation/tick clicked on the timeline: jump to its frame, select if human."""
        if self._recording is None or self._session is None:
            return
        raw_t = self._edited_to_raw(t) if t < self._session.timeline.duration else None
        if raw_t is None:
            raw_t = t
        self._seek_to(raw_t)
        self._current_t = raw_t
        self._update_state_views(raw_t)
        self._timeline.set_playhead(raw_t)
        self._update_time_label()
        if kind == "event":
            for event in self._events:
                if abs(event.start_t - raw_t) < 0.05:
                    self._status(
                        f"Event: {event.kind} {event.label} {event.start_t:.2f}s"
                        f"-{event.end_t:.2f}s (frames {event.start_frame}-{event.end_frame})"
                    )
                    return
            self._status(f"Event: {label} at {raw_t:.2f}s")
            return
        if kind != "human" or self._annotations is None:
            return
        candidates = [a for a in self._annotations if abs(a.t - raw_t) < 0.05]
        if not candidates:
            candidates = sorted(
                self._annotations,
                key=lambda a: abs(a.t - raw_t),
            )[:1]
        if candidates:
            self._on_annotation_selected(candidates[0].id)
            self._annotation_overlay.select_annotation(candidates[0].id)

    def _on_annotation_mode_toggled(self, checked: bool) -> None:
        self._annotation_mode = checked
        self._update_annotation_overlay()

    def _on_import_annotations(self) -> None:
        if self._recording is None or self._annotations is None:
            return
        if not self._perception_manifest:
            self._status("No perception results to import — run perception first")
            return
        from perception.runner import CachedAnalysis

        cached = CachedAnalysis(self._recording.directory / "perception")
        try:
            results = cached.read_results()
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to read perception results")
            QMessageBox.warning(self, "GRUI", f"Could not read perception results:\n{exc}")
            return
        total = sum(len(r.detections) for r in results)
        frame_index = self._recording.nearest_frame_index(self._current_t)
        in_frame = sum(len(r.detections) for r in results if r.frame_index == frame_index)

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Import Perception")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText(
            f"Perception produced {total} detections total. Import them as draft "
            "annotations that humans review, correct or delete?"
        )
        dialog.setInformativeText(
            "Model predictions are guesses — most will need correction or removal.\n\n"
            f"This frame has {in_frame} detections. Import just them to review "
            "frame-by-frame, or import everything at once?"
        )
        current_btn = dialog.addButton("This frame only (recommended)", QMessageBox.ButtonRole.AcceptRole)
        all_btn = dialog.addButton("All detections", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(current_btn)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is cancel_btn or clicked is None:
            return
        if clicked is current_btn:
            results = [r for r in results if r.frame_index == frame_index]
        try:
            imported = self._annotations.import_perception(
                results, frame_size=(self._recording.width, self._recording.height)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("perception import failed")
            QMessageBox.warning(self, "GRUI", f"Import failed:\n{exc}")
            return
        self._save_annotations_state()
        if imported:
            self._status(f"Imported {imported} perception detections as draft annotations")
        else:
            self._status("No new annotations (all detections already imported)")

    def _import_one_candidate(self, perception_id: str) -> Any:
        """Click on an unimported model box: import it as a draft annotation."""
        if self._recording is None or self._annotations is None:
            return None
        try:
            _, frame_part, index_part = perception_id.split(":")
            frame_index, detection_index = int(frame_part), int(index_part)
        except ValueError:
            return None
        detections = self._perception.get(frame_index)
        if not detections or not (0 <= detection_index < len(detections)):
            return None
        detection = detections[detection_index]
        imported = self._annotations.import_detection(
            frame_index,
            self._recording.frame_time(frame_index),
            detection,
            frame_size=(self._recording.width, self._recording.height),
        )
        if imported is not None:
            self._save_annotations_state()
            self._status("Imported model box as a draft annotation — rename, verify or delete it")
            return imported
        existing = next(
            (
                a
                for a in self._annotations.for_frame(frame_index)
                if annotation_dedup_key(a) == detection_dedup_key(frame_index, detection, "imported")
            ),
            None,
        )
        if existing is not None:
            self._status("This model box is already imported — selected the existing annotation")
        return existing

    def _on_prev_annotation(self) -> None:
        """Jump to the annotation before the playhead (§14 navigation)."""
        self._step_annotation(-1)

    def _on_next_annotation(self) -> None:
        """Jump to the annotation after the playhead (§14 navigation)."""
        self._step_annotation(1)

    def _step_annotation(self, direction: int) -> None:
        if self._annotations is None or not self._annotations:
            return
        annotations = sorted(self._annotations, key=lambda a: (a.t, a.frame_index))
        if direction > 0:
            target = next(
                (a for a in annotations if a.t > self._current_t + 1e-6),
                annotations[0],
            )
        else:
            target = next(
                (a for a in reversed(annotations) if a.t < self._current_t - 1e-6),
                annotations[-1],
            )
        self._seek_to(target.t)
        self._current_t = target.t
        self._selected_annotation_id = target.id
        self._ann_label_edit.setText(target.label)
        self._ann_label_edit.setEnabled(True)
        self._apply_label_btn.setEnabled(True)
        self._verify_btn.setEnabled(True)
        self._delete_ann_btn.setEnabled(True)
        self._update_state_views(target.t)
        self._timeline.set_playhead(self._raw_to_edited(target.t))
        self._update_time_label()
        self._update_annotation_overlay()
        self._annotation_overlay.select_annotation(target.id)
        self._update_annotation_status_text()
        self._status(f"Annotation {target.label or '(untitled)'} at {target.t:.2f}s")

    def _on_annotation_selected(self, annotation_id: str) -> None:
        annotation = self._annotations.get(annotation_id) if self._annotations else None
        if annotation is None and annotation_id.startswith("perception:"):
            annotation = self._import_one_candidate(annotation_id)
        self._selected_annotation_id = annotation.id if annotation else annotation_id
        if annotation is not None:
            self._ann_label_edit.setText(annotation.label)
            self._ann_label_edit.setEnabled(True)
            self._apply_label_btn.setEnabled(True)
            self._verify_btn.setEnabled(True)
            self._delete_ann_btn.setEnabled(True)
        self._update_annotation_status_text()

    def _on_annotation_moved(self, annotation_id: str, dx: float, dy: float) -> None:
        if self._annotations is None or annotation_id.startswith("perception:"):
            return
        if self._annotations.move(annotation_id, dx, dy):
            self._save_annotations_state()

    def _on_annotation_resized(self, annotation_id: str, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._annotations is None or annotation_id.startswith("perception:"):
            return
        if self._annotations.resize(annotation_id, BoundingBox(x1, y1, x2, y2)):
            self._save_annotations_state()

    def _on_annotation_created(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._recording is None or self._annotations is None:
            return
        frame_index = self._recording.nearest_frame_index(self._current_t)
        annotation = self._annotations.create(
            label="", bbox=BoundingBox(x1, y1, x2, y2),
            frame_index=frame_index, t=self._recording.frame_time(frame_index),
        )
        self._save_annotations_state()
        self._selected_annotation_id = annotation.id
        self._ann_label_edit.setEnabled(True)
        self._ann_label_edit.setText("")
        self._ann_label_edit.setFocus()
        self._annotation_overlay.select_annotation(annotation.id)
        self._update_annotation_status_text()
        self._status("New annotation created — type a label and press Rename")

    def _on_rename_annotation(self) -> None:
        if self._annotations is None or self._selected_annotation_id is None:
            return
        label = self._ann_label_edit.text().strip()
        if not label:
            self._status("Enter a label first")
            return
        if self._annotations.rename(self._selected_annotation_id, label):
            self._save_annotations_state()
            self._status(f"Renamed annotation to {label!r}")

    def _on_verify_annotation(self) -> None:
        if self._annotations is None or self._selected_annotation_id is None:
            return
        annotation = self._annotations.get(self._selected_annotation_id)
        if annotation is None:
            return
        if annotation.status == AnnotationStatus.VERIFIED:
            self._annotations.set_status(self._selected_annotation_id, AnnotationStatus.REVIEWED)
            self._status("Un-verified annotation")
        else:
            self._annotations.verify(self._selected_annotation_id)
            self._status("Verified annotation")
        self._save_annotations_state()

    def _on_delete_annotation(self) -> None:
        if self._annotations is None or self._selected_annotation_id is None:
            return
        if self._annotations.delete(self._selected_annotation_id):
            self._selected_annotation_id = None
            self._ann_label_edit.clear()
            self._save_annotations_state()
            self._status("Deleted annotation")

    def _on_annotation_undo(self) -> None:
        if self._annotations is not None and self._annotations.undo():
            self._save_annotations_state()

    def _on_annotation_redo(self) -> None:
        if self._annotations is not None and self._annotations.redo():
            self._save_annotations_state()

    def _on_save_annotations(self) -> None:
        if self._annotations is None:
            return
        self._annotations.save()
        self._status(f"Saved {len(self._annotations)} annotations")

    def _save_annotations_state(self) -> None:
        if self._annotations is None:
            return
        self._annotations.save()
        self._refresh_annotation_view()

    def _refresh_annotation_view(self) -> None:
        self._update_annotation_overlay()
        self._refresh_timeline_view()
        self._update_annotation_status_text()

    def _candidate_boxes(
        self, frame_index: int
    ) -> list[tuple[str, str, str, tuple[float, float, float, float], float | None]]:
        """Model detections for a frame, drawn as clickable overlay boxes.

        Unimported detections become dashed ``prediction`` boxes (clicking
        one imports it). Already-imported detections are drawn solid with
        their annotation id/status, so the perception view never appears
        empty after an import.
        """
        annotations_by_key = {annotation_dedup_key(a): a for a in self._annotations}
        boxes = []
        width = self._recording.width if self._recording else 0
        height = self._recording.height if self._recording else 0
        for i, detection in enumerate(self._perception.get(frame_index) or []):
            annotation = annotations_by_key.get(detection_dedup_key(frame_index, detection, "imported"))
            bbox = detection.bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            if width > 0 and height > 0:
                x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height
            if annotation is not None:
                boxes.append(
                    (annotation.id, annotation.label or "(untitled)", annotation.status.value,
                     (x1, y1, x2, y2), annotation.confidence)
                )
            else:
                boxes.append(
                    (f"perception:{frame_index}:{i}", detection.label, "prediction",
                     (x1, y1, x2, y2), detection.confidence)
                )
        return boxes

    def _update_annotation_overlay(self) -> None:
        if self._recording is None or self._annotations is None:
            self._annotation_overlay.hide()
            return
        frame_index = self._recording.nearest_frame_index(self._current_t)
        show_annotations = self._show_annotations.isChecked()
        show_candidates = self._show_perception.isChecked()
        if not show_annotations and not (show_candidates and self._perception):
            self._annotation_overlay.hide()
            return
        boxes: list[tuple[str, str, str, tuple[float, float, float, float], float | None]] = []
        if show_annotations:
            boxes.extend(self._annotation_boxes(frame_index))
        if show_candidates:
            boxes.extend(self._candidate_boxes(frame_index))
        self._annotation_overlay.set_editing(show_annotations and self._annotation_mode)
        self._annotation_overlay.setVisible(True)
        if self._selected_annotation_id not in {b[0] for b in boxes}:
            self._annotation_overlay.select_annotation(None)
        self._annotation_overlay.set_annotations(boxes)
        self._annotation_overlay.raise_()

    def _annotation_boxes(
        self, frame_index: int
    ) -> list[tuple[str, str, str, tuple[float, float, float, float], float | None]]:
        """Annotations for a frame; tolerates boxes stored in raw pixels."""
        width = self._recording.width if self._recording else 0
        height = self._recording.height if self._recording else 0
        boxes = []
        for a in self._annotations.for_frame(frame_index):
            bbox = a.bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            if width > 0 and height > 0 and max(x1, y1, x2, y2) > 1.0:
                x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height
            boxes.append(
                (a.id, a.label or "(untitled)", a.status.value, (x1, y1, x2, y2), a.confidence)
            )
        return boxes

    def _update_annotation_status_text(self) -> None:
        annotations = self._annotations
        if annotations is None:
            self._annotation_status.setText("No annotations file")
            self._edit_annotations_btn.setEnabled(False)
            self._import_annotations_btn.setEnabled(False)
            self._prev_ann_btn.setEnabled(False)
            self._next_ann_btn.setEnabled(False)
            self._undo_ann_btn.setEnabled(False)
            self._redo_ann_btn.setEnabled(False)
            self._save_annotations_btn.setEnabled(False)
            return
        total = len(annotations)
        verified = annotations.verified_count
        self._annotation_status.setText(
            f"{total} annotations ({verified} verified, {total - verified} pending)"
        )
        self._save_annotations_btn.setEnabled(True)
        self._prev_ann_btn.setEnabled(True)
        self._next_ann_btn.setEnabled(True)
        self._undo_ann_btn.setEnabled(annotations.can_undo)
        self._redo_ann_btn.setEnabled(annotations.can_redo)

    def _annotation_ticks(self) -> list[tuple[float, str, str]]:
        """Timeline ticks: human annotations (kind human) + perception (prediction),
        mapped through the edited timeline so deleted regions drop out.
        Hidden unless "Show annotations" is checked."""
        if not self._show_annotations.isChecked() or self._recording is None or self._session is None:
            return []
        raw: list[dict[str, Any]] = []
        if self._annotations is not None:
            for a in self._annotations:
                raw.append({"t": a.t, "kind": "human", "label": a.label})
        for frame_index in sorted(self._perception):
            t = self._recording.frame_time(frame_index)
            raw.append({"t": t, "kind": "prediction", "label": "perception"})
        for event in self._events:
            raw.append({"t": event.start_t, "kind": "event", "label": f"{event.kind} {event.label}"})
        ticks = []
        for item in remap_events(raw, self._session.timeline):
            ticks.append((float(item["t"]), str(item["kind"]), str(item.get("label") or "")))
        return sorted(ticks)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._teardown_reader()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if getattr(self, "_video_scroll", None) is not None:
            self._render_frame()
