"""Player window offscreen smoke tests: load, play a frame, edit, save."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.ui.keyboard_view import _GRID_PLACEMENT
from app.ui.player_window import PlayerWindow
from storage.recording import list_recordings, load_recording
from tests.fakes import build_synthetic_recording

QApplication.instance() or QApplication([])


@pytest.fixture()
def window(tmp_path):
    build_synthetic_recording(
        tmp_path / "root",
        n_frames=30,
        fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.75, "device": "keyboard", "event": "up", "code": "KeyW"},
        ],
        markers=[{"t": 0.5, "label": "checkpoint"}],
    )
    win = PlayerWindow(recordings_root=tmp_path / "root")
    win._load(win._recording_combo.itemData(0))
    deadline = time.monotonic() + 15.0
    while win._recording is None or win._video_label.pixmap() is None or win._video_label.pixmap().isNull():
        win._tick()
        QApplication.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for first frame")
        time.sleep(0.01)
    yield win
    win.close()


def test_load_shows_first_frame(window):
    assert window._recording is not None
    assert window._video_label.pixmap() is not None
    assert not window._video_label.pixmap().isNull()
    assert window._timer.isActive()  # playback ticker actually running
    assert "KeyW" in window._keyboard_view._caps
    assert "Key.f1" in window._keyboard_view._caps  # full keyboard layout
    assert "button:left" in window._keyboard_view._caps
    assert window._timeline._timeline is not None
    assert window._play_btn.isEnabled()


def test_reader_does_not_flood_queue(tmp_path):
    """Playback is paced: the reader must never decode ahead of real time."""
    from player.video_reader import VideoReader

    rec = build_synthetic_recording(tmp_path / "r", n_frames=30, fps=10)
    reader = VideoReader(rec.video_path)
    reader.start()
    assert reader.wait_ready(5.0)
    reader.set_playing(True)
    all_frames = []
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        all_frames.extend(reader.drain())
        time.sleep(0.01)
    reader.set_playing(False)
    reader.stop()
    # 1.5s at 10 fps paces to ~15 frames; a flood would decode all 30 + EOF
    assert len(all_frames) >= 5
    assert len(all_frames) < 30
    assert all(frame is not None for _, frame in all_frames)


def _write_perception(recording, by_frame):
    """Write fake cached perception results.

    by_frame: frame_index -> [labels] or [(label, confidence)] entries.
    """
    import json

    out = recording.directory / "perception"
    out.mkdir(exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {"provider": "test", "provider_version": "0", "prompts": ["test"], "count": len(by_frame)}
        ),
        encoding="utf-8",
    )
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for frame_index, entries in by_frame.items():
            def detection(entry):
                if isinstance(entry, tuple):
                    label, conf = entry
                    return {"label": label, "bbox": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0}, "confidence": conf}
                return {"label": entry, "bbox": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0}}

            row = {
                "frame_index": frame_index,
                "t": float(recording.frame_time(frame_index)),
                "prompt": "test",
                "detections": [detection(entry) for entry in entries],
            }
            fh.write(json.dumps(row) + "\n")


def test_next_perception_disabled_without_results(window):
    assert not window._next_perception_btn.isEnabled()


def test_next_perception_skips_to_detections(window):
    _write_perception(window._recording, {5: ["one"], 15: ["two"]})
    window._load_perception(window._recording)
    assert window._next_perception_btn.isEnabled()

    window._on_next_perception()
    assert window._current_t == pytest.approx(window._recording.frame_time(5))
    assert window._timeline._playhead == pytest.approx(window._recording.frame_time(5))
    assert not window._playing

    window._on_next_perception()
    assert window._current_t == pytest.approx(window._recording.frame_time(15))

    window._on_next_perception()  # past the last detection -> wraps to the first
    assert window._current_t == pytest.approx(window._recording.frame_time(5))


def test_perception_candidates_display_normalized(window):
    """Pixel-space detections must be shown as normalized 0..1 overlay boxes."""
    _write_perception(window._recording, {5: ["one"]})
    window._load_perception(window._recording)
    window._on_next_perception()  # lands on frame 5
    window._show_perception.setChecked(True)
    QApplication.processEvents()

    overlay = window._annotation_overlay
    assert not overlay.isHidden()  # explicitly asked Qt to show it
    assert len(overlay._boxes) == 1
    box = overlay._boxes[0]
    assert box.id == "perception:5:0"
    assert box.status == "prediction"
    assert 0.0 <= box.x1 <= 1.0 and 0.0 <= box.y1 <= 1.0
    assert 0.0 <= box.x2 <= 1.0 and 0.0 <= box.y2 <= 1.0


def test_click_candidate_imports_and_selects(window):
    _write_perception(window._recording, {5: ["one"]})
    window._load_perception(window._recording)
    window._on_next_perception()
    window._show_perception.setChecked(True)
    QApplication.processEvents()

    box = window._annotation_overlay._boxes[0]
    window._on_annotation_selected(box.id)

    assert len(window._annotations) == 1
    annotation = list(window._annotations)[0]
    assert annotation.status.value == "predicted"
    assert 0.0 <= annotation.bbox.x2 <= 1.0  # stored normalized, not raw pixels
    assert window._selected_annotation_id == annotation.id
    assert window._ann_label_edit.isEnabled()
    assert window._ann_label_edit.text() == "one"

    # clicking the same candidate again selects it, does not duplicate
    window._on_annotation_selected(box.id)
    assert len(window._annotations) == 1


def test_show_views_are_exclusive(window):
    _write_perception(window._recording, {5: ["one"]})
    window._load_perception(window._recording)

    window._show_annotations.setChecked(True)
    assert window._show_perception.isChecked() is False
    window._show_perception.setChecked(True)
    assert window._show_annotations.isChecked() is False


def test_annotation_view_detects_stale_pixel_boxes(window, monkeypatch):
    """Boxes stored in raw pixels (pre-normalization files) still display."""
    from perception.types import BoundingBox

    _write_perception(window._recording, {5: ["one"]})
    window._load_perception(window._recording)
    window._on_next_perception()
    window._show_perception.setChecked(True)
    QApplication.processEvents()
    window._on_annotation_selected(window._annotation_overlay._boxes[0].id)  # import

    # rewrite the stored box to raw pixels, as old files contain
    stored = list(window._annotations)[0]
    frame_size = (window._recording.width, window._recording.height)
    pixel = BoundingBox(
        x1=stored.bbox.x1 * frame_size[0],
        y1=stored.bbox.y1 * frame_size[1],
        x2=stored.bbox.x2 * frame_size[0],
        y2=stored.bbox.y2 * frame_size[1],
    )
    window._annotations.resize(stored.id, pixel)

    window._show_annotations.setChecked(True)
    window._on_next_perception()
    QApplication.processEvents()
    boxes = window._annotation_overlay._boxes
    assert len(boxes) == 1
    assert boxes[0].id == stored.id
    assert boxes[0].x2 <= 1.0  # scaled down for display despite pixel storage
    assert boxes[0].y2 <= 1.0


def test_zoom_scales_display(window):
    base = window._video_label.pixmap().size()
    assert window._zoom_slider.value() == 100
    assert window._zoom_pct_label.text() == "100%"

    window._zoom_slider.setValue(200)
    scaled = window._video_label.pixmap().size()
    assert scaled.width() > base.width()
    assert scaled.height() > base.height()
    assert window._zoom_pct_label.text() == "200%"
    assert window._video_label.minimumSize() == scaled

    window._on_zoom_fit()
    assert window._zoom_slider.value() == 100
    assert window._zoom_pct_label.text() == "100%"
    fitted = window._video_label.pixmap().size()
    assert fitted.width() < scaled.width()


def test_zoom_controls_safe_without_recording(tmp_path):
    win = PlayerWindow(recordings_root=str(tmp_path / "root"))
    win._zoom_slider.setValue(300)  # no frame loaded — must not crash
    win._zoom_out_btn.click()
    win._zoom_in_btn.click()
    win._zoom_fit_btn.click()
    assert win._video_label.pixmap() is None or win._video_label.pixmap().isNull()
    win.close()


def test_edit_cut_undo_redo(window):
    original = window._session.timeline.duration
    events_before = len(window._timeline._events)
    window._on_selection_changed((0.0, 0.2))  # covers the first key event
    assert window._timeline_sel is not None
    assert window._timeline_sel[0] == window._recording.snap_to_frame(0.0)
    assert window._timeline_sel[1] == window._recording.snap_to_frame(0.2)
    window._on_cut()
    assert window._session.timeline.duration < original
    assert window._clipboard is not None
    assert window._timeline_sel is None  # cleared after the edit
    assert len(window._timeline._events) < events_before  # events in the cut region drop
    window._on_undo()
    assert window._session.timeline.duration == original
    assert len(window._timeline._events) == events_before
    window._on_redo()
    assert window._session.timeline.duration < original


def test_select_all_and_deselect(window):
    window._on_select_all()
    assert window._timeline_sel == (0.0, window._session.timeline.duration)
    assert window._timeline._selection == window._timeline_sel
    window._on_deselect()
    assert window._timeline_sel is None
    assert window._timeline._selection is None


def test_timeline_drag_selects_region(window):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    timeline = window._timeline
    timeline.resize(800, 100)
    duration = window._session.timeline.duration
    plot = timeline._plot_rect()

    def pos_at(t):
        x = plot.left() + (t / duration) * plot.width()
        return QPointF(x, plot.center().y())

    timeline.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress, pos_at(0.2), QPointF(),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    timeline.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove, pos_at(0.7), QPointF(),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    timeline.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease, pos_at(0.7), QPointF(),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert timeline._selection is not None
    assert timeline._selection[0] == pytest.approx(0.2, abs=0.05)
    assert timeline._selection[1] == pytest.approx(0.7, abs=0.05)
    assert window._timeline_sel is not None
    assert window._timeline_sel == timeline._selection


def test_timeline_click_seeks_and_clears_selection(window):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    timeline = window._timeline
    timeline.resize(800, 100)
    duration = window._session.timeline.duration
    plot = timeline._plot_rect()
    seeks = []
    timeline.seeked.connect(seeks.append)

    window._on_select_all()
    assert timeline._selection is not None

    def pos_at(t):
        x = plot.left() + (t / duration) * plot.width()
        return QPointF(x, plot.center().y())

    timeline.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress, pos_at(0.4), QPointF(),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    timeline.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease, pos_at(0.4), QPointF(),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert seeks
    assert seeks[0] == pytest.approx(0.4, abs=0.05)
    assert timeline._selection is None
    assert window._timeline_sel is None


def test_timeline_shows_events(window):
    from app.ui.timeline_widget import _KEY_EVENT_COLOR

    timeline = window._timeline
    # keyboard events only, no key differentiation — (time, code) pairs, sorted
    assert len(timeline._events) == 2
    assert timeline._events == sorted(timeline._events)
    offset = timeline._timeline.clips[0].source_start
    assert timeline._events[0][0] == pytest.approx(0.15 - offset, abs=1e-9)
    assert timeline._events[1][0] == pytest.approx(0.75 - offset, abs=1e-9)
    assert [code for _, code in timeline._events] == ["KeyW", "KeyW"]
    assert _KEY_EVENT_COLOR.name() == "#f1c40f"  # yellow dots
    assert not timeline.grab().isNull()  # paint path renders the dots


def test_timeline_hover_shows_key_and_time(window):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    timeline = window._timeline
    timeline.resize(800, 100)
    duration = window._session.timeline.duration
    plot = timeline._plot_rect()
    offset = timeline._timeline.clips[0].source_start

    def move_to(t):
        x = plot.left() + (t / duration) * plot.width()
        timeline.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, plot.center().y()), QPointF(),
                Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    move_to(0.15 - offset)  # hover exactly on the first dot
    assert timeline._hovered is not None
    assert timeline._hovered[0] == pytest.approx(0.15 - offset, abs=1e-9)
    assert timeline._hovered[1] == "KeyW"
    assert "KeyW" in timeline._hover_label.text()
    assert "0.15" in timeline._hover_label.text()

    move_to(1.5)  # far from any dot
    assert timeline._hovered is None
    assert timeline._hover_label.text() == ""


def test_keyboard_mouse_buttons_unified(window):
    view = window._keyboard_view
    assert len(view._caps["button:left"]) == 1
    assert len(view._caps["button:right"]) == 1
    assert len(view._caps["button:middle"]) == 1
    view.set_state(set(), {"left"}, None)
    assert "#e74c3c" in view._caps["button:left"][0].styleSheet()
    assert "#e74c3c" not in view._caps["button:right"][0].styleSheet()
    view.set_state({"KeyW"}, {"left"}, (100, 50))
    assert "#e74c3c" in view._caps["KeyW"][0].styleSheet()
    assert view._mouse_surface._pos == (100.0, 50.0)


def test_keyboard_and_mouse_areas(window):
    from PySide6.QtWidgets import QGroupBox, QScrollArea

    view = window._keyboard_view
    groups = view.findChildren(QGroupBox)
    assert [group.title() for group in groups] == ["Keyboard", "Mouse"]
    assert view._keyboard_group.findChild(QScrollArea) is not None
    assert view._mouse_surface.parent() is view._mouse_group


def test_mouse_buttons_right_of_keys_no_extras(window):
    view = window._keyboard_view
    key_cols = [col for _, col, _, _, code in _GRID_PLACEMENT if not code.startswith("button:")]
    buttons = [
        (row, col, code)
        for row, col, _, _, code in _GRID_PLACEMENT
        if code.startswith("button:")
    ]
    assert all(col > max(key_cols) for _, col, _ in buttons)
    assert [code for _, _, code in buttons] == ["button:left", "button:right", "button:middle"]
    assert set(view._caps) == {code for *_, code in _GRID_PLACEMENT}  # no dynamic extras
    assert not hasattr(view, "_extras_area")


def test_keyboard_fits_at_once(window):
    view = window._keyboard_view
    min_width = view._grid.minimumSize().width()
    assert min_width < 800
    assert window.minimumWidth() >= min_width
    # every physical key + mouse button has a cap in the grid
    grid_codes = {code for _, _, _, _, code in _GRID_PLACEMENT}
    for code in grid_codes:
        assert code in view._caps


def test_no_selection_warns(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args))
    )
    window._on_cut()
    window._on_trim()
    window._on_copy()
    window._on_delete()
    assert len(warnings) == 4
    assert all(args[1] == "No Selection" for args in warnings)


# ---------------------------------------------------------------- events UI

def _write_events(recording, events):
    from perception.events import write_events

    write_events(recording.directory, events)


def test_events_loaded_into_combo_and_timeline(window):
    from perception.events import Event

    _write_events(window._recording, [
        Event(kind="appearance", label="boss", start_t=0.4, end_t=0.4,
              start_frame=4, end_frame=4, detail={}),
        Event(kind="disappearance", label="projectile", start_t=1.2, end_t=1.5,
              start_frame=12, end_frame=15, detail={"duration_s": 0.3}),
    ])
    window._load_events()

    assert len(window._events) == 2
    assert window._events_combo.count() == 2
    assert "boss" in window._events_combo.itemText(0)
    assert "1.20s-1.50s" in window._events_combo.itemText(1)
    assert "2 derived event(s)" in window._events_status.text()

    window._show_annotations.setChecked(True)  # the lane is gated by this toggle
    window._refresh_timeline_view()
    kinds = [kind for _, kind, _ in window._timeline._annotation_ticks]
    assert kinds.count("event") == 2
    assert kinds.count("human") == 0
    assert not window._timeline.grab().isNull()  # orange event squares render


def test_delete_event_button_removes_selected_event(window):
    from perception.events import Event, read_events

    _write_events(window._recording, [
        Event(kind="appearance", label="boss", start_t=0.4, end_t=0.4,
              start_frame=4, end_frame=4, detail={}),
        Event(kind="disappearance", label="projectile", start_t=1.2, end_t=1.5,
              start_frame=12, end_frame=15, detail={}),
    ])
    window._load_events()
    assert window._delete_event_btn.isEnabled()

    window._events_combo.setCurrentIndex(1)
    window._on_delete_event()
    assert len(window._events) == 1
    assert window._events_combo.count() == 1
    assert "projectile" not in window._events_combo.itemText(0)
    assert [e.label for e in read_events(window._recording.directory)] == ["boss"]

    window._events_combo.setCurrentIndex(0)
    window._on_delete_event()
    assert window._events == []
    assert window._events_combo.count() == 0
    assert window._delete_event_btn.isEnabled() is False
    window._on_delete_event()  # empty combo: no-op, no crash
    assert read_events(window._recording.directory) == []


def test_events_combo_jumps_to_event_start(window):
    from perception.events import Event

    _write_events(window._recording, [
        Event(kind="appearance", label="boss", start_t=0.4, end_t=0.4,
              start_frame=4, end_frame=4, detail={}),
        Event(kind="disappearance", label="projectile", start_t=1.2, end_t=1.5,
              start_frame=12, end_frame=15, detail={}),
    ])
    window._load_events()
    window._events_combo.setCurrentIndex(1)
    QApplication.processEvents()

    assert window._current_t == pytest.approx(1.2)
    assert abs(window._timeline._playhead - 1.2) < 0.11  # one frame duration
    assert window._recording.nearest_frame_index(1.2) == 12
    assert not window._playing


def test_events_ticks_drop_out_of_edited_regions(window):
    """Event ticks are remapped like annotations: deleted regions drop out."""
    from perception.events import Event

    _write_events(window._recording, [
        Event(kind="appearance", label="early", start_t=0.05, end_t=0.05,
              start_frame=0, end_frame=0, detail={}),
        Event(kind="appearance", label="late", start_t=2.5, end_t=2.5,
              start_frame=25, end_frame=25, detail={}),
    ])
    window._load_events()
    window._show_annotations.setChecked(True)
    window._refresh_timeline_view()
    kept = [0.05, 2.5]
    assert sorted(round(tick[0], 2) for tick in window._timeline._annotation_ticks) == kept

    # cut the first half of the recording away -> the early event disappears
    window._on_selection_changed((0.0, 1.5))
    window._on_cut()
    assert window._timeline._annotation_ticks  # late event survived
    assert all(tick[0] > 0.5 for tick in window._timeline._annotation_ticks)


# -------------------------------------------------- annotation navigation

def _seed_annotations(window):
    from perception.types import BoundingBox

    for label, fi in (("first", 1), ("second", 10), ("third", 22)):
        window._annotations.create(
            label=label,
            bbox=BoundingBox(0.1, 0.1, 0.4, 0.4),
            frame_index=fi,
            t=window._recording.frame_time(fi),
        )
    window._show_annotations.setChecked(True)


def test_annotation_next_steps_forward_and_wraps(window):
    _seed_annotations(window)

    window._on_next_annotation()
    assert window._selected_annotation_id == window._annotations.for_frame(1)[0].id
    assert window._current_t == pytest.approx(window._recording.frame_time(1))
    assert window._ann_label_edit.text() == "first"
    assert window._timeline._playhead == pytest.approx(window._current_t, abs=0.11)

    window._on_next_annotation()
    assert window._ann_label_edit.text() == "second"
    window._on_next_annotation()
    assert window._ann_label_edit.text() == "third"
    window._on_next_annotation()
    assert window._ann_label_edit.text() == "first"  # wrapped around


def test_annotation_prev_steps_backward_and_wraps(window):
    _seed_annotations(window)

    window._on_prev_annotation()
    assert window._ann_label_edit.text() == "third"  # wraps from t=0 to the last
    window._on_prev_annotation()
    assert window._ann_label_edit.text() == "second"
    window._on_prev_annotation()
    assert window._ann_label_edit.text() == "first"


def test_annotation_nav_is_noop_without_annotations(window):
    window._annotations = None
    window._annotation_status.setText("")
    before = window._current_t
    window._on_next_annotation()  # must not crash
    window._on_prev_annotation()
    assert window._current_t == before


def test_events_row_empty_without_recording(tmp_path):
    win = PlayerWindow(recordings_root=str(tmp_path / "root"))
    assert win._events_combo.isEnabled() is False
    win.close()


# -------------------------------------------------- overlay hover tooltip

def test_overlay_hover_tooltip_shows_label_status_confidence():
    from app.ui.annotation_overlay import AnnotationOverlay
    from PySide6.QtWidgets import QLabel

    label = QLabel()
    overlay = AnnotationOverlay(label)
    overlay.resize(200, 200)
    overlay.set_annotations([
        ("a1", "boss", "prediction", (0.1, 0.1, 0.3, 0.3), 0.42),
        ("a2", "crate", "verified", (0.5, 0.5, 0.7, 0.7), None),
    ])

    overlay._update_hover(40, 40)  # inside box a1
    assert overlay.toolTip() == "boss  [prediction]  confidence=0.42"
    overlay._update_hover(120, 120)  # inside box a2, no confidence
    assert overlay.toolTip() == "crate  [verified]"
    overlay._update_hover(190, 190)  # empty area
    assert overlay.toolTip() == ""
    from PySide6.QtCore import QEvent

    overlay.leaveEvent(QEvent(QEvent.Type.Leave))
    assert overlay.toolTip() == ""


def test_candidate_boxes_carry_confidence(window):
    _write_perception(window._recording, {5: [("one", 0.42)]})
    window._load_perception(window._recording)
    boxes = window._candidate_boxes(5)
    assert boxes and boxes[0][-1] == 0.42
    assert boxes[0][1] == "one"


# ---------------------------------------------- overlay edit interactions

def _mouse_event(x: float, y: float, pressed: bool = True):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    kind = QEvent.Type.MouseMove if not pressed else QEvent.Type.MouseButtonPress
    button = Qt.MouseButton.LeftButton if pressed else Qt.MouseButton.NoButton
    pos = QPointF(x, y)
    return QMouseEvent(
        kind, pos, pos, pos, button, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _make_overlay():
    from app.ui.annotation_overlay import AnnotationOverlay
    from PySide6.QtWidgets import QLabel

    label = QLabel()
    overlay = AnnotationOverlay(label)
    overlay.resize(200, 200)
    return label, overlay


def test_overlay_click_selects_box():
    label, overlay = _make_overlay()
    selected = []
    overlay.annotationSelected.connect(lambda *a: selected.append(a))
    overlay.set_annotations([
        ("a1", "boss", "prediction", (0.1, 0.1, 0.3, 0.3), 0.42),
        ("a2", "crate", "verified", (0.5, 0.5, 0.7, 0.7), None),
    ])

    overlay.mousePressEvent(_mouse_event(40, 40))
    assert selected == [("a1",)]
    assert overlay._selected_id == "a1"
    overlay.mousePressEvent(_mouse_event(40, 40))  # a second click selects again
    assert selected == [("a1",), ("a1",)]
    overlay.mousePressEvent(_mouse_event(190, 190))  # empty area: nothing selected
    assert selected == [("a1",), ("a1",)]
    overlay.mouseReleaseEvent(_mouse_event(190, 190))


def test_overlay_drag_moves_box():
    label, overlay = _make_overlay()
    moved = []
    overlay.annotationMoved.connect(lambda *a: moved.append(a))
    overlay.set_annotations([("a1", "boss", "verified", (0.1, 0.1, 0.3, 0.3), None)])
    overlay.set_editing(True)

    overlay.mousePressEvent(_mouse_event(40, 40))
    overlay.mouseMoveEvent(_mouse_event(60, 40))  # +20 px in a 200 px widget
    assert moved == [("a1", 0.1, 0.0)]
    overlay.mouseReleaseEvent(_mouse_event(60, 40))
    assert overlay._drag is None  # drag cancelled on release


def test_overlay_handle_resizes_box():
    label, overlay = _make_overlay()
    resized = []
    overlay.annotationResized.connect(lambda *a: resized.append(a))
    overlay.set_annotations([("a1", "boss", "verified", (0.1, 0.1, 0.5, 0.5), None)])
    overlay.set_editing(True)

    # bottom-right corner sits at (100, 100); press just inside it and
    # drag the corner to (140, 120)
    overlay.mousePressEvent(_mouse_event(98, 98))
    overlay.mouseMoveEvent(_mouse_event(140, 120))
    assert len(resized) == 1
    _, x1, y1, x2, y2 = resized[0]
    assert (x1, y1, x2, y2) == pytest.approx((0.1, 0.1, 0.7, 0.6))
    overlay.mouseReleaseEvent(_mouse_event(140, 120))


def test_overlay_drag_creates_box():
    label, overlay = _make_overlay()
    created = []
    overlay.annotationCreated.connect(lambda *a: created.append(a))
    overlay.set_editing(True)

    overlay.mousePressEvent(_mouse_event(20, 20))
    overlay.mouseMoveEvent(_mouse_event(120, 100))
    overlay.mouseReleaseEvent(_mouse_event(120, 100))
    assert len(created) == 1
    x1, y1, x2, y2 = created[0]
    assert (x1, y1, x2, y2) == pytest.approx((0.1, 0.1, 0.6, 0.5))

    # a tiny drag (under 12 px) is a click, not a new box
    overlay.mousePressEvent(_mouse_event(150, 150))
    overlay.mouseMoveEvent(_mouse_event(154, 154))
    overlay.mouseReleaseEvent(_mouse_event(154, 154))
    assert len(created) == 1


def test_overlay_escape_cancels_create():
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    label, overlay = _make_overlay()
    created = []
    overlay.annotationCreated.connect(lambda *a: created.append(a))
    overlay.set_editing(True)

    overlay.mousePressEvent(_mouse_event(20, 20))
    overlay.mouseMoveEvent(_mouse_event(120, 100))
    assert overlay._create_box is not None
    overlay.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, int(Qt.Key.Key_Escape), Qt.KeyboardModifier.NoModifier)
    )
    assert overlay._create_box is None
    assert created == []


# ----------------------------------------------------- manual events (GUI)

class _FakeEventDialog:
    kind = "watch"
    label = "boss"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted


def test_add_manual_event_from_selection(window, monkeypatch):
    from perception.events import read_events

    monkeypatch.setattr("app.ui.player_window.EventDialog", _FakeEventDialog)
    window._on_selection_changed((0.3, 1.2))

    window._on_add_event()

    stored = read_events(window._recording.directory)
    assert len(stored) == 1
    event = stored[0]
    assert event.kind == "watch"
    assert event.label == "boss"
    assert event.start_t == pytest.approx(window._recording.snap_to_frame(0.3))
    assert event.end_t == pytest.approx(window._recording.snap_to_frame(1.2))
    assert event.detail.get("manual") is True
    assert window._events_combo.count() == 1
    assert "watch" in window._events_combo.itemText(0)


def test_add_manual_event_maps_through_cut(window, monkeypatch):
    """Selection is in edited time; a cut shifts the mapped raw span."""
    from perception.events import read_events

    window._on_selection_changed((0.0, 0.5))
    window._on_cut()  # raw [0, 0.5) removed; edited t now starts at raw 0.5

    monkeypatch.setattr("app.ui.player_window.EventDialog", _FakeEventDialog)
    window._on_selection_changed((0.1, 0.4))
    window._on_add_event()

    stored = read_events(window._recording.directory)
    assert len(stored) == 1
    assert stored[0].start_t > 0.5  # mapped back into surviving raw time
    offset = window._session.timeline.clips[0].source_start
    assert stored[0].start_t == pytest.approx(offset + 0.1, abs=0.02)
    assert stored[0].end_t == pytest.approx(offset + 0.4, abs=0.02)


def test_add_manual_event_requires_selection(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args))
    )
    window._on_add_event()
    assert warnings and warnings[0][1] == "No Selection"


def test_save_creates_new_recording(window, tmp_path, monkeypatch):
    before = set(list_recordings(tmp_path / "root"))
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save)
    )
    window._on_save()
    after = set(list_recordings(tmp_path / "root"))
    new_dirs = after - before
    assert len(new_dirs) == 1
    saved = load_recording(next(iter(new_dirs)))
    assert saved.metadata["edited_from"]["session_id"] == window._recording.session_id


def test_save_cancel_does_nothing(window, tmp_path, monkeypatch):
    before = list_recordings(tmp_path / "root")
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel)
    )
    window._on_save()
    assert list_recordings(tmp_path / "root") == before


def test_dataset_button_requires_loaded_recording(tmp_path):
    win = PlayerWindow(recordings_root=str(tmp_path / "root"))
    assert not win._dataset_btn.isEnabled()
    win.close()


def test_dataset_button_opens_dialog(window, monkeypatch):
    from app.ui import dataset_dialog as module

    opened = []
    monkeypatch.setattr(module.DatasetDialog, "exec", lambda self: opened.append(self) or 0)
    window._dataset_btn.click()
    assert len(opened) == 1
    assert opened[0].recording is window._recording


def test_dataset_dialog_builds_dataset(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from app.ui.dataset_dialog import DatasetDialog

    dialog = DatasetDialog(window._recording)
    dialog._duration.setValue(0.2)
    dialog._fps.setValue(5)
    dialog._stride.setValue(0.1)
    out = tmp_path / "ds"
    dialog._out_edit.setText(str(out))

    messages = []
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args, **kwargs: messages.append(args))
    )
    dialog._on_build()
    assert messages, "success message shown"
    assert (out / "manifest.json").exists()
    assert (out / "samples.jsonl").exists()
    assert len(list((out / "frames").glob("frame_*.png"))) > 0
    assert dialog.result() == QDialog.DialogCode.Accepted  # accepted after success


# ------------------------------------------------------------- review queue

def test_review_btn_disabled_without_recording(tmp_path):
    win = PlayerWindow(recordings_root=str(tmp_path / "root"))
    assert not win._review_btn.isEnabled()
    win.close()


def test_review_dialog_lists_uncertainty_candidates(window):
    """Uncertainty candidates only: the low-confidence frame, not the solid one."""
    from app.ui.review_dialog import ReviewDialog
    from dataset.review import ReviewQueue

    _write_perception(window._recording, {
        5: [("one", 0.42)],
        15: [("two", 0.95)],
    })
    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    assert len(queue.pending()) == 1
    dialog = ReviewDialog(queue)
    assert dialog._list.count() == 1
    text = dialog._list.item(0).text()
    assert "frame      5" in text
    assert "low confidence 0.42" in text
    assert "priority=58" in text
    assert "15" not in text
    dialog.close()


def test_review_accept_verifies_frame_annotations(window):
    """Accepting a frame propagates: its model annotations become VERIFIED."""
    from app.ui.review_dialog import ReviewDialog
    from annotation.store import load_annotations
    from annotation.types import AnnotationStatus
    from dataset.review import ReviewQueue
    from perception.types import BoundingBox

    _write_perception(window._recording, {5: [("one", 0.42)]})
    window._annotations.create(
        label="one",
        bbox=BoundingBox(0.1, 0.1, 0.5, 0.5),
        frame_index=5,
        t=window._recording.frame_time(5),
        source="model",
        status=AnnotationStatus.PREDICTED,
    )
    window._annotations.save()

    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty", "annotation_uncertainty"])
    assert {item.frame_index for item in queue.pending()} == {5}
    dialog = ReviewDialog(queue)
    dialog._list.setCurrentRow(0)
    assert dialog._selected_item().frame_index == 5
    dialog._decide("accept")

    assert queue.verdicts.get("5") == "accepted"
    assert dialog._list.count() == 0  # moved out of pending
    fresh = load_annotations(window._recording.directory)
    assert [a for a in fresh if a.frame_index == 5][0].status == AnnotationStatus.VERIFIED
    reloaded = ReviewQueue(window._recording)
    assert reloaded.verdicts.get("5") == "accepted"  # persisted to queue.jsonl
    dialog.close()


def test_review_reject_marks_frame_annotations_rejected(window):
    from app.ui.review_dialog import ReviewDialog
    from annotation.store import load_annotations
    from annotation.types import AnnotationStatus
    from dataset.review import ReviewQueue
    from perception.types import BoundingBox

    _write_perception(window._recording, {5: [("one", 0.42)]})
    window._annotations.create(
        label="one",
        bbox=BoundingBox(0.1, 0.1, 0.5, 0.5),
        frame_index=5,
        t=window._recording.frame_time(5),
        source="model",
        status=AnnotationStatus.PREDICTED,
    )
    window._annotations.save()

    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    dialog = ReviewDialog(queue)
    dialog._list.setCurrentRow(0)
    dialog._decide("reject")

    assert queue.verdicts.get("5") == "rejected"
    fresh = load_annotations(window._recording.directory)
    assert [a for a in fresh if a.frame_index == 5][0].status == AnnotationStatus.REJECTED
    dialog.close()


def test_review_skip_leaves_annotations_untouched(window):
    from app.ui.review_dialog import ReviewDialog
    from annotation.store import load_annotations
    from annotation.types import AnnotationStatus
    from dataset.review import ReviewQueue

    _write_perception(window._recording, {5: [("one", 0.42)]})
    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    dialog = ReviewDialog(queue)
    dialog._list.setCurrentRow(0)
    dialog._decide("skip")

    assert queue.verdicts.get("5") == "skipped"
    assert load_annotations(window._recording.directory).verified_count == 0
    dialog.close()


def test_review_show_decided_restores_decided_items(window):
    from app.ui.review_dialog import ReviewDialog
    from dataset.review import ReviewQueue

    _write_perception(window._recording, {5: [("one", 0.42)]})
    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    dialog = ReviewDialog(queue)
    dialog._list.setCurrentRow(0)
    dialog._decide("accept")
    assert dialog._list.count() == 0

    dialog._show_decided.setChecked(True)
    QApplication.processEvents()
    assert dialog._list.count() == 1
    assert "→ accepted" in dialog._list.item(0).text()
    dialog.close()


def test_review_dialog_jump_closes_loop(window):
    from app.ui.review_dialog import ReviewDialog
    from dataset.review import ReviewQueue

    _write_perception(window._recording, {5: [("one", 0.42)]})
    jumped = []
    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    dialog = ReviewDialog(queue, on_jump=jumped.append)
    dialog._list.setCurrentRow(0)
    assert dialog._jump_btn.isEnabled()
    dialog._on_jump()
    assert jumped == [5]
    dialog.close()


def test_review_jump_to_seeks_player_frame(window):
    window._review_jump_to(12)
    assert window._current_t == pytest.approx(window._recording.frame_time(12))
    assert abs(window._timeline._playhead - window._recording.frame_time(12)) < 0.11
    assert not window._playing


def test_review_edit_to_sets_edit_mode(window):
    window._review_edit_to(12)
    assert window._current_t == pytest.approx(window._recording.frame_time(12))
    assert window._show_annotations.isChecked()
    assert window._edit_annotations_btn.isChecked()
    assert window._annotation_mode
    assert not window._playing


def test_review_dialog_edit_fires_callback(window):
    from app.ui.review_dialog import ReviewDialog
    from dataset.review import ReviewQueue

    _write_perception(window._recording, {5: [("one", 0.42)]})
    edited = []
    queue = ReviewQueue(window._recording)
    queue.refresh(strategies=["uncertainty"])
    dialog = ReviewDialog(queue, on_edit=edited.append)
    dialog._list.setCurrentRow(0)
    assert dialog._edit_btn.isEnabled()
    dialog._on_edit()
    assert edited == [5]
    dialog.close()


def test_review_button_opens_dialog_and_saves_queue(window, monkeypatch):
    from PySide6.QtCore import QObject, Signal
    from dataset.review import ReviewQueue

    opened = []

    class _FakeDialog(QObject):
        finished = Signal(int)

        def __init__(self, queue, on_jump=None, on_edit=None, parent=None):
            super().__init__()
            self.queue = queue
            self.on_jump = on_jump
            self.on_edit = on_edit
            opened.append(self)

        def exec(self):
            return 0

    monkeypatch.setattr("app.ui.review_dialog.ReviewDialog", _FakeDialog)
    window._review_btn.click()

    assert len(opened) == 1
    assert isinstance(window._review_queue, ReviewQueue)
    assert window._review_queue.path.exists()  # refresh() persisted the queue
    assert window._review_status.text().startswith("Review:")
    assert opened[0].on_jump.__self__ is window
    assert opened[0].on_jump.__func__ is PlayerWindow._review_jump_to
    assert opened[0].on_edit.__self__ is window
    assert opened[0].on_edit.__func__ is PlayerWindow._review_edit_to

