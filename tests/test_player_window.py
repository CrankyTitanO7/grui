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

