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
    window._on_selection_changed((0.3, 0.6))
    assert window._timeline_sel is not None
    assert window._timeline_sel[0] == window._recording.snap_to_frame(0.3)
    assert window._timeline_sel[1] == window._recording.snap_to_frame(0.6)
    window._on_cut()
    assert window._session.timeline.duration < original
    assert window._clipboard is not None
    assert window._timeline_sel is None  # cleared after the edit
    window._on_undo()
    assert window._session.timeline.duration == original
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


def test_keyboard_fits_at_once(window):
    view = window._keyboard_view
    min_width = view._grid.minimumSize().width()
    assert min_width < 700
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
