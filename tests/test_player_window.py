"""Player window offscreen smoke tests: load, play a frame, edit, save."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

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
    window._current_t = window._recording.snap_to_frame(0.3)
    window._set_boundary("in")
    window._current_t = window._recording.snap_to_frame(0.6)
    window._set_boundary("out")
    window._on_cut()
    assert window._session.timeline.duration < original
    assert window._clipboard is not None
    window._on_undo()
    assert window._session.timeline.duration == original
    window._on_redo()
    assert window._session.timeline.duration < original


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
