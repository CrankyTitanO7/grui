"""Player window episode row tests: load, add, delete, suggest, jump, ticks."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.ui.episode_suggest_dialog import EpisodeSuggestDialog
from app.ui.player_window import PlayerWindow
from dataset.episodes import Episode, read_episodes, write_episodes
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


def _seed_episodes(window, episodes):
    write_episodes(window._recording.directory, episodes)
    window._load_episodes()


def test_episodes_loaded_into_combo(window):
    _seed_episodes(window, [Episode(0.2, 1.5, reason="manual")])
    assert window._episodes_combo.count() == 1
    assert "0.20s" in window._episodes_combo.itemText(0)
    assert "1 episode" in window._episodes_status.text()
    assert window._delete_episode_btn.isEnabled()


def test_no_episodes_status_hint(window):
    assert window._episodes_combo.count() == 0
    assert not window._delete_episode_btn.isEnabled()
    assert "No episodes" in window._episodes_status.text()


def test_episode_selected_jumps(window):
    _seed_episodes(window, [Episode(0.4, 1.5, reason="manual")])
    window._on_episode_selected(0)
    assert window._current_t == pytest.approx(0.4)


def test_add_episode_from_selection(window):
    window._on_selection_changed((0.3, 1.2))
    window._on_add_episode()
    stored = read_episodes(window._recording.directory)
    assert len(stored) == 1
    episode = stored[0]
    assert episode.start == pytest.approx(window._recording.snap_to_frame(0.3))
    assert episode.end == pytest.approx(window._recording.snap_to_frame(1.2))
    assert episode.reason == "manual"
    assert window._episodes_combo.count() == 1
    assert "episode" in window._episodes_status.text()


def test_add_episode_merges_overlapping(window):
    _seed_episodes(window, [Episode(0.2, 1.5, reason="manual")])
    window._on_selection_changed((0.4, 1.1))
    window._on_add_episode()
    stored = read_episodes(window._recording.directory)
    assert len(stored) == 1
    assert stored[0].end == pytest.approx(1.5)  # merged into the existing span


def test_add_episode_requires_selection(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args))
    )
    window._on_add_episode()
    assert warnings and warnings[0][1] == "No Selection"


def test_delete_episode(window):
    _seed_episodes(
        window,
        [Episode(0.2, 1.0, reason="manual"), Episode(2.0, 2.5, reason="manual")],
    )
    window._episodes_combo.setCurrentIndex(1)
    window._on_delete_episode()
    stored = read_episodes(window._recording.directory)
    assert len(stored) == 1
    assert stored[0].start == pytest.approx(0.2)


class _FakeSuggestDialog:
    """Mimics the real dialog: on accept it writes the episodes it computed."""

    def __init__(self, recording, parent=None):
        from dataset.episodes import suggest_episodes

        self.episodes = suggest_episodes(
            recording,
            min_inactivity=0.0,
            use_markers=True,
            use_visual=False,
            use_events=False,
            use_input_changes=False,
        )


def test_suggest_episodes_applies(window, monkeypatch):
    from PySide6.QtWidgets import QDialog

    recording = window._recording

    class _Fake(_FakeSuggestDialog):
        def exec(self):
            write_episodes(recording.directory, self.episodes)
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr("app.ui.player_window.EpisodeSuggestDialog", _Fake)
    window._on_suggest_episodes()
    stored = read_episodes(recording.directory)
    assert stored  # marker boundary at 0.5s -> >= 2 episodes
    assert window._episodes_combo.count() == len(stored)


def test_episode_ticks_appear_in_timeline(window):
    _seed_episodes(window, [Episode(0.3, 1.4, reason="manual")])
    window._show_annotations.setChecked(True)
    ticks = window._annotation_ticks()
    episode_kinds = [t for t in ticks if t[1] == "episode"]
    assert len(episode_kinds) == 2  # episode start + end boundaries


def test_suggest_dialog_previews_and_applies(window):
    recording = window._recording
    dialog = EpisodeSuggestDialog(recording)
    assert dialog.episodes  # at least the full-recording fallback
    assert "episode" in dialog._preview_status.text().lower()
    dialog._on_apply()
    stored = read_episodes(recording.directory)
    assert stored == dialog.episodes
    assert dialog.result() == EpisodeSuggestDialog.DialogCode.Accepted