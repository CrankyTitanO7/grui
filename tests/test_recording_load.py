"""Read-side recording API: load_recording / list_recordings / frame helpers."""

import pytest

from storage.recording import list_recordings, load_recording
from tests.fakes import build_synthetic_recording


def test_load_round_trip(tmp_path):
    events = [
        {"t": 0.1, "device": "keyboard", "event": "down", "code": "KeyW"},
        {"t": 0.4, "device": "keyboard", "event": "up", "code": "KeyW"},
    ]
    markers = [{"t": 0.2, "label": "hello"}]
    rec = build_synthetic_recording(tmp_path / "src", n_frames=12, fps=6, events=events, markers=markers)
    loaded = load_recording(rec.directory)
    assert loaded.session_id == rec.session_id
    assert loaded.frame_times.size == 12
    assert loaded.events == events
    assert loaded.markers == markers
    assert loaded.fps == 6
    assert loaded.width == 64
    assert loaded.height == 48
    assert loaded.duration == pytest.approx(rec.duration)


def test_list_recordings(tmp_path):
    r1 = build_synthetic_recording(tmp_path / "a")
    build_synthetic_recording(tmp_path / "b")
    assert list_recordings(tmp_path / "a") == [r1.directory]
    assert len(list_recordings(tmp_path / "b")) == 1
    assert list_recordings(tmp_path / "missing") == []


def test_list_recordings_multiple_same_second(tmp_path):
    r1 = build_synthetic_recording(tmp_path / "x", session_id="same")
    r2 = build_synthetic_recording(tmp_path / "x", session_id="same")
    dirs = list_recordings(tmp_path / "x")
    assert len(dirs) == 2
    assert set(dirs) == {r1.directory, r2.directory}
    assert dirs == sorted(dirs, key=lambda p: p.name, reverse=True)


def test_load_rejects_non_recording(tmp_path):
    with pytest.raises(ValueError):
        load_recording(tmp_path / "nope")


def test_frame_helpers(tmp_path):
    rec = build_synthetic_recording(tmp_path / "s", n_frames=10, fps=10)
    assert rec.nearest_frame_index(rec.frame_times[3]) == 3
    assert rec.snap_to_frame(rec.frame_times[4] + 0.001) == rec.frame_times[4]
    assert rec.frame_time(999) == pytest.approx(999 / 10.0)
