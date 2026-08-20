"""Episode segmentation tests: signals and suggestion flags."""

import json

import numpy as np
import pytest

from dataset.episodes import (
    event_start_boundaries,
    input_change_boundaries,
    merge_episodes,
    read_episodes,
    suggest_episodes,
    write_episodes,
)
from storage.recording import RecordingData


def _recording(tmp_path, *, frame_times, events=None, markers=None):
    directory = tmp_path / "rec"
    directory.mkdir(parents=True, exist_ok=True)
    times = np.array(frame_times, dtype=float)
    return RecordingData(
        directory=directory,
        metadata={"duration": float(times[-1]) if times.size else 0.0},
        video_path=directory / "video.mp4",
        frame_times=times,
        events=events or [],
        markers=markers or [],
        fps=10.0,
        width=16,
        height=16,
    )


def _write_events(recording, starts):
    path = recording.directory / "perception"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, t in enumerate(starts):
            fh.write(
                json.dumps(
                    {
                        "kind": "appearance",
                        "label": "thing",
                        "start_t": t,
                        "end_t": t,
                        "start_frame": 0,
                        "end_frame": 0,
                    }
                )
                + "\n"
            )


def test_event_start_boundaries_from_events_file(tmp_path):
    recording = _recording(tmp_path, frame_times=list(range(10)))
    _write_events(recording, [3.0, 7.0])
    assert event_start_boundaries(recording) == [3.0, 7.0]


def test_event_start_boundaries_ignore_endpoints(tmp_path):
    recording = _recording(tmp_path, frame_times=list(range(10)))
    _write_events(recording, [0.0, 3.0, 9.0])
    assert event_start_boundaries(recording) == [3.0]


def test_event_start_boundaries_no_file(tmp_path):
    recording = _recording(tmp_path, frame_times=list(range(10)))
    assert event_start_boundaries(recording) == []


def test_input_change_boundaries_on_sharp_chord(tmp_path):
    events = [
        {"t": 0.2, "device": "keyboard", "event": "down", "code": "KeyW"},
        {"t": 0.5, "device": "keyboard", "event": "up", "code": "KeyW"},
        {"t": 0.6, "device": "keyboard", "event": "down", "code": "KeyA"},
        {"t": 0.6, "device": "keyboard", "event": "down", "code": "KeyB"},
    ]
    frame_times = [i * 0.1 for i in range(11)]
    recording = _recording(tmp_path, frame_times=frame_times, events=events)
    assert input_change_boundaries(recording) == [pytest.approx(0.6)]


def test_input_change_boundaries_ignore_small_changes(tmp_path):
    events = [
        {"t": 0.5, "device": "keyboard", "event": "down", "code": "KeyW"},
        {"t": 0.6, "device": "keyboard", "event": "up", "code": "KeyW"},
    ]
    frame_times = [i * 0.1 for i in range(11)]
    recording = _recording(tmp_path, frame_times=frame_times, events=events)
    assert input_change_boundaries(recording, min_toggle=2) == []


def test_suggest_reverts_to_full_without_signals(tmp_path):
    recording = _recording(tmp_path, frame_times=list(range(10)))
    episodes = suggest_episodes(
        recording, min_inactivity=5.0, use_markers=False,
        use_visual=False, use_events=False, use_input_changes=False,
    )
    assert len(episodes) == 1
    assert episodes[0].start == pytest.approx(0.0)
    assert episodes[0].end == pytest.approx(9.0)


def test_suggest_uses_event_signal(tmp_path):
    recording = _recording(tmp_path, frame_times=list(range(10)))
    _write_events(recording, [3.0, 7.0])
    with_events = suggest_episodes(
        recording, min_inactivity=5.0, use_markers=False,
        use_visual=False, use_events=True, use_input_changes=False,
    )
    assert len(with_events) == 3
    assert with_events[0].end == pytest.approx(3.0)
    assert with_events[1].start == pytest.approx(3.0)
    assert with_events[1].reason == "perception event"


def test_suggest_uses_input_change_signal(tmp_path):
    events = [
        {"t": 0.2, "device": "keyboard", "event": "down", "code": "KeyW"},
        {"t": 0.5, "device": "keyboard", "event": "up", "code": "KeyW"},
        {"t": 0.6, "device": "keyboard", "event": "down", "code": "KeyA"},
        {"t": 0.6, "device": "keyboard", "event": "down", "code": "KeyB"},
    ]
    frame_times = [i * 0.1 for i in range(11)]
    recording = _recording(tmp_path, frame_times=frame_times, events=events)
    episodes = suggest_episodes(
        recording, min_inactivity=5.0, use_markers=False,
        use_visual=False, use_events=False, use_input_changes=True,
    )
    assert len(episodes) == 2
    assert episodes[0].end == pytest.approx(0.6)
    assert episodes[0].reason == "input jump"
    assert episodes[1].start == pytest.approx(0.6)


def test_write_read_roundtrip(tmp_path):
    from dataset.episodes import Episode

    recording = _recording(tmp_path, frame_times=list(range(10)))
    originals = [Episode(1.0, 4.0, reason="manual"), Episode(5.0, 9.0, reason="visual")]
    path = write_episodes(recording.directory, originals)
    assert path.exists()
    loaded = read_episodes(recording.directory)
    assert loaded == originals


def test_merge_episodes_overlapping(tmp_path):
    from dataset.episodes import Episode

    merged = merge_episodes(
        [Episode(0.0, 4.0, reason="a")],
        [Episode(3.0, 8.0, reason="b")],
    )
    assert len(merged) == 1
    assert merged[0].start == pytest.approx(0.0)
    assert merged[0].end == pytest.approx(8.0)
    assert "a" in merged[0].reason and "b" in merged[0].reason