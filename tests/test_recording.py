"""Tests for raw recording directory creation and metadata."""

import json

from storage.recording import RawRecording


def test_create_layout(tmp_path):
    recording = RawRecording.create(tmp_path, "abc123", {"version": 1, "session_id": "abc123"})
    assert recording.directory.is_dir()
    assert recording.directory.name.endswith("_abc123")
    assert recording.metadata_path.name == "metadata.json"
    assert recording.video_path.name == "video.mp4"
    assert recording.events_path.name == "events.jsonl"
    assert recording.markers_path.name == "markers.jsonl"
    assert recording.frames_path.name == "frames.jsonl"
    for path in recording.files().values():
        assert path.parent == recording.directory

    metadata = recording.read_metadata()
    assert metadata["session_id"] == "abc123"
    assert metadata["version"] == 1


def test_create_same_second_unique_sessions(tmp_path):
    first = RawRecording.create(tmp_path, "s1", {})
    second = RawRecording.create(tmp_path, "s1", {})
    assert first.directory != second.directory
    assert second.directory.name.startswith(first.directory.name)
    assert second.directory.is_dir()


def test_update_metadata_merges_and_keeps_version(tmp_path):
    recording = RawRecording.create(tmp_path, "sid", {"version": 1, "a": 1})
    recording.update_metadata(duration=12.5, stats={"frames": 10})
    metadata = recording.read_metadata()
    assert metadata["a"] == 1
    assert metadata["duration"] == 12.5
    assert metadata["stats"] == {"frames": 10}
    assert metadata["version"] == 1


def test_update_metadata_is_atomic_valid_json(tmp_path):
    recording = RawRecording.create(tmp_path, "sid", {"version": 1})
    recording.update_metadata(duration=3.14)
    raw = recording.metadata_path.read_text(encoding="utf-8")
    json.loads(raw)
    assert recording.metadata_path.suffix == ".json"
