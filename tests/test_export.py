"""Export pipeline: cutting a real recording re-encodes video and remaps data."""

import cv2
import pytest

from editor.export import export_recording
from editor.timeline import EditSession
from storage.recording import list_recordings, load_recording
from tests.fakes import build_synthetic_recording


@pytest.fixture()
def source(tmp_path):
    return build_synthetic_recording(
        tmp_path / "source",
        n_frames=30,
        fps=10,
        events=[
            {"t": 0.3, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 1.7, "device": "keyboard", "event": "up", "code": "KeyW"},
        ],
        markers=[{"t": 0.5, "label": "intro"}],
    )


def _new_edited(source, out_dir, cut=(1.0, 2.0)):
    session = EditSession(source.duration, source.frame_times, initial_ranges=[(0.0, source.duration)])
    session.cut(*cut)
    saved = export_recording(source, session.timeline, out_dir)
    return load_recording(saved.directory)


def test_export_creates_new_recording(source, tmp_path):
    out = _new_edited(source, tmp_path / "out")
    assert out.directory.is_dir()
    assert out.video_path.exists()
    assert list_recordings(tmp_path / "out") == [out.directory]
    assert out.metadata["edited_from"]["session_id"] == source.session_id
    assert out.metadata["edit_clips"]


def test_export_keeps_expected_frame_count(source, tmp_path):
    out = _new_edited(source, tmp_path / "out")
    cut_start = source.snap_to_frame(1.0)
    cut_end = source.snap_to_frame(2.0)
    kept = sum(1 for t in source.frame_times if not (cut_start <= t < cut_end))
    assert out.frame_times.size == kept
    cap = cv2.VideoCapture(str(out.video_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == kept
    cap.release()


def test_export_remaps_events_and_markers(source, tmp_path):
    out = _new_edited(source, tmp_path / "out")
    events = {e["code"]: e for e in out.events if e.get("device") == "keyboard"}
    # event before cut kept at same time
    assert events["KeyW"]["event"] == "down"
    assert events["KeyW"]["t"] == pytest.approx(0.3, abs=0.05)
    # event inside cut region dropped
    assert not any(e.get("event") == "up" for e in out.events)
    # lifecycle events added
    assert any(e.get("event") == "recording_start" for e in out.events)
    assert any(e.get("event") == "recording_stop" for e in out.events)
    # marker kept
    assert [(m["t"], m["label"]) for m in out.markers if m["label"] == "intro"]


def test_export_metadata_has_edit_history(source, tmp_path):
    session = EditSession(source.duration, source.frame_times, initial_ranges=[(0.0, source.duration)])
    session.cut(1.0, 2.0)
    session.paste(3.0, session.copy(1.5, 1.8))
    out = load_recording(
        export_recording(source, session.timeline, tmp_path / "out", edit_history=session.history).directory
    )
    ops = [h["op"] for h in out.metadata["edit_history"]]
    assert ops == ["cut", "paste"]  # copy is read-only and not logged


def test_export_round_trip_loads_edits(source, tmp_path):
    session = EditSession(source.duration, source.frame_times, initial_ranges=[(0.0, source.duration)])
    session.cut(1.0, 2.0)
    saved = export_recording(source, session.timeline, tmp_path / "out")
    out = load_recording(saved.directory)
    assert out.metadata["edited_from"]["session_id"] == source.session_id


def test_export_no_edits_is_copy(source, tmp_path):
    session = EditSession(source.duration, source.frame_times, initial_ranges=[(0.0, source.duration)])
    out = load_recording(export_recording(source, session.timeline, tmp_path / "out").directory)
    assert out.frame_times.size == source.frame_times.size


def test_export_preserves_custom_event_fields(source, tmp_path):
    source.events.append({"t": 0.2, "device": "mouse", "event": "move", "x": 77, "y": 88})
    out = _new_edited(source, tmp_path / "out")
    moves = [e for e in out.events if e.get("event") == "move"]
    assert len(moves) == 1
    assert moves[0]["x"] == 77 and moves[0]["y"] == 88
