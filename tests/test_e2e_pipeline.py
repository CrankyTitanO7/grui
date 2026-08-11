"""End-to-end pipeline test: raw recording -> edit -> export -> dataset.

Chains the real components with no fakes past the initial capture:
synthetic frames are encoded with real ffmpeg, the edited timeline is
re-encoded by the export pipeline, and the exported recording feeds the
dataset builder (via the CLI). Verifies sync, remapping and sample content.
"""

from __future__ import annotations

import json

import pytest

from dataset.cli import run as dataset_cli
from editor.export import export_recording
from editor.timeline import EditSession
from storage.recording import load_recording
from tests.fakes import build_synthetic_recording


@pytest.fixture()
def source(tmp_path):
    return build_synthetic_recording(
        tmp_path / "recordings",
        n_frames=60,
        fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.75, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 1.5, "device": "mouse", "event": "move", "x": 200, "y": 100},
            {"t": 2.5, "device": "mouse", "event": "button_down", "button": "left"},
            {"t": 3.0, "device": "mouse", "event": "button_up", "button": "left"},
        ],
        markers=[{"t": 3.0, "label": "phase_two"}],
    )


def test_full_pipeline(source, tmp_path):
    # 1. edit: cut the middle region out (times snap to frame boundaries;
    #    the timeline itself starts at the first frame's capture time t0)
    t0 = float(source.frame_times[0])
    in_t, out_t = source.snap_to_frame(1.0), source.snap_to_frame(2.0)
    session = EditSession(source.duration, source.frame_times)
    session.cut(1.0, 2.0)
    expected_duration = (source.duration - t0) - (out_t - in_t)
    assert session.timeline.duration == pytest.approx(expected_duration, abs=1e-6)

    # 2. export: real re-encode into a new raw recording
    exported = export_recording(source, session.timeline, tmp_path / "recordings", edit_history=session.history)
    edited = load_recording(exported.directory)
    assert edited.metadata["edited_from"]["session_id"] == source.session_id
    assert (edited.directory / "video.mp4").stat().st_size > 0

    # frame times are monotonic and in-bounds, like the source
    assert list(edited.frame_times) == sorted(edited.frame_times)
    assert edited.frame_times[-1] == pytest.approx(edited.duration, abs=1e-9)

    # events were remapped: the KeyW block should survive (both endpoints
    # outside the cut region) and the marker too
    keys = [
        e for e in edited.events if e.get("device") == "keyboard" and e.get("code") == "KeyW"
    ]
    assert len(keys) == 2
    marker = [m for m in edited.markers if m.get("label") == "phase_two"]
    assert len(marker) == 1
    # clip2 starts at (in_t - t0); marker lands at clip2.start + (3.0 - out_t)
    assert marker[0]["t"] == pytest.approx((in_t - t0) + (3.0 - out_t), abs=1e-6)

    # 3. dataset: build from the EXPORTED recording through the real CLI
    out = tmp_path / "dataset"
    assert dataset_cli(["build", str(edited.directory), "--out", str(out),
                        "--obs-duration", "0.3", "--fps", "5", "--stride", "0.1"]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["session_id"] == edited.session_id
    assert manifest["count"] > 0

    frame_times = {}
    for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        frame_times[entry["frame_index"]] = entry["t"]
        assert (out / entry["path"]).exists()

    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["t"] for s in samples] == sorted(s["t"] for s in samples)

    # 4. sample content is consistent: window covers [t - 0.3, t] in the
    #    exported clock, and the held-key state matches the remapped events
    for sample in samples:
        t = sample["t"]
        for idx in sample["observation"]:
            assert t - 0.3 - 1e-9 <= frame_times[idx] <= t + 1e-9
        key_w = "KeyW" in sample["action"]["keys"]
        matches = [k for k in keys if k["t"] <= t + 1e-9]
        down = bool(matches) and matches[-1]["event"] == "down"
        assert key_w == down
