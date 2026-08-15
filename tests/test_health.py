"""Dataset health statistics tests (incl. the single-recording regression)."""

from __future__ import annotations

import pytest

from storage.recording import load_recording
from tests.fakes import build_synthetic_recording


def _root_with_recording(tmp_path, name="demo-1"):
    root = tmp_path / "recordings_root"
    recording = build_synthetic_recording(root, session_id=name, n_frames=30, fps=10)
    return root, recording


def test_analyze_recordings_root(tmp_path):
    from dataset.health import analyze_recordings

    root, recording = _root_with_recording(tmp_path)
    stats, per_demo = analyze_recordings(root)
    assert stats.demonstrations == 1
    assert per_demo[0].frames == len(recording.frame_times) == 30
    assert per_demo[0].duration == pytest.approx(recording.duration)
    assert stats.total_frames == 30


def test_analyze_recordings_accepts_single_recording_dir(tmp_path):
    """Regression: `grui dataset health <recording>` read all zeros before."""
    from dataset.health import analyze_recordings

    _root, recording = _root_with_recording(tmp_path)
    stats_via_root, per_demo = analyze_recordings(recording.directory)  # <- the bug case
    assert stats_via_root.demonstrations == 1
    assert stats_via_root.total_frames == 30
    assert stats_via_root.total_duration == pytest.approx(recording.duration)
    assert per_demo[0].directory == recording.directory.name


def test_analyze_recordings_empty_root(tmp_path):
    from dataset.health import analyze_recordings

    empty = tmp_path / "empty"
    empty.mkdir()
    stats, per_demo = analyze_recordings(empty)
    assert stats.demonstrations == 0
    assert per_demo == []


def test_cli_health_single_recording(tmp_path, capsys):
    from dataset.cli import run

    _root, recording = _root_with_recording(tmp_path)
    code = run(["health", str(recording.directory)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Demonstrations:            1" in out
    assert "Total frames:              30" in out
    avg = out.split("Average demonstration:")[1].split("\n")[0]
    assert "0.0s" not in avg  # not the all-zeros regression


def test_cli_health_empty_root_hints(tmp_path, capsys):
    from dataset.cli import run

    empty = tmp_path / "empty"
    empty.mkdir()
    code = run(["health", str(empty)])
    captured = capsys.readouterr()
    assert code == 0
    assert "no recordings found under" in captured.err