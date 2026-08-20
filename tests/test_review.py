"""Review strategies tests (A1): transition + coverage candidates (§16)."""

from __future__ import annotations

import json

import pytest

from annotation.store import AnnotationStore
from annotation.types import AnnotationStatus
from perception.types import BoundingBox
from tests.fakes import build_synthetic_recording


def _rec(root, name):
    return build_synthetic_recording(root, session_id=name, n_frames=30, fps=10)


def _annotate(recording, by_frame):
    store = AnnotationStore(recording.directory / "annotations")
    for frame_index, entries in by_frame.items():
        for label, status in entries:
            store.create(
                label,
                BoundingBox(0.1, 0.1, 0.5, 0.5),
                frame_index,
                recording.frame_time(frame_index),
                source="model",
                status=status,
            )
    store.save()


def _write_perception(recording, by_frame):
    """by_frame: frame_index -> [labels]; written to perception/results.jsonl."""
    out = recording.directory / "perception"
    out.mkdir(exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps({"provider": "test", "provider_version": "0", "prompts": ["p"], "count": len(by_frame)}),
        encoding="utf-8",
    )
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for frame_index, labels in by_frame.items():
            fh.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "t": recording.frame_time(frame_index),
                        "prompt": "p",
                        "detections": [
                            {"label": label, "bbox": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0}}
                            for label in labels
                        ],
                    }
                )
                + "\n"
            )


# --------------------------------------------------------- transition


def test_transition_candidates_from_input_changes(tmp_path):
    from dataset.review import transition_candidates

    recording = build_synthetic_recording(
        tmp_path / "root", session_id="demo", n_frames=30, fps=10,
        events=[
            {"t": 0.35, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 1.05, "device": "keyboard", "event": "up", "code": "KeyW"},
        ],
    )
    items = transition_candidates(recording, 10)
    frames = [i.frame_index for i in items]
    assert frames == [4, 11]  # first frame with the key active, first after release
    assert items[0].kind == "transition"
    assert "input -> KeyW" in items[0].reason
    assert "input -> idle" in items[1].reason


def test_transition_candidates_from_situation_change(tmp_path):
    from dataset.review import transition_candidates

    recording = _rec(tmp_path / "root", "demo")
    _write_perception(recording, {5: ["boss"], 7: ["boss", "enemy", "crate"], 20: ["enemy"]})
    items = transition_candidates(recording, 10)
    frames = [i.frame_index for i in items]
    assert frames == [5, 7, 20]
    first = next(i for i in items if i.frame_index == 5)
    assert "situation -> boss" in first.reason
    mid = next(i for i in items if i.frame_index == 7)
    assert "boss" in mid.reason and "enemy" in mid.reason
    # two labels toggled -> higher priority than a single label toggle
    assert mid.priority > first.priority
    assert mid.priority == pytest.approx(50.0)  # 30 + 10 * 2 toggles
    assert first.priority == pytest.approx(40.0)


def test_transition_candidates_no_changes_is_empty(tmp_path):
    from dataset.review import transition_candidates

    recording = _rec(tmp_path / "root", "demo")
    assert transition_candidates(recording, 10) == []


def test_transition_candidates_respects_limit(tmp_path):
    from dataset.review import transition_candidates

    recording = build_synthetic_recording(
        tmp_path / "root", session_id="demo", n_frames=30, fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyA"},
            {"t": 0.25, "device": "keyboard", "event": "down", "code": "KeyB"},
            {"t": 0.45, "device": "keyboard", "event": "up", "code": "KeyA"},
            {"t": 0.55, "device": "keyboard", "event": "up", "code": "KeyB"},
        ],
    )
    assert len(transition_candidates(recording, 2)) == 2


# ----------------------------------------------------------- coverage


def test_coverage_candidates_under_covered_situations(tmp_path):
    from dataset.review import coverage_candidates

    root = tmp_path / "root"
    a = _rec(root, "demo-a")
    b = _rec(root, "demo-b")
    _annotate(a, {0: [("boss", AnnotationStatus.PREDICTED)], 1: [("boss", AnnotationStatus.PREDICTED)]})
    _annotate(b, {0: [("crate", AnnotationStatus.PREDICTED)]})

    items = coverage_candidates(a, 10, recording_root=root)
    frames = sorted(i.frame_index for i in items)
    assert frames == [0, 1]
    item = items[0]
    assert item.kind == "frame"
    assert "'boss'" in item.reason and "1 of 2 demos" in item.reason
    assert item.priority == pytest.approx(60.0)  # 40 + (2 - 1) * 20

    # demo-b's crate is equally under-covered within its own recording
    items_b = coverage_candidates(b, 10, recording_root=root)
    assert [i.frame_index for i in items_b] == [0]


def test_coverage_candidates_noop_without_root(tmp_path):
    from dataset.review import coverage_candidates

    recording = _rec(tmp_path / "root", "demo")
    _annotate(recording, {0: [("boss", AnnotationStatus.PREDICTED)]})
    assert coverage_candidates(recording, 10) == []


def test_coverage_candidates_empty_root(tmp_path):
    from dataset.review import coverage_candidates

    root = tmp_path / "root"
    recording = _rec(root, "demo")
    _annotate(recording, {0: [("boss", AnnotationStatus.PREDICTED)]})
    assert coverage_candidates(recording, 10, recording_root=tmp_path / "empty") == []


def test_coverage_candidates_full_coverage_is_empty(tmp_path):
    from dataset.review import coverage_candidates

    root = tmp_path / "root"
    a = _rec(root, "demo-a")
    b = _rec(root, "demo-b")
    _annotate(a, {0: [("boss", AnnotationStatus.PREDICTED)]})
    _annotate(b, {0: [("boss", AnnotationStatus.PREDICTED)]})  # covered twice
    assert coverage_candidates(a, 10, recording_root=root) == []


def test_build_queue_forwards_recording_root(tmp_path):
    from dataset.review import build_queue

    root = tmp_path / "root"
    a = _rec(root, "demo-a")
    b = _rec(root, "demo-b")
    _annotate(a, {0: [("boss", AnnotationStatus.PREDICTED)]})
    _annotate(b, {0: [("crate", AnnotationStatus.PREDICTED)]})
    items = build_queue(a, strategies=["coverage", "transition"], recording_root=root)
    assert items and all(i.kind in ("frame", "transition") for i in items)
    with pytest.raises(ValueError, match="unknown review strategy"):
        build_queue(a, strategies=["nope"])


# ----------------------------------------------------------------- CLI


def test_cli_review_build_transition_and_coverage(tmp_path, capsys):
    from dataset.cli import run

    root = tmp_path / "root"
    a = build_synthetic_recording(
        root, session_id="demo-a", n_frames=30, fps=10,
        events=[
            {"t": 0.35, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 1.05, "device": "keyboard", "event": "up", "code": "KeyW"},
        ],
    )
    b = _rec(root, "demo-b")
    _annotate(a, {0: [("boss", AnnotationStatus.PREDICTED)]})
    _annotate(b, {0: [("crate", AnnotationStatus.PREDICTED)]})

    code = run(
        [
            "review", "build", str(a.directory),
            "--strategy", "transition", "--strategy", "coverage",
            "--recording-root", str(root),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "candidates" in out and "pending" in out
    assert (a.directory / "review" / "queue.jsonl").exists()

    code = run(["review", "list", str(a.directory)])
    assert code == 0
    listing = capsys.readouterr().out
    assert "transition" in listing and "under-covered" in listing


def test_cli_review_unknown_strategy_rejected(tmp_path, capsys):
    from dataset.cli import run

    recording = _rec(tmp_path / "root", "demo")
    with pytest.raises(SystemExit):
        run(["review", str(recording.directory), "build", "--strategy", "nope"])