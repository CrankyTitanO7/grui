"""Event-discovery tests — no models, no GPU; synthetic sightings only.

Covers the rule framework (clustering, appearance/disappearance),
sighting loading from annotations vs perception results, persistence
round-trip, and the `grui perception events` CLI end-to-end.
"""

from __future__ import annotations

import json

import pytest

from annotation.store import AnnotationStore
from perception.events import (
    AppearanceRule,
    DisappearanceRule,
    Event,
    TrackSighting,
    cluster_presence,
    detect_events,
    load_sightings,
    make_rules,
    read_events,
    render_events,
    sightings_from_annotations,
    sightings_from_perception,
    write_events,
)
from perception.types import BoundingBox, Detection, PerceptionResult
from tests.fakes import build_synthetic_recording


def _sightings(*triples: tuple[str, float]) -> list[TrackSighting]:
    """Build sightings from (label, t) pairs; frame_index follows fps=10."""
    return [
        TrackSighting(label=label, t=t, frame_index=int(round(t * 10)), source="annotation")
        for label, t in triples
    ]


def test_cluster_splits_at_gap():
    sightings = _sightings(
        ("boss", 0.0), ("boss", 0.5), ("boss", 3.0), ("boss", 3.5),
        ("projectile", 1.0),
    )
    clusters = cluster_presence(sightings, gap_s=2.0)
    boss_clusters = [c.sightings for c in clusters if c.label == "boss"]
    assert len(boss_clusters) == 2
    assert [s.t for s in boss_clusters[0]] == [0.0, 0.5]
    assert [s.t for s in boss_clusters[1]] == [3.0, 3.5]
    assert [c.label for c in clusters] == ["boss", "boss", "projectile"]


def test_appearance_disappearance_events():
    sightings = _sightings(
        ("boss", 0.0), ("boss", 0.5), ("boss", 3.0), ("boss", 3.5),
        ("projectile", 1.0),
    )
    events = detect_events(sightings, [AppearanceRule(gap_s=2.0), DisappearanceRule(gap_s=2.0)])
    appearances = [e for e in events if e.kind == "appearance"]
    disappearances = [e for e in events if e.kind == "disappearance"]
    assert [(e.label, e.start_t) for e in appearances] == [
        ("boss", 0.0), ("projectile", 1.0), ("boss", 3.0),
    ]
    assert [(e.label, e.start_t) for e in disappearances] == [
        ("boss", 0.0), ("projectile", 1.0), ("boss", 3.0),
    ]
    boss_dis = [e for e in disappearances if e.label == "boss"]
    assert boss_dis[0].end_t == 0.5
    assert boss_dis[0].detail["duration_s"] == 0.5


def test_min_sightings_filters():
    sightings = _sightings(("boss", 0.0), ("boss", 0.2))
    events = detect_events(sightings, [AppearanceRule(gap_s=2.0, min_sightings=3)])
    assert events == []


def test_detect_events_sorted_and_deterministic():
    sightings = _sightings(("a", 2.0), ("a", 2.1), ("b", 0.5))
    first = detect_events(sightings, [AppearanceRule(gap_s=1.0)])
    second = detect_events(sightings, [AppearanceRule(gap_s=1.0)])
    assert [(e.label, e.start_t) for e in first] == [("b", 0.5), ("a", 2.0)]
    assert first == second


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown event rule"):
        make_rules(["nope"], gap_s=2.0)


def test_gap_must_be_positive():
    with pytest.raises(ValueError, match="gap_s"):
        AppearanceRule(gap_s=0)


def test_event_roundtrip():
    event = Event(
        kind="disappearance", label="boss",
        start_t=1.0, end_t=2.0, start_frame=10, end_frame=20,
        detail={"sightings": 5},
    )
    rebuilt = Event.from_dict(event.to_dict())
    assert rebuilt == event


def test_write_read_events_roundtrip(tmp_path):
    events = [
        Event(kind="appearance", label="boss", start_t=0.0, end_t=0.0,
              start_frame=0, end_frame=0, detail={"sightings": 1}),
    ]
    write_events(tmp_path, events)
    assert read_events(tmp_path) == events
    assert read_events(tmp_path / "missing") == []


def test_render_events_empty_and_nonempty():
    assert render_events([]) == "No events detected."
    lines = render_events([
        Event(kind="appearance", label="boss", start_t=1.5, end_t=1.5,
              start_frame=15, end_frame=15, detail={"sightings": 4}),
    ]).splitlines()
    assert "1 event(s):" in lines[0]
    assert "appearance" in lines[2] and "boss" in lines[2]


# ----------------------------------------------------------- sightings sources

def test_sightings_from_annotations(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=40, fps=10)
    store = AnnotationStore.load(recording.directory / "annotations")
    store.create("boss", BoundingBox(1.0, 2.0, 3.0, 4.0), frame_index=5,
                 t=recording.frame_time(5))
    store.save()
    sightings = sightings_from_annotations(recording.directory)
    assert [(s.label, s.frame_index, s.source) for s in sightings] == [("boss", 5, "annotation")]


def test_sightings_from_perception(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=40, fps=10)
    directory = recording.directory
    perception_dir = directory / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    (perception_dir / "results.jsonl").write_text(
        json.dumps(PerceptionResult(
            frame_index=3, t=recording.frame_time(3), prompt="boss",
            detections=[Detection("boss", BoundingBox(1, 2, 3, 4), 0.9)],
        ).to_dict()) + "\n",
        encoding="utf-8",
    )
    (perception_dir / "manifest.json").write_text('{"format_version": 1}', encoding="utf-8")
    sights = sightings_from_perception(directory)
    assert [(s.label, s.frame_index, s.source) for s in sights] == [("boss", 3, "perception")]
    with pytest.raises(ValueError):
        sightings_from_perception(tmp_path)  # not a recording


def test_load_sightings_auto_prefers_annotations(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=40, fps=10)
    store = AnnotationStore.load(recording.directory / "annotations")
    store.create("boss", BoundingBox(1, 2, 3, 4), frame_index=5, t=recording.frame_time(5))
    store.save()
    assert load_sightings(recording.directory, "auto")[0].source == "annotation"
    with pytest.raises(FileNotFoundError):
        load_sightings(recording.directory, "perception")
    with pytest.raises(ValueError):
        load_sightings(recording.directory, "bogus")


# ---------------------------------------------------------------- CLI

@pytest.fixture()
def annotated_recording(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=100, fps=10)
    store = AnnotationStore.load(recording.directory / "annotations")
    for t in (0.3, 0.6, 0.9):
        store.create("boss", BoundingBox(1, 2, 3, 4),
                     frame_index=recording.nearest_frame_index(t),
                     t=recording.snap_to_frame(t))
    store.create("projectile", BoundingBox(5, 6, 7, 8),
                 frame_index=recording.nearest_frame_index(4.0),
                 t=recording.snap_to_frame(4.0))
    store.save()
    return recording


def _run_perception_cli(capsys, *argv: str) -> int:
    from perception.cli import run

    code = run(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_events_end_to_end(capsys, annotated_recording):
    code, out, err = _run_perception_cli(capsys, "events", str(annotated_recording.directory))
    assert code == 0, err
    assert "2 event(s)" in out or "4 event(s)" in out
    events = read_events(annotated_recording.directory)
    assert len(events) == 4  # appearance + disappearance per label/cluster
    assert {e.kind for e in events} == {"appearance", "disappearance"}
    assert {e.label for e in events} == {"boss", "projectile"}
    manifest_exists = (annotated_recording.directory / "perception" / "events.jsonl").exists()
    assert manifest_exists


def test_cli_events_dry_run_writes_nothing(capsys, annotated_recording):
    code, out, _ = _run_perception_cli(
        capsys, "events", "--dry-run", "--rule", "appearance", str(annotated_recording.directory)
    )
    assert code == 0
    assert "1 event(s)" in out or "2 event(s)" in out
    assert not (annotated_recording.directory / "perception" / "events.jsonl").exists()


def test_cli_events_json_output(capsys, annotated_recording):
    code, out, _ = _run_perception_cli(capsys, "events", "--json", str(annotated_recording.directory))
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list) and payload
    assert {"kind", "label", "start_t", "end_t"} <= set(payload[0])


def test_cli_events_no_sightings(capsys, tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=20, fps=10)
    code, out, _ = _run_perception_cli(capsys, "events", str(recording.directory))
    assert code == 0
    assert "No sightings" in out


def test_cli_events_unknown_rule(capsys, annotated_recording):
    code, out, err = _run_perception_cli(
        capsys, "events", "--rule", "bogus", str(annotated_recording.directory)
    )
    assert code == 1
    assert "unknown event rule" in err