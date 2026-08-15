"""Dataset coverage analysis tests (§18): situations, presence, per-demo matrix."""

from __future__ import annotations

import json

import pytest

from annotation.store import AnnotationStore
from annotation.types import AnnotationStatus
from perception.types import BoundingBox
from tests.fakes import build_synthetic_recording


def _rec(root, name):
    return build_synthetic_recording(
        root, session_id=name, n_frames=30, fps=10
    )


def _annotate(recording, by_frame):
    """by_frame: frame_index -> [(label, status)]; written to the annotations layer."""
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


def _two_demos(tmp_path):
    root = tmp_path / "demos"
    a = _rec(root, "demo-a")
    b = _rec(root, "demo-b")
    _annotate(a, {
        0: [("boss", AnnotationStatus.PREDICTED)] * 1,
        1: [("boss", AnnotationStatus.VERIFIED)],
        2: [("boss", AnnotationStatus.REJECTED)],  # rejected: not coverage
        3: [("boss", AnnotationStatus.PREDICTED), ("enemy", AnnotationStatus.PREDICTED)],
        4: [("boss", AnnotationStatus.PREDICTED), ("enemy", AnnotationStatus.PREDICTED)],
        5: [("projectile", AnnotationStatus.PREDICTED)],
    })
    _annotate(b, {
        0: [("boss", AnnotationStatus.PREDICTED)],
        1: [("boss", AnnotationStatus.PREDICTED)],
        2: [("boss", AnnotationStatus.PREDICTED), ("projectile", AnnotationStatus.PREDICTED)],
    })
    return a, b


def test_analyze_annotations_source(tmp_path):
    from dataset.coverage import analyze

    a, b = _two_demos(tmp_path)
    report = analyze([a, b])
    assert report.source == "annotations"
    assert report.total_analysed_frames == 5 + 3

    assert report.labels["boss"] == 7  # rejected frame excluded; 4 in demo-a + 3 in demo-b
    assert report.labels["enemy"] == 2
    assert report.labels["projectile"] == 2

    situations = report.situations
    assert situations[("boss",)] == 2 + 2  # demo-a frames 0,1; demo-b frames 0,1
    assert situations[("boss", "enemy")] == 2
    assert situations[("boss", "projectile")] == 1
    assert situations[("projectile",)] == 1


def test_under_covered_flags_few_demo_situations(tmp_path):
    from dataset.coverage import analyze

    a, b = _two_demos(tmp_path)
    report = analyze([a, b])
    under = dict((name, demos) for name, demos, _ in report.under_covered(2))
    assert under == {
        "boss + enemy": 1,
        "boss + projectile": 1,
        "projectile": 1,  # frame 5 of demo-a; demo-b's projectile co-occurs with boss
    }
    # lower the floor: nothing is under-covered with min_demos=1
    assert report.under_covered(1) == []


def test_analyze_auto_falls_back_to_perception(tmp_path):
    from dataset.coverage import analyze

    rec = _rec(tmp_path / "root", "perc-demo")
    _write_perception(rec, {
        2: ["enemy"],
        3: ["enemy", "boss"],
    })
    report = analyze([rec])
    assert report.source == "perception"
    assert report.labels["enemy"] == 2
    assert report.labels["boss"] == 1
    assert report.situations[("enemy",)] == 1
    assert report.situations[("boss", "enemy")] == 1


def test_analyze_annotations_preferred_over_perception(tmp_path):
    from dataset.coverage import analyze

    rec = _rec(tmp_path / "root", "both-demo")
    _annotate(rec, {5: [("boss", AnnotationStatus.PREDICTED)]})
    _write_perception(rec, {6: ["enemy"]})
    report = analyze([rec])
    assert report.source == "annotations"
    assert report.labels["boss"] == 1
    assert report.labels["enemy"] == 0


def test_frame_situations_missing_source_raises(tmp_path):
    from dataset.coverage import frame_situations

    rec = _rec(tmp_path / "root", "bare-demo")
    with pytest.raises(ValueError, match="no annotations layer"):
        frame_situations(rec, "annotations")
    with pytest.raises(ValueError, match="no perception results"):
        frame_situations(rec, "perception")
    source, by_frame = frame_situations(rec, "auto")  # silent only on auto
    assert source == "auto"
    assert by_frame == {}


def test_render_report_bars_and_warnings(tmp_path):
    from dataset.coverage import analyze, render_report

    a, b = _two_demos(tmp_path)
    report = analyze([a, b])
    text = render_report(report, min_demos=2)
    assert "Coverage Report (2 demonstrations) - source: annotations" in text
    assert "Situation coverage - label sets co-occurring on a frame" in text
    assert "boss + enemy" in text
    assert "Label presence" in text
    assert "#" in text
    assert "Per-demonstration coverage" in text
    assert "[x]" in text
    assert "! under-covered: boss + enemy - only 1 of 2 demonstrations" in text


def test_render_report_empty(tmp_path):
    from dataset.coverage import analyze, render_report

    rec = _rec(tmp_path / "root", "bare")
    text = render_report(analyze([rec]))
    assert "No frames carry labels" in text


def test_report_to_dict_json_shape(tmp_path):
    from dataset.coverage import analyze, report_to_dict

    a, b = _two_demos(tmp_path)
    data = report_to_dict(analyze([a, b]))
    assert data["source"] == "annotations"
    assert [d["demo"] for d in data["demos"]] == [a.directory.name, b.directory.name]
    assert data["aggregate"]["analysed_frames"] == 8
    assert data["aggregate"]["situations"]["boss + enemy"] == 2
    assert json.dumps(data)  # serializable


def test_cli_coverage_report(tmp_path, capsys):
    from dataset.cli import run

    _two_demos(tmp_path)
    code = run(["coverage", str(tmp_path / "demos")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Coverage Report (2 demonstrations) - source: annotations" in out
    assert "! under-covered: boss + enemy" in out


def test_cli_coverage_single_recording(tmp_path, capsys):
    from dataset.cli import run

    a, _b = _two_demos(tmp_path)
    code = run(["coverage", str(a.directory)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Coverage Report (1 demonstration)" in out


def test_cli_coverage_json(tmp_path, capsys):
    from dataset.cli import run

    _two_demos(tmp_path)
    code = run(["coverage", str(tmp_path / "demos"), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["aggregate"]["labels"]["boss"] == 7


def test_cli_coverage_missing_source_fails(tmp_path, capsys):
    from dataset.cli import run

    a, _b = _two_demos(tmp_path)
    code = run(["coverage", str(a.directory), "--source", "perception"])
    err = capsys.readouterr().err
    assert code == 1
    assert "no perception results" in err


def test_cli_coverage_only_filters(tmp_path, capsys):
    from dataset.cli import run

    _two_demos(tmp_path)
    code = run(["coverage", str(tmp_path / "demos"), "--only", "demo-b"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Coverage Report (1 demonstration)" in out
    assert "demo-a" not in out