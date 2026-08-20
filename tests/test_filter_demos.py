"""A2 (section 19): rare-action demo filtering - chords, filter-demos CLI, report section."""

from __future__ import annotations

import pytest

from dataset.health import ActionDistribution
from tests.fakes import build_synthetic_recording


def _rec(root, name):
    return build_synthetic_recording(root, session_id=name, n_frames=30, fps=10)


def _chord_recording(root, name):
    """Both KeyA and KeySPACE held together from ~frame 1 through frame 9."""
    return build_synthetic_recording(
        root,
        session_id=name,
        n_frames=30,
        fps=10,
        events=[
            {"t": 0.02, "device": "keyboard", "event": "down", "code": "KeyA"},
            {"t": 0.05, "device": "keyboard", "event": "down", "code": "KeySPACE"},
            {"t": 0.95, "device": "keyboard", "event": "up", "code": "KeyA"},
            {"t": 0.96, "device": "keyboard", "event": "up", "code": "KeySPACE"},
        ],
    )


# --------------------------------------------------------- parsing


def test_normalize_chord_code():
    from dataset.health import normalize_chord_code

    assert normalize_chord_code("A") == "KeyA"
    assert normalize_chord_code("SPACE") == "KeySPACE"
    assert normalize_chord_code("KeyQ") == "KeyQ"
    assert normalize_chord_code("mouse:left") == "mouse:left"
    assert normalize_chord_code("None") == "None"
    assert normalize_chord_code(" F1 ") == "KeyF1"


def test_parse_chord():
    from dataset.health import parse_chord

    assert parse_chord("A + SPACE") == ("KeyA", "KeySPACE")
    assert parse_chord("KeyQ") == ("KeyQ",)
    assert parse_chord("mouse:left + KeyZ") == ("KeyZ", "mouse:left")


# --------------------------------------------------------- chord detection


def test_recording_has_chord(tmp_path):
    from dataset.health import recording_has_chord

    recording = _chord_recording(tmp_path / "root", "demo")
    assert recording_has_chord(recording, ("KeyA", "KeySPACE"))
    assert recording_has_chord(recording, ("KeyA",))  # subset of a held chord counts
    assert not recording_has_chord(recording, ("KeyA", "KeyQ"))


# --------------------------------------------------------- filter_demos


def test_filter_demos_matching_demo_with_counts(tmp_path):
    from dataset.health import action_distribution, filter_demos

    root = tmp_path / "root"
    matching = _chord_recording(root, "demo-a")
    _rec(root, "demo-b")

    (name, total, count) = filter_demos(root, [("KeyA", "KeySPACE")])[0]
    assert name == matching.directory.name
    assert total == len(matching.frame_times)
    assert count == action_distribution(matching).chords[("KeyA", "KeySPACE")]
    assert count > 0


def test_filter_demos_or_semantics_multiple_contains(tmp_path):
    from dataset.health import filter_demos

    root = tmp_path / "root"
    matching = _chord_recording(root, "demo-a")
    _rec(root, "demo-b")

    (name, _, _) = filter_demos(root, [("KeyQ",), ("KeyA", "KeySPACE")])[0]
    assert name == matching.directory.name


def test_filter_demos_no_match(tmp_path):
    from dataset.health import filter_demos

    root = tmp_path / "root"
    _rec(root, "one")
    _rec(root, "two")
    assert filter_demos(root, [("KeyQ", "KeyZ")]) == []


def test_filter_demos_single_recording_directory(tmp_path):
    from dataset.health import filter_demos

    recording = _chord_recording(tmp_path / "root", "demo-a")
    (name, _, count) = filter_demos(recording.directory, [("KeyA", "KeySPACE")])[0]
    assert name == recording.directory.name
    assert count > 0


def test_filter_demos_skips_corrupt_demo(tmp_path):
    from dataset.health import filter_demos

    root = tmp_path / "root"
    matching = _chord_recording(root, "good")
    corrupt = _rec(root, "corrupt")
    (corrupt.directory / "metadata.json").unlink()

    matches = filter_demos(root, [("KeyA", "KeySPACE")])
    assert [name for name, _, _ in matches] == [matching.directory.name]


# --------------------------------------------------------- rare action labels


def test_rare_action_labels_relative_threshold():
    from dataset.health import rare_action_labels

    dist = ActionDistribution(samples=100)
    dist.keys["KeyA"] = 60
    dist.keys["KeyB"] = 2
    dist.keys["KeyC"] = 6
    dist.idle_frames = 32
    labels = rare_action_labels(dist)
    assert {label for label, _ in labels} == {"KeyB"}  # 0.02 < 0.05 * 0.60
    assert dict(labels)["KeyB"] == pytest.approx(0.02)


def test_rare_action_labels_empty_without_movement_share():
    from dataset.health import rare_action_labels

    dist = ActionDistribution(samples=10)
    dist.idle_frames = 10
    assert rare_action_labels(dist) == []


# --------------------------------------------------------- coverage report section


def _perception(recording, by_frame):
    import json

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


def test_render_report_rare_actions_section(tmp_path):
    from dataset.coverage import analyze, render_report

    root = tmp_path / "root"
    dominant = build_synthetic_recording(
        root, session_id="demo-a", n_frames=300, fps=10,
        events=[
            {"t": 0.01, "device": "keyboard", "event": "down", "code": "KeyA"},
            {"t": 5.01, "device": "keyboard", "event": "up", "code": "KeyA"},
            {"t": 8.01, "device": "keyboard", "event": "down", "code": "KeyQ"},
            {"t": 8.11, "device": "keyboard", "event": "up", "code": "KeyQ"},
        ],
    )
    plain = _rec(root, "demo-b")
    _perception(dominant, {0: ["boss"], 5: ["boss"]})
    _perception(plain, {0: ["crate"]})

    recordings = [dominant, plain]
    report = analyze(recordings, source="auto")
    out = render_report(report, recordings=recordings)
    assert "Rare actions" in out
    assert "KeyQ" in out
    assert "demo-a" in out

    without = render_report(report)
    assert "Rare actions" not in without


# --------------------------------------------------------- CLI


def test_cli_filter_demos_matching_root(tmp_path, capsys):
    from dataset.cli import run

    root = tmp_path / "root"
    matching = _chord_recording(root, "demo-a")
    _rec(root, "demo-b")

    code = run(["filter-demos", str(root), "--contains", "A + SPACE"])
    assert code == 0
    out = capsys.readouterr().out
    assert matching.directory.name in out
    assert "matching frame(s)" in out
    assert "demo-b" not in out


def test_cli_filter_demos_no_match(tmp_path, capsys):
    from dataset.cli import run

    root = tmp_path / "root"
    _rec(root, "one")
    code = run(["filter-demos", str(root), "--contains", "B + C"])
    assert code == 0
    assert "no demonstrations" in capsys.readouterr().out


def test_cli_filter_demos_requires_contains(tmp_path, capsys):
    from dataset.cli import run

    root = tmp_path / "root"
    _rec(root, "one")
    code = run(["filter-demos", str(root)])
    assert code == 2


def test_cli_filter_demos_single_recording_dir(tmp_path, capsys):
    from dataset.cli import run

    matching = _chord_recording(tmp_path / "root", "demo-a")
    code = run(["filter-demos", str(matching.directory), "--contains", "KeyA", "--contains", "A + SPACE"])
    assert code == 0
    assert matching.directory.name in capsys.readouterr().out