"""Dataset builder tests: window sampling, sync, actions, CLI."""

from __future__ import annotations

import json

import pytest

from dataset.build import DatasetConfig, build_dataset
from tests.fakes import build_synthetic_recording


@pytest.fixture()
def recording(tmp_path):
    return build_synthetic_recording(
        tmp_path / "root",
        n_frames=30,
        fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.75, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 0.3, "device": "mouse", "event": "move", "x": 100, "y": 50},
            {"t": 0.4, "device": "mouse", "event": "move", "x": 120, "y": 55},
            {"t": 0.5, "device": "mouse", "event": "button_down", "button": "left"},
            {"t": 0.8, "device": "mouse", "event": "button_up", "button": "left"},
        ],
    )


def test_build_dataset_sync_and_actions(recording, tmp_path):
    config = DatasetConfig(observation_duration=0.2, fps=5, stride=0.1)
    out = build_dataset(recording, config, tmp_path / "ds")

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["source"]["session_id"] == recording.session_id
    assert manifest["config"] == {
        "observation_duration": 0.2,
        "fps": 5.0,
        "stride": 0.1,
        "prediction_horizon": 0.2,
    }
    assert manifest["screen"] == {"width": recording.width, "height": recording.height}
    assert manifest["time_base"] == "seconds since session start"

    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    assert vocab == {"keys": ["KeyW"], "buttons": ["left"]}

    t0 = float(recording.frame_times[0])
    t1 = recording.duration
    expected_first = t0 + 0.2
    expected_count = int((t1 - expected_first + 1e-9) // 0.1) + 1
    assert manifest["count"] == expected_count > 0

    frame_times = {}
    for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        frame_times[entry["frame_index"]] = entry["t"]
        assert entry["t"] == pytest.approx(recording.frame_time(entry["frame_index"]))
        assert (out / entry["path"]).exists()
    assert manifest["observation_frames"] == len(frame_times)

    samples = [
        json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(samples) == expected_count
    times = [s["t"] for s in samples]
    assert times == sorted(times)
    assert times[0] == pytest.approx(expected_first)

    for sample in samples:
        t = sample["t"]
        assert sample["observation"], "every sample has a window"
        assert sample["observation"] == sorted(sample["observation"])
        for idx in sample["observation"]:
            assert t - 0.2 - 1e-9 <= frame_times[idx] <= t + 1e-9
            assert (out / "frames" / f"frame_{idx}.png").exists()
        action = sample["action"]
        if 0.15 <= t <= 0.75:
            assert "KeyW" in action["keys"]
        else:
            assert "KeyW" not in action["keys"]
        if 0.5 <= t <= 0.8:
            assert "left" in action["buttons"]
        else:
            assert "left" not in action["buttons"]
        if t >= 0.4:
            assert action["mouse"]["x"] == 120 and action["mouse"]["y"] == 55
        elif t >= 0.3:
            assert action["mouse"]["x"] == 100 and action["mouse"]["y"] == 50
        else:
            assert action["mouse"] is None

    last_pos = None
    for sample in samples:
        mouse = sample["action"]["mouse"]
        if mouse is None:
            continue
        if last_pos is None:
            assert (mouse["dx"], mouse["dy"]) == (0, 0)
        else:
            assert (mouse["dx"], mouse["dy"]) == (
                mouse["x"] - last_pos[0],
                mouse["y"] - last_pos[1],
            )
        last_pos = (mouse["x"], mouse["y"])


def test_build_dataset_is_deterministic(recording, tmp_path):
    config = DatasetConfig(observation_duration=0.2, fps=5, stride=0.1)
    first = build_dataset(recording, config, tmp_path / "ds1")
    second = build_dataset(recording, config, tmp_path / "ds2")
    assert (second / "samples.jsonl").read_bytes() == (first / "samples.jsonl").read_bytes()
    assert (second / "manifest.json").read_bytes() == (first / "manifest.json").read_bytes()
    assert (second / "frames.jsonl").read_bytes() == (first / "frames.jsonl").read_bytes()
    assert (second / "vocab.json").read_bytes() == (first / "vocab.json").read_bytes()
    assert sorted(p.name for p in (second / "frames").iterdir()) == sorted(
        p.name for p in (first / "frames").iterdir()
    )


def test_build_dataset_invalid_config_rejected(recording, tmp_path):
    with pytest.raises(ValueError, match="stride"):
        build_dataset(recording, DatasetConfig(stride=0), tmp_path / "ds")
    with pytest.raises(ValueError, match="horizon"):
        build_dataset(recording, DatasetConfig(prediction_horizon=-1), tmp_path / "ds")


def test_build_dataset_too_short_rejected(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=3, fps=10)
    with pytest.raises(ValueError, match="too short"):
        build_dataset(recording, DatasetConfig(observation_duration=2.0), tmp_path / "ds")


def test_build_dataset_missing_video_rejected(tmp_path):
    recording = build_synthetic_recording(tmp_path / "root", n_frames=3, fps=10)
    recording.video_path.unlink()
    with pytest.raises(ValueError, match="no video"):
        build_dataset(recording, DatasetConfig(), tmp_path / "ds")


def test_cli_build(recording, tmp_path, capsys):
    from dataset.cli import run

    out = tmp_path / "cli_ds"
    code = run(
        [
            "build",
            str(recording.directory),
            "--out",
            str(out),
            "--obs-duration",
            "0.2",
            "--fps",
            "5",
            "--stride",
            "0.1",
        ]
    )
    assert code == 0
    assert (out / "manifest.json").exists()
    assert (out / "samples.jsonl").exists()
    assert (out / "frames").is_dir()
    assert f"built dataset: {out}" in capsys.readouterr().out


def test_cli_build_default_out_dir(recording, tmp_path, capsys):
    from dataset.cli import run

    code = run(["build", str(recording.directory), "--obs-duration", "0.2", "--fps", "5", "--stride", "0.1"])
    assert code == 0
    default = recording.directory.parent / f"{recording.directory.name}_dataset"
    assert default.is_dir()


def test_cli_build_bad_path(tmp_path, capsys):
    from dataset.cli import run

    assert run(["build", str(tmp_path / "nope")]) == 1
    assert "error:" in capsys.readouterr().err
