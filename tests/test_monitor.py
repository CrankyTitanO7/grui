"""Training monitor tests: progress bar and metrics logging."""

from __future__ import annotations

import io
import json
import sys
import time

import pytest

torch = pytest.importorskip("torch")


def test_progress_bar_quiet_when_not_tty(monkeypatch):
    from ml.monitor import ProgressBar

    monkeypatch.setenv("GRUI_NO_PROGRESS", "")
    monkeypatch.setattr("sys.stdout", io.StringIO())
    bar = ProgressBar(10, prefix="epoch 1/5")
    bar.update(5, loss=0.5)
    bar.close()
    assert bar.enabled is False
    assert "epoch 1/5" not in sys.stdout.getvalue()


def test_progress_bar_renders_when_enabled(monkeypatch):
    from ml.monitor import ProgressBar

    stream = io.StringIO()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr(stream, "isatty", lambda: True)
    bar = ProgressBar(10, prefix="epoch 1/5")
    assert bar.enabled is True
    bar.update(5, loss=0.5)
    bar.update(5, loss=0.4)
    bar.close()
    output = stream.getvalue()
    assert "epoch 1/5" in output
    assert "5/10" in output and "10/10" in output
    assert "loss=0.45" in output  # running average of 0.5 and 0.4
    assert "eta" in output


def test_progress_bar_no_progress_flag(monkeypatch):
    from ml.monitor import ProgressBar

    stream = io.StringIO()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr(stream, "isatty", lambda: True)
    assert ProgressBar(10, enabled=False).enabled is False
    assert not stream.getvalue()


def test_metrics_logger_roundtrip(tmp_path):
    from ml.monitor import MetricsLogger

    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.write({"event": "epoch", "epoch": 1, "loss": 1.5})
    logger.write({"event": "epoch", "epoch": 2, "loss": 1.25})
    logger.write({"event": "summary", "epochs": 2, "final_loss": 1.25})
    records = MetricsLogger.read(path)
    assert [r["epoch"] for r in records if r["event"] == "epoch"] == [1, 2]
    assert records[-1]["event"] == "summary"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["loss"] == 1.5


def test_train_writes_metrics_file(tmp_path):
    from dataset.build import DatasetConfig, build_dataset
    from ml.monitor import MetricsLogger
    from ml.train import main as train_main
    from tests.fakes import build_synthetic_recording

    recording = build_synthetic_recording(
        tmp_path / "recordings",
        n_frames=60,
        fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.75, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 0.5, "device": "mouse", "event": "move", "x": 100, "y": 50},
        ],
    )
    dataset = tmp_path / "dataset"
    build_dataset(recording, DatasetConfig(observation_duration=0.4, fps=5, stride=0.1), dataset)
    ckpt = tmp_path / "ckpt.pt"
    code = train_main(
        ["--dataset", str(dataset), "--out", str(ckpt),
         "--epochs", "2", "--batch-size", "4", "--hidden", "16", "--no-progress"]
    )
    assert code == 0
    records = MetricsLogger.read(tmp_path / "ckpt.metrics.jsonl")  # default path
    epochs = [r for r in records if r["event"] == "epoch"]
    assert len(epochs) == 2
    for record in epochs:
        assert 0.0 <= record["key_acc"] <= 1.0
        assert 0.0 <= record["button_acc"] <= 1.0
        assert record["dx_mae"] >= 0.0 and record["dy_mae"] >= 0.0
    assert records[-1]["event"] == "summary"
    assert records[-1]["checkpoint"] == str(ckpt)
