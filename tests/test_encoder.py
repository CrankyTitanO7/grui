"""Real FFmpeg encoder tests with synthetic frames (no screen capture)."""

import json
import queue as queue_module

import numpy as np
import pytest

from recorder.clock import SessionClock
from recorder.config import EncoderConfig
from recorder.encoder import FFmpegEncoder
from storage.event_writer import EventWriter


@pytest.fixture()
def clock():
    return SessionClock()


def _random_frame():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_encodes_frames_to_mp4(tmp_path, clock):
    frame_queue = queue_module.Queue()
    frames_writer = EventWriter(tmp_path / "frames.jsonl")
    frames_writer.start()
    encoder = FFmpegEncoder(
        EncoderConfig(),
        tmp_path / "video.mp4",
        frames_writer,
        frame_queue,
        clock,
        fps=10,
    )
    encoder.start()

    for _ in range(30):
        frame_queue.put((clock.now(), _random_frame()))
    frame_queue.put(None)
    encoder.stop()
    frames_writer.stop()

    assert encoder.returncode == 0
    assert encoder.error is None
    assert encoder.frames_encoded == 30

    video = tmp_path / "video.mp4"
    assert video.exists()
    assert video.stat().st_size > 1000

    lines = frames_writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    entries = [json.loads(line) for line in lines]
    assert [e["frame_index"] for e in entries] == list(range(30))
    timestamps = [e["t"] for e in entries]
    assert timestamps == sorted(timestamps)


def test_no_frames_produces_no_video(tmp_path, clock):
    frame_queue = queue_module.Queue()
    frames_writer = EventWriter(tmp_path / "frames.jsonl")
    frames_writer.start()
    encoder = FFmpegEncoder(
        EncoderConfig(),
        tmp_path / "video.mp4",
        frames_writer,
        frame_queue,
        clock,
        fps=10,
    )
    encoder.start()
    frame_queue.put(None)
    encoder.stop()
    frames_writer.stop()

    assert not (tmp_path / "video.mp4").exists()
    assert encoder.returncode is None
    assert encoder.error is None
    assert encoder.frames_encoded == 0
    assert (tmp_path / "frames.jsonl").read_text(encoding="utf-8") == ""


def test_frames_are_encoded_in_queue_order(tmp_path, clock):
    frame_queue = queue_module.Queue()
    frames_writer = EventWriter(tmp_path / "frames.jsonl")
    frames_writer.start()
    encoder = FFmpegEncoder(
        EncoderConfig(),
        tmp_path / "video.mp4",
        frames_writer,
        frame_queue,
        clock,
        fps=10,
    )
    encoder.start()

    t = clock.now()
    for _ in range(10):
        frame_queue.put((t, _random_frame()))
        t += 0.033
    frame_queue.put(None)
    encoder.stop()
    frames_writer.stop()

    entries = [
        json.loads(line) for line in frames_writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 10
    assert [e["t"] for e in entries] == sorted(e["t"] for e in entries)
    assert [e["frame_index"] for e in entries] == list(range(10))
