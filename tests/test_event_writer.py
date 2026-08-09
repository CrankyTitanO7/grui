"""Tests for the thread-safe JSONL event writer."""

import json
import threading

import pytest

from storage.event_writer import EventWriter


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_write_and_stop(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.start()
    for i in range(50):
        writer.write({"t": i / 10, "device": "keyboard", "event": "down"})
    writer.stop()

    lines = _read_lines(path)
    assert len(lines) == 50
    assert lines[0]["t"] == 0.0
    assert lines[49]["t"] == 4.9
    assert writer.written == 50


def test_write_from_many_threads(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.start()

    def worker():
        for i in range(200):
            writer.write({"t": i, "thread": threading.current_thread().name})

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.stop()

    lines = _read_lines(path)
    assert len(lines) == 800
    assert all(isinstance(line["t"], int) for line in lines)


def test_write_before_start_returns_false(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl")
    assert writer.write({"t": 0}) is False
    assert writer.written == 0


def test_stop_creates_empty_file(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.start()
    writer.stop()
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_stop_is_idempotent(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl")
    writer.start()
    writer.write({"a": 1})
    writer.stop()
    writer.stop()
    assert writer.written == 1


def test_full_queue_drops_without_hanging(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, max_queue=8)
    writer.start()
    for i in range(500):
        writer.write({"t": i})
    writer.stop()

    assert writer.written + writer.dropped == 500
    assert writer.dropped > 0
    assert len(_read_lines(path)) == writer.written


def test_stop_drains_remaining_events(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.start()
    for i in range(1000):
        writer.write({"t": i})
    writer.stop()
    assert writer.written == 1000
    assert len(_read_lines(path)) == 1000


@pytest.mark.parametrize("n", [0, 1, 2, 250])
def test_arbitrary_write_counts(tmp_path, n):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.start()
    for i in range(n):
        writer.write({"t": i, "device": "mouse"})
    writer.stop()
    assert writer.written == n
    assert len(_read_lines(path)) == n
