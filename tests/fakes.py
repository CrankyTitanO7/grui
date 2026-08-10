"""Fake capture components used to exercise the session without hardware.

These mirror the constructor signatures of the real components in
``recorder/`` so the session can be patched wholesale in tests.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class FakeScreen:
    """Pushes a few synthetic frames into the frame queue when started."""

    def __init__(self, config, clock, frame_queue, *, is_paused=None, error_cb=None):
        self.config = config
        self.clock = clock
        self.frame_queue = frame_queue
        self.is_paused = is_paused or (lambda: False)
        self.error_cb = error_cb
        self.frames_captured = 0
        self.frames_dropped = 0
        self.errors = 0
        self._stop = threading.Event()

    def start(self):
        for _ in range(5):
            self.frame_queue.put((self.clock.now(), np.zeros((4, 8, 3), dtype=np.uint8)))
            self.frames_captured += 1

    def stop(self, timeout: float = 10.0):
        self._stop.set()


class FakeEncoder:
    """Consumes frames in a thread and records them via the frames writer."""

    def __init__(self, config, video_path, frames_writer, frame_queue, clock, fps):
        self.config = config
        self.video_path = video_path
        self.frames_writer = frames_writer
        self.frame_queue = frame_queue
        self.clock = clock
        self.fps = fps
        self.frames_encoded = 0
        self.returncode = 0
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            _, frame = item
            self.frames_writer.write(
                {"frame_index": self.frames_encoded, "t": self.clock.now()}
            )
            self.frames_encoded += 1
            time.sleep(0.001)

    def stop(self, timeout: float = 30.0):
        if self._thread.is_alive():
            self._thread.join(timeout)
        self.video_path.write_bytes(b"fake-video-bytes")


class FakeKeyboard:
    def __init__(self, clock, sink, is_paused=None):
        self.clock = clock
        self.sink = sink
        self.is_paused = is_paused or (lambda: False)

    def start(self):
        if not self.is_paused():
            self.sink.write(
                {"t": self.clock.now(), "device": "keyboard", "event": "down", "code": "KeyW"}
            )

    def stop(self):
        pass


class FakeMouse:
    def __init__(self, clock, sink, is_paused=None):
        self.clock = clock
        self.sink = sink
        self.is_paused = is_paused or (lambda: False)

    def start(self):
        if not self.is_paused():
            self.sink.write(
                {"t": self.clock.now(), "device": "mouse", "event": "move", "x": 1, "y": 2}
            )

    def stop(self):
        pass


def patch_session_components(monkeypatch, session_module):
    """Replace all hardware-touching pieces of the session with fakes."""
    monkeypatch.setattr(session_module, "ScreenRecorder", FakeScreen)
    monkeypatch.setattr(session_module, "KeyboardRecorder", FakeKeyboard)
    monkeypatch.setattr(session_module, "MouseRecorder", FakeMouse)
    monkeypatch.setattr(session_module, "FFmpegEncoder", FakeEncoder)
    monkeypatch.setattr(session_module, "resolve_monitor_size", lambda index: (1920, 1080))


def build_synthetic_recording(
    root,
    *,
    session_id=None,
    n_frames=30,
    fps=10,
    size=(48, 64),
    events=None,
    markers=None,
):
    """Create a real on-disk recording (true ffmpeg encode) for player/export tests.

    Frame times are captured with a real SessionClock, so ``RecordingData.frame_times``
    is exact; callers should compute expectations from the loaded data rather than
    assuming absolute values.
    """
    import json
    import queue
    import uuid

    import numpy as np

    from recorder.clock import SessionClock
    from recorder.config import EncoderConfig
    from recorder.encoder import FFmpegEncoder
    from storage.event_writer import EventWriter
    from storage.recording import RawRecording, load_recording

    session_id = session_id or f"synth-{uuid.uuid4().hex[:8]}"
    height, width = size
    metadata = {
        "version": 1,
        "session_id": session_id,
        "platform": "test",
        "screen": {"width": width, "height": height, "fps": fps, "monitor_index": 0},
        "input": {"keyboard": True, "mouse": True},
        "duration": 0.0,
    }
    recording = RawRecording.create(root, session_id, metadata)

    frame_queue = queue.Queue()
    frames_writer = EventWriter(recording.frames_path)
    frames_writer.start()
    clock = SessionClock()
    encoder = FFmpegEncoder(EncoderConfig(), recording.video_path, frames_writer, frame_queue, clock, fps)
    encoder.start()
    rng = np.random.default_rng(7)
    t = clock.now()
    for _ in range(n_frames):
        frame_queue.put((t, rng.integers(0, 256, (height, width, 3), dtype=np.uint8)))
        t += 1.0 / fps
    frame_queue.put(None)
    encoder.stop()
    frames_writer.stop()

    for path, rows in ((recording.events_path, events or []), (recording.markers_path, markers or [])):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    recording.update_metadata(duration=t)
    return load_recording(recording.directory)
