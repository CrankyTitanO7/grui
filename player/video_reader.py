"""Threaded, frame-accurate video reader based on OpenCV.

The reader owns the ``cv2.VideoCapture`` (not thread-safe across threads)
and is controlled with a small command queue: seek to a frame index, or
toggle play. When playing it decodes ahead into a bounded frame queue; the
UI thread drains that queue on a timer and paces playback itself, so
display stays exactly in sync with the ``frames.jsonl`` timestamps.

Frames are tuples ``(frame_index, bgr_ndarray)``; a ``None`` frame marks
end-of-video. Seeking flushes the queue so stale frames are never shown.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.02


class VideoReader(threading.Thread):
    """Decode-ahead reader for a single mp4 file."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(name="video-reader", daemon=True)
        self.path = Path(path)
        self._commands: queue.Queue = queue.Queue()
        self._frames: queue.Queue = queue.Queue(maxsize=6)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self.error: str | None = None
        self.fps = 30.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.playing = False

    # ------------------------------------------------------------ control

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until the video metadata is available. True on success."""
        return self._ready.wait(timeout)

    def seek(self, frame_index: int) -> None:
        """Seek to a frame index (flush + decode that frame)."""
        self._commands.put(("seek", max(0, frame_index)))

    def set_playing(self, playing: bool) -> None:
        self._commands.put(("play", bool(playing)))

    def drain(self) -> list[tuple[int, np.ndarray | None]]:
        """Non-blocking drain of decoded frames."""
        frames = []
        while True:
            try:
                frames.append(self._frames.get_nowait())
            except queue.Empty:
                return frames

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout)

    # ------------------------------------------------------------ thread

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            self.error = f"could not open video: {self.path}"
            self._ready.set()
            logger.error("%s", self.error)
            return
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._ready.set()

        pos = 0
        self._next_frame_at = 0.0
        while not self._stop.is_set():
            try:
                command = self._commands.get(timeout=_POLL_INTERVAL)
            except queue.Empty:
                command = None
            if command is not None:
                kind = command[0]
                if kind == "seek":
                    pos = int(command[1])
                    self._flush()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, frame = cap.read()
                    if ok:
                        self._push(pos, frame)
                        pos += 1
                    else:
                        self._push(pos, None)
                elif kind == "play":
                    self.playing = bool(command[1])
                continue
            if self.playing:
                # Pace to real time so the bounded queue stays nearly empty
                # instead of flooding the UI (which would drop frames).
                now = time.monotonic()
                if now < self._next_frame_at:
                    continue
                ok, frame = cap.read()
                if not ok:
                    self.playing = False
                    self._push(pos, None)
                    continue
                self._push(pos, frame)
                pos += 1
                self._next_frame_at = now + 1.0 / max(self.fps, 1.0)

        cap.release()

    def _push(self, frame_index: int, frame: np.ndarray | None) -> None:
        # Block with back-pressure instead of dropping; the UI drains on a
        # timer, so a full queue only means the UI is momentarily busy.
        while not self._stop.is_set():
            try:
                self._frames.put((frame_index, frame), timeout=0.05)
                return
            except queue.Full:
                continue

    def _flush(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return
