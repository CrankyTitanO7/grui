"""Screen capture using MSS.

The capture loop runs in its own thread so a slow encoder (or slow disk)
never blocks input capture or the UI. Frames are pushed as ``(t, frame)``
tuples into a bounded queue; the encoder thread drains it. Frames are
timestamped with the session clock immediately after grabbing, so their
timestamps correspond to the moment the pixels were captured.

No individual PNG/JPEG files are written during normal recording — frames
stream straight from the capture buffer into the encoder queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from recorder.clock import SessionClock
from recorder.config import ScreenConfig

logger = logging.getLogger(__name__)

_PAUSE_SLEEP = 0.05
_ERROR_SLEEP = 0.1
_MAX_CONSECUTIVE_ERRORS = 5


def list_monitors() -> list[dict[str, int]]:
    """Describe available monitors for the UI.

    Returns one dict per monitor: ``{"index": 0, "width": ..., "height": ...}``.
    """
    import mss

    with mss.mss() as sct:
        return [
            {"index": i, "width": int(m["width"]), "height": int(m["height"])}
            for i, m in enumerate(sct.monitors[1:])
        ]


def _resolve_region(monitors: list[dict[str, int]], monitor_index: int) -> dict[str, int]:
    """Map a configured monitor index to an MSS region.

    ``-1`` selects the combined region of all monitors; ``n`` selects the
    (n+1)-th entry of ``mss.monitors`` (monitor 0 is the first monitor).
    """
    if monitor_index == -1:
        return monitors[0]
    idx = monitor_index + 1
    if idx >= len(monitors):
        raise ValueError(
            f"monitor_index {monitor_index} out of range "
            f"({len(monitors) - 1} monitors available)"
        )
    return monitors[idx]


def resolve_monitor_size(monitor_index: int) -> tuple[int, int]:
    """Resolution (width, height) of the configured monitor region."""
    import mss

    with mss.mss() as sct:
        region = _resolve_region(sct.monitors, monitor_index)
        return int(region["width"]), int(region["height"])


def _bgra_to_bgr(sct_img: Any) -> np.ndarray:
    """Convert an MSS screenshot (BGRA bytes) to a detached BGR uint8 array.

    MSS reuses its internal buffer between grabs, so a copy is required for
    queued frames. The result is contiguous BGR, ready for the encoder.
    """
    height, width = sct_img.height, sct_img.width
    bgra = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape((height, width, 4))
    return np.ascontiguousarray(bgra[:, :, :3])


class ScreenRecorder:
    """Threaded MSS capture loop pushing timestamped frames into a queue."""

    def __init__(
        self,
        config: ScreenConfig,
        clock: SessionClock,
        frame_queue: "queue.Queue[tuple[float, np.ndarray]]",
        *,
        is_paused: Callable[[], bool] | None = None,
        error_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.frame_queue = frame_queue
        self._is_paused = is_paused or (lambda: False)
        self._error_cb = error_cb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.errors = 0

    def start(self) -> None:
        """Start the capture thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="screen-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Request the capture thread to stop and wait for it.

        Safe to call from the capture thread itself (returns immediately).
        """
        if threading.current_thread() is self._thread:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("screen capture thread did not stop within %.1fs", timeout)

    def _run(self) -> None:
        import mss

        interval = 1.0 / self.config.fps
        try:
            with mss.mss() as sct:
                region = _resolve_region(sct.monitors, self.config.monitor_index)
                next_deadline = self.clock.now() + interval
                paused = False
                while not self._stop.is_set():
                    if self._is_paused():
                        if not paused:
                            paused = True
                        time.sleep(_PAUSE_SLEEP)
                        continue
                    if paused:
                        paused = False
                        next_deadline = self.clock.now() + interval
                    try:
                        sct_img = sct.grab(region)
                    except Exception as exc:  # noqa: BLE001 - capture must survive errors
                        self.errors += 1
                        logger.error("screen grab failed (%d): %s", self.errors, exc)
                        if self.errors >= _MAX_CONSECUTIVE_ERRORS:
                            self._notify_error(f"screen capture failed repeatedly: {exc}")
                            return
                        time.sleep(_ERROR_SLEEP)
                        continue
                    t = self.clock.now()
                    frame = _bgra_to_bgr(sct_img)
                    try:
                        self.frame_queue.put_nowait((t, frame))
                        self.frames_captured += 1
                    except queue.Full:
                        self.frames_dropped += 1
                    delay = next_deadline - self.clock.now()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        logger.warning("screen capture behind schedule by %.1f ms", -delay * 1e3)
                    next_deadline += interval
                    if next_deadline < self.clock.now() - interval * 4:
                        next_deadline = self.clock.now() + interval
        except Exception:  # noqa: BLE001
            logger.exception("screen capture thread crashed")
            self._notify_error("screen capture thread crashed")

    def _notify_error(self, message: str) -> None:
        if self._error_cb is not None:
            try:
                self._error_cb(message)
            except Exception:  # noqa: BLE001
                logger.exception("screen error callback failed")
