"""Recording session: coordinates screen, input and encoding components.

A session owns one monotonic clock, a raw recording directory, the JSONL
writers for events/markers/frame-times, and all capture components. It is
the only object the UI needs to talk to.

Lifecycle: ``start()`` -> (``pause()`` / ``resume()``)* -> ``stop()``.
Components run in their own threads and communicate through queues, so no
slow component can block input capture.

Public state transitions (published to observers)::

    IDLE -> STARTING -> RECORDING -> STOPPING -> IDLE
                         |  ^
                         v  |          (pause/resume)
                       PAUSED
                     (any) -> ERROR   (component failure)
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from recorder.clock import SessionClock
from recorder.config import RecorderConfig
from recorder.encoder import FFmpegEncoder
from recorder.keyboard import KeyboardRecorder
from recorder.mouse import MouseRecorder
from recorder.screen import ScreenRecorder, resolve_monitor_size
from storage.event_writer import EventWriter
from storage.recording import RawRecording

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


StateCallback = Callable[[SessionState], None]

_STARTABLE = {SessionState.IDLE, SessionState.ERROR}
_RUNNING = {SessionState.STARTING, SessionState.RECORDING, SessionState.PAUSED}
_ANNOTATABLE = {SessionState.RECORDING, SessionState.PAUSED}


def _normalize_platform(system: str) -> str:
    lowered = system.lower()
    if lowered.startswith("win"):
        return "windows"
    if lowered.startswith("darwin"):
        return "macos"
    if lowered.startswith("linux"):
        return "linux"
    return lowered


class RecordingSession:
    """Central coordinator for one recording."""

    def __init__(self, config: RecorderConfig, *, session_id: str | None = None) -> None:
        self.config = config
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._clock: SessionClock | None = None
        self._state = SessionState.IDLE
        self._error_message: str | None = None
        self._observers: list[StateCallback] = []
        self._lock = threading.RLock()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.frame_queue_size)
        self._screen: ScreenRecorder | None = None
        self._encoder: FFmpegEncoder | None = None
        self._keyboard: KeyboardRecorder | None = None
        self._mouse: MouseRecorder | None = None
        self._events: EventWriter | None = None
        self._markers: EventWriter | None = None
        self._frames: EventWriter | None = None
        self._recording: RawRecording | None = None
        self._paused = False
        self._pause_total = 0.0
        self._pause_started: float | None = None

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def clock(self) -> SessionClock | None:
        return self._clock

    @property
    def recording_dir(self) -> Path | None:
        return self._recording.directory if self._recording is not None else None

    def elapsed(self) -> float:
        """Seconds since the session started (0 if not started)."""
        return self._clock.now() if self._clock is not None else 0.0

    def register_observer(self, callback: StateCallback) -> None:
        with self._lock:
            self._observers.append(callback)

    def _set_state(self, state: SessionState) -> None:
        self._state = state
        with self._lock:
            observers = list(self._observers)
        for callback in observers:
            try:
                callback(state)
            except Exception:  # noqa: BLE001
                logger.exception("state observer failed")

    def _paused_getter(self) -> bool:
        return self._paused

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Begin recording. Raises RuntimeError if not startable."""
        with self._lock:
            if self._state not in _STARTABLE:
                raise RuntimeError(f"cannot start recording from state {self._state.value}")
            if self._state == SessionState.ERROR:
                self._frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
                self._paused = False
                self._pause_total = 0.0
                self._pause_started = None
                self._error_message = None
            self._set_state(SessionState.STARTING)
        try:
            self._start_components()
        except Exception:
            logger.exception("failed to start recording")
            self._cleanup_after_failure()
            raise
        self._set_state(SessionState.RECORDING)

    def stop(self) -> None:
        """Stop recording and finalize all files. Idempotent."""
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPING):
                return
            if self._state not in _RUNNING | {SessionState.ERROR}:
                raise RuntimeError(f"cannot stop recording from state {self._state.value}")
            was_error = self._error_message is not None
            self._set_state(SessionState.STOPPING)
        try:
            self._write_event("recording_stop")
        except Exception:  # noqa: BLE001
            logger.exception("failed to write recording_stop event")
        self._stop_components()
        self._finalize_metadata()
        self._set_state(SessionState.ERROR if was_error else SessionState.IDLE)

    def pause(self) -> None:
        """Pause capture. Only valid while recording."""
        with self._lock:
            if self._state != SessionState.RECORDING:
                return
            self._write_event("pause")
            self._paused = True
            self._pause_started = self._clock.now()
            self._set_state(SessionState.PAUSED)

    def resume(self) -> None:
        """Resume after :meth:`pause`."""
        with self._lock:
            if self._state != SessionState.PAUSED:
                return
            if self._pause_started is not None:
                self._pause_total += self._clock.now() - self._pause_started
                self._pause_started = None
            self._paused = False
            self._write_event("resume")
            self._set_state(SessionState.RECORDING)

    def add_annotation(self, label: str) -> float:
        """Record a human annotation (arbitrary label string). Returns its time."""
        label = label.strip()
        if not label:
            raise ValueError("annotation label must not be empty")
        with self._lock:
            if self._state not in _ANNOTATABLE:
                raise RuntimeError(f"cannot annotate from state {self._state.value}")
            t = self._clock.now()
            self._markers.write({"t": t, "type": "annotation", "label": label})
        return t

    # ------------------------------------------------------------ internals

    def _start_components(self) -> None:
        self._clock = SessionClock()
        width, height = resolve_monitor_size(self.config.screen.monitor_index)
        metadata = {
            "version": RawRecording.FORMAT_VERSION,
            "session_id": self.session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platform": _normalize_platform(platform.system()),
            "screen": {
                "width": width,
                "height": height,
                "fps": self.config.screen.fps,
                "monitor_index": self.config.screen.monitor_index,
            },
            "input": {"keyboard": True, "mouse": True},
            "duration": 0.0,
        }
        self._recording = RawRecording.create(self.config.output_dir, self.session_id, metadata)

        self._events = EventWriter(self._recording.events_path)
        self._markers = EventWriter(self._recording.markers_path)
        self._frames = EventWriter(self._recording.frames_path)
        self._events.start()
        self._markers.start()
        self._frames.start()

        self._screen = ScreenRecorder(
            self.config.screen,
            self._clock,
            self._frame_queue,
            is_paused=self._paused_getter,
            error_cb=self._on_component_error,
        )
        self._encoder = FFmpegEncoder(
            self.config.encoder,
            self._recording.video_path,
            self._frames,
            self._frame_queue,
            self._clock,
            self.config.screen.fps,
        )
        self._keyboard = KeyboardRecorder(self._clock, self._events, self._paused_getter)
        self._mouse = MouseRecorder(self._clock, self._events, self._paused_getter)

        self._screen.start()
        self._keyboard.start()
        self._mouse.start()
        self._encoder.start()
        if self._state != SessionState.STARTING:
            raise RuntimeError("recording aborted by component error during start")
        self._write_event("recording_start")

    def _stop_components(self) -> None:
        if self._paused:
            self._paused = False
            if self._pause_started is not None and self._clock is not None:
                self._pause_total += self._clock.now() - self._pause_started
                self._pause_started = None
        # Stop the screen first so no new frames are queued, then signal the
        # encoder to drain and finish.
        try:
            if self._screen is not None:
                self._screen.stop()
        except Exception:  # noqa: BLE001
            logger.exception("failed to stop screen cleanly")
        self._frame_queue.put(None)  # encoder sentinel
        # Best-effort: one failing component must not prevent the rest from
        # being shut down cleanly (e.g. a component that failed mid-start).
        for name, component in (
            ("encoder", self._encoder),
            ("keyboard", self._keyboard),
            ("mouse", self._mouse),
            ("events", self._events),
            ("markers", self._markers),
            ("frames", self._frames),
        ):
            if component is None:
                continue
            try:
                component.stop()
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop %s cleanly", name)

    def _finalize_metadata(self) -> None:
        if self._recording is None or self._clock is None:
            return

        def count(writer: EventWriter | None) -> int:
            return writer.written if writer is not None else 0

        stats = {
            "frames_captured": self._screen.frames_captured if self._screen else 0,
            "frames_dropped": self._screen.frames_dropped if self._screen else 0,
            "frames_encoded": self._encoder.frames_encoded if self._encoder else 0,
            "events_written": count(self._events),
            "markers_written": count(self._markers),
            "pause_duration": round(self._pause_total, 6),
            "encoder_returncode": self._encoder.returncode if self._encoder else None,
            "encoder_error": self._encoder.error if self._encoder else None,
        }
        files = {
            name: (path.stat().st_size if path.exists() else 0)
            for name, path in self._recording.files().items()
        }
        try:
            self._recording.update_metadata(duration=self._clock.now(), stats=stats, files=files)
        except Exception:  # noqa: BLE001
            logger.exception("failed to finalize metadata")

    def _cleanup_after_failure(self) -> None:
        self._error_message = "recording start failed"
        try:
            self._write_event("recording_error", message=self._error_message)
        except Exception:  # noqa: BLE001
            logger.exception("failed to write error event")
        self._stop_components()
        self._finalize_metadata()
        self._set_state(SessionState.ERROR)

    def _write_event(self, name: str, **extra) -> None:
        if self._events is None or self._clock is None:
            return
        self._events.write({"t": self._clock.now(), "device": "session", "event": name, **extra})

    def _on_component_error(self, message: str) -> None:
        logger.error("component error: %s", message)
        self._error_message = message
        try:
            self._write_event("recording_error", message=message)
        except Exception:  # noqa: BLE001
            logger.exception("failed to write error event")
        with self._lock:
            if self._state not in _RUNNING:
                return
            self._set_state(SessionState.ERROR)
        threading.Thread(target=self.stop, name="session-error-cleanup", daemon=True).start()
