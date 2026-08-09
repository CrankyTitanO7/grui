"""Global keyboard input capture via pynput.

Events are enqueued through the session's event sink on the pynput listener
thread and timestamped with the shared session clock at the moment they
arrive. The listener thread never blocks on disk I/O.

Event schema (one JSON object per line in ``events.jsonl``)::

    {"t": 4.12031, "device": "keyboard", "event": "down", "code": "KeyW"}
    {"t": 4.81244, "device": "keyboard", "event": "down", "code": "Key.space", "char": " "}
    {"t": 4.90011, "device": "keyboard", "event": "up", "code": "Key.space", "char": " "}

Canonical key codes: char keys -> ``KeyW``, special keys -> ``Key.space``,
unknown keys -> ``Key.vk_<vk>``. The schema is device-agnostic so further
input devices (controller, touch, ...) can be added without redesign.
"""

from __future__ import annotations

import logging
from typing import Callable

from recorder.clock import SessionClock
from storage.event_writer import EventWriter

logger = logging.getLogger(__name__)


def serialize_key(key) -> tuple[str, str | None]:
    """Return ``(code, char)`` for a pynput key object.

    ``char`` is ``None`` when the key has no printable representation.
    """
    try:
        from pynput.keyboard import Key, KeyCode
    except ImportError:
        return str(key), None
    if isinstance(key, KeyCode):
        if key.char:
            return f"Key{key.char.upper()}", key.char
        return f"Key.vk_{key.vk}", None
    if isinstance(key, Key):
        return f"Key.{key.name}", None
    return str(key), None


class KeyboardRecorder:
    """Global keyboard listener writing timestamped events to a sink."""

    def __init__(
        self,
        clock: SessionClock,
        sink: EventWriter,
        is_paused: Callable[[], bool] | None = None,
    ) -> None:
        self.clock = clock
        self.sink = sink
        self._is_paused = is_paused or (lambda: False)
        self._listener = None

    def start(self) -> None:
        """Start the listener (non-blocking)."""
        if self._listener is not None:
            return
        from pynput.keyboard import Listener

        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        """Stop the listener and wait for it to unwind."""
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
            listener.join(timeout=5.0)

    def _on_press(self, key) -> None:
        self._emit(key, "down")

    def _on_release(self, key) -> None:
        self._emit(key, "up")

    def _emit(self, key, event: str) -> None:
        try:
            if self._is_paused():
                return
            code, char = serialize_key(key)
            record: dict = {"t": self.clock.now(), "device": "keyboard", "event": event, "code": code}
            if char is not None:
                record["char"] = char
            self.sink.write(record)
        except Exception:  # noqa: BLE001 - never let capture die
            logger.exception("keyboard event capture failed")
