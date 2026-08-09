"""Global mouse input capture via pynput.

Event schema (one JSON object per line in ``events.jsonl``)::

    {"t": 5.10222, "device": "mouse", "event": "move", "x": 731, "y": 412, "dx": 12, "dy": -3}
    {"t": 5.81222, "device": "mouse", "event": "button_down", "button": "left", "x": 731, "y": 412}
    {"t": 6.01000, "device": "mouse", "event": "scroll", "dx": 0, "dy": -1, "x": 731, "y": 412}

``dx``/``dy`` are deltas relative to the previous position; they are omitted
on the first event after the listener starts. Position is included so the
future dataset builder can derive both absolute position and delta streams.
"""

from __future__ import annotations

import logging
from typing import Callable

from recorder.clock import SessionClock
from storage.event_writer import EventWriter

logger = logging.getLogger(__name__)


def button_name(button) -> str:
    """Canonical name for a pynput mouse button (``left``, ``right``, ...)."""
    try:
        from pynput.mouse import Button
    except ImportError:
        return str(button)
    names = {Button.left: "left", Button.right: "right", Button.middle: "middle"}
    return names.get(button, f"button_{getattr(button, 'name', button)}")


class MouseRecorder:
    """Global mouse listener writing timestamped events to a sink."""

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
        self._last_pos: tuple[int, int] | None = None

    def start(self) -> None:
        """Start the listener (non-blocking)."""
        if self._listener is not None:
            return
        from pynput.mouse import Listener

        self._listener = Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._last_pos = None
        self._listener.start()

    def stop(self) -> None:
        """Stop the listener and wait for it to unwind."""
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
            listener.join(timeout=5.0)

    def _on_move(self, x: int, y: int) -> None:
        try:
            if self._is_paused():
                return
            dx = dy = None
            if self._last_pos is not None:
                dx = x - self._last_pos[0]
                dy = y - self._last_pos[1]
            self._last_pos = (x, y)
            record: dict = {"t": self.clock.now(), "device": "mouse", "event": "move", "x": x, "y": y}
            if dx is not None:
                record["dx"] = dx
                record["dy"] = dy
            self.sink.write(record)
        except Exception:  # noqa: BLE001 - never let capture die
            logger.exception("mouse move capture failed")

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        try:
            if self._is_paused():
                return
            self.sink.write(
                {
                    "t": self.clock.now(),
                    "device": "mouse",
                    "event": "button_down" if pressed else "button_up",
                    "button": button_name(button),
                    "x": x,
                    "y": y,
                }
            )
        except Exception:  # noqa: BLE001
            logger.exception("mouse click capture failed")

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        try:
            if self._is_paused():
                return
            self.sink.write(
                {
                    "t": self.clock.now(),
                    "device": "mouse",
                    "event": "scroll",
                    "dx": dx,
                    "dy": dy,
                    "x": x,
                    "y": y,
                }
            )
        except Exception:  # noqa: BLE001
            logger.exception("mouse scroll capture failed")
