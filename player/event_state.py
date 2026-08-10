"""Reconstruct input state at any timestamp from ``events.jsonl``.

Builds sorted change lists once at load, then answers queries for any
playback time: which keyboard keys were held, which mouse buttons were
held, and the last known mouse position. Times are the same monotonic
``t`` values used by frames and markers, so the keyboard UI stays in exact
sync with the video.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

logger = logging.getLogger(__name__)

_KEYBOARD_PREFIX = "key"
_MOUSE_PREFIX = "button"


class KeyStateTimeline:
    """Queryable held-key / held-button / mouse-position timeline."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._changes: list[tuple[float, str]] = []  # (t, "<prefix>:<code>")
        self._moves: list[tuple[float, int, int]] = []  # (t, x, y)
        self._parse(events)

    def _parse(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            t = event.get("t")
            device = event.get("device")
            kind = event.get("event")
            if t is None:
                continue
            if device == "keyboard" and kind in ("down", "up"):
                code = event.get("code")
                if code:
                    self._changes.append((t, f"{_KEYBOARD_PREFIX}:{code}"))
            elif device == "mouse" and kind in ("button_down", "button_up"):
                button = event.get("button")
                if button:
                    self._changes.append((t, f"{_MOUSE_PREFIX}:{button}"))
            elif device == "mouse" and kind == "move":
                x, y = event.get("x"), event.get("y")
                if x is not None and y is not None:
                    self._moves.append((t, int(x), int(y)))
        self._changes.sort(key=lambda item: item[0])
        self._moves.sort(key=lambda item: item[0])

    # ------------------------------------------------------------ queries

    def active_codes_at(self, t: float, prefix: str) -> set[str]:
        """Codes (keyboard or button) held at time ``t``."""
        state: set[str] = set()
        for change_t, code in self._changes:
            if change_t > t:
                break
            if code in state:
                state.discard(code)
            else:
                state.add(code)
        return {code.split(":", 1)[1] for code in state if code.startswith(f"{prefix}:")}

    def active_keys_at(self, t: float) -> set[str]:
        return self.active_codes_at(t, _KEYBOARD_PREFIX)

    def active_buttons_at(self, t: float) -> set[str]:
        return self.active_codes_at(t, _MOUSE_PREFIX)

    def mouse_at(self, t: float) -> tuple[int, int] | None:
        """Last mouse position at or before ``t`` (None if unknown)."""
        if not self._moves:
            return None
        idx = bisect.bisect_right([m[0] for m in self._moves], t) - 1
        if idx < 0:
            return None
        _, x, y = self._moves[idx]
        return x, y

    @property
    def used_codes(self) -> set[str]:
        """Keyboard codes that appear in the events (for UI layout)."""
        return {
            code.split(":", 1)[1]
            for _, code in self._changes
            if code.startswith(f"{_KEYBOARD_PREFIX}:")
        }

    @property
    def used_buttons(self) -> set[str]:
        return {
            code.split(":", 1)[1]
            for _, code in self._changes
            if code.startswith(f"{_MOUSE_PREFIX}:")
        }
