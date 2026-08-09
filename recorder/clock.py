"""Monotonic clock shared by every component of a recording session.

All timestamps in the project derive from ``time.perf_counter_ns()`` so that
screen frames, input events, annotations and lifecycle events live on a
single timeline. Never use wall-clock time to synchronize anything.

A :class:`SessionClock` is created when a recording starts; ``t=0`` is the
session start and every captured timestamp is stored as seconds since then.
"""

from __future__ import annotations

import time

_NS_PER_SECOND = 1e9


class SessionClock:
    """Monotonic clock anchored to the start of a recording session."""

    __slots__ = ("_start_ns",)

    def __init__(self) -> None:
        self._start_ns = time.perf_counter_ns()

    @property
    def start_ns(self) -> int:
        """Session start on the monotonic timeline, in nanoseconds."""
        return self._start_ns

    def now_ns(self) -> int:
        """Current monotonic time in nanoseconds."""
        return time.perf_counter_ns()

    def now(self) -> float:
        """Seconds since session start (``t=0`` at clock creation)."""
        return (self.now_ns() - self._start_ns) / _NS_PER_SECOND
