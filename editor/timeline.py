"""Clip-based timeline model with edit operations and undo/redo.

A :class:`Timeline` is an ordered list of non-overlapping clips. Each clip
references a source range ``[source_start, source_end)`` of the original
recording and has a position ``start`` on the edited timeline. Clips are
contiguous (no gaps), so the edited duration is simply the end of the last
clip.

Operations work in *edited* coordinates and are always snapped to frame
boundaries, keeping video and events in exact sync. Copy/paste duplicates
source ranges (and therefore the events inside them), which is exactly what
duplicating footage implies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Clip:
    """A reference to a source range placed on the edited timeline."""

    source_start: float
    source_end: float
    start: float = 0.0

    @property
    def length(self) -> float:
        return self.source_end - self.source_start

    def source_time(self, edited_t: float) -> float:
        """Map an edited-timeline time to source time (within this clip)."""
        return self.source_start + (edited_t - self.start)

    def edited_time(self, source_t: float) -> float:
        """Map a source time to edited-timeline time (within this clip)."""
        return self.start + (source_t - self.source_start)


class Timeline:
    """Ordered, contiguous set of clips referencing the source recording."""

    def __init__(self, source_duration: float, frame_times: np.ndarray | None = None) -> None:
        self.source_duration = source_duration
        self._frame_times = (
            np.asarray(frame_times, dtype=np.float64) if frame_times is not None else np.array([])
        )
        self.clips: list[Clip] = []

    # ------------------------------------------------------------ basics

    @property
    def duration(self) -> float:
        if not self.clips:
            return 0.0
        last = self.clips[-1]
        return last.start + last.length

    def snap(self, t: float) -> float:
        """Snap a time to the nearest frame boundary (clamped to the source)."""
        if self._frame_times.size:
            idx = int(np.argmin(np.abs(self._frame_times - t)))
            return float(self._frame_times[idx])
        return float(min(max(t, 0.0), self.source_duration))

    def snapshot(self) -> list[list[float]]:
        """Deep copy of the current clip list as source ranges."""
        return [[c.source_start, c.source_end] for c in self.clips]

    def load_ranges(self, ranges: list[list[float]] | None) -> None:
        """Replace the timeline with the given source ranges (in order)."""
        self._rebuild(ranges if ranges is not None else [])

    def _rebuild(self, ranges: list[list[float]]) -> None:
        cleaned = []
        for start, end in ranges:
            start, end = self.snap(start), self.snap(end)
            if end - start > 1e-9:
                cleaned.append((start, end))
        cursor = 0.0
        rebuilt = []
        for a, b in cleaned:
            rebuilt.append(Clip(a, b, cursor))
            cursor += b - a
        self.clips = rebuilt

    # ------------------------------------------------------------ regions

    def _region_parts(self, in_t: float, out_t: float) -> tuple[list, list]:
        """Split the timeline into (kept ranges, removed ranges) for [in_t, out_t)."""
        kept: list[list[float]] = []
        removed: list[list[float]] = []
        for clip in self.clips:
            clip_end = clip.start + clip.length
            if clip_end <= in_t or clip.start >= out_t:
                kept.append([clip.source_start, clip.source_end])
                continue
            lo = max(clip.start, in_t)
            hi = min(clip_end, out_t)
            removed.append([clip.source_time(lo), clip.source_time(hi)])
            if clip.start < in_t:
                kept.append([clip.source_start, clip.source_time(in_t)])
            if clip_end > out_t:
                kept.append([clip.source_time(out_t), clip.source_end])
        return kept, removed

    def remove_region(self, in_t: float, out_t: float) -> None:
        """Delete the edited-time region [in_t, out_t)."""
        if out_t - in_t <= 1e-9:
            return
        kept, _ = self._region_parts(in_t, out_t)
        self._rebuild(kept)

    def cut_region(self, in_t: float, out_t: float) -> "Timeline":
        """Remove [in_t, out_t) and return the removed region as a clipboard."""
        if out_t - in_t <= 1e-9:
            return Timeline(self.source_duration, self._frame_times)
        kept, removed = self._region_parts(in_t, out_t)
        clipboard = Timeline(self.source_duration, self._frame_times)
        clipboard._rebuild(removed)
        self._rebuild(kept)
        return clipboard

    def copy_region(self, in_t: float, out_t: float) -> "Timeline":
        """Return a clipboard with the source ranges inside [in_t, out_t)."""
        _, removed = self._region_parts(in_t, out_t)
        clipboard = Timeline(self.source_duration, self._frame_times)
        clipboard._rebuild(removed)
        return clipboard

    def trim_to_region(self, in_t: float, out_t: float) -> None:
        """Keep only the edited-time region [in_t, out_t)."""
        if out_t - in_t <= 1e-9:
            self.clips = []
            return
        inside: list[list[float]] = []
        for clip in self.clips:
            clip_end = clip.start + clip.length
            lo = max(clip.start, in_t)
            hi = min(clip_end, out_t)
            if hi - lo > 1e-9:
                inside.append([clip.source_time(lo), clip.source_time(hi)])
        self._rebuild(inside)

    def paste(self, t: float, clipboard: "Timeline") -> None:
        """Insert the clipboard's clips at edited time ``t``, shifting the rest."""
        if not clipboard.clips:
            return
        t = self.snap(t)
        before: list[list[float]] = []
        after: list[list[float]] = []
        for clip in self.clips:
            (before if clip.start < t else after).append([clip.source_start, clip.source_end])
        inserted = [[c.source_start, c.source_end] for c in clipboard.clips]
        self._rebuild(before + inserted + after)


def remap_events(events: list[dict[str, Any]], timeline: Timeline) -> list[dict[str, Any]]:
    """Map events through the timeline, duplicating events for pasted clips.

    Events whose time falls inside a clip are transformed into the clip's
    edited position; events in removed source ranges are dropped; events
    without a ``t`` key are dropped.
    """
    out: list[dict[str, Any]] = []
    for clip in timeline.clips:
        lo, hi = clip.source_start, clip.source_end
        for event in events:
            t = event.get("t")
            if t is None:
                continue
            if lo <= t < hi:
                out.append({**event, "t": clip.edited_time(t)})
    out.sort(key=lambda e: e["t"])
    return out


class EditSession:
    """Timeline plus undo/redo history and an operation log."""

    def __init__(
        self,
        source_duration: float,
        frame_times: np.ndarray | None = None,
        initial_ranges: list[list[float]] | None = None,
    ) -> None:
        self.timeline = Timeline(source_duration, frame_times)
        self.timeline.load_ranges(initial_ranges if initial_ranges is not None else [[0.0, source_duration]])
        self._undo: list[list[list[float]]] = []
        self._redo: list[list[list[float]]] = []
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------ ops

    def cut(self, in_t: float, out_t: float) -> "Timeline | None":
        if out_t - in_t <= 1e-9:
            return None
        self._record("cut", in_t=in_t, out_t=out_t)
        return self.timeline.cut_region(in_t, out_t)

    def delete(self, in_t: float, out_t: float) -> None:
        if out_t - in_t <= 1e-9:
            return
        self._record("delete", in_t=in_t, out_t=out_t)
        self.timeline.remove_region(in_t, out_t)

    def copy(self, in_t: float, out_t: float) -> "Timeline | None":
        if out_t - in_t <= 1e-9:
            return None
        return self.timeline.copy_region(in_t, out_t)

    def paste(self, t: float, clipboard: "Timeline | None") -> None:
        if clipboard is None or not clipboard.clips:
            return
        self._record("paste", at=t, clips=clipboard.snapshot())
        self.timeline.paste(t, clipboard)

    def trim(self, in_t: float, out_t: float) -> None:
        self._record("trim", in_t=in_t, out_t=out_t)
        self.timeline.trim_to_region(in_t, out_t)

    def reset(self) -> None:
        self._record("reset")
        self.timeline.load_ranges([[0.0, self.timeline.source_duration]])

    # ------------------------------------------------------------ undo/redo

    def _record(self, op: str, **params: Any) -> None:
        self._undo.append(self.timeline.snapshot())
        self._redo.clear()
        self.history.append({"op": op, **params})
        if len(self._undo) > 200:
            self._undo.pop(0)
        if len(self.history) > 500:
            self.history.pop(0)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.timeline.snapshot())
        self.timeline.load_ranges(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.timeline.snapshot())
        self.timeline.load_ranges(self._redo.pop())
        return True
