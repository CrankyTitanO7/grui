"""Episode segmentation: divide long recordings into logical episodes.

Segmentation is *derived metadata*: suggestions are computed from signals in
the raw recording (input inactivity, user markers, visual changes) and
persisted as ``<recording>/episodes.jsonl``. The raw recording itself is
never modified, and the user may overwrite the suggestions with manual
boundaries (``grui dataset episodes set``).

Each episode is a half-open ``[start, end)`` range in session time, the same
domain as ``frames.jsonl``/``events.jsonl``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from player.event_state import KeyStateTimeline
from storage.recording import RecordingData, load_recording

logger = logging.getLogger(__name__)

# default inactivity gap (seconds) that suggests an episode boundary
_DEFAULT_MIN_INACTIVITY_S = 5.0
# marker label prefix that always creates an episode boundary
_MARKER_BOUNDARY_PREFIX = "episode:"


@dataclass(frozen=True)
class Episode:
    start: float
    end: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            reason=str(data.get("reason") or ""),
        )


# --------------------------------------------------------------------- signals


def input_gap_boundaries(
    recording: RecordingData, *, min_inactivity: float = _DEFAULT_MIN_INACTIVITY_S
) -> list[float]:
    """Timestamps where the user was inactive >= ``min_inactivity`` seconds.

    A boundary is placed in the middle of the inactive gap. Uses the input
    events, not the frame times, so a static scene with no input counts as
    inactivity.
    """
    events = sorted(
        (float(e["t"]) for e in recording.events if e.get("t") is not None)
    )
    if not events:
        return []
    boundaries: list[float] = []
    for previous, current in zip(events, events[1:]):
        if current - previous >= min_inactivity:
            boundaries.append((previous + current) / 2.0)
    return boundaries


def marker_boundaries(recording: RecordingData) -> list[float]:
    """User markers delimit episodes.

    Every marker labelled ``episode:<name>`` creates a boundary at its time;
    with ``use_all_markers=True`` all markers do.
    """
    boundaries = []
    for marker in recording.markers:
        label = str(marker.get("label") or "")
        t = marker.get("t")
        if t is None:
            continue
        if label.startswith(_MARKER_BOUNDARY_PREFIX):
            boundaries.append(float(t))
    return boundaries


def visual_change_boundaries(
    recording: RecordingData,
    *,
    max_episode_s: float = 120.0,
    threshold: float = 18.0,
    stride: int = 30,
) -> list[float]:
    """Boundaries where consecutive sampled frames differ dramatically.

    Uses a cheap mean-frame-difference on downscaled frames (optionally
    decoded if ``max_episode_s`` limits the scan region). Never re-records
    anything — the video is only read.
    """
    if len(recording.frame_times) < 2:
        return []
    import cv2

    wanted = list(range(0, len(recording.frame_times), stride))
    wanted_set = set(wanted)
    boundaries: list[float] = []
    prev_frame: Any = None
    cap = cv2.VideoCapture(str(recording.video_path))
    index = 0
    try:
        while wanted_set:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
                if prev_frame is not None:
                    diff = float(cv2.absdiff(small, prev_frame).mean())
                    if diff > threshold:
                        boundaries.append(float(recording.frame_time(index)))
                prev_frame = small
                wanted_set.discard(index)
            index += 1
    finally:
        cap.release()
    return boundaries


# ------------------------------------------------------------------- assembly


def suggest_episodes(
    recording: RecordingData,
    *,
    min_inactivity: float = _DEFAULT_MIN_INACTIVITY_S,
    use_markers: bool = True,
    use_visual: bool = False,
    max_episode_s: float | None = None,
) -> list[Episode]:
    """Suggest episodes from inactivity, markers and (optionally) visuals.

    Boundaries from all enabled signals are merged, deduplicated and turned
    into contiguous ``[start, end)`` episodes. ``max_episode_s`` optionally
    splits any episode longer than that at the nearest boundary (or evenly).
    """
    boundaries: dict[float, list[str]] = {}
    for t in input_gap_boundaries(recording, min_inactivity=min_inactivity):
        boundaries.setdefault(t, []).append("inactivity")
    if use_markers:
        for t in marker_boundaries(recording):
            boundaries.setdefault(t, []).append("marker")
    if use_visual:
        for t in visual_change_boundaries(recording):
            boundaries.setdefault(t, []).append("visual")

    if not boundaries:
        duration = recording.duration
        if max_episode_s and duration > max_episode_s:
            episodes = []
            start = 0.0
            while start < duration - 1e-6:
                end = min(start + max_episode_s, duration)
                episodes.append(Episode(start, end, reason="length"))
                start = end
            return episodes
        return [Episode(0.0, duration, reason="full")]

    points = sorted(boundaries)
    if points and max_episode_s and recording.duration > max_episode_s:
        merged: list[float] = []
        cursor = 0.0
        for t in points:
            if t - cursor >= max_episode_s:
                merged.append(t)
                cursor = t
            elif points[-1] - cursor <= max_episode_s:
                pass
        points = merged or points

    episodes: list[Episode] = []
    start = 0.0
    for t in points:
        if t <= start + 1e-6 or t >= recording.duration:
            continue
        reason = " / ".join(boundaries[t])
        episodes.append(Episode(start, t, reason=reason))
        start = t
    if start < recording.duration - 1e-6 or not episodes:
        episodes.append(Episode(start, recording.duration, reason="full"))
    return episodes


# ----------------------------------------------------------------- persistence


def read_episodes(recording: RecordingData | Path | str) -> list[Episode]:
    """Load user-set episodes from ``<recording>/episodes.jsonl`` (or [])."""
    if isinstance(recording, RecordingData):
        recording_dir = recording.directory
    else:
        recording_dir = Path(recording)
    path = recording_dir / "episodes.jsonl"
    episodes: list[Episode] = []
    if not path.exists():
        return episodes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                episodes.append(Episode.from_dict(json.loads(line)))
            except (ValueError, KeyError) as exc:
                logger.warning("skipping bad episode line in %s: %s", path, exc)
    return episodes


def write_episodes(recording_dir: Path | str, episodes: list[Episode]) -> Path:
    """Persist episodes as derived metadata next to the recording."""
    path = Path(recording_dir) / "episodes.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for episode in sorted(episodes, key=lambda e: e.start):
            fh.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
    return path


def merge_episodes(a: list[Episode], b: list[Episode]) -> list[Episode]:
    """Merge two episode lists by concatenating ranges (for manual adjustment)."""
    all_episodes = list(a) + list(b)
    all_episodes.sort(key=lambda e: e.start)
    merged: list[Episode] = []
    for episode in all_episodes:
        if merged and episode.start <= merged[-1].end + 1e-6:
            previous = merged[-1]
            merged[-1] = Episode(
                previous.start, max(previous.end, episode.end),
                reason=" / ".join(r for r in (previous.reason, episode.reason) if r),
            )
        else:
            merged.append(episode)
    return merged