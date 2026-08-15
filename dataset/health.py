"""Dataset health: statistics, action distribution and quality reports.

Analyzes raw recordings (and their derived perception/annotation layers)
without modifying anything. Reports are human-readable summaries:

* dataset statistics (demonstrations, duration, frames, annotations, …)
* action distribution (keys, mouse buttons, movement, chords, idle)
* quality issues (missing frames, timestamp gaps, idle periods, imbalance,
  short/long demonstrations, perception failures, inconsistent annotations)

Everything here is pure analysis over finished recordings — recording,
playback and editing never call into this module.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from player.event_state import KeyStateTimeline
from storage.recording import RecordingData, list_recordings, load_recording

logger = logging.getLogger(__name__)

# thresholds (seconds); both directions configurable
_LONG_DEMO_S = 3600.0
_SHORT_DEMO_S = 5.0
_IDLE_WARN_S = 30.0
_FRAME_GAP_TOLERANCE = 3.0  # x median frame interval
_IMBALANCE_RATIO = 0.25  # relative share of the most common action


# --------------------------------------------------------------------------- stats


@dataclass(frozen=True)
class DatasetStatistics:
    demonstrations: int = 0
    total_duration: float = 0.0
    total_frames: int = 0
    average_duration: float = 0.0
    annotations: int = 0  # markers (human events during recording)
    perception_predictions: int = 0  # detections across perception results


@dataclass(frozen=True)
class RecordingStats:
    directory: str
    duration: float
    frames: int
    fps: float
    markers: int
    perception_predictions: int


def recording_statistics(recording: RecordingData) -> RecordingStats:
    """Per-demonstration statistics (raw + derived layers)."""
    perception_predictions = 0
    cached = recording.directory / "perception" / "results.jsonl"
    if cached.exists():
        for line in cached.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            perception_predictions += len(row.get("detections") or [])
    return RecordingStats(
        directory=recording.directory.name,
        duration=recording.duration,
        frames=len(recording.frame_times),
        fps=recording.fps,
        markers=len(recording.markers),
        perception_predictions=perception_predictions,
    )


def dataset_statistics(recordings: list[RecordingData]) -> DatasetStatistics:
    """Aggregate statistics over many demonstrations."""
    stats = [recording_statistics(r) for r in recordings]
    return DatasetStatistics(
        demonstrations=len(stats),
        total_duration=sum(s.duration for s in stats),
        total_frames=sum(s.frames for s in stats),
        average_duration=(sum(s.duration for s in stats) / len(stats)) if stats else 0.0,
        annotations=sum(s.markers for s in stats),
        perception_predictions=sum(s.perception_predictions for s in stats),
    )


# --------------------------------------------------------------------------- action distribution


@dataclass
class ActionDistribution:
    """Fraction-of-time each action was active (sampled at frame times)."""

    samples: int = 0
    keys: Counter[str] = field(default_factory=Counter)
    buttons: Counter[str] = field(default_factory=Counter)
    chords: Counter[tuple[str, ...]] = field(default_factory=Counter)
    mouse_movement_frames: int = 0
    idle_frames: int = 0

    def key_fraction(self, code: str) -> float:
        return self.keys[code] / self.samples if self.samples else 0.0

    def button_fraction(self, code: str) -> float:
        return self.buttons[code] / self.samples if self.samples else 0.0

    def chord_fraction(self, chord: tuple[str, ...]) -> float:
        return self.chords[chord] / self.samples if self.samples else 0.0

    @property
    def mouse_movement_fraction(self) -> float:
        return self.mouse_movement_frames / self.samples if self.samples else 0.0

    @property
    def idle_fraction(self) -> float:
        return self.idle_frames / self.samples if self.samples else 0.0

    def imbalance_warnings(self, *, ignore_idle: bool = False) -> list[str]:
        """Warn when one action dominates or another is near-absent."""
        warnings: list[str] = []
        for label, fraction in self._shares(ignore_idle=ignore_idle):
            if not label:
                continue
            if fraction < 0.02:
                warnings.append(f"{label} appears very rarely ({fraction * 100:.1f}%)")
        return warnings

    def _shares(self, *, ignore_idle: bool) -> list[tuple[str, float]]:
        shares = [
            (code, self.key_fraction(code)) for code in self.keys
        ] + [
            (f"mouse:{code}", self.button_fraction(code)) for code in self.buttons
        ]
        if self.mouse_movement_fraction > 0.02:
            shares.append(("MOUSE_MOVE", self.mouse_movement_fraction))
        if not ignore_idle:
            shares.append(("None", self.idle_fraction))
        return sorted(shares, key=lambda item: -item[1])


def action_distribution(recording: RecordingData) -> ActionDistribution:
    """Sample the input state at every frame time => action distribution.

    Idle means no key held, no mouse button held and no mouse movement since
    the previous sampled frame.
    """
    keys = KeyStateTimeline(recording.events)
    times = recording.frame_times
    dist = ActionDistribution(samples=len(times))
    prev_pos = None
    for t in times:
        active_keys = sorted(keys.active_keys_at(t))
        active_buttons = sorted(keys.active_buttons_at(t))
        pos = keys.mouse_at(t)
        moved = prev_pos is not None and pos is not None and (pos != prev_pos)
        if moved:
            dist.mouse_movement_frames += 1
        prev_pos = pos
        for code in active_keys:
            dist.keys[code] += 1
        for code in active_buttons:
            dist.buttons[code] += 1
        if len(active_keys) > 1:
            dist.chords[tuple(active_keys)] += 1
        if not active_keys and not active_buttons and not moved:
            dist.idle_frames += 1
    return dist


# --------------------------------------------------------------------------- quality


@dataclass(frozen=True)
class QualityIssue:
    """One detected dataset-health problem."""

    category: str  # missing_frames | timestamp_gap | no_input | idle | duplicate_frames | imbalance | interaction | short | long | perception | annotation
    severity: str  # error | warning | info
    message: str

    def render(self) -> str:
        mark = {"error": "✗", "warning": "⚠", "info": "·"}.get(self.severity, "·")
        return f"{mark} [{self.category}] {self.message}"


def _frame_gaps(recording: RecordingData) -> list[tuple[float, float]]:
    """(gap_seconds, previous_time) for gaps above the tolerance."""
    if len(recording.frame_times) < 2:
        return []
    times = recording.frame_times
    intervals = np.diff(times)
    median = float(np.median(intervals)) if intervals.size else 0.0
    tolerance = max(_FRAME_GAP_TOLERANCE * median, 0.1)
    gaps = []
    for i in range(1, len(times)):
        gap = float(times[i] - times[i - 1])
        if gap > tolerance:
            gaps.append((gap, float(times[i - 1])))
    return gaps


def _long_idle_periods(recording: RecordingData) -> list[tuple[float, float]]:
    """(start, duration) of input-inactive stretches longer than the threshold."""
    keys = KeyStateTimeline(recording.events)
    if len(recording.frame_times) < 2:
        return []
    times = recording.frame_times
    idle_start: float | None = None
    periods: list[tuple[float, float]] = []
    prev_pos = None
    for t in times:
        active = keys.active_keys_at(t) or keys.active_buttons_at(t)
        pos = keys.mouse_at(t)
        moved = prev_pos is not None and pos is not None and (pos != prev_pos)
        prev_pos = pos
        if not active and not moved:
            if idle_start is None:
                idle_start = t
        elif idle_start is not None:
            duration = t - idle_start
            if duration >= _IDLE_WARN_S:
                periods.append((idle_start, duration))
            idle_start = None
    if idle_start is not None and times[-1] - idle_start >= _IDLE_WARN_S:
        periods.append((idle_start, float(times[-1] - idle_start)))
    return periods


def _duplicate_frame_suspects(recording: RecordingData, *, stride: int = 30, check: bool = True) -> int:
    """Cheap duplicate-frame check: downscaled hash of sampled frames.

    Only decoded when ``check`` is True (it reads the video — keep it
    optional for large datasets).
    """
    if not check or recording.frame_times.size < 2:
        return 0
    import cv2

    wanted = list(range(0, recording.frame_times.size, stride))
    wanted_set = set(wanted)
    cap = cv2.VideoCapture(str(recording.video_path))
    duplicates = 0
    prev_hash: tuple | None = None
    index = 0
    try:
        while wanted_set:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
                mean = float(small.mean())
                std = float(small.std())
                h = (round(mean, 2), round(std, 2), int(small[8, 8].mean()), int(small[24, 24].mean()))
                if prev_hash == h:
                    duplicates += 1
                prev_hash = h
                wanted_set.discard(index)
            index += 1
    finally:
        cap.release()
    return duplicates


def create_quality_issues(
    recording: RecordingData,
    *,
    check_duplicates: bool = False,
) -> list[QualityIssue]:
    """Analyze one recording and return all detected issues."""
    issues: list[QualityIssue] = []

    # --- corrupted / missing pieces
    if not recording.video_path.exists():
        issues.append(QualityIssue("corrupted", "error", "video.mp4 is missing"))
    if recording.frame_times.size == 0:
        issues.append(QualityIssue("corrupted", "error", "frames.jsonl is empty/missing"))
        return issues
    if not recording.events:
        issues.append(QualityIssue("no_input", "warning", "no input events at all — nothing to imitate"))

    # --- timestamp discontinuities
    for gap, at in _frame_gaps(recording):
        issues.append(
            QualityIssue("timestamp_gap", "warning", f"timestamp discontinuity of {gap:.2f}s at t={at:.2f}")
        )

    # --- idle periods
    for start, duration in _long_idle_periods(recording):
        issues.append(
            QualityIssue("idle", "warning", f"{duration:.0f}s without any input starting at t={start:.1f}")
        )
    dist = action_distribution(recording)
    if dist.idle_fraction > 0.6:
        issues.append(
            QualityIssue("interaction", "warning",
                         f"very little interaction: {dist.idle_fraction * 100:.0f}% of frames are idle")
        )

    # --- duplicates
    if check_duplicates:
        duplicates = _duplicate_frame_suspects(recording)
        if duplicates:
            issues.append(
                QualityIssue("duplicate_frames", "warning",
                             f"~{duplicates} near-duplicate sampled frames (stride 30)")
            )

    # --- action imbalance
    for warning in dist.imbalance_warnings():
        issues.append(QualityIssue("imbalance", "warning", warning))

    # --- durations
    if recording.duration < _SHORT_DEMO_S:
        issues.append(QualityIssue("short", "warning", f"very short demonstration ({recording.duration:.1f}s)"))
    if recording.duration > _LONG_DEMO_S:
        issues.append(
            QualityIssue("long", "info", f"long demonstration ({recording.duration / 3600:.1f}h) — consider episode segmentation")
        )

    # --- perception failures (derived layer)
    cached = recording.directory / "perception" / "manifest.json"
    if cached.exists():
        try:
            manifest = json.loads(cached.read_text(encoding="utf-8"))
            if manifest.get("count", 0) == 0:
                issues.append(QualityIssue("perception", "info", "perception ran but produced zero records"))
        except ValueError:
            issues.append(QualityIssue("perception", "error", "perception manifest is unreadable"))
    results = recording.directory / "perception" / "results.jsonl"
    if results.exists():
        detections_total = 0
        for line in results.read_text(encoding="utf-8").splitlines():
            try:
                detections_total += len(json.loads(line).get("detections") or [])
            except ValueError:
                pass
        if detections_total == 0:
            issues.append(QualityIssue("perception", "warning", "perception found no objects in any frame"))

    # --- inconsistent annotations (markers + derived annotation layer)
    for marker in recording.markers:
        if not marker.get("label"):
            issues.append(QualityIssue("annotation", "warning", "a marker has no label"))
    load_store = True
    try:
        from annotation.store import load_annotations as _load_annotations

        store = _load_annotations(recording.directory)
        if store and len(store) > 0:
            weird = [a for a in store if a.bbox.x2 <= a.bbox.x1 or a.bbox.y2 <= a.bbox.y1]
            if weird:
                issues.append(
                    QualityIssue("annotation", "warning", f"{len(weird)} annotations have inverted/zero-size boxes")
                )
    except Exception:  # noqa: BLE001 - annotation layer is optional
        load_store = False
        del load_store
    return issues


# --------------------------------------------------------------------------- reports


def render_dataset_statistics(stats: DatasetStatistics) -> str:
    lines = [
        "Dataset Statistics",
        "==================",
        f"Demonstrations:            {stats.demonstrations}",
        f"Total duration:            {stats.total_duration / 60:.1f} min",
        f"Total frames:              {stats.total_frames}",
        f"Average demonstration:     {stats.average_duration:.1f}s",
        f"Annotations (markers):     {stats.annotations}",
        f"Perception predictions:    {stats.perception_predictions}",
    ]
    if stats.total_duration:
        lines.append(f"Effective frame rate:     {stats.total_frames / stats.total_duration:.1f} fps")
    return "\n".join(lines)


def render_action_distribution(dist: ActionDistribution) -> str:
    lines = ["Action Distribution", "=================="]
    if not dist.samples:
        lines.append("(no frames)")
        return "\n".join(lines)
    shares = dist._shares(ignore_idle=False)
    if not shares:
        lines.append("(no actions recorded)")
        return "\n".join(lines)
    for label, fraction in shares:
        bar = "█" * max(1, int(fraction * 30))
        lines.append(f"{label:<16} {bar} {fraction * 100:5.1f}%")
    if dist.chords:
        lines.append("\nCombinations/chords:")
        for chord, count in dist.chords.most_common(10):
            frac = dist.chord_fraction(chord)
            if frac > 0.005:
                lines.append(f"  {' + '.join(chord):<28} {count:>5} samples ({frac * 100:.1f}%)")
    lines.append(f"\nSamples (frames): {dist.samples}")
    return "\n".join(lines)


def render_quality_report(recording: RecordingData, issues: list[QualityIssue]) -> str:
    lines = [
        f"Quality Report — {recording.directory.name}",
        "================",
        f"duration {recording.duration:.1f}s · {len(recording.frame_times)} frames "
        f"· fps {recording.fps:.0f} · {len(recording.events)} events · {len(recording.markers)} markers",
    ]
    by_severity = {"error": [], "warning": [], "info": []}
    for issue in issues:
        by_severity[issue.severity].append(issue)
    if not issues:
        lines.append("\nNo issues detected.")
        return "\n".join(lines)
    for severity in ("error", "warning", "info"):
        if by_severity[severity]:
            lines.append(f"\n{severity.upper()} ({len(by_severity[severity])}):")
            for issue in by_severity[severity]:
                lines.append("  " + issue.render())
    return "\n".join(lines)


def analyze_recordings(root: Path | str) -> tuple[DatasetStatistics, list[RecordingStats]]:
    """Statistics over every recording under ``root`` (or ``root`` itself).

    ``root`` may be a recordings root directory or a single recording
    directory — both are accepted, mirroring the full dataset layer.
    """
    root = Path(root)
    if (root / "metadata.json").exists():
        recordings = [load_recording(root)]
    else:
        recordings = [load_recording(p) for p in list_recordings(root)]
    return dataset_statistics(recordings), [recording_statistics(r) for r in recordings]