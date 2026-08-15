"""Automatic event discovery over perception/annotation tracks.

Turns raw sightings into higher-level events, e.g.::

    boss appears
            |
    projectile appears
            |
    projectile disappears

The framework is intentionally simple and extensible: a rule consumes a
flat series of :class:`TrackSighting` records and returns a list of
:class:`Event` spans. GRUI does not assign persistent object identities,
so sightings are grouped per label; consecutive sightings closer than
``gap_s`` form one cluster (a "presence episode").

Built-in rules:

* ``appearance`` — one event at the start of every presence cluster;
* ``disappearance`` — one event per cluster, spanning the whole presence
  (start_t/start_frame = first sighting, end_t/end_frame = last).

Sightings come from derived data only — either the annotation store
(``<recording>/annotations/annotations.jsonl``, preferred: it includes
human-verified truth) or the perception results
(``<recording>/perception/results.jsonl``). Events are stored as derived
data in ``<recording>/perception/events.jsonl``. Raw recordings are never
modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from annotation.store import AnnotationStore, load_annotations
from perception.runner import CachedAnalysis
from storage.recording import load_recording

logger = logging.getLogger(__name__)

_EVENTS_FILENAME = "events.jsonl"


@dataclass(frozen=True)
class TrackSighting:
    """One label sighting at a frame (no object identity — grouped by label)."""

    label: str
    t: float  # seconds since session start (frames.jsonl clock)
    frame_index: int
    source: str  # "annotation" | "perception"


@dataclass(frozen=True)
class Event:
    """A high-level event with a time span on the recording clock."""

    kind: str  # rule name, e.g. "appearance" | "disappearance"
    label: str
    start_t: float
    end_t: float
    start_frame: int
    end_frame: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            kind=str(data["kind"]),
            label=str(data["label"]),
            start_t=float(data["start_t"]),
            end_t=float(data["end_t"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            detail=dict(data.get("detail") or {}),
        )


class EventRule(Protocol):
    """A plug-in: turn a flat sighting series into events."""

    name: str

    def detect(self, sightings: Sequence[TrackSighting]) -> list[Event]:
        ...


@dataclass(frozen=True)
class _PresenceCluster:
    """Consecutive sightings of one label, separated by < gap_s."""

    label: str
    sightings: list[TrackSighting]

    @property
    def first(self) -> TrackSighting:
        return self.sightings[0]

    @property
    def last(self) -> TrackSighting:
        return self.sightings[-1]


class SightingRule:
    """Base for rules that operate on per-label presence clusters."""

    name = "base"

    def __init__(self, gap_s: float = 2.0, min_sightings: int = 1) -> None:
        if gap_s <= 0:
            raise ValueError(f"gap_s must be > 0 (got {gap_s})")
        self.gap_s = gap_s
        self.min_sightings = min_sightings

    def detect(self, sightings: Sequence[TrackSighting]) -> list[Event]:
        events = []
        for cluster in cluster_presence(sightings, self.gap_s, self.min_sightings):
            events.append(self._event_for(cluster))
        return events

    def _event_for(self, cluster: _PresenceCluster) -> Event:  # pragma: no cover - abstract
        raise NotImplementedError


class AppearanceRule(SightingRule):
    """An event at the start of every presence cluster (a label showed up)."""

    name = "appearance"

    def _event_for(self, cluster: _PresenceCluster) -> Event:
        first = cluster.first
        return Event(
            kind=self.name,
            label=cluster.label,
            start_t=first.t,
            end_t=first.t,
            start_frame=first.frame_index,
            end_frame=first.frame_index,
            detail={"sightings": len(cluster.sightings)},
        )


class DisappearanceRule(SightingRule):
    """An event spanning each presence cluster (the label was around and went away)."""

    name = "disappearance"

    def _event_for(self, cluster: _PresenceCluster) -> Event:
        first, last = cluster.first, cluster.last
        return Event(
            kind=self.name,
            label=cluster.label,
            start_t=first.t,
            end_t=last.t,
            start_frame=first.frame_index,
            end_frame=last.frame_index,
            detail={
                "sightings": len(cluster.sightings),
                "duration_s": round(last.t - first.t, 6),
            },
        )


_RULE_REGISTRY: dict[str, type[SightingRule]] = {
    AppearanceRule.name: AppearanceRule,
    DisappearanceRule.name: DisappearanceRule,
}


def available_rules() -> list[str]:
    return sorted(_RULE_REGISTRY)


def make_rules(names: Sequence[str], gap_s: float) -> list[SightingRule]:
    """Instantiate the named rules (as given by ``--rule``; default: all)."""
    if not names:
        names = available_rules()
    rules = []
    for name in names:
        try:
            rules.append(_RULE_REGISTRY[name](gap_s=gap_s))
        except KeyError as exc:
            raise ValueError(
                f"unknown event rule {name!r} (available: {', '.join(available_rules())})"
            ) from exc
    return rules


def cluster_presence(
    sightings: Sequence[TrackSighting],
    gap_s: float,
    min_sightings: int = 1,
) -> list[_PresenceCluster]:
    """Group per-label sightings into presence clusters at >= gap_s breaks."""
    by_label: dict[str, list[TrackSighting]] = {}
    for sighting in sightings:
        by_label.setdefault(sighting.label, []).append(sighting)
    clusters: list[_PresenceCluster] = []
    for label, series in by_label.items():
        series = sorted(series, key=lambda s: (s.t, s.frame_index))
        current: list[TrackSighting] = []
        for sighting in series:
            if current and sighting.t - current[-1].t >= gap_s:
                if len(current) >= min_sightings:
                    clusters.append(_PresenceCluster(label, current))
                current = []
            current.append(sighting)
        if len(current) >= min_sightings:
            clusters.append(_PresenceCluster(label, current))
    return clusters


# ------------------------------------------------------------ sighting sources

def sightings_from_annotations(recording_dir: Path | str) -> list[TrackSighting]:
    """Sightings from the annotation store (human-verified truth preferred)."""
    store = load_annotations(recording_dir)
    return [
        TrackSighting(label=a.label or "(untitled)", t=a.t, frame_index=a.frame_index, source="annotation")
        for a in store
    ]


def sightings_from_perception(recording_dir: Path | str) -> list[TrackSighting]:
    """Sightings from cached perception results only."""
    recording = load_recording(recording_dir)
    cached = CachedAnalysis(recording.directory / "perception")
    if not cached.exists:
        raise FileNotFoundError(
            f"no perception results in {cached.directory} — run `grui perception analyze` first"
        )
    sightings = []
    for result in cached.read_results():
        for detection in result.detections:
            sightings.append(
                TrackSighting(
                    label=detection.label,
                    t=result.t,
                    frame_index=result.frame_index,
                    source="perception",
                )
            )
    return sightings


def load_sightings(recording_dir: Path | str, source: str = "auto") -> list[TrackSighting]:
    """Sightings for a recording; ``auto`` prefers annotations, falls back to perception."""
    root = Path(recording_dir)
    if source not in ("auto", "annotations", "perception"):
        raise ValueError(f"unknown sighting source {source!r} (auto|annotations|perception)")
    has_annotations = (root / "annotations" / "annotations.jsonl").exists()
    if source == "annotations" or (source == "auto" and has_annotations):
        return sightings_from_annotations(root)
    if source == "auto" and not has_annotations and not (root / "perception" / "results.jsonl").exists():
        raise FileNotFoundError(
            f"no annotation store and no perception results under {root} — "
            "annotate the recording or run `grui perception analyze` first"
        )
    return sightings_from_perception(root)


# ------------------------------------------------------------ detection

def detect_events(
    sightings: Sequence[TrackSighting],
    rules: Sequence[EventRule],
) -> list[Event]:
    """Run every rule over the sightings; results are deterministic and sorted."""
    events: list[Event] = []
    for rule in rules:
        events.extend(rule.detect(sightings))
    events.sort(key=lambda e: (e.start_t, e.kind, e.label))
    return events


# ------------------------------------------------------------ persistence

def events_path(recording_dir: Path | str) -> Path:
    return Path(recording_dir) / "perception" / _EVENTS_FILENAME


def write_events(recording_dir: Path | str, events: Sequence[Event]) -> None:
    """Store events as derived data next to the perception results (raw untouched)."""
    path = events_path(recording_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for event in sorted(events, key=lambda e: (e.start_t, e.kind, e.label)):
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_events(recording_dir: Path | str) -> list[Event]:
    """Load previously written events (empty if none)."""
    path = events_path(recording_dir)
    if not path.exists():
        return []
    events = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(Event.from_dict(json.loads(line)))
    except (ValueError, OSError) as exc:
        logger.warning("failed to read events %s: %s", path, exc)
    return events


def render_events(events: Sequence[Event]) -> str:
    """Human-readable listing for the CLI."""
    if not events:
        return "No events detected."
    lines = [f"{len(events)} event(s):", ""]
    for event in events:
        span = (
            f"{event.start_t:.2f}s" if event.end_t == event.start_t
            else f"{event.start_t:.2f}s -> {event.end_t:.2f}s ({event.detail.get('duration_s', 0)}s)"
        )
        lines.append(
            f"{span:<28} {event.kind:<14} {event.label}"
            f"  (frames {event.start_frame}-{event.end_frame})"
        )
    return "\n".join(lines)