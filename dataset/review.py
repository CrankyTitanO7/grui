"""Active-learning / review queue infrastructure.

A :class:`ReviewQueue` selects frames/blocks worth a human look, with an
extensible set of ranking strategies:

* perception uncertainty (low-confidence detections)
* rare actions and unusual combinations
* visual novelty (frames that differ strongly from their neighbors)
* annotation uncertainty / disagreement
* more strategies can register later (training loss, model disagreement, …)

Queue state persists as derived metadata (`<recording>/review/queue.jsonl`),
distinct from raw data, perception and annotations. Verdicts
(accept/reject/skip) update the *annotation* layer — never the model output
or the raw recording.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from annotation.store import load_annotations
from annotation.types import AnnotationStatus
from dataset.health import action_distribution
from storage.recording import RecordingData

logger = logging.getLogger(__name__)

_MAX_PRIORITY = 100.0


@dataclass(frozen=True)
class ReviewItem:
    """One candidate for human review."""

    frame_index: int
    t: float  # seconds since session start
    reason: str  # short human-readable reason, e.g. "low confidence (0.42)"
    priority: float  # 0..100, higher = review first
    kind: str = "frame"  # frame | detection | action
    data: dict[str, Any] = field(default_factory=dict)  # e.g. detection dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "t": self.t,
            "reason": self.reason,
            "priority": round(float(self.priority), 2),
            "kind": self.kind,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewItem":
        return cls(
            frame_index=int(data["frame_index"]),
            t=float(data["t"]),
            reason=str(data.get("reason") or ""),
            priority=float(data.get("priority") or 0.0),
            kind=str(data.get("kind") or "frame"),
            data=dict(data.get("data") or {}),
        )


class ReviewStrategy(Protocol):
    name: str

    def candidates(self, recording: RecordingData, limit: int) -> list[ReviewItem]:
        ...


# --------------------------------------------------------------- strategies


def uncertainty_candidates(recording: RecordingData, limit: int) -> list[ReviewItem]:
    """Frames with low-confidence detections (``conf < 0.6``), highest first."""
    from perception.runner import CachedAnalysis

    items: list[ReviewItem] = []
    cached = CachedAnalysis(recording.directory / "perception")
    if not cached.exists:
        return items
    for result in cached.read_results():
        for detection in result.detections:
            conf = detection.confidence
            if conf is None or conf >= 0.6:
                continue
            items.append(
                ReviewItem(
                    frame_index=result.frame_index,
                    t=result.t,
                    reason=f"low confidence {conf:.2f} ({detection.label!r})",
                    priority=_MAX_PRIORITY * (1.0 - conf),
                    kind="detection",
                    data={"label": detection.label, "bbox": detection.bbox.to_dict(), "confidence": conf},
                )
            )
    items.sort(key=lambda i: i.priority, reverse=True)
    return items[:limit]


def rare_action_candidates(recording: RecordingData, limit: int) -> list[ReviewItem]:
    """Frames where a rare action/combination is active.

    A key/button is "rare" when its share of sampled frames is below 5% of
    the most frequent action's share. Frames with such an action active get
    a priority based on rarity; the first occurrence of each rare action is
    prioritized (segments, not every frame).
    """
    from player.event_state import KeyStateTimeline

    dist = action_distribution(recording)
    if not dist.samples:
        return []
    top = max(
        [dist.key_fraction(c) for c in dist.keys]
        + [dist.button_fraction(c) for c in dist.buttons]
        + [0.0]
    )
    if top <= 0:
        return []
    rare_keys = {c for c in dist.keys if dist.key_fraction(c) <= 0.05 * top}
    rare_buttons = {c for c in dist.buttons if dist.button_fraction(c) <= 0.05 * top}
    if not rare_keys and not rare_buttons:
        return []
    keys = KeyStateTimeline(recording.events)
    items: list[ReviewItem] = []
    for frame_index, t in enumerate(recording.frame_times):
        active = set(keys.active_keys_at(t)) & rare_keys
        buttons = set(keys.active_buttons_at(t)) & rare_buttons
        if not active and not buttons:
            continue
        codes = sorted(active | buttons)
        items.append(
            ReviewItem(
                frame_index=frame_index,
                t=float(t),
                reason="rare action: " + " + ".join(codes),
                priority=70.0,
                kind="action",
                data={"codes": codes},
            )
        )
        if len(items) >= limit:
            break
    return items


def visual_novelty_candidates(recording: RecordingData, limit: int) -> list[ReviewItem]:
    """Frames that differ strongly from the previous frame (scene transitions)."""
    if len(recording.frame_times) < 2:
        return []
    import cv2
    import numpy as np

    stride = max(1, len(recording.frame_times) // (limit * 8))
    wanted = list(range(0, len(recording.frame_times), stride))
    wanted_set = set(wanted)
    caps: dict[int, float] = {}
    prev: Any = None
    cap = cv2.VideoCapture(str(recording.video_path))
    index = 0
    try:
        while wanted_set:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
                if prev is not None:
                    diff = float(cv2.absdiff(small, prev).mean())
                    if diff > 25.0:
                        caps[index] = diff
                prev = small
                wanted_set.discard(index)
            index += 1
    finally:
        cap.release()
    items = [
        ReviewItem(
            frame_index=index,
            t=float(recording.frame_time(index)),
            reason="novel visual state",
            priority=min(_MAX_PRIORITY, 50.0 + diff),
            kind="frame",
        )
        for index, diff in sorted(caps.items(), key=lambda kv: -kv[1])[:limit]
    ]
    return items


def annotation_uncertainty_candidates(recording: RecordingData, limit: int) -> list[ReviewItem]:
    """Model-proposed annotations nobody reviewed yet (status: predicted)."""
    store = load_annotations(recording.directory)
    items = []
    for annotation in store:
        if annotation.status != AnnotationStatus.PREDICTED:
            continue
        items.append(
            ReviewItem(
                frame_index=annotation.frame_index,
                t=annotation.t,
                reason=f"unreviewed prediction {annotation.label!r}",
                priority=60.0 if annotation.confidence is None else min(60.0, 40.0 + (1.0 - annotation.confidence) * 40.0),
                kind="detection",
                data={"annotation_id": annotation.id, "label": annotation.label},
            )
        )
    items.sort(key=lambda i: i.priority, reverse=True)
    return items[:limit]


STRATEGIES: dict[str, Callable[[RecordingData, int], list[ReviewItem]]] = {
    "uncertainty": uncertainty_candidates,
    "rare_action": rare_action_candidates,
    "novelty": visual_novelty_candidates,
    "annotation_uncertainty": annotation_uncertainty_candidates,
}


def build_queue(
    recording: RecordingData,
    *,
    strategies: list[str] | None = None,
    limit: int = 200,
) -> list[ReviewItem]:
    """Combine candidates from the selected strategies, deduplicated by frame."""
    selected = strategies or sorted(STRATEGIES)
    unknown = [s for s in selected if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown review strategy(s): {', '.join(unknown)}")
    by_frame: dict[int, ReviewItem] = {}
    for name in selected:
        for item in STRATEGIES[name](recording, limit):
            existing = by_frame.get(item.frame_index)
            if existing is None or item.priority > existing.priority:
                by_frame[item.frame_index] = item
    return sorted(by_frame.values(), key=lambda i: -i.priority)[:limit]


# ------------------------------------------------------------------ persistence


class ReviewQueue:
    """Persisted queue plus verdicts for one recording."""

    def __init__(self, recording: RecordingData) -> None:
        self.recording = recording
        self.directory = recording.directory / "review"
        self.path = self.directory / "queue.jsonl"
        self.items: list[ReviewItem] = []
        self.verdicts: dict[str, str] = {}  # key -> accepted | rejected | skipped
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "verdict" in row:
                self.verdicts[str(row["frame_index"])] = row["verdict"]
            else:
                self.items.append(ReviewItem.from_dict(row))

    def refresh(self, **kwargs: Any) -> list[ReviewItem]:
        """Rebuild candidates (keeps prior verdicts)."""
        seen_verdicts = dict(self.verdicts)
        self.items = build_queue(self.recording, **kwargs)
        self.verdicts = seen_verdicts
        self._save()
        return self.items

    def pending(self) -> list[ReviewItem]:
        return [i for i in self.items if str(i.frame_index) not in self.verdicts]

    def accept(self, frame_index: int, *, propagate: bool = True) -> None:
        """Mark a frame reviewed; verify its model annotations when present."""
        self.verdicts[str(frame_index)] = "accepted"
        if propagate:
            store = load_annotations(self.recording.directory)
            changed = False
            for annotation in store.for_frame(frame_index):
                if annotation.source in ("model", "imported") and annotation.status in (
                    AnnotationStatus.PREDICTED,
                    AnnotationStatus.REVIEWED,
                ):
                    store.verify(annotation.id)
                    changed = True
            if changed:
                store.save()
        self._save()

    def reject(self, frame_index: int) -> None:
        self.verdicts[str(frame_index)] = "rejected"
        store = load_annotations(self.recording.directory)
        changed = False
        for annotation in store.for_frame(frame_index):
            if annotation.source in ("model", "imported") and annotation.status != AnnotationStatus.REJECTED:
                store.set_status(annotation.id, AnnotationStatus.REJECTED)
                changed = True
        if changed:
            store.save()
        self._save()

    def skip(self, frame_index: int) -> None:
        self.verdicts[str(frame_index)] = "skipped"
        self._save()

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for item in sorted(self.items, key=lambda i: -i.priority):
                fh.write(json.dumps(item.to_dict()) + "\n")
            for frame_index, verdict in sorted(self.verdicts.items()):
                fh.write(json.dumps({"frame_index": int(frame_index), "verdict": verdict}) + "\n")