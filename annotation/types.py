"""Generic annotation data structures — *not* tied to any perception model.

An :class:`Annotation` is the human-facing record for one localized object
on one frame. It carries:

* ``label`` / ``bbox`` — the current (possibly human-corrected) values;
* ``source`` — where it came from (``model``, ``human``, ``imported``,
  ``derived``);
* ``status`` — an explicit workflow state (predicted, reviewed, verified,
  corrected, rejected);
* ``prediction`` — the *original* model output, frozen, so a human
  correction never destroys the model's proposal;
* ``history`` — a list of :class:`Revision` records describing how the
  annotation reached its current state (the raw model output is always
  recoverable from ``prediction``).

Everything round-trips through JSON via ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from perception.types import BoundingBox, Detection


class AnnotationStatus(str, Enum):
    """Explicit lifecycle state of an annotation."""

    PREDICTED = "predicted"  # a model proposed it; nobody looked at it yet
    REVIEWED = "reviewed"  # a human looked at it
    VERIFIED = "verified"  # a human confirmed the label/box as correct
    CORRECTED = "corrected"  # a human changed label and/or bbox
    REJECTED = "rejected"  # a human marked it as wrong

    @classmethod
    def from_value(cls, value: Any) -> "AnnotationStatus":
        try:
            return cls(value)
        except ValueError:
            # Unknown statuses from future versions degrade to PREDICTED.
            return cls.PREDICTED

    def to_dict(self) -> str:
        return self.value


@dataclass(frozen=True)
class DetectionProvenance:
    """The unmodified output of a perception provider for one object."""

    label: str
    bbox: BoundingBox
    confidence: float | None = None
    provider: str = ""
    prompt: str = ""

    @classmethod
    def from_detection(
        cls, detection: Detection, provider: str = "", prompt: str = ""
    ) -> "DetectionProvenance":
        return cls(
            label=detection.label,
            bbox=detection.bbox,
            confidence=detection.confidence,
            provider=provider,
            prompt=prompt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "provider": self.provider,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionProvenance":
        return cls(
            label=str(data["label"]),
            bbox=BoundingBox.from_dict(data["bbox"]),
            confidence=data.get("confidence"),
            provider=str(data.get("provider") or ""),
            prompt=str(data.get("prompt") or ""),
        )


@dataclass(frozen=True)
class Revision:
    """One recorded modification to an annotation."""

    action: str  # created | renamed | moved | resized | verified | corrected | rejected | deleted
    at: str  # ISO timestamp
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "at": self.at, "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Revision":
        return cls(
            action=str(data.get("action") or ""),
            at=str(data.get("at") or ""),
            detail=dict(data.get("detail") or {}),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Annotation:
    """One localized object with full provenance."""

    label: str
    bbox: BoundingBox
    frame_index: int
    t: float  # seconds since session start (from frames.jsonl)
    source: str = "model"  # model | human | imported | derived
    status: AnnotationStatus = AnnotationStatus.PREDICTED
    confidence: float | None = None
    notes: str = ""
    prediction: DetectionProvenance | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: list[Revision] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # ----------------------------------------------------------- mutations
    # Each mutation records a Revision so the full history is preserved.

    def rename(self, label: str) -> "Annotation":
        self.history.append(
            Revision("renamed", _now_iso(), {"from": self.label, "to": label})
        )
        self.label = label
        self._touch()
        return self

    def move(self, dx: float, dy: float) -> "Annotation":
        moved = BoundingBox(
            x1=self.bbox.x1 + dx,
            y1=self.bbox.y1 + dy,
            x2=self.bbox.x2 + dx,
            y2=self.bbox.y2 + dy,
        )
        self.history.append(
            Revision("moved", _now_iso(), {"from": self.bbox.to_dict(), "to": moved.to_dict()})
        )
        self.bbox = moved
        self._touch()
        return self

    def resize(self, bbox: BoundingBox) -> "Annotation":
        self.history.append(
            Revision("resized", _now_iso(), {"from": self.bbox.to_dict(), "to": bbox.to_dict()})
        )
        self.bbox = bbox
        self._touch()
        return self

    def set_verified(self, source: str | None = None) -> "Annotation":
        self.history.append(Revision("verified", _now_iso(), {"source": source or self.source}))
        self.status = AnnotationStatus.VERIFIED
        if source:
            self.source = source
        self._touch()
        return self

    def set_status(self, status: AnnotationStatus) -> "Annotation":
        if status == self.status:
            return self
        self.history.append(
            Revision("status", _now_iso(), {"from": self.status.value, "to": status.value})
        )
        self.status = status
        self._touch()
        return self

    def set_notes(self, notes: str) -> "Annotation":
        self.notes = notes
        self._touch()
        return self

    def _touch(self) -> None:
        self.updated_at = _now_iso()

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "bbox": self.bbox.to_dict(),
            "frame_index": self.frame_index,
            "t": self.t,
            "source": self.source,
            "status": self.status.to_dict(),
            "confidence": self.confidence,
            "notes": self.notes,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "history": [h.to_dict() for h in self.history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        return cls(
            label=str(data["label"]),
            bbox=BoundingBox.from_dict(data["bbox"]),
            frame_index=int(data["frame_index"]),
            t=float(data["t"]),
            source=str(data.get("source") or "model"),
            status=AnnotationStatus.from_value(data.get("status") or "predicted"),
            confidence=data.get("confidence"),
            notes=str(data.get("notes") or ""),
            prediction=(
                DetectionProvenance.from_dict(data["prediction"]) if data.get("prediction") else None
            ),
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            history=[Revision.from_dict(h) for h in data.get("history") or []],
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )