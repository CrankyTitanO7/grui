"""Structured types for perception results (provider-agnostic).

Everything a perception provider returns is converted into these types
before it leaves the provider, so the rest of GRUI never sees a model's
native output format. All types round-trip through JSON via
``to_dict`` / ``from_dict``, which is what the on-disk ``results.jsonl``
and ``manifest.json`` use.

Model-generated predictions are kept separate from human verification:
``Detection.source`` is ``"model"`` by default and ``verified`` is
``None`` until a human confirms/rejects/edits the box (a future
annotation workflow writes its own records instead of overwriting these).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (top-left to bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
        )


@dataclass(frozen=True)
class Detection:
    """One localized object produced by a perception provider."""

    label: str
    bbox: BoundingBox
    confidence: float | None = None
    source: str = "model"  # "model" now; future: "human" for verified edits
    verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(
            label=str(data["label"]),
            bbox=BoundingBox.from_dict(data["bbox"]),
            confidence=data.get("confidence"),
            source=str(data.get("source") or "model"),
            verified=data.get("verified"),
        )


@dataclass(frozen=True)
class PerceptionResult:
    """Per-frame result for one prompt (one line of ``results.jsonl``)."""

    frame_index: int
    t: float  # seconds since session start, from frames.jsonl
    prompt: str
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "t": self.t,
            "prompt": self.prompt,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerceptionResult":
        return cls(
            frame_index=int(data["frame_index"]),
            t=float(data["t"]),
            prompt=str(data["prompt"]),
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
        )


@dataclass(frozen=True)
class ProviderInfo:
    """Human-readable description of a registered provider."""

    name: str
    version: str
    available: bool
    model: str | None = None
    description: str = ""
    install_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionManifest:
    """Identifies one perception run over one recording (manifest.json)."""

    format_version: int = 1
    provider: str = ""
    provider_version: str = ""
    model: str | None = None
    source_session_id: str = ""
    source_recording: str = ""
    sampling: dict[str, Any] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerceptionManifest":
        return cls(
            format_version=int(data.get("format_version") or 1),
            provider=str(data.get("provider") or ""),
            provider_version=str(data.get("provider_version") or ""),
            model=data.get("model"),
            source_session_id=str(data.get("source_session_id") or ""),
            source_recording=str(data.get("source_recording") or ""),
            sampling=dict(data.get("sampling") or {}),
            prompts=list(data.get("prompts") or []),
            count=int(data.get("count") or 0),
        )
