"""Dataset versioning, diffs and safe train/val/test splitting.

Derived datasets are immutable *versions*: each version references its
source recordings, the selected episodes, the annotations included
(filtered by status), excluded data, perception results and the
preprocessing configuration that produced it. Original recordings are never
touched; a new version is created by transforming a previous one.

Splitting is done per *demonstration/episode*, never per adjacent frame, so
the same moment cannot leak across train/validation/test. Splits are
deterministic (seeded hash of the demonstration identity).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation.store import load_annotations
from annotation.types import AnnotationStatus
from dataset.health import DatasetStatistics, dataset_statistics, recording_statistics
from storage.recording import RecordingData, list_recordings, load_recording

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """One immutable dataset snapshot."""

    name: str  # e.g. "v1"
    sources: list[str] = field(default_factory=list)  # recording directory names
    excluded: list[str] = field(default_factory=list)  # recordings/episodes excluded
    episodes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # recording -> episodes
    annotation_statuses: list[str] = field(default_factory=lambda: [
        AnnotationStatus.VERIFIED.value,
        AnnotationStatus.CORRECTED.value,
    ])
    include_perception: bool = True
    preprocessing: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sources": list(self.sources),
            "excluded": list(self.excluded),
            "episodes": {k: list(v) for k, v in self.episodes.items()},
            "annotation_statuses": list(self.annotation_statuses),
            "include_perception": self.include_perception,
            "preprocessing": dict(self.preprocessing),
            "parent": self.parent,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVersion":
        return cls(
            name=str(data["name"]),
            sources=list(data.get("sources") or []),
            excluded=list(data.get("excluded") or []),
            episodes={str(k): list(v) for k, v in (data.get("episodes") or {}).items()},
            annotation_statuses=list(data.get("annotation_statuses") or []),
            include_perception=bool(data.get("include_perception", True)),
            preprocessing=dict(data.get("preprocessing") or {}),
            parent=data.get("parent"),
            created_at=str(data.get("created_at") or ""),
            notes=str(data.get("notes") or ""),
        )


def next_version_name(existing: list[str]) -> str:
    """v1, v2, ... — the next free name."""
    numbers = []
    for name in existing:
        if name.startswith("v") and name[1:].isdigit():
            numbers.append(int(name[1:]))
    return f"v{max(numbers, default=0) + 1}"


# ------------------------------------------------------------------ version repo


class VersionStore:
    """Versions of one dataset root (``<root>/versions.json`` metadata)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.path = self.root / "versions.json"

    def load(self) -> list[DatasetVersion]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning("unreadable version metadata: %s", self.path)
            return []
        return [DatasetVersion.from_dict(item) for item in data.get("versions", [])]

    def add(self, version: DatasetVersion) -> None:
        versions = [v for v in self.load() if v.name != version.name]
        versions.append(version)
        versions.sort(key=lambda v: v.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"versions": [v.to_dict() for v in versions]}, indent=2),
            encoding="utf-8",
        )

    def get(self, name: str) -> DatasetVersion:
        for version in self.load():
            if version.name == name:
                return version
        raise KeyError(f"no dataset version named {name!r} in {self.root}")


def create_version(
    root: Path | str,
    recordings_root: Path | str | None = None,
    *,
    name: str | None = None,
    sources: list[Path | str] | None = None,
    excluded: list[str] | None = None,
    episodes: dict[str, list[dict[str, Any]]] | None = None,
    annotation_statuses: list[str] | None = None,
    include_perception: bool = True,
    preprocessing: dict[str, Any] | None = None,
    parent: str | None = None,
    notes: str = "",
) -> DatasetVersion:
    """Create and register a dataset version.

    ``sources`` defaults to all recordings under ``recordings_root`` (the
    dataset root's parent by default). Nothing is copied — a version is pure
    metadata referencing the raw recordings.
    """
    store = VersionStore(root)
    recordings_root = Path(recordings_root) if recordings_root else Path(root).parent
    if sources is None:
        sources = [p.name for p in list_recordings(recordings_root)]
    sources = [str(s) if isinstance(s, Path) else s for s in sources]
    if not sources:
        raise ValueError(f"no recordings found under {recordings_root}")
    version = DatasetVersion(
        name=name or next_version_name([v.name for v in store.load()]),
        sources=sources,
        excluded=list(excluded or []),
        episodes=dict(episodes or {}),
        annotation_statuses=list(
            annotation_statuses
            or [
                AnnotationStatus.VERIFIED.value,
                AnnotationStatus.CORRECTED.value,
            ]
        ),
        include_perception=include_perception,
        preprocessing=dict(preprocessing or {}),
        parent=parent,
        notes=notes,
    )
    store.add(version)
    return version


def version_statistics(
    version: DatasetVersion, recordings_root: Path | str
) -> DatasetStatistics:
    """Statistics of the recordings referenced by a version."""
    root = Path(recordings_root)
    recordings: list[RecordingData] = []
    for name in version.sources:
        if name in version.excluded:
            continue
        directory = root / name
        if not (directory / "metadata.json").exists():
            logger.warning("version references missing recording: %s", directory)
            continue
        recordings.append(load_recording(directory))
    stats = dataset_statistics(recordings)
    # annotations counted from the derived annotation layer (status-filtered)
    statuses = {AnnotationStatus(s) for s in version.annotation_statuses}
    annotations = 0
    verified = 0
    for recording in recordings:
        store = load_annotations(recording.directory)
        matched = store.filter(statuses=statuses)
        annotations += len(matched)
        verified += store.verified_count
    stats = DatasetStatistics(
        demonstrations=stats.demonstrations,
        total_duration=stats.total_duration,
        total_frames=stats.total_frames,
        average_duration=stats.average_duration,
        annotations=annotations,
        perception_predictions=stats.perception_predictions,
    )
    return stats


# ------------------------------------------------------------------------- diff


@dataclass(frozen=True)
class VersionDiff:
    """Changes between two dataset versions (``new`` vs ``old``)."""

    old_name: str
    new_name: str
    added: list[str]
    removed: list[str]
    frames_delta: int
    duration_delta: float
    annotations_delta: int
    verified_delta: int
    perception_delta: int

    def render(self) -> str:
        lines = [f"Dataset {self.old_name} → {self.new_name}"]
        for name in self.added:
            lines.append(f"+ demonstration {name}")
        for name in self.removed:
            lines.append(f"- demonstration {name}")
        lines.append(f"{self.frames_delta:+d} frames")
        lines.append(f"{self.duration_delta:+.1f}s")
        lines.append(f"{self.annotations_delta:+d} annotations (verified {self.verified_delta:+d})")
        lines.append(f"{self.perception_delta:+d} perception predictions")
        return "\n".join(lines)


def diff_versions(old: DatasetVersion, new: DatasetVersion, recordings_root: Path | str) -> VersionDiff:
    """Diff ``old`` → ``new`` (added/removed recordings, sizes, annotations)."""
    old_set = set(old.sources) - set(old.excluded)
    new_set = set(new.sources) - set(new.excluded)
    old_stats = version_statistics(old, recordings_root)
    new_stats = version_statistics(new, recordings_root)
    return VersionDiff(
        old_name=old.name,
        new_name=new.name,
        added=sorted(new_set - old_set),
        removed=sorted(old_set - new_set),
        frames_delta=new_stats.total_frames - old_stats.total_frames,
        duration_delta=new_stats.total_duration - old_stats.total_duration,
        annotations_delta=new_stats.annotations - old_stats.annotations,
        verified_delta=new_stats.annotations - old_stats.annotations,
        perception_delta=new_stats.perception_predictions - old_stats.perception_predictions,
    )


# ----------------------------------------------------------------------- split


@dataclass(frozen=True)
class DatasetSplit:
    """Deterministic per-demonstration train/val/test assignment."""

    train: list[str]
    validation: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"train": list(self.train), "validation": list(self.validation), "test": list(self.test)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSplit":
        return cls(
            train=list(data.get("train") or []),
            validation=list(data.get("validation") or []),
            test=list(data.get("test") or []),
        )


def split_demonstrations(
    recordings_root: Path | str,
    *,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
    only: list[str] | None = None,
) -> DatasetSplit:
    """Deterministic split by whole demonstrations (never adjacent frames).

    Demonstrations are hashed with ``seed`` so the split is reproducible and
    stable across runs. Raises ``ValueError`` for invalid fractions.
    """
    if train < 0 or validation < 0 or test < 0 or abs(train + validation + test - 1.0) > 1e-6:
        raise ValueError(f"train/validation/test must be non-negative and sum to 1 (got {train}, {validation}, {test})")
    names = [p.name for p in list_recordings(recordings_root)]
    if only is not None:
        names = [n for n in names if n in set(only)]
    if not names:
        raise ValueError(f"no recordings under {recordings_root} to split")
    hashed = sorted(
        (int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) / 2**32, name)
        for name in names
    )
    hashed.sort(key=lambda pair: pair[0])
    n = len(hashed)
    n_val = int(round(n * validation))
    n_test = int(round(n * test))
    n_train = n - n_val - n_test
    train_names = [name for _, name in hashed[:n_train]]
    val_names = [name for _, name in hashed[n_train : n_train + n_val]]
    test_names = [name for _, name in hashed[n_train + n_val :]]
    return DatasetSplit(train=train_names, validation=val_names, test=test_names)


def save_split(root: Path | str, recording_dirs: list[Path | str], split: DatasetSplit) -> Path:
    """Write ``split.json`` into a built dataset dir (as version metadata)."""
    path = Path(root)
    data = {
        "split": split.to_dict(),
        "sources": [str(d) for d in recording_dirs],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.mkdir(parents=True, exist_ok=True)
    out = path / "split.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out