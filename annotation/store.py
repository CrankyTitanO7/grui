"""Annotation store: load/edit/persist annotations for one recording.

Derived data separation::

    <recording>/
        video.mp4, events.jsonl, frames.jsonl, markers.jsonl   # raw, untouched
        perception/        # model proposals (never overwritten)
        annotations/
            manifest.json
            annotations.jsonl    # one Annotation per line

The store never writes into the raw recording files, never modifies the
perception results, and preserves the original model prediction on every
annotation that came from a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from annotation.types import Annotation, AnnotationStatus, DetectionProvenance
from perception.types import BoundingBox, Detection

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_EMPTY = ()


class AnnotationStore:
    """In-memory annotation set with history undo/redo and JSON persistence."""

    def __init__(
        self,
        directory: Path | str,
        *,
        annotations: list[Annotation] | None = None,
        source_recording: str = "",
        source_session_id: str = "",
    ) -> None:
        self.directory = Path(directory)
        self._annotations = list(annotations) if annotations else []
        self._undo: list[list[Annotation]] = []
        self._redo: list[list[Annotation]] = []
        self.source_recording = source_recording
        self.source_session_id = source_session_id
        self._dirty = False

    # ------------------------------------------------------------ paths

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def annotations_path(self) -> Path:
        return self.directory / "annotations.jsonl"

    @property
    def exists(self) -> bool:
        return self.annotations_path.exists()

    # ------------------------------------------------------------ load/save

    @classmethod
    def load(cls, directory: Path | str) -> "AnnotationStore":
        """Load a store from ``<directory>/annotations`` (empty if none)."""
        directory = Path(directory)
        manifest: dict[str, Any] = {}
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                logger.warning("unreadable annotation manifest: %s", manifest_path)
                manifest = {}
        annotations: list[Annotation] = []
        path = directory / "annotations.jsonl"
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            annotations.append(Annotation.from_dict(json.loads(line)))
            except (ValueError, OSError) as exc:
                logger.warning("failed to read annotations %s: %s", path, exc)
        return cls(
            directory,
            annotations=annotations,
            source_recording=str(manifest.get("source_recording") or ""),
            source_session_id=str(manifest.get("source_session_id") or ""),
        )

    def save(self) -> None:
        """Write annotations (+ manifest) next to the recording. Raw data untouched."""
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": _FORMAT_VERSION,
            "source_recording": self.source_recording,
            "source_session_id": self.source_session_id,
            "count": len(self._annotations),
        }
        _write_json_atomic(self.manifest_path, manifest)
        _write_jsonl(self.annotations_path, [a.to_dict() for a in self._annotations])
        self._dirty = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": {
                "format_version": _FORMAT_VERSION,
                "source_recording": self.source_recording,
                "source_session_id": self.source_session_id,
            },
            "annotations": [a.to_dict() for a in self._annotations],
        }

    # ------------------------------------------------------------ queries

    def __len__(self) -> int:
        return len(self._annotations)

    def __iter__(self):
        return iter(self._annotations)

    def get(self, annotation_id: str) -> Annotation | None:
        for annotation in self._annotations:
            if annotation.id == annotation_id:
                return annotation
        return None

    def for_frame(self, frame_index: int) -> list[Annotation]:
        return [a for a in self._annotations if a.frame_index == frame_index]

    def filter(
        self,
        *,
        statuses: set[AnnotationStatus] | None = None,
        sources: set[str] | None = None,
        labels: set[str] | None = None,
    ) -> list[Annotation]:
        """Annotations matching the given status/source/label filters."""
        out = self._annotations
        if statuses is not None:
            out = [a for a in out if a.status in statuses]
        if sources is not None:
            out = [a for a in out if a.source in sources]
        if labels is not None:
            out = [a for a in out if a.label in labels]
        return out

    @property
    def verified_count(self) -> int:
        return sum(1 for a in self._annotations if a.status == AnnotationStatus.VERIFIED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for a in self._annotations if a.status == AnnotationStatus.REJECTED)

    # ------------------------------------------------------------ creation

    def create(
        self,
        label: str,
        bbox,
        frame_index: int,
        t: float,
        *,
        source: str = "human",
        status: AnnotationStatus = AnnotationStatus.REVIEWED,
        confidence: float | None = None,
        notes: str = "",
        prediction: DetectionProvenance | None = None,
    ) -> Annotation:
        """Create a new annotation (no model prediction involved, or with one)."""
        annotation = Annotation(
            label=label,
            bbox=bbox,
            frame_index=int(frame_index),
            t=float(t),
            source=source,
            status=status,
            confidence=confidence,
            notes=notes,
            prediction=prediction,
        )
        self._record_history("create", annotation)
        self._annotations.append(annotation)
        self._dirty = True
        return annotation

    def import_perception(self, results, *, provider: str = "imported", frame_size: tuple[int, int] | None = None) -> int:
        """Import perception detections as annotations (proposals, not truth).

        ``results`` are :class:`~perception.runner.CachedAnalysis` read with
        ``read_results()`` (list of PerceptionResult). Each detection becomes
        one annotation with the original model output preserved in
        ``prediction`` (``provider`` records where the proposal came from).
        Dedup is per detection box — (frame, label, box, provider) — so two
        same-label objects on one frame stay two individually editable
        annotations; repeat imports of the same box are not duplicated.

        Perception boxes are in absolute video pixels; annotations and the
        overlay use normalized 0..1 coordinates, so pass ``frame_size``
        ``(width, height)`` to convert at import time.
        """
        if frame_size:
            results = normalize_detection_results(results, *frame_size)
        imported = 0
        existing = {annotation_dedup_key(a) for a in self._annotations}
        for result in results:
            for detection in result.detections:
                prediction = DetectionProvenance.from_detection(detection, provider=provider)
                key = detection_dedup_key(result.frame_index, detection, provider)
                if key in existing:
                    continue
                existing.add(key)
                self.create(
                    detection.label,
                    detection.bbox,
                    result.frame_index,
                    result.t,
                    source="model",
                    status=AnnotationStatus.PREDICTED,
                    confidence=detection.confidence,
                    prediction=prediction,
                )
                imported += 1
        return imported

    def import_detection(
        self,
        frame_index: int,
        t: float,
        detection: Detection,
        *,
        provider: str = "imported",
        frame_size: tuple[int, int] | None = None,
    ) -> Annotation | None:
        """Import a single model detection as one draft annotation.

        Dedup matches :meth:`import_perception` — per detection box
        (frame, label, box, provider) — so clicking an already-imported box
        returns ``None`` instead of creating a copy, while a second
        same-label box on the same frame imports independently.
        Perception detections are in video pixels; pass ``frame_size`` to
        store normalized boxes.
        """
        if frame_size:
            detection = _normalize_detection(detection, *frame_size)
        existing = {annotation_dedup_key(a) for a in self._annotations}
        key = detection_dedup_key(frame_index, detection, provider)
        if key in existing:
            return None
        prediction = DetectionProvenance.from_detection(detection, provider=provider)
        return self.create(
            detection.label,
            detection.bbox,
            int(frame_index),
            float(t),
            source="model",
            status=AnnotationStatus.PREDICTED,
            confidence=detection.confidence,
            prediction=prediction,
        )

    # ------------------------------------------------------------ editing

    def rename(self, annotation_id: str, label: str) -> Annotation | None:
        annotation = self._require(annotation_id)
        self._record_history("rename", annotation)
        annotation.rename(label)  # the model's prediction field is untouched
        self._dirty = True
        return self.get(annotation_id)

    def move(self, annotation_id: str, dx: float, dy: float) -> Annotation | None:
        annotation = self._require(annotation_id)
        self._record_history("move", annotation)
        annotation.move(dx, dy)
        self._dirty = True
        return self.get(annotation_id)

    def resize(self, annotation_id: str, bbox) -> Annotation | None:
        annotation = self._require(annotation_id)
        self._record_history("resize", annotation)
        annotation.resize(bbox)
        self._dirty = True
        return self.get(annotation_id)

    def delete(self, annotation_id: str) -> bool:
        """Remove an annotation (undoable; the model's prediction is never erased)."""
        annotation = self.get(annotation_id)
        if annotation is None:
            return False
        self._record_history("delete", annotation)
        self._annotations = [a for a in self._annotations if a.id != annotation_id]
        self._dirty = True
        return True

    def set_status(self, annotation_id: str, status: AnnotationStatus) -> Annotation | None:
        annotation = self._require(annotation_id)
        self._record_history("status", annotation)
        annotation.set_status(status)
        self._dirty = True
        return self.get(annotation_id)

    def verify(self, annotation_id: str) -> Annotation | None:
        annotation = self._require(annotation_id)
        self._record_history("verify", annotation)
        annotation.set_verified()
        self._dirty = True
        return self.get(annotation_id)

    # ------------------------------------------------------------ history

    def _record_history(self, op: str, annotation: Annotation | None) -> None:
        self._undo.append(self._copy())
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._redo.clear()

    def _copy(self) -> list[Annotation]:
        return [Annotation.from_dict(a.to_dict()) for a in self._annotations]

    def _snapshot(self) -> list[Annotation]:
        return self._copy()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._copy())
        self._annotations = self._undo.pop()
        self._dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._copy())
        self._annotations = self._redo.pop()
        self._dirty = True
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ------------------------------------------------------------ internals

    def _require(self, annotation_id: str) -> Annotation:
        annotation = self.get(annotation_id)
        if annotation is None:
            raise KeyError(f"no annotation with id {annotation_id!r}")
        return annotation


def load_annotations(recording_dir: Path | str) -> AnnotationStore:
    """Load the annotation store for a raw recording directory."""
    root = Path(recording_dir)
    return AnnotationStore.load(root / "annotations")


def _rounded_box(bbox: BoundingBox | None) -> tuple | None:
    if bbox is None:
        return None
    return (
        round(bbox.x1, 1), round(bbox.y1, 1),
        round(bbox.x2, 1), round(bbox.y2, 1),
    )


def annotation_dedup_key(annotation: Annotation) -> tuple:
    """Identity of an annotation's origin box: (frame, label, box, provider)."""
    if annotation.prediction is not None:
        label, provider = annotation.prediction.label, annotation.prediction.provider
        bbox = annotation.prediction.bbox
    else:
        label, provider = annotation.label, annotation.source
        bbox = annotation.bbox
    return (annotation.frame_index, label, _rounded_box(bbox), provider)


def detection_dedup_key(frame_index: int, detection: Detection, provider: str) -> tuple:
    """Dedup key for a model detection, matching :func:`annotation_dedup_key`."""
    return (int(frame_index), detection.label, _rounded_box(detection.bbox), provider)


def _normalize_detection(detection: Detection, width: int, height: int) -> Detection:
    """Clone a Detection with its bbox converted to normalized 0..1 coords."""
    if detection.bbox is None or width <= 0 or height <= 0:
        return detection
    bbox = detection.bbox
    return replace(
        detection,
        bbox=BoundingBox(
            x1=bbox.x1 / width,
            y1=bbox.y1 / height,
            x2=bbox.x2 / width,
            y2=bbox.y2 / height,
        ),
    )


def normalize_detection_results(
    results: list[Any], width: int, height: int
) -> list[Any]:
    """Clone PerceptionResults with detection bboxes normalized to 0..1."""
    from perception.types import PerceptionResult

    return [
        PerceptionResult(
            frame_index=result.frame_index,
            t=result.t,
            prompt=result.prompt,
            detections=[_normalize_detection(d, width, height) for d in result.detections],
        )
        for result in results
    ]


def save_annotations(store: AnnotationStore) -> None:
    store.save()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")