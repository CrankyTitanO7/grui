"""Annotation: human-verifiable labels over recorded frames.

GRUI keeps the layers separate::

    raw recording  ->  perception (model proposals)  ->  annotations (human truth)

:class:`~annotation.store.AnnotationStore` loads and edits annotations for
one recording and persists them as a derived artifact
(``<recording>/annotations/..``). Model output is never overwritten — the
original prediction is preserved on each annotation (``annot.prediction``),
so GRUI can always answer "what did the model think?" and "what did the
human correct it to?".
"""

from __future__ import annotations

from annotation.store import AnnotationStore, load_annotations, save_annotations
from annotation.types import (
    Annotation,
    AnnotationStatus,
    DetectionProvenance,
    Revision,
)

__all__ = [
    "Annotation",
    "AnnotationStatus",
    "AnnotationStore",
    "DetectionProvenance",
    "Revision",
    "load_annotations",
    "save_annotations",
]