"""Tests for AnnotationStore import/dedup behavior (perception -> annotations)."""

from pathlib import Path

import pytest

from annotation.store import AnnotationStore
from annotation.types import AnnotationStatus
from perception.types import BoundingBox, Detection, PerceptionResult


@pytest.fixture()
def store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path)


DETECTION = Detection(label="person", bbox=BoundingBox(10, 20, 110, 220), confidence=0.87)


def test_import_detection_creates_draft_annotation(store: AnnotationStore) -> None:
    imported = store.import_detection(3, 1.5, DETECTION)
    assert imported is not None
    assert len(store) == 1
    annotation = store.get(imported.id)
    assert annotation.source == "model"
    assert annotation.status == AnnotationStatus.PREDICTED
    assert annotation.frame_index == 3
    assert annotation.prediction is not None
    assert annotation.prediction.provider == "imported"
    assert annotation.prediction.label == "person"


def test_import_detection_dedups(store: AnnotationStore) -> None:
    store.import_detection(3, 1.5, DETECTION)
    assert store.import_detection(3, 1.5, DETECTION) is None
    assert len(store) == 1


def test_import_perception_matches_single_import_dedup(store: AnnotationStore) -> None:
    result = PerceptionResult(
        frame_index=3, t=1.5, prompt="p", detections=[DETECTION]
    )
    assert store.import_detection(3, 1.5, DETECTION) is not None
    assert store.import_perception([result]) == 0
    assert len(store) == 1


def test_import_detection_normalizes_with_frame_size(store: AnnotationStore) -> None:
    imported = store.import_detection(3, 1.5, DETECTION, frame_size=(640, 480))
    annotation = store.get(imported.id)
    assert annotation.bbox.to_dict() == {
        "x1": 10 / 640, "y1": 20 / 480,
        "x2": 110 / 640, "y2": 220 / 480,
    }
    assert annotation.prediction.bbox.to_dict() == annotation.bbox.to_dict()


def test_import_perception_normalizes_with_frame_size(store: AnnotationStore) -> None:
    result = PerceptionResult(
        frame_index=3, t=1.5, prompt="p", detections=[DETECTION]
    )
    assert store.import_perception([result], frame_size=(640, 480)) == 1
    annotation = list(store)[0]
    assert annotation.bbox.x1 == pytest.approx(10 / 640)
    assert annotation.bbox.y1 == pytest.approx(20 / 480)
    assert annotation.bbox.x2 == pytest.approx(110 / 640)
    assert annotation.bbox.y2 == pytest.approx(220 / 480)


def test_import_perception_normalized_matches_single_normalized(store: AnnotationStore) -> None:
    result = PerceptionResult(
        frame_index=3, t=1.5, prompt="p", detections=[DETECTION]
    )
    assert store.import_detection(3, 1.5, DETECTION, frame_size=(640, 480)) is not None
    assert store.import_perception([result], frame_size=(640, 480)) == 0
    assert len(store) == 1


def test_import_perception_batch_counts_only_new(store: AnnotationStore) -> None:
    results = [
        PerceptionResult(
            frame_index=0, t=0.0, prompt="p",
            detections=[DETECTION, Detection(label="chair", bbox=BoundingBox(0, 0, 5, 5))],
        ),
        PerceptionResult(
            frame_index=4, t=2.0, prompt="p",
            detections=[Detection(label="dog", bbox=BoundingBox(1, 1, 2, 2))],
        ),
    ]
    assert store.import_perception(results) == 3
    assert store.import_perception(results) == 0
    assert len(store) == 3