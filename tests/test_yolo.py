"""YOLO provider tests — no ultralytics, no model weights required.

Uses a fake detector to exercise prompt matching, prediction wiring and
CLI plumbing, and forces the ultralytics error paths via sys.modules /
monkeypatched availability so everything runs on CPU-only CI.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from perception import BoundingBox, Detection, get, is_registered
from perception.registry import _PROVIDERS
from perception.types import ProviderInfo
from tests.fakes import build_synthetic_recording
from perception.base import provider_info, with_options
from perception.providers.yolo import YoloProvider, _as_class_index, _to_rgb


class FakeBox:
    def __init__(self, cls_idx, xyxy, conf):
        self.cls = np.array([cls_idx])
        self.xyxy = np.array([[float(v) for v in xyxy]])
        self.conf = np.array([conf]) if conf is not None else None


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    """Records predict() kwargs and returns configured boxes."""

    def __init__(self, results=None):
        self.results = results if results is not None else [FakeResult([])]
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


NAMES = {0: "person", 1: "car", 2: "truck"}


def _provider(**kwargs):
    kwargs.setdefault("detector", FakeDetector())
    kwargs.setdefault("names", dict(NAMES))
    return YoloProvider(**kwargs)


@pytest.fixture()
def recording(tmp_path):
    return build_synthetic_recording(tmp_path / "root", n_frames=30, fps=10)


# ------------------------------------------------------------ registration

def test_yolo_registered_with_metadata():
    assert is_registered("yolo")
    provider = get("yolo")
    assert provider.name == "yolo"
    assert "fixed-vocabulary" in provider.description.lower()
    assert "grui[yolo]" in provider.install_hint
    assert provider.warnings
    assert isinstance(provider.version, str)
    info = provider_info(provider)
    assert isinstance(info, ProviderInfo)
    assert info.name == "yolo"
    assert info.model == "user-provided .pt weights file"


def test_yolo_availability_monkeypatched(monkeypatch):
    from perception.providers import yolo

    monkeypatch.setattr(yolo, "_backend_importable", lambda: False)
    assert YoloProvider().is_available() is False
    monkeypatch.setattr(yolo, "_backend_importable", lambda: True)
    assert YoloProvider().is_available() is True


# -------------------------------------------------------------- loading

def test_missing_weights_raise_clear_error(tmp_path):
    provider = _provider(model=tmp_path / "missing.pt", detector=None)
    with pytest.raises(RuntimeError, match="YOLO weights file not found"):
        provider.prepare()
    with pytest.raises(RuntimeError, match="does not download"):
        provider.prepare()
    with pytest.raises(RuntimeError, match="--allow-download"):
        provider.prepare()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="YOLO weights file not found"):
        provider.analyze(frame, ["person"])


def test_allow_download_skips_weights_check(monkeypatch, tmp_path):
    # When --allow-download is set, the missing file no longer blocks, so
    # the next failure is the (forced) missing ultralytics package.
    backend = sys.modules.get("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    try:
        provider = YoloProvider(model=tmp_path / "missing.pt", allow_download=True)
        with pytest.raises(RuntimeError, match="ultralytics"):
            provider.prepare()
    finally:
        monkeypatch.setitem(sys.modules, "ultralytics", backend)


def test_missing_ultralytics_raise_clear_error(monkeypatch, tmp_path):
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"fake")
    backend = sys.modules.get("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    try:
        provider = YoloProvider(model=weights)
        with pytest.raises(RuntimeError, match=r'pip install "grui\[yolo\]"'):
            provider.prepare()
    finally:
        monkeypatch.setitem(sys.modules, "ultralytics", backend)


# ------------------------------------------------------------- options

def test_with_options_identity_and_override():
    provider = YoloProvider(model="a.pt", conf=0.25, allow_download=True)
    assert with_options(provider, model=None, conf=None, device=None) is provider
    assert with_options(provider, model="a.pt", conf=0.25) is provider
    reconfigured = with_options(provider, conf=0.4)
    assert reconfigured is not provider
    assert reconfigured._conf == 0.4
    assert reconfigured._allow_download is True  # preserved
    moved = with_options(provider, device="cuda:1")
    assert moved._device == "cuda:1"
    other_model = with_options(provider, model="b.pt")
    assert other_model._model_path == "b.pt"
    assert other_model._conf == 0.25  # untouched option kept


# ----------------------------------------------------------- inference

def test_prompt_matching_and_detection():
    detector = FakeDetector(
        [FakeResult([FakeBox(0, (1, 2, 3, 4), 0.9), FakeBox(1, (5, 6, 7, 8), 0.7)])]
    )
    provider = _provider(detector=detector)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = provider.analyze(frame, ["person", "car"])

    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].bbox == BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
    assert detections[0].confidence == 0.9
    assert detections[0].source == "model"
    assert detections[1].label == "car"
    assert detections[1].confidence == 0.7

    assert len(detector.calls) == 1
    call = detector.calls[0]
    assert call["conf"] == 0.25  # default threshold
    assert call["classes"] == [0, 1]
    assert call["device"] is None
    assert call["verbose"] is False
    assert isinstance(call["source"], np.ndarray)


def test_case_insensitive_and_digit_prompts():
    detector = FakeDetector(
        [FakeResult([FakeBox(0, (0, 0, 10, 10), 0.5)])]
    )
    provider = _provider(detector=detector)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert provider.analyze(frame, ["PERSON"])[0].label == "person"
    assert provider.analyze(frame, ["  person "])[0].label == "person"
    assert provider.analyze(frame, ["0"])[0].label == "person"  # class index
    classes = {tuple(c["classes"]) for c in detector.calls}
    assert classes == {(0,)}


def test_digit_prompt_outside_vocabulary_no_detections():
    detector = FakeDetector()
    provider = _provider(detector=detector)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert provider.analyze(frame, ["9"]) == []
    assert provider.analyze(frame, ["0.5"]) == []  # not a plain index
    assert detector.calls == []


def test_unknown_prompt_never_calls_predict():
    detector = FakeDetector()
    provider = _provider(detector=detector)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert provider.analyze(frame, ["spaceship"]) == []
    assert detector.calls == []


def test_default_labels_fallback():
    detector = FakeDetector(
        [FakeResult([FakeBox(2, (1, 1, 5, 5), 0.8)])]
    )
    provider = _provider(detector=detector, default_labels=["truck"])
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = provider.analyze(frame, ["nonsense prompt"])
    assert len(detections) == 1
    assert detections[0].label == "truck"
    assert detector.calls[0]["classes"] == [2]


def test_empty_results_handled():
    detector = FakeDetector([FakeResult([])])
    provider = _provider(detector=detector)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert provider.analyze(frame, ["person"]) == []

    detector = FakeDetector([])
    provider = _provider(detector=detector)
    assert provider.analyze(frame, ["person"]) == []


def test_conf_and_device_plumbing():
    detector = FakeDetector([FakeResult([FakeBox(0, (0, 0, 1, 1), 0.6)])])
    provider = _provider(detector=detector)
    reconfigured = with_options(provider, conf=0.5, device="cuda:1")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    reconfigured.analyze(frame, ["person"])
    assert detector.calls[0]["conf"] == 0.5
    assert detector.calls[0]["device"] == "cuda:1"


def test_to_rgb_conversion():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]  # BGR -> RGB flips the first and last channel
    rgb = _to_rgb(frame)
    assert rgb.shape == frame.shape
    assert rgb[0, 0].tolist() == [30, 20, 10]
    gray = np.zeros((2, 2), dtype=np.uint8)
    assert _to_rgb(gray) is gray  # non-color frames passed through


def test_as_class_index():
    assert _as_class_index("0") == 0
    assert _as_class_index(" 12 ") == 12
    assert _as_class_index("person") is None
    assert _as_class_index("0.5") is None


# ----------------------------------------------------------------- CLI

def test_cli_analyze_yolo_wiring(recording, capsys, monkeypatch):
    from perception.cli import run
    from perception.providers import yolo

    monkeypatch.setattr(yolo, "_backend_importable", lambda: True)
    detector = FakeDetector([FakeResult([FakeBox(0, (1, 2, 3, 4), 0.9)])])
    original = _PROVIDERS["yolo"]
    _PROVIDERS["yolo"] = _provider(detector=detector)
    try:
        code = run(
            [
                "analyze",
                str(recording.directory),
                "--provider",
                "yolo",
                "--model",
                "fake.pt",
                "--conf",
                "0.6",
                "--prompt",
                "person",
                "--every",
                "10",
            ]
        )
    finally:
        _PROVIDERS["yolo"] = original
    assert code == 0
    assert (recording.directory / "perception" / "results.jsonl").exists()
    assert detector.calls
    assert detector.calls[0]["conf"] == 0.6  # --conf reached the detector
    assert detector.calls[0]["classes"] == [0]


def test_cli_analyze_yolo_missing_weights(monkeypatch, recording, capsys):
    from perception.cli import run
    from perception.providers import yolo

    monkeypatch.setattr(yolo, "_backend_importable", lambda: True)
    code = run(
        [
            "analyze",
            str(recording.directory),
            "--provider",
            "yolo",
            "--model",
            str(recording.directory / "missing.pt"),
            "--prompt",
            "person",
            "--every",
            "10",
        ]
    )
    assert code == 1
    assert "YOLO weights file not found" in capsys.readouterr().err