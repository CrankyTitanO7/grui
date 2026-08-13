"""Tests for the ml/locate adapter against the real locate-anything PyPI API.

These exercise ``ml.locate.load_locator`` and ``PipLocator`` with fake
clients shaped like the installed ``locate-anything`` wrapper (whose
constructor takes ``model_name``/``device_map``/... and whose ``detect``
returns ``{"detections": [{"label", "bbox_pixels"}]}``). No model, GPU or
Hugging Face access is required.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ml import locate


class _FakeClient:
    """Mirrors locate_anything.LocateAnything.detect's contract."""

    def __init__(self, detections=None) -> None:
        self.calls: list[tuple] = []
        self._detections = detections or []

    def detect(self, image, categories=None, draw=True, **kwargs):
        self.calls.append((image, list(categories or []), draw))
        return {"detections": self._detections, "counts": {}, "total": len(self._detections)}


@pytest.fixture()
def fake_pip_backend(monkeypatch):
    """Install a fake ``locate_anything`` module for import time."""

    def install(cls=None):
        module = types.ModuleType("locate_anything")
        if cls is not None:
            module.LocateAnything = cls
        monkeypatch.setitem(sys.modules, "locate_anything", module)

    return install


@pytest.fixture()
def no_worker_backend(monkeypatch):
    """Force the Eagle-repo worker backend to be unimportable."""

    def install(cls=None):
        module = types.ModuleType("locateanything_worker")
        if cls is not None:
            module.LocateAnythingWorker = cls
        monkeypatch.setitem(sys.modules, "locateanything_worker", module)

    return install


def test_load_locator_pip_backend_takes_no_device_kwarg(fake_pip_backend, no_worker_backend):
    """The wrapper has no ``device`` kwarg; loading must not pass one."""

    class RealisticWrapper:
        def __init__(self, model_name=None, device_map="auto", torch_dtype=None, hf_token=None):
            assert device_map == "auto"
            assert model_name is None

    fake_pip_backend(RealisticWrapper)
    no_worker_backend("unused")
    locator = locate.load_locator(device="cpu")  # would raise TypeError if device leaked
    assert isinstance(locator, locate.PipLocator)


def test_load_locator_import_error_falls_through(fake_pip_backend, no_worker_backend):
    fake_pip_backend()  # module present but without LocateAnything -> ImportError
    no_worker_backend()  # same for the worker
    with pytest.raises(RuntimeError, match="not installed"):
        locate.load_locator()


def test_load_locator_worker_backend_gets_device(fake_pip_backend, no_worker_backend):
    class FakeWorker:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

    fake_pip_backend()  # no usable pip backend -> fall through to the worker
    no_worker_backend(FakeWorker)
    locator = locate.load_locator(device="cuda:1")
    assert isinstance(locator, locate.WorkerLocator)
    assert locator._worker.device == "cuda:1"
    assert locator._worker.model_name == "nvidia/LocateAnything-3B"


def test_load_locator_quantize_uses_inrepo_worker(fake_pip_backend, no_worker_backend, monkeypatch):
    """Quantization must win over installed backends and use the in-repo worker."""

    class FakeHfWorker:
        def __init__(self, device="cuda", quantize="4bit", max_tokens=1024):
            self.device = device
            self.quantize = quantize
            self.max_tokens = max_tokens

    monkeypatch.setattr(locate, "HfLocateAnythingWorker", FakeHfWorker)
    fake_pip_backend()  # even with a usable pip backend installed...
    no_worker_backend()  # ...and no worker backend
    locator = locate.load_locator(device="cuda:1", quantize="4bit", max_tokens=512)
    assert isinstance(locator, locate.WorkerLocator)
    assert locator._worker.device == "cuda:1"
    assert locator._worker.quantize == "4bit"
    assert locator._worker.max_tokens == 512


def test_hf_worker_parse_boxes_and_points():
    boxes = locate.HfLocateAnythingWorker.parse_boxes(
        "stuff <box><100><200><300><400></box> tail", 1920, 1080
    )
    assert boxes == [
        {"x1": 192.0, "y1": 216.0, "x2": 576.0, "y2": 432.0}
    ]
    points = locate.HfLocateAnythingWorker.parse_points(
        "<box><500><250></box>", 1920, 1080
    )
    assert points == [{"x": 960.0, "y": 270.0}]


def test_scale_result_scales_back_to_original_size():
    result = {
        "boxes": [{"x1": 50.0, "y1": 25.0, "x2": 150.0, "y2": 75.0}],
        "points": [{"x": 100.0, "y": 50.0}],
    }
    scaled = locate._scale_result(result, (1000, 800), (1920, 1080))
    assert scaled["boxes"][0] == {"x1": 96.0, "y1": 33.75, "x2": 288.0, "y2": 101.25}
    assert scaled["points"][0] == {"x": 192.0, "y": 67.5}


def test_scale_result_noop_when_same_size():
    result = {"boxes": [{"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}], "points": []}
    assert locate._scale_result(result, (1920, 1080), (1920, 1080)) is result


def test_load_image_downscales_to_max_pixels(tmp_path):
    import cv2
    import numpy as np

    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:] = (10, 200, 30)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)

    image, orig = locate._load_image(path, max_pixels=20_000)
    assert orig == (400, 200)
    assert image.width * image.height <= 20_000
    assert image.width / image.height == pytest.approx(2.0, abs=0.2)

    full, full_orig = locate._load_image(path)
    assert full.size == (400, 200)
    assert full_orig == (400, 200)


def test_enrich_dataset_scales_boxes_back_to_original(tmp_path):
    import cv2
    import json
    import numpy as np

    ds = tmp_path / "ds"
    frames = ds / "frames"
    frames.mkdir(parents=True)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (255, 255, 0)
    (frames / "f0.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    (ds / "manifest.json").write_text("{}", encoding="utf-8")
    (ds / "frames.jsonl").write_text(
        json.dumps({"frame_index": 0, "t": 0.0, "path": "frames/f0.png"}) + "\n",
        encoding="utf-8",
    )

    class FakeLocator:
        def locate(self, image, prompt, task):
            return {
                "boxes": [{"x1": 20.0, "y1": 20.0, "x2": 40.0, "y2": 40.0}],
                "points": [{"x": 30.0, "y": 30.0}],
            }

    out = locate.enrich_dataset(
        ds, ["button"], "ground_gui",
        every=1, locator=FakeLocator(), max_pixels=2_000,
    )
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    # 200x100 (20k px) downscales to ~63x31; boxes/points must come back to
    # the original 200x100 frame coordinates.
    assert record["boxes"][0]["x1"] == pytest.approx(20.0 * 200 / 63, abs=0.5)
    assert record["boxes"][0]["x2"] == pytest.approx(40.0 * 200 / 63, abs=0.5)
    assert record["boxes"][0]["y1"] == pytest.approx(20.0 * 100 / 31, abs=0.5)
    assert record["boxes"][0]["y2"] == pytest.approx(40.0 * 100 / 31, abs=0.5)
    assert record["points"][0] == pytest.approx(
        {"x": 30.0 * 200 / 63, "y": 30.0 * 100 / 31}, abs=0.5
    )


def test_pip_locator_parses_boxes():
    client = _FakeClient(
        detections=[
            {"label": "boss", "bbox_pixels": [10, 20, 30, 40]},
            {"label": "boss", "bbox_pixels": [100, 200, 300, 400]},
        ]
    )
    locator = locate.PipLocator(client)
    result = locator.locate(np.zeros((64, 64, 3), dtype=np.uint8), "boss", "ground_gui")
    assert result["boxes"] == [
        {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
    ]
    assert result["points"] == []
    _, categories, draw = client.calls[0]
    assert categories == ["boss"]
    assert draw is False  # skip annotated-image rendering during analysis


def test_pip_locator_rejects_unsupported_tasks():
    locator = locate.PipLocator(_FakeClient())
    with pytest.raises(ValueError, match="LocateAnythingWorker"):
        locator.locate(np.zeros((4, 4, 3), dtype=np.uint8), "p", "point")