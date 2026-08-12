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