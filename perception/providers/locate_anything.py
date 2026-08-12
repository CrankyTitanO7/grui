"""NVIDIA LocateAnything provider (``perception/providers/locate_anything.py``).

Wraps LocateAnything behind the generic :class:`~perception.base.PerceptionProvider`
interface. All NVIDIA-specific logic lives here (and in ``ml.locate``, the
existing backend loader this provider delegates to) — nothing else in GRUI
imports LocateAnything.

Backends (same negotiation as ``ml.locate.load_locator``):

* the ``locate-anything`` PyPI wrapper (``pip install "grui[locate]"``);
* the Eagle repo's ``LocateAnythingWorker`` on ``PYTHONPATH``.

Requirements / licensing (read before use):

* model: ``nvidia/LocateAnything-3B`` on Hugging Face — **gated**, ~6 GB
  download, requires ``huggingface-cli login``; check its license before
  commercial use (GRUI does not bundle or silently download weights);
* hardware: CUDA GPU with several GB of VRAM; CPU inference is impractical;
* the model is loaded lazily on the first ``analyze`` call — plain GRUI
  startup, recording and playback never touch it;
* inference runs entirely on local images; nothing is uploaded.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from typing import Any

import numpy as np

from perception.types import BoundingBox, Detection

MODEL_ID = "nvidia/LocateAnything-3B"
TASK = "ground_gui"  # GUI element grounding: prompt -> boxes


def _backend_importable() -> bool:
    """True when a backend can be imported (module-level only, no model load)."""
    try:
        importlib.import_module("locate_anything")
        return True
    except ImportError:
        pass
    try:
        importlib.import_module("locateanything_worker")
        return True
    except ImportError:
        return False


class LocateAnythingProvider:
    """Locate natural-language concepts in frames with LocateAnything-3B."""

    name = "locate_anything"
    model = MODEL_ID
    description = "Ground arbitrary natural-language concepts (\"the save button\", ...) in frames."
    install_hint = (
        'pip install "grui[locate]"   (PyPI wrapper; or put the Eagle repo\'s '
        "locateanything_worker.py on PYTHONPATH). The nvidia/LocateAnything-3B "
        "model is gated on Hugging Face — run `huggingface-cli login` first, "
        "and check its license before commercial use."
    )
    warnings = [
        "LocateAnything-3B is a ~6 GB model, gated on Hugging Face (`huggingface-cli login`).",
        "It needs a CUDA GPU with several GB of VRAM; CPU inference is not practical.",
        "Model weights are downloaded by the model itself on first use — GRUI never bundles or silently downloads them.",
        "Check the nvidia/LocateAnything-3B license before commercial use.",
    ]

    def __init__(self, device: str = "cuda", locator: Any | None = None) -> None:
        self._device = device
        self._locator = locator

    @property
    def version(self) -> str:
        try:
            return metadata.version("locate-anything")
        except PackageNotFoundError:
            return "worker-backend"

    def is_available(self) -> bool:
        """Import-level availability check; never instantiates the model."""
        return _backend_importable()

    def prepare(self) -> None:
        """Load the model backend up front (downloads weights on first use)."""
        self._load()

    def with_options(self, **options: Any) -> "LocateAnythingProvider":
        """Reconfigure (e.g. ``device="cuda:1"``); returns a new instance."""
        device = options.get("device")
        if device in (None, self._device):
            return self
        return LocateAnythingProvider(device=device)

    def _load(self) -> Any:
        """Lazily load the backend locator (imports/instantiates the model)."""
        if self._locator is None:
            from ml.locate import load_locator

            try:
                self._locator = load_locator(self._device)
            except Exception as exc:  # noqa: BLE001 - surface any model-load failure clearly
                raise RuntimeError(
                    "LocateAnything could not be loaded:\n"
                    f"    {exc}\n"
                    '    Install the optional dependencies: pip install "grui[locate]"\n'
                    "    The nvidia/LocateAnything-3B model is gated on Hugging Face\n"
                    "    (`huggingface-cli login`) and needs a CUDA GPU."
                ) from exc
        return self._locator

    def analyze(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Ground each prompt in one BGR frame; returns all detections."""
        locator = self._load()
        detections: list[Detection] = []
        for prompt in prompts:
            try:
                result = locator.locate(self._to_pil(frame), prompt, TASK)
            except Exception as exc:  # noqa: BLE001 - surface inference failures clearly
                raise RuntimeError(
                    f"LocateAnything inference failed for prompt {prompt!r}: {exc}"
                ) from exc
            for box in result.get("boxes", []):
                detections.append(
                    Detection(
                        label=prompt,
                        bbox=BoundingBox(
                            x1=float(box["x1"]),
                            y1=float(box["y1"]),
                            x2=float(box["x2"]),
                            y2=float(box["y2"]),
                        ),
                        confidence=None,
                        source="model",
                    )
                )
        return detections

    @staticmethod
    def _to_pil(frame: np.ndarray):
        import cv2
        from PIL import Image

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
