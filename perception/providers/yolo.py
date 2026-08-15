"""YOLO perception provider (``perception/providers/yolo.py``).

A thin wrapper around Ultralytics YOLO behind the generic
:class:`~perception.base.PerceptionProvider` interface.

Design constraints (all enforced here):

* **optional** — plain GRUI never imports this module at startup and never
  loads ``ultralytics``; the import happens lazily on the first
  ``analyze``/``prepare`` call;
* **no bundled weights** — the model file is a path the user owns;
* **no silent downloads** — the provider *never* calls Ultralytics' hub
  downloader. If the weight file does not exist, ``prepare``/``analyze``
  raise a clear error instead of fetching anything;
* **isolated** — everything YOLO-specific lives here.

Prompts map to class names: each prompt is matched (case-insensitively)
against the model's ``names`` table. Detections whose class name matches at
least one prompt are returned, labelled with the class name. A model without
a ``names`` table returns labelled boxes only for prompts passed as class
indices (integers).
"""

from __future__ import annotations

import importlib
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import numpy as np

from perception.types import BoundingBox, Detection

YOLO_MODEL_WEIGHTS = (
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov9c.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
)


def _backend_importable() -> bool:
    try:
        importlib.import_module("ultralytics")
        return True
    except ImportError:
        return False


class YoloProvider:
    """Detect objects with an Ultralytics YOLO model (fixed class vocabulary).

    ``model`` is a path to a weights file (``.pt``). Unlike the configure-to-
    download hub names, GRUI does not download YOLO weights: pass an existing
    file (e.g. ``yolov8n.pt`` downloaded by you from
    https://github.com/ultralytics/assets/releases).
    """

    name = "yolo"
    description = "Fixed-vocabulary object detection with Ultralytics YOLO (class-name prompts)."
    install_hint = (
        'pip install "grui[yolo]"   (ultralytics). YOLO weights are NOT '
        "downloaded by GRUI: download a .pt file yourself, e.g. yolov8n.pt "
        "from https://github.com/ultralytics/assets/releases, and pass --model."
    )
    warnings = [
        "YOLO detects only the fixed class vocabulary baked into the weights (typically COCO 80 classes).",
        "Weights are not bundled or downloaded by GRUI — provide your own .pt file.",
        "CPU inference works but is slow; a CUDA GPU (ultralytics picks it up automatically) is recommended.",
    ]
    model = "user-provided .pt weights file"

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        *,
        conf: float = 0.25,
        device: str | None = None,
        allow_download: bool = False,
        detector: Any | None = None,
        names: dict[int, str] | None = None,
        default_labels: list[str] | None = None,
    ) -> None:
        self._model_path = str(model)
        self._conf = float(conf)
        self._device = device
        self._allow_download = bool(allow_download)
        self._detector = detector
        self._names: dict[int, str] | None = names
        self._default_labels = list(default_labels) if default_labels else None

    @property
    def version(self) -> str:
        try:
            return metadata.version("ultralytics")
        except PackageNotFoundError:
            return "not-installed"

    def is_available(self) -> bool:
        """Import-level availability only — never loads/creates a model."""
        return _backend_importable()

    def with_options(self, **options: Any) -> "YoloProvider":
        model = options.get("model")
        conf = options.get("conf")
        device = options.get("device")
        if (
            model in (None, self._model_path)
            and conf in (None, self._conf)
            and device in (None, self._device)
        ):
            return self
        return YoloProvider(
            model=self._model_path if model is None else model,
            conf=self._conf if conf is None else conf,
            device=self._device if device is None else device,
            allow_download=self._allow_download,
            detector=self._detector,
            names=self._names,
            default_labels=self._default_labels,
        )

    # ------------------------------------------------------------ loading

    def prepare(self) -> None:
        """Load the model (validates the weights file exists first)."""
        self._load()

    def _load(self) -> Any:
        if self._detector is not None:
            return self._detector
        path = Path(self._model_path)
        if not self._allow_download:
            if not path.exists():
                raise RuntimeError(
                    f"YOLO weights file not found: {path}\n"
                    "GRUI does not download YOLO weights. Download a .pt model "
                    "(e.g. yolov8n.pt from https://github.com/ultralytics/assets/releases) "
                    "and pass --model <path-to-weights>. "
                    "Alternatively pass --allow-download to let ultralytics fetch it."
                )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO provider needs the ultralytics package:\n"
                '    pip install "grui[yolo]"'
            ) from exc
        try:
            self._detector = YOLO(str(path), task="detect")
        except Exception as exc:  # noqa: BLE001 - surface load failures clearly
            raise RuntimeError(f"could not load YOLO model {path}: {exc}") from exc
        if self._names is None and hasattr(self._detector, "names"):
            try:
                names = self._detector.names
                if isinstance(names, dict):
                    self._names = {int(k): str(v) for k, v in names.items()}
            except Exception:  # noqa: BLE001
                self._names = None
        return self._detector

    # ------------------------------------------------------------ inference

    def analyze(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        detector = self._load()
        names = self._names or {}
        wanted: set[int] = set()
        prompt_labels: list[str] = []
        default_labels = self._default_labels or []
        for prompt in prompts:
            as_index = _as_class_index(prompt)
            if as_index is not None and as_index in names:
                wanted.add(as_index)
                prompt_labels.append(names[as_index])
            else:
                for index, name in names.items():
                    if name.lower() == prompt.strip().lower():
                        wanted.add(index)
                        prompt_labels.append(name)
        valid = [i for i in wanted if i in names]
        if valid:
            result = detector.predict(
                source=_to_rgb(frame),
                conf=self._conf,
                device=self._device,
                classes=list(valid),
                verbose=False,
            )
            detections = result[0] if result else None
            if detections is None:
                return []
            boxes = detections.boxes if detections is not None and hasattr(detections, "boxes") else None
            if boxes is None:
                return []
            out: list[Detection] = []
            for box in boxes:
                cls_idx = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                if cls_idx not in valid:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                out.append(
                    Detection(
                        label=names.get(cls_idx, prompt_labels[valid.index(cls_idx)] if cls_idx in valid else f"class_{cls_idx}"),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=conf,
                        source="model",
                    )
                )
            return out
        # No class-name match: fall back to `default_labels` (all classes) when
        # the user configured them, otherwise return nothing for unknown prompts.
        if default_labels and self._names:
            valid = [i for i, n in self._names.items() if n.lower() in {l.lower() for l in default_labels}]
            if not valid:
                return []
            result = detector.predict(
                source=_to_rgb(frame), conf=self._conf, device=self._device,
                classes=list(valid), verbose=False,
            )
            boxes = result[0].boxes if result else None
            if boxes is None:
                return []
            return [
                Detection(
                    label=self._names.get(int(b.cls[0]), f"class_{int(b.cls[0])}"),
                    bbox=BoundingBox(x1=float(b.xyxy[0][0]), y1=float(b.xyxy[0][1]),
                                     x2=float(b.xyxy[0][2]), y2=float(b.xyxy[0][3])),
                    confidence=float(b.conf[0]) if hasattr(b.conf, "__len__") else float(b.conf),
                    source="model",
                )
                for b in boxes
            ]
        return []

    @staticmethod
    def _names_from_labels(labels: list[str]) -> dict[int, str]:
        return {i: label for i, label in enumerate(labels)}


def _as_class_index(prompt: str) -> int | None:
    text = prompt.strip()
    if text.isdigit():
        return int(text)
    return None


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR (GRUI's convention) to RGB for ultralytics."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


__all__ = ["YoloProvider", "_backend_importable"]