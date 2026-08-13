"""LocateAnything enrichment: text-prompted localization over a dataset.

Runs a LocateAnything-3B worker (GUI grounding, object detection, pointing,
text detection) over sampled frames of a built dataset and writes
``locations.jsonl``: one JSON record per (frame, prompt) with the boxes and
points the model returned.

The model is a heavy, gated, GPU-only dependency — ~6 GB download, Hugging
Face login, CUDA with several GB of VRAM, and seconds per frame. It is
loaded lazily (``load_locator``), and the CLI prints the costs up front
unless ``--iknow`` is passed. Two backends are supported:

* the ``locate-anything`` PyPI wrapper (``pip install "grui[locate]"``)
  — object detection with categories only;
* the Eagle repo's ``LocateAnythingWorker`` (``locateanything_worker.py``
  on ``PYTHONPATH``) — full task support including GUI grounding.

VRAM can be cut via ``--quantize 8bit|4bit`` (bitsandbytes; ~4 GB / ~2 GB
instead of ~8 GB at bf16, loaded in-repo with Transformers so either backend
works) and ``--max-pixels N`` (downscales frames before inference to shrink
the vision encoder; boxes are rescaled to the original frame size on output).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

WARNINGS = [
    "LocateAnything-3B is a ~6 GB model, gated on Hugging Face (`huggingface-cli login`).",
    "It needs a CUDA GPU; ~6-8 GB VRAM at bf16, ~2-4 GB with --quantize 4bit/8bit.",
    "Inference is slow (a 3B VLM): frames are sampled every --every N, not all.",
    "This uses research/community code; localization quality varies by prompt.",
]

_TASKS = ("ground_gui", "detect", "point", "detect_text")


class Locator:
    """Adapter over a LocateAnything worker: image + prompt -> boxes/points.

    Subclasses must implement :meth:`locate`, returning
    ``{"boxes": [{"x1", "y1", "x2", "y2"}], "points": [{"x", "y"}]}``.
    """

    def locate(self, image, prompt: str, task: str) -> dict:
        raise NotImplementedError


class PipLocator(Locator):
    """Wraps the ``locate-anything`` PyPI package (detect only)."""

    def __init__(self, client) -> None:
        self._client = client

    def locate(self, image, prompt: str, task: str) -> dict:
        if task not in ("ground_gui", "detect"):
            raise ValueError(
                f"task {task!r} needs the full LocateAnythingWorker "
                "(add the Eagle repo's locateanything_worker.py to PYTHONPATH)"
            )
        result = self._client.detect(image, categories=[prompt], draw=False)
        boxes = [
            {"x1": float(d["bbox_pixels"][0]), "y1": float(d["bbox_pixels"][1]),
             "x2": float(d["bbox_pixels"][2]), "y2": float(d["bbox_pixels"][3])}
            for d in result.get("detections", [])
        ]
        return {"boxes": boxes, "points": []}


class WorkerLocator(Locator):
    """Wraps the Eagle repo's ``LocateAnythingWorker`` (all tasks)."""

    def __init__(self, worker) -> None:
        self._worker = worker

    def locate(self, image, prompt: str, task: str) -> dict:
        width, height = image.size
        if task == "ground_gui":
            answer = self._worker.ground_gui(image, prompt)["answer"]
        elif task == "detect":
            answer = self._worker.detect(image, [prompt])["answer"]
        elif task == "point":
            answer = self._worker.point(image, prompt)["answer"]
        elif task == "detect_text":
            answer = self._worker.detect_text(image)["answer"]
        else:
            raise ValueError(f"unknown task: {task}")
        boxes = [
            {"x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]}
            for b in self._worker.parse_boxes(answer, width, height)
        ]
        points = [
            {"x": p["x"], "y": p["y"]}
            for p in self._worker.parse_points(answer, width, height)
        ]
        return {"boxes": boxes, "points": points}


class HfLocateAnythingWorker:
    """In-repo LocateAnything worker with bitsandbytes quantization.

    Mirrors the Eagle repo's ``LocateAnythingWorker`` surface (the methods
    :class:`WorkerLocator` calls) but loads ``nvidia/LocateAnything-3B``
    directly with Transformers so ``load_in_4bit``/``load_in_8bit`` can be
    applied, cutting inference VRAM from ~8 GB (bf16) to ~2-4 GB. Used by
    :func:`load_locator` whenever quantization is requested, regardless of
    which external backend is installed.

    Requires ``transformers`` (and ``bitsandbytes`` for the quantized load),
    which the model needs anyway. Constructing it is the expensive part —
    pass the instance around, don't rebuild it per frame.
    """

    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        quantize: str = "4bit",
        max_tokens: int = 1024,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.quantize = quantize
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        kwargs: dict = {"trust_remote_code": True}
        if quantize == "4bit":
            kwargs.update(load_in_4bit=True, device_map="auto")
        elif quantize == "8bit":
            kwargs.update(load_in_8bit=True, device_map="auto")
        else:  # "none"
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_path, **kwargs)
        if quantize == "none":
            self.model.to(device)
        self.model.eval()

    def predict(self, image, question: str, **kwargs) -> dict:
        """Run one perception query; returns ``{"answer": ...}``."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        apply_template = getattr(
            self.processor, "py_apply_chat_template", None
        ) or self.processor.apply_chat_template
        text = apply_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.model.dtype)
        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                generation_mode="hybrid",
                do_sample=False,
                verbose=False,
            )
        return {"answer": response[0] if isinstance(response, tuple) else response}

    # ---- Task conveniences (same prompts as the Eagle worker) ----

    def detect(self, image, categories: list[str], **kwargs) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image, phrase: str, **kwargs) -> dict:
        return self.predict(image, f"Point to: {phrase}.", **kwargs)

    def detect_text(self, image, **kwargs) -> dict:
        return self.predict(image, "Detect all the text in box format.", **kwargs)

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points


def load_locator(
    device: str = "cuda",
    quantize: str = "none",
    max_tokens: int = 1024,
) -> Locator:
    """Load a LocateAnything backend.

    ``quantize`` is ``"none"`` (bf16, ~8 GB VRAM), ``"8bit"`` (~4 GB) or
    ``"4bit"`` (~2 GB). The quantized path uses :class:`HfLocateAnythingWorker`
    so it works whether the PyPI wrapper or the Eagle worker is installed;
    unquantized runs keep the existing backend negotiation.
    """
    if quantize != "none":
        try:
            worker = HfLocateAnythingWorker(
                device=device, quantize=quantize, max_tokens=max_tokens
            )
        except ImportError as exc:
            raise RuntimeError(
                "Quantized LocateAnything needs the Transformers/bitsandbytes stack:\n"
                "  pip install transformers bitsandbytes accelerate\n"
                f"    ({exc})"
            ) from exc
        return WorkerLocator(worker)
    try:
        from locate_anything import LocateAnything
    except ImportError:  # backend not installed; try the next one
        pass
    else:
        # the PyPI wrapper picks the best device itself (device_map="auto");
        # it has no `device` kwarg, so nothing is passed here.
        return PipLocator(LocateAnything())
    try:
        from locateanything_worker import LocateAnythingWorker
    except ImportError:
        pass
    else:
        return WorkerLocator(LocateAnythingWorker("nvidia/LocateAnything-3B", device=device))
    raise RuntimeError(
        "LocateAnything is not installed.\n"
        "  pip install \"grui[locate]\"        # PyPI wrapper (detect task)\n"
        "or put the Eagle repo's locateanything_worker.py on PYTHONPATH\n"
        "  (full tasks). The nvidia/LocateAnything-3B model is gated:\n"
        "  run `huggingface-cli login` once first."
    )


def _load_image(path: Path, max_pixels: int | None = None):
    """Load a frame as RGB PIL, optionally downscaled.

    Returns ``(image, orig_size)`` where ``orig_size`` is the on-disk
    ``(width, height)``. When ``max_pixels`` is set, larger frames are
    resized (aspect-ratio preserved) to fit within that many pixels so the
    vision encoder uses less VRAM; callers scale results back with
    :func:`_scale_result`.
    """
    import cv2
    from PIL import Image

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"could not read frame image: {path}")
    orig_size = (frame.shape[1], frame.shape[0])  # (width, height)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if max_pixels and frame.shape[1] * frame.shape[0] > max_pixels:
        h, w = frame.shape[:2]
        scale = (max_pixels / (w * h)) ** 0.5
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(frame), orig_size


def _scale_result(result: dict, from_size, to_size) -> dict:
    """Rescale boxes/points from ``from_size`` into ``to_size`` (w, h) space."""
    from_w, from_h = from_size
    to_w, to_h = to_size
    if (from_w, from_h) == (to_w, to_h):
        return result
    sx, sy = to_w / from_w, to_h / from_h
    return {
        "boxes": [
            {"x1": b["x1"] * sx, "y1": b["y1"] * sy,
             "x2": b["x2"] * sx, "y2": b["y2"] * sy}
            for b in result["boxes"]
        ],
        "points": [{"x": p["x"] * sx, "y": p["y"] * sy} for p in result["points"]],
    }


def enrich_dataset(
    dataset_dir: Path | str,
    prompts: list[str],
    task: str,
    *,
    every: int = 10,
    out: Path | str | None = None,
    locator: Locator | None = None,
    max_pixels: int | None = None,
    log=print,
) -> Path:
    """Locate each prompt on every Nth frame; returns the output path."""
    if task not in _TASKS:
        raise ValueError(f"task must be one of {_TASKS} (got {task!r})")
    if not prompts:
        raise ValueError("give at least one prompt")
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    frames_path = root / "frames.jsonl"
    if not manifest_path.exists() or not frames_path.exists():
        raise ValueError(f"not a built dataset: {root}")
    entries = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
    ]
    sampled = [e for i, e in enumerate(entries) if i % max(1, every) == 0]
    if not sampled:
        raise ValueError(f"no frames to process in {root}")
    locator = locator or load_locator()
    out_path = Path(out) if out else root / "locations.jsonl"
    started = time.monotonic()
    records = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in sampled:
            image, orig_size = _load_image(root / entry["path"], max_pixels=max_pixels)
            for prompt in prompts:
                result = locator.locate(image, prompt, task)
                if max_pixels:
                    result = _scale_result(result, image.size, orig_size)
                fh.write(
                    json.dumps(
                        {
                            "frame_index": entry["frame_index"],
                            "t": entry["t"],
                            "prompt": prompt,
                            "task": task,
                            "boxes": result["boxes"],
                            "points": result["points"],
                        }
                    )
                    + "\n"
                )
                records += 1
    log(
        f"located {len(sampled)} frames x {len(prompts)} prompts "
        f"({records} records) -> {out_path} ({time.monotonic() - started:.1f}s)"
    )
    return out_path


def _marker_prompts(dataset_dir: Path | str) -> list[str]:
    """Marker labels from the source recording, usable as prompts."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    recording_dir = manifest.get("source", {}).get("recording_dir")
    if not recording_dir or not Path(recording_dir).exists():
        return []
    from storage.recording import load_recording

    recording = load_recording(recording_dir)
    return [str(m.get("label")) for m in recording.markers if m.get("label")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui locate",
        description="Enrich a dataset with LocateAnything boxes/points (locations.jsonl).",
    )
    parser.add_argument("--dataset", required=True, metavar="DIR", help="built dataset directory")
    parser.add_argument("--prompt", action="append", metavar="TEXT",
                        help="localization prompt (repeatable)")
    parser.add_argument("--markers", action="store_true",
                        help="also use the source recording's marker labels as prompts")
    parser.add_argument("--task", default="ground_gui", choices=_TASKS,
                        help="ground_gui (default), detect, point or detect_text")
    parser.add_argument("--every", type=int, default=10,
                        help="process every Nth frame (default: 10)")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="output path (default: <dataset>/locations.jsonl)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu (cpu is not practical)")
    parser.add_argument(
        "--quantize", default="none", choices=("none", "8bit", "4bit"),
        help="bitsandbytes weight quantization to cut VRAM: 8bit ~4 GB, "
             "4bit ~2 GB (default: none / bf16 ~8 GB; needs transformers+bitsandbytes)",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=None, metavar="N",
        help="downscale frames to at most N pixels before inference to shrink the "
             "vision encoder's VRAM; boxes/points are rescaled to the original "
             "frame size in the output",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="max new tokens per inference (default: 1024)",
    )
    parser.add_argument("--iknow", action="store_true",
                        help="skip the model-cost warnings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prompts = list(args.prompt or [])
        if args.markers:
            prompts += _marker_prompts(args.dataset)
        seen: set[str] = set()
        prompts = [p for p in prompts if not (p in seen or seen.add(p))]
        if not prompts:
            print("error: give --prompt TEXT (or --markers)", file=sys.stderr)
            return 2
        if not args.iknow:
            for warning in WARNINGS:
                print(f"warning: {warning}", file=sys.stderr)
            try:
                answer = input("Continue? [y/N] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer != "y":
                print("aborted.", file=sys.stderr)
                return 1
        locator = load_locator(args.device, args.quantize, args.max_tokens)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        out = enrich_dataset(
            args.dataset, prompts, args.task,
            every=args.every, out=args.out, locator=locator,
            max_pixels=args.max_pixels,
        )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"locations: {out}")
    return 0
