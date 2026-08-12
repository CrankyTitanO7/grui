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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WARNINGS = [
    "LocateAnything-3B is a ~6 GB model, gated on Hugging Face (`huggingface-cli login`).",
    "It needs a CUDA GPU with ~6-8 GB VRAM; CPU inference is not practical.",
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


def load_locator(device: str = "cuda") -> Locator:
    """Load a LocateAnything backend. Raises ``RuntimeError`` with setup hints."""
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


def _load_image(path: Path):
    import cv2
    from PIL import Image

    frame = cv2.imread(str(path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def enrich_dataset(
    dataset_dir: Path | str,
    prompts: list[str],
    task: str,
    *,
    every: int = 10,
    out: Path | str | None = None,
    locator: Locator | None = None,
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
            image = _load_image(root / entry["path"])
            for prompt in prompts:
                result = locator.locate(image, prompt, task)
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
        locator = load_locator(args.device)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        out = enrich_dataset(
            args.dataset, prompts, args.task,
            every=args.every, out=args.out, locator=locator,
        )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"locations: {out}")
    return 0
