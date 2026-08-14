"""Perception runner: analyze an existing recording with a provider.

Analysis is a pure, derived operation over a finished recording::

    existing recording
            |
            v
    frames.jsonl          -> frame timestamps (source of truth)
            |
            v
    select sampled frames -> decode from video.mp4 (never re-recorded)
            |
            v
    provider.analyze(...)  -> detections
            |
            v
    recordings/<session>/perception/
        manifest.json
        results.jsonl

Nothing in the raw recording is ever modified. ``results.jsonl`` keeps
``frame_index`` and ``t`` taken from ``frames.jsonl`` (the same clock
the dataset builder and player use), one JSON record per (frame, prompt).

Runs are cached: if the same recording + provider + prompts + sampling
were analyzed before (identical manifest) the existing results are
returned without recomputation.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from perception.base import PerceptionProvider
from perception.types import BoundingBox, PerceptionManifest, PerceptionResult
from storage.recording import RecordingData, load_recording

_FORMAT_VERSION = 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def every_for_fps(recording_fps: float, fps: float) -> int:
    """``--fps F`` sampling: the stride that yields ~F frames per second."""
    if fps <= 0:
        raise ValueError(f"--fps must be > 0 (got {fps})")
    return max(1, int(round(recording_fps / fps)))


def select_frame_indices(
    frame_count: int,
    recording_fps: float,
    *,
    every: int | None = None,
    fps: float | None = None,
) -> list[int]:
    """Sampled frame indices within ``frame_count`` frames.

    ``every`` takes precedence when both are given; ``fps`` is converted
    with :func:`every_for_fps`.
    """
    if every is None and fps is not None:
        every = every_for_fps(recording_fps, fps)
    if every is None:
        raise ValueError("give either --every N or --fps F")
    if every < 1:
        raise ValueError(f"sampling stride must be >= 1 (got {every})")
    return list(range(0, frame_count, every))


def _decode_frames(
    video_path: Path, wanted: list[int], max_pixels: int | None = None
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int]]]:
    """Decode the listed frame indices in one sequential pass (BGR).

    With ``max_pixels`` set, frames larger than that many pixels are
    downscaled (aspect-ratio preserved) so the vision encoder uses less
    VRAM; the original ``(width, height)`` of every decoded frame is
    returned too, so :func:`_rescale` can map detections back to
    original-frame coordinates before the results are written.
    """
    import cv2

    wanted_set = set(wanted)
    frames: dict[int, np.ndarray] = {}
    orig_sizes: dict[int, tuple[int, int]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    index = 0
    try:
        while wanted_set:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                orig_sizes[index] = (frame.shape[1], frame.shape[0])
                frames[index] = _downscale(frame, max_pixels)
                wanted_set.discard(index)
            index += 1
    finally:
        cap.release()
    if wanted_set:
        raise ValueError(
            f"video {video_path} ended before frames "
            f"{sorted(wanted_set)[:5]}{'...' if len(wanted_set) > 5 else ''} "
            f"({len(frames)}/{len(wanted)} decoded)"
        )
    return frames, orig_sizes


def _downscale(frame: np.ndarray, max_pixels: int | None) -> np.ndarray:
    """Aspect-preserving resize of ``frame`` to at most ``max_pixels``."""
    if not max_pixels or max_pixels < 1:
        return frame
    import cv2

    h, w = frame.shape[:2]
    if w * h <= max_pixels:
        return frame
    scale = (max_pixels / (w * h)) ** 0.5
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _rescale(
    detections: list[Any], from_size: tuple[int, int], to_size: tuple[int, int]
) -> list[Any]:
    """Scale Detection bboxes from ``from_size`` into ``to_size`` (w, h)."""
    from_w, from_h = from_size
    to_w, to_h = to_size
    if (from_w, from_h) == (to_w, to_h):
        return detections
    sx, sy = to_w / from_w, to_h / from_h
    scaled = []
    for detection in detections:
        if detection.bbox is None:
            scaled.append(detection)
            continue
        bbox = detection.bbox
        scaled.append(
            dataclasses.replace(
                detection,
                bbox=BoundingBox(
                    x1=bbox.x1 * sx,
                    y1=bbox.y1 * sy,
                    x2=bbox.x2 * sx,
                    y2=bbox.y2 * sy,
                ),
            )
        )
    return scaled


def _manifest(
    recording: RecordingData,
    provider: PerceptionProvider,
    sampling: dict[str, Any],
    prompts: list[str],
    count: int,
) -> PerceptionManifest:
    return PerceptionManifest(
        format_version=_FORMAT_VERSION,
        provider=str(provider.name),
        provider_version=str(provider.version),
        model=getattr(provider, "model", None),
        source_session_id=recording.session_id,
        source_recording=recording.directory.name,
        sampling=sampling,
        prompts=list(prompts),
        count=count,
    )


class CachedAnalysis:
    """The on-disk outcome of one perception run."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.results_path = self.directory / "results.jsonl"

    @property
    def exists(self) -> bool:
        return self.manifest_path.exists() and self.results_path.exists()

    def read_manifest(self) -> PerceptionManifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            return PerceptionManifest.from_dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        except (ValueError, KeyError):
            return None

    def read_results(self) -> list[PerceptionResult]:
        return [PerceptionResult.from_dict(row) for row in _read_jsonl(self.results_path)]


def analyze_recording(
    recording: RecordingData | Path | str,
    provider: PerceptionProvider,
    prompts: list[str],
    *,
    every: int | None = None,
    fps: float | None = None,
    max_pixels: int | None = None,
    out_dir: Path | str | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> CachedAnalysis:
    """Analyze an existing recording with a perception provider.

    Returns the derived ``perception/`` directory inside the recording
    (``manifest.json`` + ``results.jsonl``). If a matching cached run
    already exists it is returned without re-running the model, unless
    ``force`` is set. Raises ``ValueError`` for bad input and
    ``RuntimeError`` when the provider cannot run.
    """
    if isinstance(recording, (str, Path)):
        recording = load_recording(recording)
    if max_pixels is not None and max_pixels < 1:
        raise ValueError(f"max_pixels must be >= 1 (got {max_pixels})")
    prompts = [str(p).strip() for p in prompts]
    if not prompts or any(not p for p in prompts):
        raise ValueError("give at least one non-empty prompt")
    seen: set[str] = set()
    prompts = [p for p in prompts if not (p in seen or seen.add(p))]
    if recording.frame_times.size == 0:
        raise ValueError(f"recording has no frames.jsonl: {recording.directory}")
    if not recording.video_path.exists():
        raise ValueError(f"recording has no video: {recording.video_path}")

    frame_count = recording.frame_times.size
    indices = select_frame_indices(frame_count, recording.fps, every=every, fps=fps)
    if not indices:
        raise ValueError("no frames selected for perception analysis")
    if every is None and fps is not None:
        stride = every_for_fps(recording.fps, fps)
    else:
        stride = max(1, int(every))
    sampling = {
        "fps": float(fps) if fps is not None else float(recording.fps / max(1, stride)),
        "every": stride,
        "frames": len(indices),
        "max_pixels": max_pixels,
    }

    out = CachedAnalysis(Path(out_dir) if out_dir else recording.directory / "perception")
    expected_count = len(indices) * len(prompts)  # one record per (frame, prompt)
    for stale in (out.directory / "results.jsonl.tmp", out.directory / "manifest.json.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass
    if not force:
        existing = out.read_manifest()
        wanted = _manifest(recording, provider, sampling, prompts, expected_count)
        if existing is not None and existing.to_dict() == wanted.to_dict():
            log(f"cached perception results: {out.results_path}")
            return out

    if not provider.is_available():
        raise RuntimeError(
            f"perception provider {provider.name!r} is unavailable on this machine. "
            f"Install its optional dependencies first (see the provider's docs)."
        )

    # Load the model (download weights on first use) BEFORE creating any
    # artifacts, so a slow or failing load never leaves a partial run behind.
    prepare = getattr(provider, "prepare", None)
    if callable(prepare):
        prepare()
    log(f"analyzing {len(indices)} frames x {len(prompts)} prompts with {provider.name!r}…")

    out.directory.mkdir(parents=True, exist_ok=True)
    tmp_results = out.results_path.with_name("results.jsonl.tmp")
    tmp_manifest = out.manifest_path.with_name("manifest.json.tmp")
    started = time.monotonic()
    records = 0
    try:
        frames, orig_sizes = _decode_frames(
            recording.video_path, indices, max_pixels=max_pixels
        )
        with tmp_results.open("w", encoding="utf-8") as fh:
            for index in sorted(frames):
                frame = frames[index]
                height, width = frame.shape[:2]
                orig_width, orig_height = orig_sizes[index]
                for prompt in prompts:
                    detections = provider.analyze(frame, [prompt])
                    detections = _rescale(
                        detections, (width, height), (orig_width, orig_height)
                    )
                    result = PerceptionResult(
                        frame_index=index,
                        t=float(recording.frame_time(index)),
                        prompt=prompt,
                        detections=list(detections),
                    )
                    fh.write(json.dumps(result.to_dict()) + "\n")
                    records += 1
        manifest = _manifest(recording, provider, sampling, prompts, expected_count)
        tmp_manifest.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_results.replace(out.results_path)
        tmp_manifest.replace(out.manifest_path)
    except BaseException:
        for tmp in (tmp_results, tmp_manifest):
            try:
                tmp.unlink()
            except OSError:
                pass
        if records == 0 and not out.exists:
            try:
                out.directory.rmdir()  # remove the empty dir we just created
            except OSError:
                pass
        raise
    log(
        f"analyzed {len(frames)} frames x {len(prompts)} prompts "
        f"({records} records) -> {out.results_path} "
        f"({time.monotonic() - started:.1f}s)"
    )
    return out
