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

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from perception.base import PerceptionProvider
from perception.types import PerceptionManifest, PerceptionResult
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


def _decode_frames(video_path: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode the listed frame indices in one sequential pass (BGR)."""
    import cv2

    wanted_set = set(wanted)
    frames: dict[int, np.ndarray] = {}
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
                frames[index] = frame
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
    return frames


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
    }

    out = CachedAnalysis(Path(out_dir) if out_dir else recording.directory / "perception")
    expected_count = len(indices) * len(prompts)  # one record per (frame, prompt)
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

    out.directory.mkdir(parents=True, exist_ok=True)
    frames = _decode_frames(recording.video_path, indices)
    started = time.monotonic()
    records: list[PerceptionResult] = []
    with out.results_path.open("w", encoding="utf-8") as fh:
        for index in sorted(frames):
            frame = frames[index]
            for prompt in prompts:
                detections = provider.analyze(frame, [prompt])
                result = PerceptionResult(
                    frame_index=index,
                    t=float(recording.frame_time(index)),
                    prompt=prompt,
                    detections=list(detections),
                )
                fh.write(json.dumps(result.to_dict()) + "\n")
                records.append(result)
    manifest = _manifest(recording, provider, sampling, prompts, expected_count)
    out.manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"analyzed {len(frames)} frames x {len(prompts)} prompts "
        f"({len(records)} records) -> {out.results_path} "
        f"({time.monotonic() - started:.1f}s)"
    )
    return out
