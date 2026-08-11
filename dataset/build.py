"""Convert a raw recording into temporally aligned observation->action samples.

For each sample time ``t`` (every ``stride`` seconds) the observation is the
video window ``[t - observation_duration, t]`` resampled to ``fps`` by
picking the nearest recorded frames; the action is the human input state at
``t``: held keys, held mouse buttons, and the pointer position with
``dx``/``dy`` since the previous sample. All times are in the recording's
session clock (seconds since session start) — the same domain as
``frames.jsonl`` and ``events.jsonl``.

Frames shared by overlapping windows are decoded exactly once in a single
sequential video pass and written as ``frames/frame_<index>.png``; samples
reference frame indices, so nothing is duplicated on disk. The result is
fully deterministic: the same recording and config produce byte-identical
output. ``prediction_horizon`` is not used for sampling (the observation
ends at the action time) — it is recorded in the manifest for downstream
trainers that offset the target.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from player.event_state import KeyStateTimeline
from storage.recording import RecordingData

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-generation parameters (not part of the raw recording)."""

    observation_duration: float = 3.0  # seconds of video history per sample
    fps: float = 15.0  # observation sampling rate
    stride: float = 1.0  # seconds between consecutive sample times
    prediction_horizon: float = 0.2  # seconds between observation end and target

    def validate(self) -> None:
        if self.observation_duration <= 0 or self.fps <= 0 or self.stride <= 0:
            raise ValueError(
                f"invalid dataset config: observation_duration, fps and stride must be > 0 "
                f"(got {self.observation_duration}, {self.fps}, {self.stride})"
            )
        if self.prediction_horizon < 0:
            raise ValueError(
                f"invalid dataset config: prediction_horizon must be >= 0 (got {self.prediction_horizon})"
            )


def _sample_times(t0: float, t1: float, stride: float) -> list[float]:
    """Ascending sample times in ``[t0, t1]``, one every ``stride`` seconds."""
    if t1 <= t0:
        return []
    count = int(math.floor((t1 - t0 + 1e-9) / stride)) + 1
    return [t0 + k * stride for k in range(count)]


def _observation_indices(recording: RecordingData, t: float, config: DatasetConfig) -> list[int]:
    """Nearest frame indices covering ``[t - duration, t]`` at ``config.fps``."""
    n_obs = int(round(config.observation_duration * config.fps))
    indices = []
    for k in range(n_obs + 1):
        tau = t - config.observation_duration + k / config.fps
        indices.append(recording.nearest_frame_index(tau))
    return indices


def _decode_frames(video_path: Path, needed: list[int]) -> dict[int, Any]:
    """Decode the listed frame indices in one sequential pass (BGR ndarrays)."""
    wanted = set(needed)
    frames: dict[int, Any] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    index = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = frame
                wanted.discard(index)
            index += 1
    finally:
        cap.release()
    if wanted:
        raise ValueError(
            f"video {video_path} ended before frames {sorted(wanted)[:5]}{'...' if len(wanted) > 5 else ''} "
            f"({len(frames)}/{len(needed)} decoded)"
        )
    return frames


def build_dataset(
    recording: RecordingData,
    config: DatasetConfig,
    out_dir: Path | str,
) -> Path:
    """Build a dataset from one raw recording. Returns the output directory.

    Raises ``ValueError`` for invalid config or unusable recordings.
    """
    config.validate()
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.iterdir():
        if stale.is_file():
            stale.unlink()

    if not recording.video_path.exists():
        raise ValueError(f"recording has no video: {recording.video_path}")
    if recording.frame_times.size == 0:
        raise ValueError(f"recording has no frames.jsonl: {recording.directory}")

    t0 = float(recording.frame_times[0])
    t1 = recording.duration
    if t1 - t0 < config.observation_duration:
        raise ValueError(
            f"recording is too short: {t1 - t0:.2f}s of frames, need at least "
            f"{config.observation_duration:.2f}s for observation windows"
        )

    times = _sample_times(t0 + config.observation_duration, t1, config.stride)
    needed = sorted({i for t in times for i in _observation_indices(recording, t, config)})
    frames = _decode_frames(recording.video_path, needed)

    keys = KeyStateTimeline(recording.events)
    prev_pos: tuple[int, int] | None = None

    samples: list[dict[str, Any]] = []
    for t in times:
        pos = keys.mouse_at(t)
        if pos is None:
            mouse = None
        else:
            dx = pos[0] - prev_pos[0] if prev_pos is not None else 0
            dy = pos[1] - prev_pos[1] if prev_pos is not None else 0
            mouse = {"x": pos[0], "y": pos[1], "dx": dx, "dy": dy}
            prev_pos = pos
        samples.append(
            {
                "t": t,
                "observation": _observation_indices(recording, t, config),
                "action": {
                    "keys": sorted(keys.active_keys_at(t)),
                    "buttons": sorted(keys.active_buttons_at(t)),
                    "mouse": mouse,
                },
            }
        )

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample) + "\n")

    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as fh:
        for index in sorted(frames):
            fh.write(
                json.dumps(
                    {
                        "frame_index": index,
                        "t": recording.frame_time(index),
                        "path": f"frames/frame_{index}.png",
                    }
                )
                + "\n"
            )
    for index, frame in frames.items():
        cv2.imwrite(str(frames_dir / f"frame_{index}.png"), frame)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": _FORMAT_VERSION,
                "source": {
                    "session_id": recording.session_id,
                    "recording_dir": str(recording.directory),
                },
                "config": asdict(config),
                "count": len(samples),
                "observation_frames": len(frames),
                "screen": {"width": recording.width, "height": recording.height},
                "time_base": "seconds since session start",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir
