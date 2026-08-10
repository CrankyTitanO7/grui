"""Raw demonstration directory, metadata management and read-side model.

The raw recording format is versioned from the beginning because future
dataset-generation algorithms must be able to consume old recordings::

    recordings/<YYYY-MM-DD_HH-MM-SS>_<session-id>/
        metadata.json   # format version, session info, screen config
        video.mp4       # encoded screen capture
        events.jsonl    # keyboard / mouse / lifecycle events
        markers.jsonl   # human annotations
        frames.jsonl    # frame_index -> capture time (for exact sync)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


def _read_frame_times(path: Path) -> np.ndarray:
    """Frame capture times indexed by frame index (from ``frames.jsonl``)."""
    entries = _read_jsonl(path)
    if not entries:
        return np.array([], dtype=np.float64)
    size = max(int(e["frame_index"]) for e in entries) + 1
    times = np.zeros(size, dtype=np.float64)
    for e in entries:
        times[int(e["frame_index"])] = float(e["t"])
    return times


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (write to temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class RawRecording:
    """The files that make up one raw demonstration."""

    FORMAT_VERSION = 1

    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self._metadata = metadata

    @classmethod
    def create(cls, root: Path, session_id: str, metadata: dict[str, Any]) -> "RawRecording":
        """Create a fresh recording directory.

        The directory is named ``<timestamp>_<session_id>``; if that name is
        already taken (e.g. a retried session in the same second) a numeric
        suffix is appended so no recording ever overwrites another.
        """
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = root / f"{stamp}_{session_id}"
        counter = 2
        while directory.exists():
            directory = root / f"{stamp}_{session_id}_{counter}"
            counter += 1
        directory.mkdir(parents=True, exist_ok=False)
        recording = cls(directory, dict(metadata))
        _write_json_atomic(recording.metadata_path, recording._metadata)
        return recording

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def video_path(self) -> Path:
        return self.directory / "video.mp4"

    @property
    def events_path(self) -> Path:
        return self.directory / "events.jsonl"

    @property
    def markers_path(self) -> Path:
        return self.directory / "markers.jsonl"

    @property
    def frames_path(self) -> Path:
        return self.directory / "frames.jsonl"

    def files(self) -> dict[str, Path]:
        """All recording files by logical name."""
        return {
            "metadata": self.metadata_path,
            "video": self.video_path,
            "events": self.events_path,
            "markers": self.markers_path,
            "frames": self.frames_path,
        }

    def read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def update_metadata(self, **fields: Any) -> None:
        """Merge fields into the metadata and rewrite it atomically."""
        self._metadata.update(fields)
        _write_json_atomic(self.metadata_path, self._metadata)


@dataclass
class RecordingData:
    """Read-side view of a raw recording (used by the player/editor)."""

    directory: Path
    metadata: dict[str, Any]
    video_path: Path
    frame_times: np.ndarray  # capture time per frame index (float64)
    events: list[dict[str, Any]]
    markers: list[dict[str, Any]]
    fps: float
    width: int
    height: int

    @property
    def duration(self) -> float:
        if self.frame_times.size:
            return float(self.frame_times[-1])
        return float(self.metadata.get("duration") or 0.0)

    @property
    def session_id(self) -> str:
        return str(self.metadata.get("session_id") or "")

    def frame_time(self, frame_index: int) -> float:
        """Capture time of a video frame index (falls back to index/fps)."""
        if 0 <= frame_index < self.frame_times.size:
            return float(self.frame_times[frame_index])
        return float(frame_index) / self.fps if self.fps else 0.0

    def nearest_frame_index(self, t: float) -> int:
        """Video frame index whose capture time is closest to ``t``."""
        if self.frame_times.size:
            idx = int(np.argmin(np.abs(self.frame_times - t)))
            return max(0, idx)
        return max(0, int(round(t * self.fps))) if self.fps else 0

    def snap_to_frame(self, t: float) -> float:
        """Nearest frame capture time, so edits always land on frame boundaries."""
        return self.frame_time(self.nearest_frame_index(t))


def load_recording(directory: Path | str) -> RecordingData:
    """Load a raw recording directory into a :class:`RecordingData`.

    Raises ``ValueError`` if the directory is not a recording.
    """
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"not a recording (no metadata.json): {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    screen = metadata.get("screen") or {}
    fps = float(screen.get("fps") or 30.0)
    return RecordingData(
        directory=directory,
        metadata=metadata,
        video_path=directory / "video.mp4",
        frame_times=_read_frame_times(directory / "frames.jsonl"),
        events=_read_jsonl(directory / "events.jsonl"),
        markers=_read_jsonl(directory / "markers.jsonl"),
        fps=fps,
        width=int(screen.get("width") or 0),
        height=int(screen.get("height") or 0),
    )


def list_recordings(root: Path | str) -> list[Path]:
    """All recording directories (containing metadata.json) under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
