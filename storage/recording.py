"""Raw demonstration directory and metadata management.

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
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
