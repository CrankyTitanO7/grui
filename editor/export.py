"""Export an edited timeline as a new raw recording.

The source recording is never modified: a fresh recording directory is
created, the kept video frames are re-encoded with FFmpeg, and events,
markers and frame times are remapped through the timeline into the new
timestamps.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from editor.timeline import Clip, Timeline, remap_events
from recorder.encoder import _find_ffmpeg
from recorder.config import EncoderConfig
from storage.recording import RawRecording, RecordingData

logger = logging.getLogger(__name__)


def _ffmpeg_args(
    exe: str, video_path: Path, width: int, height: int, fps: float, config: EncoderConfig
) -> list[str]:
    return [
        exe,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", config.codec,
        "-preset", config.preset,
        "-crf", str(config.crf),
        "-pix_fmt", config.pix_fmt_out,
        "-y",
        str(video_path),
    ]


def _clip_mask(frame_times: np.ndarray, clip: "Clip", last: float) -> np.ndarray:
    """Boolean mask of source frames inside a clip.

    The end boundary is exclusive, except for a clip ending at the very last
    frame of the recording (its end is the only ``<`` boundary that would
    otherwise exclude the final frame).
    """
    start_mask = frame_times >= clip.source_start
    if clip.source_end >= last:
        return start_mask & (frame_times <= last)
    return start_mask & (frame_times < clip.source_end)


def _remap_frame_times(frame_times: np.ndarray, timeline: Timeline) -> tuple[np.ndarray, list[int]]:
    """Map source frame times through the timeline.

    Returns (new_times, kept_source_indices). Only frames inside at least one
    clip are kept; their new times are monotonic (clips are contiguous).
    """
    last = float(frame_times[-1]) if frame_times.size else 0.0
    kept = np.zeros(frame_times.size, dtype=bool)
    for clip in timeline.clips:
        kept |= _clip_mask(frame_times, clip, last)
    source_indices = np.flatnonzero(kept)
    new_times: list[float] = []
    for clip in timeline.clips:
        new_times.extend(clip.edited_time(t) for t in frame_times[_clip_mask(frame_times, clip, last)].tolist())
    return np.asarray(new_times, dtype=np.float64), [int(i) for i in source_indices]


def export_recording(
    source: RecordingData,
    timeline: Timeline,
    output_root: Path | str,
    *,
    edit_history: list[dict[str, Any]] | None = None,
    encoder_config: EncoderConfig | None = None,
) -> RawRecording:
    """Export the edited timeline as a new raw recording directory."""
    output_root = Path(output_root)
    encoder_config = encoder_config or EncoderConfig()
    session_id = uuid.uuid4().hex[:12]

    metadata = {
        "version": RawRecording.FORMAT_VERSION,
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": source.metadata.get("platform", ""),
        "screen": {
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
            "monitor_index": source.metadata.get("screen", {}).get("monitor_index"),
        },
        "input": source.metadata.get("input", {"keyboard": True, "mouse": True}),
        "edited_from": {
            "session_id": source.session_id,
            "path": str(source.directory),
        },
        "edit_history": edit_history or [],
        "edit_clips": timeline.snapshot(),
        "duration": timeline.duration,
    }
    recording = RawRecording.create(output_root, session_id, metadata)

    new_frame_times, kept_indices = _remap_frame_times(source.frame_times, timeline)
    frames_encoded = _encode_video(source, kept_indices, recording.video_path, encoder_config)

    events = remap_events(source.events, timeline)
    markers = remap_events(source.markers, timeline)
    events = [
        {"t": 0.0, "device": "session", "event": "recording_start"},
        *events,
        {"t": timeline.duration, "device": "session", "event": "recording_stop"},
    ]
    events.sort(key=lambda e: e["t"])

    _write_jsonl(recording.events_path, events)
    _write_jsonl(recording.markers_path, markers)
    _write_jsonl(
        recording.frames_path,
        [{"frame_index": i, "t": float(t)} for i, t in enumerate(new_frame_times)],
    )

    stats = {
        "frames_encoded": frames_encoded,
        "frames_dropped": 0,
        "events_written": len(events),
        "markers_written": len(markers),
        "edited": True,
        "encoder_returncode": 0,
    }
    files = {
        name: (path.stat().st_size if path.exists() else 0)
        for name, path in recording.files().items()
    }
    recording.update_metadata(duration=timeline.duration, stats=stats, files=files)
    return recording


def _encode_video(
    source: RecordingData, kept_indices: list[int], video_path: Path, config: EncoderConfig
) -> int:
    """Re-encode the kept source frames into ``video_path``. Returns count."""
    if not kept_indices:
        logger.info("nothing to encode; no video written")
        return 0
    import cv2

    cap = cv2.VideoCapture(str(source.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {source.video_path}")
    try:
        width = source.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = source.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("could not determine source video dimensions")
        fps = source.fps or 30.0

        keep = set(kept_indices)
        exe = _find_ffmpeg()
        proc = subprocess.Popen(
            _ffmpeg_args(exe, video_path, width, height, fps, config),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        encoded = 0
        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index in keep:
                    proc.stdin.write(frame.tobytes())
                    encoded += 1
                frame_index += 1
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"video export failed: {exc}") from exc
        finally:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            returncode = proc.wait(timeout=60)
            stderr = proc.stderr.read() if proc.stderr else b""
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')[:500]}"
            )
        return encoded
    finally:
        cap.release()


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
