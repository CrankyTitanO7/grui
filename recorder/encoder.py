"""Frame encoding via an FFmpeg subprocess.

Frames are consumed from the shared frame queue in a dedicated thread and
piped into ffmpeg as raw ``bgr24`` frames. The subprocess is started lazily
on the first frame, so stopping before any frame is captured does not create
an empty video file.

The capture time of every actually-encoded frame is written to
``frames.jsonl`` (``frame_index`` -> ``t``) so downstream consumers can
synchronize video content with input events exactly, even if frames were
dropped under backpressure.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

from recorder.clock import SessionClock
from recorder.config import EncoderConfig

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Locate an ffmpeg executable: bundled imageio-ffmpeg, then PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg not found; install imageio-ffmpeg or add ffmpeg to PATH")


class FFmpegEncoder:
    """Threaded encoder that pipes queued frames into an ffmpeg process."""

    def __init__(
        self,
        config: EncoderConfig,
        video_path: Path,
        frames_writer,
        frame_queue: "queue.Queue",
        clock: SessionClock,
        fps: int,
    ) -> None:
        self.config = config
        self.video_path = Path(video_path)
        self.frames_writer = frames_writer
        self.frame_queue = frame_queue
        self.clock = clock
        self.fps = fps
        self._thread: threading.Thread | None = None
        self.frames_encoded = 0
        self.returncode: int | None = None
        self.error: str | None = None

    def start(self) -> None:
        """Start the encoder thread (non-blocking)."""
        self._thread = threading.Thread(target=self._run, name="video-encoder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Wait for the encoder thread to finish after the queue sentinel.

        The session must put ``None`` into the frame queue before calling
        this. Safe to call from the encoder thread itself.
        """
        if threading.current_thread() is self._thread:
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("encoder thread did not stop within %.1fs", timeout)

    def _run(self) -> None:
        first = self.frame_queue.get()  # blocks until first frame or sentinel
        if first is None:
            logger.info("no frames captured; skipping video encoding")
            return
        t0, frame0 = first
        height, width = frame0.shape[:2]
        try:
            exe = _find_ffmpeg()
        except RuntimeError as exc:
            self.error = str(exc)
            logger.error("%s", self.error)
            return

        args = [
            exe,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self.config.codec,
            "-preset", self.config.preset,
            "-crf", str(self.config.crf),
            "-pix_fmt", self.config.pix_fmt_out,
            "-y",
            str(self.video_path),
        ]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            self._write_frame(proc, frame0, t0)
            while True:
                item = self.frame_queue.get()
                if item is None:
                    break
                t, frame = item
                self._write_frame(proc, frame, t)
        except (BrokenPipeError, OSError) as exc:
            self.error = f"video encoding failed: {exc}"
            logger.error("%s", self.error)
        finally:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            stderr = b""
            try:
                self.returncode = proc.wait(timeout=10)
                stderr = proc.stderr.read()
            except Exception:  # noqa: BLE001
                self.returncode = None
            if self.returncode not in (0, None):
                self.error = (
                    f"ffmpeg exited with code {self.returncode}: "
                    f"{stderr.decode(errors='replace')[:500]}"
                )
                logger.error("%s", self.error)

    def _write_frame(self, proc: subprocess.Popen, frame, t: float) -> None:
        proc.stdin.write(frame.tobytes())
        if self.frames_writer is not None:
            self.frames_writer.write({"frame_index": self.frames_encoded, "t": t})
        self.frames_encoded += 1
