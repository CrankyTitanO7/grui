"""Training monitor: progress bar and machine-readable metrics logging.

Dependency-free so ``grui train`` stays light. The progress bar updates in
place with ``\\r`` and disables itself when stdout is not a terminal (or
when ``GRUI_NO_PROGRESS`` is set); ``--no-progress`` on the CLI is the same
flag. Metrics are appended as JSON Lines so an external process can tail or
plot them while training runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_BAR_WIDTH = 20


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def progress_enabled() -> bool:
    if os.environ.get("GRUI_NO_PROGRESS"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class ProgressBar:
    """Single-line progress bar with running loss, elapsed time and ETA."""

    def __init__(self, total: int, prefix: str = "", *, enabled: bool | None = None) -> None:
        self.total = max(1, total)
        self.prefix = prefix
        self.enabled = progress_enabled() if enabled is None else enabled
        self._start = time.monotonic()
        self._last = 0.0
        self._step = 0
        self.loss = 0.0

    def update(self, step: int = 1, loss: float | None = None) -> None:
        """Advance by ``step`` batches; ``loss`` is the batch's loss value."""
        self._step += step
        if loss is not None:
            # running average over all batches so far
            total = self.loss * (self._step - step) + loss * step
            self.loss = total / self._step
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < 0.1 and self._step < self.total:
            return  # throttle repaints to ~10 Hz
        self._last = now
        elapsed = now - self._start
        rate = self._step / max(elapsed, 1e-9)
        remaining = (self.total - self._step) / rate if rate else 0.0
        filled = int(_BAR_WIDTH * self._step / self.total)
        bar = "[" + "#" * filled + "-" * (_BAR_WIDTH - filled) + "]"
        line = (
            f"\r{self.prefix} {bar} {self._step}/{self.total} "
            f"loss={self.loss:.4f} {_format_seconds(elapsed)}s "
            f"eta {_format_seconds(remaining)}"
        )
        sys.stdout.write(line + " " * 4)
        sys.stdout.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stdout.write("\r" + " " * 8)
            sys.stdout.write("\r")
            sys.stdout.flush()


class MetricsLogger:
    """Append-only JSON Lines log of per-epoch training metrics."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @classmethod
    def read(cls, path: Path | str) -> list[dict]:
        """Read all records back (used by tests and tooling)."""
        items = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
