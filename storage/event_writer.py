"""Thread-safe JSONL writer with a dedicated background thread.

Capture components enqueue small dicts; a single writer thread appends them
to the file. Input capture therefore never blocks on disk I/O. Events are
flushed to disk periodically so a crash loses at most the most recent batch.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_QUEUE = 65536
_FLUSH_EVERY = 100
_SENTINEL = None


class EventWriter:
    """Order-preserving, thread-safe appender of JSON objects to a file."""

    def __init__(self, path: Path | str, *, max_queue: int = _DEFAULT_MAX_QUEUE) -> None:
        self.path = Path(path)
        self._max_queue = max_queue
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._dropped = 0
        self.written = 0

    @property
    def dropped(self) -> int:
        """Number of events dropped because the write queue was full."""
        return self._dropped

    def start(self) -> None:
        """Start the writer thread. Must be called before :meth:`write`."""
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread = threading.Thread(
            target=self._run, name=f"jsonl-writer-{self.path.name}", daemon=True
        )
        self._thread.start()

    def write(self, event: dict[str, Any]) -> bool:
        """Enqueue one event. Returns True if accepted, False if dropped."""
        if not self._started:
            logger.warning("write to %s before start ignored", self.path)
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self._dropped += 1
            logger.error("event queue full; dropped event for %s (total dropped=%d)", self.path, self._dropped)
            return False

    def stop(self) -> None:
        """Drain all queued events, stop the thread and flush the file."""
        with self._lock:
            if not self._started:
                return
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("writer thread for %s did not stop", self.path)

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                self.written += 1
                if self.written % _FLUSH_EVERY == 0:
                    fh.flush()
            fh.flush()
