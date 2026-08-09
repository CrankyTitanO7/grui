"""Manual end-to-end smoke test with REAL capture components.

Records ~2.5s of the actual screen + input into a temp directory, then
prints the results. Run: python tests/manual_e2e.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.config import RecorderConfig, ScreenConfig
from recorder.session import RecordingSession, SessionState


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = RecorderConfig(
            output_dir=Path(tmp),
            screen=ScreenConfig(fps=15, monitor_index=0),
        )
        session = RecordingSession(config)
        states = []
        session.register_observer(states.append)

        session.start()
        assert session.state == SessionState.RECORDING
        print(f"recording into {session.recording_dir}")
        time.sleep(2.5)
        session.add_annotation("smoke_test_marker")
        session.stop()

        assert session.state == SessionState.IDLE, session.state
        d = Path(session.recording_dir)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        n_events = sum(1 for _ in (d / "events.jsonl").open(encoding="utf-8"))
        n_frames = sum(1 for _ in (d / "frames.jsonl").open(encoding="utf-8"))
        n_markers = sum(1 for _ in (d / "markers.jsonl").open(encoding="utf-8"))
        video_size = (d / "video.mp4").stat().st_size
        print(f"states: {[s.value for s in states]}")
        print(f"screen: {meta['screen']}")
        print(f"duration: {meta['duration']:.2f}s")
        print(f"stats: {meta['stats']}")
        print(f"video.mp4: {video_size} bytes")
        print(f"events.jsonl: {n_events} lines")
        print(f"frames.jsonl: {n_frames} lines")
        print(f"markers.jsonl: {n_markers} lines")
        assert n_frames >= 15, "too few frames encoded"
        assert n_markers == 1
        assert video_size > 1000
        assert meta["stats"]["encoder_returncode"] == 0, meta["stats"]["encoder_error"]
        print("E2E SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
