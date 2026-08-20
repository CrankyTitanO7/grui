"""Session lifecycle tests using fake capture components (no hardware)."""

import json
import threading
import time

import pytest

from recorder import session as session_module
from recorder.config import RecorderConfig
from recorder.session import RecordingSession, SessionState
from tests.fakes import FakeScreen, patch_session_components


@pytest.fixture()
def patched(monkeypatch):
    patch_session_components(monkeypatch, session_module)
    return monkeypatch


def _make_config(tmp_path):
    return RecorderConfig(output_dir=tmp_path)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_full_lifecycle(tmp_path, patched):
    config = _make_config(tmp_path)
    session = RecordingSession(config)
    states = []
    session.register_observer(states.append)

    assert session.state == SessionState.IDLE
    session.start()
    assert session.state == SessionState.RECORDING
    assert states[0] == SessionState.STARTING
    assert states[1] == SessionState.RECORDING
    assert session.clock is not None
    assert session.elapsed() >= 0

    session.add_annotation("boss_start")
    session.add_annotation("phase_transition")
    session.stop()

    assert session.state == SessionState.IDLE
    assert states[-1] == SessionState.IDLE
    assert SessionState.STOPPING in states

    directory = session.recording_dir
    assert directory is not None and directory.is_dir()
    assert (directory / "metadata.json").exists()
    assert (directory / "video.mp4").exists()
    assert (directory / "events.jsonl").exists()
    assert (directory / "markers.jsonl").exists()
    assert (directory / "frames.jsonl").exists()

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["version"] == 1
    assert metadata["session_id"] == session.session_id
    assert metadata["screen"] == {"width": 1920, "height": 1080, "fps": 30, "monitor_index": 0}
    assert metadata["duration"] > 0
    assert isinstance(metadata["platform"], str) and metadata["platform"]
    assert metadata["stats"]["frames_encoded"] == 5
    assert metadata["stats"]["frames_captured"] == 5

    events = _read_jsonl(directory / "events.jsonl")
    kinds = {e["event"] for e in events}
    assert "recording_start" in kinds
    assert "recording_stop" in kinds
    devices = {e["device"] for e in events}
    assert {"session", "keyboard", "mouse"} <= devices
    timestamps = [e["t"] for e in events]
    assert all(t >= 0 for t in timestamps)

    markers = _read_jsonl(directory / "markers.jsonl")
    assert [m["label"] for m in markers] == ["boss_start", "phase_transition"]
    assert all(m["type"] == "annotation" for m in markers)

    frames = _read_jsonl(directory / "frames.jsonl")
    assert [f["frame_index"] for f in frames] == list(range(5))
    assert [f["t"] for f in frames] == sorted(f["t"] for f in frames)


def test_multiple_sessions_are_independent(tmp_path, patched):
    config = _make_config(tmp_path)
    expected_dirs = []
    for _ in range(2):
        session = RecordingSession(config)
        session.start()
        session.add_annotation("attack")
        session.stop()
        assert session.recording_dir is not None
        expected_dirs.append(session.recording_dir)

    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 2
    for directory in expected_dirs:
        assert directory in dirs
        markers = _read_jsonl(directory / "markers.jsonl")
        assert markers[0]["label"] == "attack"
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["session_id"] == directory.name.rsplit("_", 1)[1]


def test_pause_resume_writes_lifecycle_events(tmp_path, patched):
    config = _make_config(tmp_path)
    session = RecordingSession(config)
    session.start()
    session.pause()
    assert session.state == SessionState.PAUSED
    session.resume()
    assert session.state == SessionState.RECORDING
    session.stop()

    events = _read_jsonl(session.recording_dir / "events.jsonl")
    kinds = [e["event"] for e in events]
    assert kinds.index("pause") < kinds.index("resume") < kinds.index("recording_stop")
    metadata = json.loads((session.recording_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["stats"]["pause_duration"] >= 0


def test_pause_blocks_input_and_screen(tmp_path, patched):
    config = _make_config(tmp_path)
    session = RecordingSession(config)
    session.start()
    session.pause()
    session.add_annotation("during_pause")
    session.resume()
    session.stop()

    markers = _read_jsonl(session.recording_dir / "markers.jsonl")
    assert len(markers) == 1  # annotation still works while paused


def test_stop_without_start_is_noop(tmp_path, patched):
    session = RecordingSession(_make_config(tmp_path))
    session.stop()
    assert session.state == SessionState.IDLE


def test_double_stop_is_safe(tmp_path, patched):
    session = RecordingSession(_make_config(tmp_path))
    session.start()
    session.stop()
    session.stop()
    assert session.state == SessionState.IDLE


def test_empty_annotation_rejected(tmp_path, patched):
    session = RecordingSession(_make_config(tmp_path))
    session.start()
    with pytest.raises(ValueError):
        session.add_annotation("   ")
    session.stop()


def test_annotation_before_start_rejected(tmp_path, patched):
    session = RecordingSession(_make_config(tmp_path))
    with pytest.raises(RuntimeError):
        session.add_annotation("attack")


def test_start_twice_rejected(tmp_path, patched):
    session = RecordingSession(_make_config(tmp_path))
    session.start()
    with pytest.raises(RuntimeError):
        session.start()
    session.stop()


def test_start_failure_cleans_up_to_error_state(tmp_path, monkeypatch):
    class ExplodingScreen(FakeScreen):
        def start(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(session_module, "ScreenRecorder", ExplodingScreen)
    patch_rest(session_module, monkeypatch)

    session = RecordingSession(_make_config(tmp_path))
    states = []
    session.register_observer(states.append)
    with pytest.raises(RuntimeError):
        session.start()
    assert session.state == SessionState.ERROR
    assert session.error_message == "recording start failed"
    assert states[-1] == SessionState.ERROR


def _fake_keyboard():
    from tests.fakes import FakeKeyboard

    return FakeKeyboard


def _fake_mouse():
    from tests.fakes import FakeMouse

    return FakeMouse


def _fake_encoder():
    from tests.fakes import FakeEncoder

    return FakeEncoder


def test_restart_after_error(tmp_path, monkeypatch):
    fail = {"on": True}

    class FlakyScreen(FakeScreen):
        def start(self):
            if fail["on"]:
                raise RuntimeError("boom")
            super().start()

    monkeypatch.setattr(session_module, "ScreenRecorder", FlakyScreen)
    patch_rest(session_module, monkeypatch)

    session = RecordingSession(_make_config(tmp_path))
    with pytest.raises(RuntimeError):
        session.start()
    assert session.state == SessionState.ERROR

    fail["on"] = False
    session.start()
    assert session.state == SessionState.RECORDING
    session.stop()
    assert session.state == SessionState.IDLE


def patch_rest(session_module_, monkeypatch):
    monkeypatch.setattr(session_module_, "KeyboardRecorder", _fake_keyboard())
    monkeypatch.setattr(session_module_, "MouseRecorder", _fake_mouse())
    monkeypatch.setattr(session_module_, "FFmpegEncoder", _fake_encoder())
    monkeypatch.setattr(session_module_, "resolve_monitor_size", lambda index: (1920, 1080))


def test_component_error_during_recording_stops_cleanly(tmp_path, monkeypatch):
    class FailingScreen(FakeScreen):
        def start(self):
            super().start()
            threading.Timer(0.05, lambda: self.error_cb("simulated capture failure")).start()

    monkeypatch.setattr(session_module, "ScreenRecorder", FailingScreen)
    patch_rest(session_module, monkeypatch)

    session = RecordingSession(_make_config(tmp_path))
    session.start()
    deadline = time.monotonic() + 10
    while session.state != SessionState.ERROR and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.state == SessionState.ERROR

    # final state stays ERROR after cleanup, files still finalized
    session.stop()
    assert session.state == SessionState.ERROR
    metadata = json.loads((session.recording_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["stats"]["frames_encoded"] == 5


def test_stop_during_start_is_safe(tmp_path, monkeypatch):
    blocked = threading.Event()

    class SlowScreen(FakeScreen):
        def start(self):
            blocked.wait(timeout=10)
            super().start()

    monkeypatch.setattr(session_module, "ScreenRecorder", SlowScreen)
    patch_rest(session_module, monkeypatch)

    session = RecordingSession(_make_config(tmp_path))

    def do_start():
        try:
            session.start()
        except RuntimeError:
            pass

    thread = threading.Thread(target=do_start)
    thread.start()
    time.sleep(0.05)
    session.stop()  # must not raise or deadlock
    blocked.set()
    thread.join(timeout=5)
    assert session.state in (SessionState.IDLE, SessionState.ERROR)
