"""Perception layer tests — no NVIDIA GPU, no model weights required.

Uses a deterministic fake provider for the pipeline and monkeypatches
backend availability for the LocateAnything provider, so everything runs
on CPU-only CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from perception import (
    BoundingBox,
    Detection,
    PerceptionManifest,
    PerceptionResult,
    get,
    is_registered,
    list_providers,
    register,
)
from perception.base import PerceptionProvider, provider_info
from perception.registry import _PROVIDERS
from perception.runner import analyze_recording, every_for_fps, select_frame_indices
from perception.types import ProviderInfo
from tests.fakes import build_synthetic_recording


class FakePerceptionProvider:
    """Deterministic provider: one fixed box per prompt, no model involved."""

    name = "fake_perception_test"
    version = "1.0.0"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def analyze(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        self.calls += 1
        return [
            Detection(label=prompt, bbox=BoundingBox(1.0, 2.0, 3.0, 4.0), confidence=0.9)
            for prompt in prompts
        ]


@pytest.fixture()
def recording(tmp_path):
    return build_synthetic_recording(tmp_path / "root", n_frames=30, fps=10)


@pytest.fixture()
def fake_provider():
    provider = FakePerceptionProvider()
    register(provider)
    yield provider
    _PROVIDERS.pop(provider.name, None)


# ------------------------------------------------------------ interface

def test_provider_interface(fake_provider):
    assert isinstance(fake_provider, PerceptionProvider)
    assert fake_provider.name == "fake_perception_test"
    assert fake_provider.version == "1.0.0"
    assert fake_provider.is_available() is True
    detections = fake_provider.analyze(np.zeros((8, 8, 3), dtype=np.uint8), ["boss"])
    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)


def test_provider_info(fake_provider):
    info = provider_info(fake_provider)
    assert isinstance(info, ProviderInfo)
    assert info.name == fake_provider.name
    assert info.available is True
    assert info.model == "fake-model"
    assert info.to_dict()["version"] == "1.0.0"


# ---------------------------------------------------------- serialization

def test_bounding_box_serialization_round_trip():
    box = BoundingBox(x1=812.5, y1=231, x2=1057, y2=614.25)
    data = box.to_dict()
    assert data == {"x1": 812.5, "y1": 231.0, "x2": 1057.0, "y2": 614.25}
    restored = BoundingBox.from_dict(data)
    assert restored == box
    assert json.loads(json.dumps(data)) == data


def test_detection_serialization_round_trip():
    detection = Detection(label="boss", bbox=BoundingBox(1, 2, 3, 4), confidence=0.94)
    restored = Detection.from_dict(detection.to_dict())
    assert restored == detection
    assert restored.source == "model"
    assert restored.verified is None
    no_conf = Detection(label="x", bbox=BoundingBox(0, 0, 1, 1))
    assert Detection.from_dict(no_conf.to_dict()).confidence is None


def test_perception_result_serialization_round_trip():
    result = PerceptionResult(
        frame_index=1204,
        t=40.133,
        prompt="boss",
        detections=[Detection(label="boss", bbox=BoundingBox(1, 2, 3, 4), confidence=0.9)],
    )
    restored = PerceptionResult.from_dict(result.to_dict())
    assert restored == result
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_manifest_creation_and_round_trip():
    manifest = PerceptionManifest(
        provider="locate_anything",
        provider_version="0.1.0",
        model="nvidia/LocateAnything-3B",
        source_session_id="abc123",
        source_recording="2026-08-08_22-51-03_abc123",
        sampling={"fps": 2.0, "every": 15, "frames": 4},
        prompts=["boss", "projectile"],
        count=8,
    )
    restored = PerceptionManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert json.loads(json.dumps(manifest.to_dict()))["format_version"] == 1


# ------------------------------------------------------------ registration

def test_provider_registration(fake_provider):
    assert is_registered("fake_perception_test")
    assert get("fake_perception_test") is fake_provider
    assert any(p.name == "fake_perception_test" for p in list_providers())
    with pytest.raises(KeyError, match="unknown perception provider"):
        get("no_such_provider")


# --------------------------------------------------------------- sampling

def test_every_for_fps():
    assert every_for_fps(30.0, 2.0) == 15
    assert every_for_fps(30.0, 2.5) == 12
    assert every_for_fps(10.0, 2.0) == 5
    assert every_for_fps(30.0, 60.0) == 1
    with pytest.raises(ValueError, match="fps"):
        every_for_fps(30.0, 0)


def test_select_frame_indices():
    assert select_frame_indices(30, 10.0, every=5) == [0, 5, 10, 15, 20, 25]
    assert select_frame_indices(30, 10.0, fps=2.0) == [0, 5, 10, 15, 20, 25]
    assert select_frame_indices(30, 10.0, fps=2.0)[-1] < 30
    with pytest.raises(ValueError, match="either --every"):
        select_frame_indices(30, 10.0)
    with pytest.raises(ValueError, match=">= 1"):
        select_frame_indices(30, 10.0, every=0)


# ------------------------------------------------------ pipeline (no GPU)

def test_analyze_recording_pipeline(recording, fake_provider):
    out = analyze_recording(
        recording, fake_provider, ["boss", "projectile"], fps=2.0
    )
    assert out.exists
    manifest = out.read_manifest()
    assert manifest.format_version == 1
    assert manifest.provider == "fake_perception_test"
    assert manifest.provider_version == "1.0.0"
    assert manifest.model == "fake-model"
    assert manifest.source_session_id == recording.session_id
    assert manifest.source_recording == recording.directory.name
    assert manifest.prompts == ["boss", "projectile"]
    assert manifest.sampling["every"] == 5  # 10 fps / 2 fps
    expected_records = len(select_frame_indices(30, 10.0, fps=2.0)) * 2
    assert manifest.count == expected_records

    results = out.read_results()
    assert len(results) == expected_records
    seen = {(r.frame_index, r.prompt) for r in results}
    assert len(seen) == expected_records
    for result in results:
        assert result.detections
        assert result.t == pytest.approx(recording.frame_time(result.frame_index))
        assert result.t == pytest.approx(recording.frame_times[result.frame_index])
        assert result.detections[0].confidence == 0.9
    assert out.results_path.parent == recording.directory / "perception"
    assert out.manifest_path.parent == recording.directory / "perception"


def test_prompt_handling_dedupes(recording, fake_provider):
    out = analyze_recording(recording, fake_provider, ["boss", "boss", "projectile"], every=10)
    manifest = out.read_manifest()
    assert manifest.prompts == ["boss", "projectile"]
    prompts = {r.prompt for r in out.read_results()}
    assert prompts == {"boss", "projectile"}
    with pytest.raises(ValueError, match="prompt"):
        analyze_recording(recording, fake_provider, [], every=10)
    with pytest.raises(ValueError, match="prompt"):
        analyze_recording(recording, fake_provider, ["  "], every=10)


def test_recording_unchanged_after_analysis(recording, fake_provider):
    snapshot = {
        path.name: path.read_bytes()
        for path in recording.directory.iterdir()
        if path.is_file()
    }
    analyze_recording(recording, fake_provider, ["boss"], every=10)
    for path in recording.directory.iterdir():
        if path.is_file() and path.name != "perception":
            assert path.read_bytes() == snapshot[path.name], f"{path.name} was modified"
    assert (recording.directory / "perception" / "manifest.json").exists()
    assert (recording.directory / "perception" / "results.jsonl").exists()


def test_caching_reuses_results(recording, fake_provider):
    first = analyze_recording(recording, fake_provider, ["boss", "projectile"], every=10)
    calls_after_first = fake_provider.calls
    assert calls_after_first > 0
    second = analyze_recording(recording, fake_provider, ["boss", "projectile"], every=10)
    assert fake_provider.calls == calls_after_first  # cached: no recompute
    assert first.results_path == second.results_path
    forced = analyze_recording(recording, fake_provider, ["boss", "projectile"], every=10, force=True)
    assert fake_provider.calls > calls_after_first
    assert forced.results_path == first.results_path
    different_prompt = analyze_recording(recording, fake_provider, ["boss"], every=10)
    assert fake_provider.calls > calls_after_first  # different prompts: not cached


def test_analyze_requires_frames_and_video(recording, fake_provider):
    recording.video_path.unlink()
    with pytest.raises(ValueError, match="no video"):
        analyze_recording(recording, fake_provider, ["boss"], every=10)


def test_prepare_failure_leaves_no_artifacts(recording, tmp_path):
    class PrepareFailsProvider(FakePerceptionProvider):
        def prepare(self) -> None:
            raise RuntimeError("model download failed")

    provider = PrepareFailsProvider()
    with pytest.raises(RuntimeError, match="model download failed"):
        analyze_recording(recording, provider, ["boss"], every=10)
    assert not (recording.directory / "perception").exists()


def test_prepare_called_before_analyze(recording):
    class PreparedProvider(FakePerceptionProvider):
        def __init__(self) -> None:
            super().__init__()
            self.prepared = 0

        def prepare(self) -> None:
            self.prepared += 1

    provider = PreparedProvider()
    analyze_recording(recording, provider, ["boss"], every=10)
    assert provider.prepared == 1
    assert provider.calls > 0


def test_failed_rerun_preserves_previous_results(recording, fake_provider):
    good = analyze_recording(recording, fake_provider, ["boss"], every=10)
    snapshot = good.results_path.read_bytes()

    class CrashProvider(FakePerceptionProvider):
        def analyze(self, frame, prompts):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("CUDA out of memory")
            return super().analyze(frame, prompts)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        analyze_recording(recording, CrashProvider(), ["boss"], every=10, force=True)
    assert good.results_path.read_bytes() == snapshot  # previous run intact
    assert not (recording.directory / "perception" / "results.jsonl.tmp").exists()
    assert not (recording.directory / "perception" / "manifest.json.tmp").exists()


# --------------------------------------------- missing optional dependency

def test_locate_anything_missing_dependency(monkeypatch, recording):
    from perception.providers import locate_anything

    monkeypatch.setattr(locate_anything, "_backend_importable", lambda: False)
    provider = locate_anything.LocateAnythingProvider()
    assert provider.is_available() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        analyze_recording(recording, provider, ["boss"], every=10)


def test_locate_anything_load_failure_is_clean(monkeypatch):
    from perception.providers import locate_anything

    def boom(_device):
        raise ImportError("no transformers installed")

    monkeypatch.setattr("ml.locate.load_locator", boom)
    provider = locate_anything.LocateAnythingProvider()
    with pytest.raises(RuntimeError, match="could not be loaded"):
        provider.analyze(np.zeros((8, 8, 3), dtype=np.uint8), ["boss"])


def test_locate_anything_converts_locator_output(monkeypatch):
    from perception.providers import locate_anything

    class GoodLocator:
        def locate(self, image, prompt, task):
            if prompt == "nothing here":
                return {"boxes": [], "points": []}
            return {"boxes": [{"x1": 10, "y1": 20, "x2": 30, "y2": 40}], "points": []}

    provider = locate_anything.LocateAnythingProvider(locator=GoodLocator())
    detections = provider.analyze(np.zeros((64, 64, 3), dtype=np.uint8), ["the save button"])
    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "the save button"
    assert detection.bbox == BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    assert detection.confidence is None
    assert detection.source == "model"
    empty = provider.analyze(np.zeros((64, 64, 3), dtype=np.uint8), ["nothing here"])
    assert empty == []


def test_locate_anything_inference_failure_is_clean():
    from perception.providers import locate_anything

    class BrokenLocator:
        def locate(self, image, prompt, task):
            raise RuntimeError("CUDA out of memory")

    provider = locate_anything.LocateAnythingProvider(locator=BrokenLocator())
    with pytest.raises(RuntimeError, match="inference failed"):
        provider.analyze(np.zeros((8, 8, 3), dtype=np.uint8), ["boss"])


def test_locate_anything_version_never_loads_model(monkeypatch):
    from perception.providers import locate_anything

    monkeypatch.setattr(locate_anything, "_backend_importable", lambda: False)
    provider = locate_anything.LocateAnythingProvider()
    assert isinstance(provider.version, str)  # metadata lookup, no model load
    assert provider.model == "nvidia/LocateAnything-3B"


def test_with_options_device_override():
    from perception.base import with_options
    from perception.providers.locate_anything import LocateAnythingProvider

    provider = LocateAnythingProvider(device="cuda")
    assert with_options(provider, device=None) is provider
    reconfigured = with_options(provider, device="cuda:1")
    assert reconfigured is not provider
    assert reconfigured._device == "cuda:1"
    class NoOptions:
        def is_available(self) -> bool:
            return False
    assert with_options(NoOptions(), device="x") is not None  # no with_options -> unchanged
    assert with_options(NoOptions(), device="x").is_available() is False


# --------------------------------------------------------------- CLI

def test_cli_providers(fake_provider, capsys):
    from perception.cli import run

    assert run(["providers"]) == 0
    output = capsys.readouterr().out
    assert "fake_perception_test" in output
    assert "available: yes" in output

    assert run(["providers", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "fake_perception_test" for item in data)


def test_cli_analyze_with_fake_provider(recording, fake_provider, capsys):
    from perception.cli import run

    code = run(
        [
            "analyze",
            str(recording.directory),
            "--provider",
            "fake_perception_test",
            "--prompt",
            "boss",
            "--fps",
            "2",
        ]
    )
    assert code == 0
    assert (recording.directory / "perception" / "results.jsonl").exists()


def test_cli_analyze_validation(recording, fake_provider, capsys):
    from perception.cli import run

    assert run(["analyze", str(recording.directory), "--provider", "fake_perception_test"]) == 2
    assert "prompt" in capsys.readouterr().err

    assert (
        run(
            [
                "analyze",
                str(recording.directory),
                "--provider",
                "fake_perception_test",
                "--prompt",
                "boss",
                "--fps",
                "2",
                "--every",
                "5",
            ]
        )
        == 2
    )
    assert "either --fps or --every" in capsys.readouterr().err

    assert run(["analyze", str(recording.directory.parent / "missing"), "--prompt", "boss"]) == 1
    assert "error:" in capsys.readouterr().err

    assert (
        run(
            [
                "analyze",
                str(recording.directory),
                "--provider",
                "no_such_provider",
                "--prompt",
                "boss",
            ]
        )
        == 1
    )
    assert "unknown perception provider" in capsys.readouterr().err


def test_cli_analyze_unavailable_provider(recording, capsys):
    from perception.cli import run

    class UnavailableProvider:
        name = "fake_unavailable_test"
        version = "0.0.0"

        def is_available(self) -> bool:
            return False

        def analyze(self, frame, prompts):
            raise AssertionError("must not be called")

    register(UnavailableProvider())
    try:
        code = run(
            [
                "analyze",
                str(recording.directory),
                "--provider",
                "fake_unavailable_test",
                "--prompt",
                "boss",
            ]
        )
    finally:
        _PROVIDERS.pop("fake_unavailable_test", None)
    assert code == 1
    assert "unavailable" in capsys.readouterr().err


def test_cli_analyze_missing_args_exit_2():
    from perception.cli import run

    with pytest.raises(SystemExit) as excinfo:
        run(["analyze"])
    assert excinfo.value.code == 2


# ------------------------------------------------------------ config

def test_perception_config_defaults_disabled():
    from perception.config import PerceptionConfig

    config = PerceptionConfig()
    assert config.enabled is False
    assert config.provider is None
    assert config.prompts == []
    assert config.fps > 0
