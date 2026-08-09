"""Tests for typed recorder configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from recorder.config import RecorderConfig, ScreenConfig


def test_defaults():
    config = RecorderConfig()
    assert config.screen.fps == 30
    assert config.screen.monitor_index == 0
    assert config.output_dir == Path("recordings")
    assert config.encoder.codec == "libx264"
    assert config.frame_queue_size >= 1


def test_custom_values():
    config = RecorderConfig(
        screen=ScreenConfig(fps=60, monitor_index=1),
        output_dir=Path("tmp/out"),
    )
    assert config.screen.fps == 60
    assert config.screen.monitor_index == 1
    assert config.output_dir == Path("tmp/out")


def test_invalid_values_rejected():
    with pytest.raises(ValidationError):
        RecorderConfig(screen=ScreenConfig(fps=0))
    with pytest.raises(ValidationError):
        RecorderConfig(screen=ScreenConfig(fps=1000))
    with pytest.raises(ValidationError):
        RecorderConfig(screen=ScreenConfig(monitor_index=-2))
    with pytest.raises(ValidationError):
        RecorderConfig(frame_queue_size=0)
