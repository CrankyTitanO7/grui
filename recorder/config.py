"""Typed configuration for a recording session (pydantic).

None of these settings are hardcoded into the recorder beyond these defaults;
the same recording can later be consumed by dataset builders with completely
independent parameters.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ScreenConfig(BaseModel):
    """Screen capture settings."""

    fps: int = Field(default=30, ge=1, le=240)
    monitor_index: int = Field(
        default=0,
        ge=-1,
        description="-1 captures all monitors combined; 0 is the first monitor",
    )


class EncoderConfig(BaseModel):
    """FFmpeg video encoding settings."""

    codec: str = "libx264"
    preset: str = "veryfast"
    crf: int = Field(default=20, ge=0, le=51)
    pix_fmt_out: str = "yuv420p"


class RecorderConfig(BaseModel):
    """Top-level recorder configuration."""

    output_dir: Path = Field(default_factory=lambda: Path("recordings"))
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    frame_queue_size: int = Field(default=8, ge=1, le=1024)
