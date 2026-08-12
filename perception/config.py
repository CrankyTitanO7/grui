"""Perception settings. Everything is off by default.

Perception is never enabled automatically: ``enabled`` is ``False``, the
provider is ``None`` until chosen, and no model is loaded at startup. The
values here are defaults for the CLI and the UI dialog.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PerceptionConfig(BaseModel):
    """Top-level perception configuration."""

    enabled: bool = Field(default=False, description="never enabled by default")
    provider: str | None = Field(default=None, description="provider name, e.g. locate_anything")
    fps: float = Field(default=2.0, gt=0, description="analysis sampling rate (frames/second)")
    prompts: list[str] = Field(default_factory=list, description="default natural-language prompts")
