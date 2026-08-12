"""Perception: optional, provider-based visual analysis of recordings.

GRUI records demonstrations independently of any perception model. This
package adds a generic perception-provider interface (``PerceptionProvider``)
plus providers (LocateAnything, future models) that analyze already-recorded
frames and attach structured detections as a derived artifact next to the
raw recording. Nothing here is imported by the recorder, player or dataset
builder at startup, and the model stack is only loaded lazily when an
analysis is explicitly requested.
"""

from __future__ import annotations

from perception import providers  # noqa: F401  (registers built-in providers)
from perception.base import PerceptionProvider, provider_info, with_options
from perception.registry import get, is_registered, list_providers, register
from perception.types import BoundingBox, Detection, PerceptionManifest, PerceptionResult, ProviderInfo

__all__ = [
    "PerceptionProvider",
    "provider_info",
    "with_options",
    "BoundingBox",
    "Detection",
    "PerceptionManifest",
    "PerceptionResult",
    "ProviderInfo",
    "get",
    "is_registered",
    "list_providers",
    "register",
]
