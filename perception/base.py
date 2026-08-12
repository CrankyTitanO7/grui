"""The generic perception-provider interface.

GRUI's recording layer knows nothing about perception providers. A
provider is any object satisfying :class:`PerceptionProvider`; it
receives an arbitrary image (a recorded frame, a dataset frame, a
screenshot, ...) plus one or more natural-language prompts, and returns
structured :class:`~perception.types.Detection` objects. Providers may
be registered in the registry so the CLI/UI can list and select them
without importing model code at GRUI startup.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from perception.types import Detection, ProviderInfo


@runtime_checkable
class PerceptionProvider(Protocol):
    """Minimal interface every perception provider must satisfy."""

    name: str
    version: str

    def is_available(self) -> bool:
        """True when the provider's model stack can be loaded on this machine."""
        ...

    def analyze(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Run the model on one image.

        ``frame`` is a BGR ndarray (the screen-capture convention used
        throughout GRUI); ``prompts`` is a list of natural-language
        queries. The provider may treat each prompt independently and
        returns every detection across all prompts.
        """
        ...


def provider_info(provider: PerceptionProvider) -> ProviderInfo:
    """Build :class:`ProviderInfo` from a provider instance (duck-typed)."""
    return ProviderInfo(
        name=str(provider.name),
        version=str(provider.version),
        available=bool(provider.is_available()),
        model=getattr(provider, "model", None),
        description=str(getattr(provider, "description", "")),
        install_hint=str(getattr(provider, "install_hint", "")),
    )


def with_options(provider: PerceptionProvider, **options: Any) -> PerceptionProvider:
    """Return a provider reconfigured with ``options``.

    Providers may expose a ``with_options(**kwargs)`` method (e.g. to swap
    the inference device); providers without one are returned unchanged.
    """
    factory = getattr(provider, "with_options", None)
    if callable(factory):
        return factory(**options)
    return provider
