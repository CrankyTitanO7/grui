"""Registry of available perception providers.

Providers self-register at import time (``perception.providers`` is
imported by the package, which imports the built-in providers). GRUI
never imports model code at startup — registry entries are lightweight
provider objects; the heavy model load happens lazily on the first
``analyze`` call.
"""

from __future__ import annotations

from perception.base import PerceptionProvider

_PROVIDERS: dict[str, PerceptionProvider] = {}


def register(provider: PerceptionProvider) -> None:
    """Register a provider under its ``name`` (replaces an existing one)."""
    _PROVIDERS[provider.name] = provider


def get(name: str) -> PerceptionProvider:
    """Look up a provider by name. Raises ``KeyError`` if unknown."""
    try:
        return _PROVIDERS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown perception provider {name!r} "
            f"(available: {', '.join(sorted(_PROVIDERS)) or 'none'})"
        ) from exc


def list_providers() -> list[PerceptionProvider]:
    """All registered providers, sorted by name."""
    return [_PROVIDERS[name] for name in sorted(_PROVIDERS)]


def is_registered(name: str) -> bool:
    return name in _PROVIDERS
