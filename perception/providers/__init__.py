"""Built-in perception providers. Importing this module registers them."""

from __future__ import annotations

from perception.registry import register

from .locate_anything import LocateAnythingProvider

register(LocateAnythingProvider())
