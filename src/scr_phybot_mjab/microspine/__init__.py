"""Microspine adhesion module."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from scr_phybot_mjab.microspine.config import (
  MicrospineAffordanceCfg,
  MicrospineCfg,
)

__all__ = [
  "MicrospineAffordance",
  "MicrospineAffordanceCfg",
  "MicrospineCfg",
  "MicrospinePhysics",
  "add_adhesion_actuators",
]

_RUNTIME_EXPORTS = {
  "MicrospineAffordance": (
    "scr_phybot_mjab.microspine.affordance",
    "MicrospineAffordance",
  ),
  "MicrospinePhysics": (
    "scr_phybot_mjab.microspine.physics",
    "MicrospinePhysics",
  ),
  "add_adhesion_actuators": (
    "scr_phybot_mjab.microspine.physics",
    "add_adhesion_actuators",
  ),
}


def __getattr__(name: str) -> Any:
  """Load public exports on demand to avoid package-level import cycles."""
  try:
    module_name, attr_name = _RUNTIME_EXPORTS[name]
  except KeyError as exc:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

  value = getattr(import_module(module_name), attr_name)
  globals()[name] = value
  return value
