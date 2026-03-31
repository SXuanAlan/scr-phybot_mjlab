"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion parameters.

  Fields are placeholders for two tuning approaches:
  - Path A: Contact-Triggered Equality Constraint
  - Path B: Custom Force Model (xfrc_applied)
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  max_penetration: float = 0.15
  """Maximum allowed contact penetration depth [m]."""

  penetration_hysteresis: float = 0.05
  """Hysteresis band for adhesion engagement/disengagement [m]."""
