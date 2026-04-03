"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion via MuJoCo adhesion actuators.

  Each end-effector has a grid of contact spheres. An adhesion actuator
  is added per sphere; MuJoCo applies adhesion force only at real contact
  points, so no manual engage/disengage thresholds are needed.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  adhesion_force: float = 50.0
  """Total adhesion force across all spheres [N].
  Split equally among spheres (force_per_sphere = adhesion_force / num_spheres)."""

  ctrl_value: float = 1.0
  """Adhesion actuator control signal [0, 1]. Actual force = gain * ctrl.
  Lower values reduce contact oscillation at the cost of weaker adhesion."""
