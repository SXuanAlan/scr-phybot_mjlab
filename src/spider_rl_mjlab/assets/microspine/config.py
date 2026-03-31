"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion via custom force model (xfrc_applied).

  Spring-damper PD controller that engages when end-effectors contact the wall
  (detected via MuJoCo contact data), recording anchor positions and applying
  restoring forces to resist shear loads.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  # -- Engagement thresholds --

  disengage_distance: float = 0.2
  """Separation from anchor [m] at which spines fully disengage."""

  # -- Normal spring-damper (toward wall) --

  k_normal: float = 5000.0
  """Normal contact stiffness [N/m]."""

  c_normal: float = 150.0
  """Normal contact damping [Ns/m]."""

  f_normal_max: float = 500.0
  """Normal force clamp [N]."""

  # -- Tangential spring-damper (resist shear) --

  k_tangential: float = 20000.0
  """Tangential contact stiffness [N/m]."""

  c_tangential: float = 500.0
  """Tangential contact damping [Ns/m]."""

  f_tangential_max: float = 500.0
  """Tangential force clamp [N]."""

  # -- Anisotropy --

  aniso_ratio: float = 0.3
  """Stiffness/damping ratio for release direction vs grasp direction.
  grasp direction = gravity direction (downward). Release = upward."""

  # -- Lateral --

  f_lateral_max: float = 200.0
  """Lateral (y) force clamp [N]."""
