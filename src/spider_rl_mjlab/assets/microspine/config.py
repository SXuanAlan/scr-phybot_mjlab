"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion parameters.

  Path A: Contact-triggered equality constraint approach.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  max_penetration: float = 0.15
  """Maximum allowed contact penetration depth [m]."""

  penetration_hysteresis: float = 0.05
  """Hysteresis band for adhesion engagement/disengagement [m]."""

  # --- Path A: Equality constraint parameters ---

  engage_threshold: float = 0.5
  """Minimum normal force [N] to trigger engagement. Low threshold to engage quickly."""

  disengage_normal_disp: float = 0.005
  """Normal displacement threshold [m] to trigger disengagement (5mm)."""

  constraint_solref: tuple[float, float] = (0.001, 1.0)
  """Constraint solver reference: (timeconst, dampratio). Very stiff for 300N holding."""

  constraint_solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)
  """Constraint solver impedance. High d0/dwidth for strong holding."""

  hand_sub_geom_prefix: str = "right_hand_c_"
  """Prefix for hand sub-geom names to match in contact detection."""

  wall_geom_name: str = "wall/wall_geom"
  """Scoped name of the wall geom for contact detection."""

  wall_body_name: str = "wall/wall_body"
  """Scoped name of the wall body for constraint anchoring."""
