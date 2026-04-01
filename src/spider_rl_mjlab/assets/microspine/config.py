"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion parameters.

  Models the physical behavior of microspine arrays: once spines engage
  with a surface (detected via contact penetration), they provide an
  adhesion (pull-in) force that maintains normal contact, enabling
  friction to resist tangential loads.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  max_penetration: float = 0.15
  """Maximum allowed contact penetration depth [m]."""

  penetration_hysteresis: float = 0.05
  """Hysteresis band for adhesion engagement/disengagement [m]."""

  adhesion_force: float = 50.0
  """Adhesion (pull-in) force per EE body when fully engaged [N].
  In actuator mode this is the gain of the adhesion actuator."""

  engage_threshold: float = 0.001
  """Minimum penetration depth to consider spines engaged [m]."""

  disengage_threshold: float = 0.01
  """Separation distance at which spines fully disengage [m]."""

  wall_normal: tuple[float, float, float] = (-1.0, 0.0, 0.0)
  """Inward normal direction of the wall surface (pointing from wall toward robot).
  Default (-1,0,0) means the wall face is at +x and the robot approaches from -x."""

  contact_offset: float = 0.082
  """Local x offset from EE body origin to contact surface [m].
  Positive = contact surface in +x local direction (forearm_hand).
  Negative = contact surface in -x local direction (phybot climbing)."""

  use_actuator: bool = False
  """If True, use MuJoCo adhesion actuators instead of external wrench.
  Adhesion actuators work through the constraint solver at contact points,
  providing more stable and physically accurate adhesion."""
