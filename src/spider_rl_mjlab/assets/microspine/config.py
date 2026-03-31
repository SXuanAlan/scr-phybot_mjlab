"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion parameters.

  Path B: Custom spring-damper-friction force model applied via xfrc_applied.
  Implements anisotropic friction with state-machine-based contact tracking.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  max_penetration: float = 0.15
  """Maximum allowed contact penetration depth [m]."""

  penetration_hysteresis: float = 0.05
  """Hysteresis band for adhesion engagement/disengagement [m]."""

  # -- Normal spring-damper parameters --

  k_normal: float = 50000.0
  """Normal contact stiffness [N/m]. Spring constant for penetration resistance."""

  c_normal: float = 200.0
  """Normal contact damping [N*s/m]. Damping on approach/separation velocity."""

  # -- Engagement parameters --

  engage_depth: float = 0.003
  """Depth of penetration required to engage microspine hooks [m]."""

  disengage_depth: float = -0.001
  """Negative penetration (gap) at which hooks disengage [m]."""

  # -- Tangential spring-damper parameters --

  k_tangential: float = 100000.0
  """Tangential contact stiffness [N/m]. Spring constant for shear resistance."""

  c_tangential: float = 500.0
  """Tangential contact damping [N*s/m]. Damping on sliding velocity."""

  # -- Friction parameters (anisotropic) --

  mu_kinetic: float = 2.0
  """Kinetic friction coefficient before engagement."""

  mu_grasp: float = 12.0
  """Effective friction coefficient along grasp direction (downward, -z).
  High value simulates microspine hooks embedding into surface asperities."""

  mu_release: float = 1.5
  """Effective friction coefficient against grasp direction (upward, +z).
  Low value simulates easy release when pulling spines out."""

  mu_lateral: float = 5.0
  """Friction coefficient in the lateral (y) direction."""

  # -- Grasp axis --

  grasp_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Direction in world frame along which spines have maximum grip (e.g. downward)."""

  # -- Adhesion parameters --

  adhesion_depth: float = 0.001
  """Maximum gap distance [m] at which adhesion pull-back force acts."""

  k_adhesion: float = 10000.0
  """Adhesion pull-back stiffness [N/m]."""

  f_adhesion_max: float = 50.0
  """Maximum adhesion pull-back force [N]."""

  # -- Slip tracking --

  max_cumulative_slip: float = 0.015
  """Maximum cumulative tangential slip [m] before full disengage."""

  # -- Wall geometry (will be computed at runtime, these are defaults) --

  wall_normal: tuple[float, float, float] = (-1.0, 0.0, 0.0)
  """Outward normal of the wall surface facing the robot (in world frame)."""

  # -- Sub-geom offset (hand contact surface offset from body frame origin) --

  hand_contact_offset_x: float = 0.08
  """X-offset of contact sub-geoms from hand body origin in local frame [m]."""
