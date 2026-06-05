"""Microspine configuration dataclasses."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import mujoco

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


@dataclasses.dataclass(kw_only=True)
class MicrospineCfg:
  """Configuration for MuJoCo microspine adhesion actuators.

  This config owns the physical layout: end-effector names, contact-sphere
  grid, wall contact pairs, and adhesion actuator strength. Engage/disengage
  decisions live in ``MicrospineAffordanceCfg``.
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  adhesion_force: float = 50.0
  """Per-EE adhesion force [N], split evenly across that EE's spheres."""

  ctrl_value: float = 1.0
  """Adhesion actuator control signal [0, 1]."""

  affordance_name: str | None = None
  """Command-manager term name that provides the per-EE engage state."""

  spheres_per_ee_rows: int = 2
  """Rows in the per-EE contact-sphere grid."""

  spheres_per_ee_cols: int = 4
  """Columns in the per-EE contact-sphere grid."""

  wall_geom_name: str = "wall/wall_geom"
  """Scoped wall geom name used by generated sphere-wall contact pairs."""

  wall_pair_friction: tuple[float, float, float, float, float] = (
    30.0,
    30.0,
    1.0,
    1.0,
    1.0,
  )
  """MuJoCo pair friction for microspine sphere-to-wall contacts."""

  wall_pair_solref: tuple[float, float] = (0.02, 5.0)
  """MuJoCo pair solref for microspine sphere-to-wall contacts."""

  wall_pair_solimp: tuple[float, float, float, float, float] = (
    0.9,
    0.99,
    0.001,
    0.5,
    2.0,
  )
  """MuJoCo pair solimp for microspine sphere-to-wall contacts."""

  wall_pair_margin: float = 0.01
  """MuJoCo pair margin for microspine sphere-to-wall contacts."""

  wall_pair_gap: float = 0.01
  """MuJoCo pair gap for microspine sphere-to-wall contacts."""

  debug: bool = False
  """Print engage-state diagnostics. This causes CPU-GPU sync and is slow."""

  @property
  def spheres_per_ee(self) -> int:
    """Number of contact spheres per end-effector."""
    return self.spheres_per_ee_rows * self.spheres_per_ee_cols

  @property
  def sphere_geom_names(self) -> tuple[str, ...]:
    """All microspine sphere geom names derived from EE names and grid shape."""
    return tuple(
      f"{ee}_c_{i}_{j}"
      for ee in self.ee_body_names
      for i in range(self.spheres_per_ee_rows)
      for j in range(self.spheres_per_ee_cols)
    )

  def add_contact_pairs_and_actuators(
    self, spec: mujoco.MjSpec, robot_prefix: str = "robot"
  ) -> None:
    """Add wall contact pairs and per-sphere adhesion actuators."""
    from scr_phybot_mjab.microspine.physics import add_adhesion_actuators

    for gname in self.sphere_geom_names:
      pair = spec.add_pair()
      pair.geomname1 = f"{robot_prefix}/{gname}"
      pair.geomname2 = self.wall_geom_name
      pair.condim = 6
      pair.friction = list(self.wall_pair_friction)
      pair.solref = list(self.wall_pair_solref)
      pair.solimp = list(self.wall_pair_solimp)
      pair.solreffriction = list(self.wall_pair_solref)
      pair.margin = self.wall_pair_margin
      pair.gap = self.wall_pair_gap

    add_adhesion_actuators(
      spec,
      self.ee_body_names,
      self.adhesion_force,
      robot_prefix=robot_prefix,
      rows=self.spheres_per_ee_rows,
      cols=self.spheres_per_ee_cols,
    )

  def make_physics_event_cfg(
    self,
    affordance_name: str | None = None,
    entity_name: str = "robot",
  ) -> EventTermCfg:
    """Build the step event term that applies physical microspine adhesion."""
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from scr_phybot_mjab.microspine.physics import MicrospinePhysics

    return EventTermCfg(
      func=MicrospinePhysics,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg(entity_name),
        "microspine_cfg": dataclasses.replace(
          self,
          affordance_name=affordance_name,
        ),
      },
    )


@dataclasses.dataclass(kw_only=True)
class MicrospineAffordanceCfg:
  """Configuration for the environment-side microspine state machine."""

  ee_body_names: tuple[str, ...]
  asset_name: str = "robot"
  contact_sensor_name: str
  spheres_per_ee: int = 1
  engage_force_z_threshold: float = 0.0
  engage_normal_force_threshold: float = 0.0
  engage_min_fraction: float = 1.0
  engage_min_spheres: int = 0
  engage_parallel_cos_threshold: float = 0.0
  engage_consecutive_steps: int = 1
  disengage_force_z_threshold: float = 0.0
  disengage_consecutive_steps: int = 1
  reengage_cooldown_steps: int = 0
  face_local_axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
  wall_entity_name: str = "wall"
  initial_on: bool = True
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv):
    """Build the runtime command term lazily to avoid registration-time cycles."""
    from scr_phybot_mjab.microspine.affordance import MicrospineAffordance

    return MicrospineAffordance(self, env)
