"""Microspine configuration dataclasses."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclasses.dataclass(kw_only=True)
class MicrospineCfg:
  """Configuration for microspine wall contact and external adhesion forces.

  This config owns the physical layout: end-effector names, contact-sphere
  grid, wall contact pairs, external adhesion force strength, and affordance
  engage/disengage thresholds.
  """

  ee_body_names: tuple[str, ...]
  """End-effector body names that have microspine contact surfaces."""

  adhesion_force: float
  """Per-EE adhesion force [N], split evenly across that EE's spheres."""

  spheres_per_ee_rows: int
  """Rows in the per-EE contact-sphere grid."""

  spheres_per_ee_cols: int
  """Columns in the per-EE contact-sphere grid."""

  ctrl_value: float = 1.0
  """External adhesion force scale [0, 1]."""

  affordance_name: str | None = None
  """Command-manager term name that provides the per-EE engage state."""

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

  engage_force_z_threshold: float = 10
  """Minimum summed wall-on-EE world-z force [N] required to engage. 0 disables."""

  engage_normal_force_threshold: float = 3.0
  """Minimum per-sphere normal compression [N] used by the engage condition."""

  engage_min_fraction: float = 0.25
  """Fraction of an EE's spheres that must pass the normal-force threshold."""

  engage_min_spheres: int = 2
  """If > 0, overrides engage_min_fraction with an absolute sphere count."""

  engage_parallel_cos_threshold: float = 0.25
  """Minimum dot(face axis, wall inward normal) required to engage."""

  engage_consecutive_steps: int = 5
  """Number of consecutive engage-condition steps required before enabling."""

  disengage_force_z_threshold: float = 3.0
  """Release when summed wall-on-EE world-z force is below -threshold. 0 disables."""

  disengage_consecutive_steps: int = 5
  """Number of consecutive disengage-candidate steps required before release."""

  reengage_cooldown_steps: int = 15
  """Steps after release during which re-engagement is blocked."""

  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
  """Local EE axis that should face the wall inward normal."""

  wall_entity_name: str = "wall"
  """Scene entity name used for wall orientation checks."""

  initial_on: bool = True
  """Initial per-EE engaged state after reset."""


@dataclasses.dataclass(kw_only=True)
class MicrospineAffordanceCfg:
  """Configuration for the environment-side microspine state machine."""

  microspine_cfg: MicrospineCfg
  asset_name: str = "robot"
  contact_sensor_name: str
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv):
    """Build the runtime command term lazily to avoid registration-time cycles."""
    from scr_phybot_mjab.microspine.affordance import MicrospineAffordance

    return MicrospineAffordance(self, env)
