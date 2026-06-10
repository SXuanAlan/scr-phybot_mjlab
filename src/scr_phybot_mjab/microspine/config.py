"""Microspine configuration dataclasses."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclasses.dataclass(kw_only=True)
class MicrospineLimitSurfaceCfg:
  """Spine limit-surface (hold envelope) parameters.

  The anchor load is decomposed in the engage frame (t̂ = rake/engage
  direction down the wall, l̂ = lateral, n̂ = outward wall normal). An engaged
  spine HOLDs iff all of:

    1. f_t >= -f_rev_max                       (small reverse-shear tolerance)
    2. f_n <= mu_adhesion * max(f_t, 0) + f_n0 (tension grows with engaged shear)
    3. |f_l| <= mu_lateral * max(f_t, 0) + f_l0
    4. f_t <= f_t_max                          (ultimate asperity/spine strength)

  All ``*_range`` fields are per-env uniform sampling bounds drawn at reset —
  built-in domain randomization. They map 1:1 to bench pull-off tests (see
  docs/microspine_calibration.md and scripts/fit_limit_surface.py).
  """

  enable: bool = True
  """Evaluate the hold envelope. False keeps engaged spines latched forever."""

  rake_mode: str = "gravity"
  """Engage direction source: "gravity" (gravity projected onto the wall) or
  "ee_local" (EE-local axis projected onto the wall)."""

  ee_rake_local_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """EE-local rake axis used when ``rake_mode == "ee_local"``."""

  f_t_max_range: tuple[float, float] = (300.0, 500.0)
  """Ultimate engaged shear capacity [N]. ~1.7-2.9x a 2-EE stance load."""

  f_rev_max_range: tuple[float, float] = (15.0, 25.0)
  """Reverse-shear (pushed up the wall) capacity [N]."""

  mu_adhesion_range: tuple[float, float] = (0.3, 0.5)
  """Tension capacity per unit engaged shear (pull-off cone slope) [-]."""

  f_n0_range: tuple[float, float] = (10.0, 20.0)
  """Tension capacity at zero engaged shear [N]."""

  mu_lateral_range: tuple[float, float] = (0.5, 0.7)
  """Lateral capacity per unit engaged shear [-]."""

  f_l0_range: tuple[float, float] = (10.0, 20.0)
  """Lateral capacity at zero engaged shear [N]."""

  hold_grace_steps: int = 14
  """Policy steps after an engage transition before hold checks apply. The
  constraint only produces force on the next decimation loop, so the load is
  zero on the engage step itself. The window doubles as the re-grip seating
  phase (``MicrospineCfg.reanchor_after_grace``): each grace step re-seats the
  anchor at the settled pad, walking pose preload out to the structural
  equilibrium. Empirically (90° spawn-pose hang) convergence needs >10
  rounds: grace 6 → 31% engaged / mass releases, 10 → 93%, 14 → 100% with
  hold margins 0.2-0.8. 14 steps = 0.28 s of seating, the upper end of
  physical spine engagement times; a fresh engagement is unkillable for that
  window, but it cannot carry load either (the reseated anchor sheds force
  every step), so cyclic re-engagement only creeps down the wall."""

  violation_consecutive_steps: int = 2
  """Debounce for conditions 1-3. Condition 4 (ultimate shear) is instant."""

  release_mode: str = "binary"
  """"binary" releases on any violation. "slip_drag" (anchor dragging /
  stick-slip) is reserved for a later milestone and currently rejected."""


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

  disengage_consecutive_steps: int = 3
  """Number of consecutive disengage-candidate steps required before release."""

  reengage_cooldown_steps: int = 30
  """Steps after release during which re-engagement is blocked."""

  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
  """Local EE axis that should face the wall inward normal."""

  wall_entity_name: str = "wall"
  """Scene entity name used for wall orientation checks."""

  initial_on: bool = True
  """Initial per-EE engaged state after reset."""

  attachment_mode: str = "adhesion"
  """Physical grip mechanism: "adhesion" (legacy: external wall-normal force +
  high-friction contacts) or "constraint" (per-EE connect equality "virtual
  anchor" toggled at runtime; anisotropy expressed by the limit surface)."""

  anchor_solref: tuple[float, float] = (0.01, 1.0)
  """Connect-equality solref. timeconst = 5 * dt keeps implicitfast stable."""

  anchor_solimp: tuple[float, float, float, float, float] = (
    0.9,
    0.95,
    0.001,
    0.5,
    2.0,
  )
  """Connect-equality solimp."""

  anchor_project_to_wall: bool = True
  """Project the engage-time anchor point onto the wall surface plane.

  Without this, anchoring a pad that slightly penetrates the wall (spawn
  pose, or pressing in before engaging) locks the penetration in: the wall
  contact pushes the pad out against the anchor and the constraint reads a
  phantom pull-off tension, tripping the limit surface."""

  reanchor_after_grace: bool = True
  """Re-seat the anchor at the settled pad during the grace window ("re-grip").

  Anchoring a PD-held pose locks in whatever pad tilt / penetration the pose
  has; the contacts then push against the anchor and the standoff reads as
  large phantom pull-off tension (spawn pose: >100 N on the feet) that a real
  spine envelope correctly refuses to hold. Physically, a spine array seats
  into the asperities under its initial load. Re-writing the anchor to the
  *unprojected* settled pad position on every grace step (executor engage-age
  1 .. hold_grace_steps - 1) lets the pad walk out of the standoff so the
  first limit-surface check sees the true structural load of the pose. A
  policy could in principle ratchet by cyclic re-engagement, but the gain is
  ~mm per engage cycle."""

  physical_pair_friction: tuple[float, float, float, float, float] = (
    0.8,
    0.8,
    0.005,
    1.0e-4,
    1.0e-4,
  )
  """Sphere-wall pair friction in constraint mode. Tangential grip is the
  anchor's job; contacts only handle compression and pre-engage sliding."""

  physical_pair_condim: int = 3
  """Sphere-wall pair condim in constraint mode (saves 16x3 efc rows vs 6)."""

  physical_pair_solref: tuple[float, float] = (0.02, 1.0)
  """Sphere-wall pair solref in constraint mode (drops the dampratio=5
  anti-jitter patch needed by the high-friction legacy contacts)."""

  limit_surface: MicrospineLimitSurfaceCfg = dataclasses.field(
    default_factory=MicrospineLimitSurfaceCfg
  )
  """Hold envelope evaluated by the affordance in constraint mode."""


@dataclasses.dataclass(kw_only=True)
class MicrospineAffordanceCfg:
  """Configuration for the environment-side microspine state machine."""

  microspine_cfg: MicrospineCfg
  asset_name: str = "robot"
  contact_sensor_name: str
  physics_event_name: str = "microspine_adhesion"
  """Event-term name of the physics executor. In constraint mode the
  affordance reads per-EE anchor loads from it for the limit surface."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv):
    """Build the runtime command term lazily to avoid registration-time cycles."""
    from scr_phybot_mjab.microspine.affordance import MicrospineAffordance

    return MicrospineAffordance(self, env)
