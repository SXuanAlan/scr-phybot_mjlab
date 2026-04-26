"""Microspine configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MicrospineCfg:
  """Configuration for microspine adhesion via MuJoCo adhesion actuators.

  Each end-effector has a grid of contact spheres. An adhesion actuator
  is added per sphere.

  Engage logic (when engage_force_threshold > 0):
    - Per-hand latch: adhesion for a hand is OFF by default.
    - When ALL spheres on that hand simultaneously detect >= threshold
      normal force against the wall, the hand latches to ENGAGED.
    - Once engaged, adhesion stays on until:
      (a) episode reset, OR
      (b) friction force along the hand's up direction exceeds
          disengage_friction_threshold (if > 0).
  """

  ee_body_names: tuple[str, ...] = ("right_hand",)
  """End-effector body names that have microspine contact surfaces."""

  adhesion_force: float = 50.0
  """Per-hand adhesion force when engaged [N]. Each latched hand pulls its
  contact face into the wall with up to this many newtons, independently of
  other hands. Internally split evenly across the 8 contact spheres on the
  hand (force_per_sphere = adhesion_force / spheres_per_hand)."""

  ctrl_value: float = 1.0
  """Adhesion actuator control signal [0, 1]. Actual force = gain * ctrl.
  Lower values reduce contact oscillation at the cost of weaker adhesion."""

  engage_force_threshold: float = 0.0
  """Minimum normal contact force [N] required per sphere to latch adhesion.
  When > 0, a hand latches when the fraction of spheres exceeding this
  threshold reaches ``engage_min_fraction``. Set to 0 to disable (always on)."""

  engage_min_fraction: float = 1.0
  """Fraction of spheres on a hand (in [0, 1]) that must simultaneously exceed
  ``engage_force_threshold`` to latch the hand. Default 1.0 = all spheres
  required. Lower values accept partial contact (e.g. 0.75 = at least 6 of 8
  spheres). If ``engage_min_spheres`` > 0 it overrides this fraction."""

  engage_min_spheres: int = 0
  """If > 0, overrides ``engage_min_fraction``: a hand latches as soon as at
  least this many spheres simultaneously exceed ``engage_force_threshold``.
  Use 1 for a lenient "any contact is enough" policy."""

  engage_parallel_cos_threshold: float = 0.0
  """If > 0, additionally require the hand's contact face to be sufficiently
  parallel to the wall surface before engagement. Specifically, the dot
  product between the hand's contact-face normal (local -x, in world) and the
  wall's inward normal must be >= this threshold. 0 disables the check.
  cos(30°) ≈ 0.866 is a reasonable "roughly flat against the wall" value."""

  wall_entity_name: str = "wall"
  """Scene entity name of the wall, used to query its current world pose for
  the parallel-hand engagement check. Only consulted when
  ``engage_parallel_cos_threshold`` > 0."""

  disengage_friction_threshold: float = 0.0
  """Friction force [N] along hand's up direction that causes disengage.
  When > 0 and a hand is engaged, if the total friction force on that hand
  projected onto hand_up_local exceeds this value, the hand disengages.
  Set to 0 to disable (only reset disengages)."""

  disengage_consecutive_steps: int = 1
  """Disengage hysteresis: require ``fric_up > threshold`` for this many
  consecutive steps before actually releasing. 1 = no hysteresis (release
  immediately on first violation). Larger values suppress bang-bang
  oscillation caused by CCD jitter in the solver."""

  disengage_pull_threshold: float = 0.0
  """Disengage when the kinematic chain pulls the EE outward (away from wall)
  with force exceeding this value [N]. Approximated as
  ``adhesion_force − sum(wall_normal_force_per_sphere)``: a fully-pressed EE
  has ``contact ≈ adhesion`` so ``pull_force ≈ 0``; a fully-lifted EE has
  ``contact ≈ 0`` so ``pull_force ≈ adhesion``. Threshold should be set
  somewhere between, e.g. 0.7–0.9 × ``adhesion_force``. Set to 0 to disable.
  This is the natural "policy lifts limb to release" criterion for climbing
  gait, complementing or replacing the slip-overload-style
  ``disengage_friction_threshold``."""

  disengage_pull_consecutive_steps: int = 1
  """Pull-disengage hysteresis. Same semantics as
  ``disengage_consecutive_steps`` but for the pull-based criterion."""

  disengage_lift_velocity_threshold: float = 0.0
  """Disengage when the EE body's world-frame velocity along ``+z`` exceeds
  this value [m/s]. The natural "policy lifts limb" signal — independent of
  body-local axes (no per-EE flip needed) and not conflated with gravity-
  supporting friction. For wall climbing, set above the commanded climb
  rate (so that body-along-wall motion doesn't trigger) but below the
  velocity reached during a deliberate stepping motion. 0 disables."""

  disengage_lift_velocity_consecutive_steps: int = 1
  """Lift-velocity disengage hysteresis. Same semantics as
  ``disengage_consecutive_steps``."""

  reengage_cooldown_steps: int = 0
  """After an engaged hand disengages, block re-engagement for this many
  steps. 0 disables the cooldown. Combined with the disengage hysteresis,
  this breaks single-step engage/disengage oscillation."""

  hand_up_local: tuple[float, float, float] | tuple[tuple[float, float, float], ...] = (0.0, 0.0, 1.0)
  """Local direction in each EE body frame that defines 'up' for the disengage
  friction check. Either a single 3-tuple (broadcast to every EE) or a tuple
  of 3-tuples with one entry per EE in ``ee_body_names``. Default (0,0,1)
  = local +z axis. Used only when ``friction_axis_world`` is ``None``."""

  friction_axis_world: tuple[float, float, float] | None = None
  """If set, the friction-based disengage check projects onto this WORLD-frame
  axis instead of each EE's body-rotated ``hand_up_local``. Useful when 'up'
  is the same world direction across every EE (e.g. ``(0, 0, 1)`` for
  vertical-wall climbing) so per-EE body-local flips become unnecessary.
  None falls back to the legacy body-local behavior."""

  spheres_per_ee_rows: int = 2
  """Rows in the per-EE contact-sphere grid. Sphere body/geom names follow
  ``{ee_body_name}_c_{i}_{j}`` for i in [rows], j in [cols]. Default 2 keeps
  phybot_big_v2's 2x4 = 8-sphere layout."""

  spheres_per_ee_cols: int = 4
  """Cols in the per-EE contact-sphere grid. Default 4 keeps phybot_big_v2's
  2x4 = 8-sphere layout."""

  wall_geom_name: str = "wall/wall_geom"
  """Scoped name of the wall geom used for contact force detection."""

  debug: bool = False
  """Print per-step engage diagnostics (causes CPU-GPU sync, slow)."""

  disengage_threshold_start: float = 0.0
  """Curriculum: starting value of the disengage friction threshold [N].
  Linearly ramps to ``disengage_friction_threshold`` over
  ``disengage_threshold_ramp_steps`` global env-step calls. 0 disables the
  curriculum (uses ``disengage_friction_threshold`` directly)."""

  disengage_threshold_ramp_steps: int = 0
  """Number of global env-step calls over which the disengage threshold
  linearly ramps from ``disengage_threshold_start`` to
  ``disengage_friction_threshold``. 0 disables the curriculum."""

  adhesion_force_start: float = 0.0
  """If > 0, enables per-hand engage ramp: starting effective adhesion force [N]
  applied the instant a hand latches. Ramps exponentially to
  ``adhesion_force_end`` over ``adhesion_ramp_duration_s`` seconds of continuous
  engagement. Actuator gain is fixed at spec time to ``adhesion_force``;
  the ramp is applied by scaling the per-hand ctrl."""

  adhesion_force_end: float = 0.0
  """If > 0, enables the ramp: end value of effective adhesion force [N].
  Must be <= ``adhesion_force`` (the hard actuator ceiling)."""

  adhesion_ramp_duration_s: float = 0.0
  """Seconds of continuous engagement over which each hand's effective adhesion
  force ramps exponentially from start to end. 0 disables the ramp."""
