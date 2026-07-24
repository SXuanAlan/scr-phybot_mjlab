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
  direction down the wall, l̂ = lateral, n̂ = outward wall normal). Two hold
  envelopes are available via ``model``:

  ``"wedge"`` (legacy piecewise-linear cone): an engaged spine HOLDs iff

    1. f_t >= -f_rev_max                       (small reverse-shear tolerance)
    2. f_n <= mu_adhesion * max(f_t, 0) + f_n0 (tension grows with engaged shear)
    3. |f_l| <= mu_lateral * max(f_t, 0) + f_l0
    4. f_t <= f_t_max                          (ultimate asperity/spine strength)

  ``"wang2017"`` (closed 3D surface, Wang/Jiang/Cutkosky IJRR 2017 eqs 1-4,
  9-12; see scr_phybot_mjab/microspine/wang_limit_surface.py): HOLD iff the
  load magnitude stays inside the direction-dependent capacity

    |(f_t, f_l, max(f_n, 0))| <= F0 * max(R(theta, phi), r_floor)

  where F0 is sampled from ``f_t_max_range`` (F0 = pure rake-shear capacity)
  and R is the normalized closed surface: tension capacity exists only with
  simultaneous engaged shear and the forward lobe closes at the origin, pure
  lateral holds ~0.57 F0, against-rake ~0.04 F0. The other wedge ranges are
  ignored in this mode. Compression is judged at phi = 0; pressing in rides
  the ordinary sphere-wall contacts (the paper's Sec 3.4 friction extension).

  All ``*_range`` fields are per-env uniform sampling bounds drawn at reset —
  built-in domain randomization. Wedge ranges map 1:1 to bench pull-off tests
  (scripts/fit_limit_surface.py); the wang2017 shape parameters are fit to
  the paper's Fig 8 anchors (scripts/plot_wang_limit_surface.py).
  """

  enable: bool = True
  """Evaluate the hold envelope. False keeps engaged spines latched forever."""

  model: str = "wedge"
  """Hold-envelope model: "wedge" (legacy linear cone) or "wang2017"."""

  rake_mode: str = "gravity"
  """Engage direction source: "gravity" (gravity projected onto the wall) or
  "ee_local" (EE-local axis projected onto the wall). "ee_local" is the
  physical model — spines rake along a pad-fixed direction, so capacity
  follows the pad orientation and a yawed pad pays for it (under "gravity"
  the envelope is blind to pad yaw: measured trained-policy rollouts latched
  hands 30-60 deg off-rake at full capacity)."""

  ee_rake_local_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """EE-local rake axis used when ``rake_mode == "ee_local"``."""

  f_t_max_range: tuple[float, float] = (300.0, 500.0)
  """Ultimate engaged shear capacity [N]. ~1.7-2.9x a 2-EE stance load.
  In wang2017 mode this doubles as F0, the pure rake-shear capacity that
  scales the whole closed surface (the paper's palm demonstrated ~700 N)."""

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

  # --- wang2017 shape parameters (dimensionless surface, scaled by F0) ------
  # Defaults are fit to the paper's single-tile 3D limit surface (Fig 8):
  # tension peak 0.12*F0 at ~15 deg elevation, pure lateral 0.57*F0, pure
  # reverse 0.043*F0, forward lobe closing near 30 deg, smooth in azimuth.
  # Verify any change with scripts/plot_wang_limit_surface.py.

  wang_mu_tip: float = 0.42
  """Spine-tip/asperity friction coefficient (paper Fig 5 uses ~0.4). Sets
  the slip threshold psi_c = arctan(1/mu) and, with beta and alpha, the
  forward-lobe closing elevation beta + 90 - alpha/2 - psi_c. Also controls
  the against-rake residual: arctan(1/mu) must sit just below
  -beta + 90 - alpha/2 (67.5 deg at the defaults) or reverse capacity either
  vanishes (mu <= 0.40) or grows too strong (mu >= 0.5)."""

  wang_delta_psi_deg: float = 90.0
  """Width of the usable asperity-slope population (paper eq 9's delta-psi).
  An effective surface-roughness parameter: larger = slower capacity falloff
  with loading angle."""

  wang_beta_deg: float = 15.0
  """Spine inclination angle from the surface plane (paper Sec 2.3)."""

  wang_alpha_deg: float = 15.0
  """Spine tip included angle (paper Sec 2.3)."""

  wang_mu_channel: float = 0.2
  """Spine/channel friction mu_s in the channel-slip factor (paper eq 4)."""

  wang_c_i: float = 50.0
  """Tip-engaged asperity strength relative to weak side/shaft contact
  (paper eq 2). Larger sharpens the rake anisotropy (smaller against-rake
  and lateral capacity)."""

  wang_r_floor: float = 0.02
  """Isotropic residual capacity as a fraction of F0. Models the stochastic
  scraps of adhesion measured where the deterministic surface predicts zero
  (paper Fig 6/8 scatter) and keeps margins well-conditioned as R -> 0.
  0.02 * F0(400..650) ~ 8-13 N, matching the old f_n0/f_rev bench feel."""

  wang_margin_eps: float = 0.02
  """Margin normalization softener as a fraction of F0: margin =
  (cap - |F|) / (cap + eps * F0). Keeps the normalized margin finite and
  smooth for loads pointed at low-capacity directions."""

  wang_lut_theta_bins: int = 181
  """Azimuth bins (0..180 deg) of the precomputed capacity table."""

  wang_lut_phi_bins: int = 91
  """Elevation bins (0..90 deg) of the precomputed capacity table."""

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
  window. Its equality can carry transient load inside each policy interval,
  but the next re-seat discards accumulated anchor error, so it has no fixed
  cross-step reference until grace ends. Cyclic re-engagement can therefore
  ratchet the reference and creep along the wall."""

  violation_consecutive_steps: int = 1
  """Violation debounce (1 = release on the first violating step).

  The 2-step debounce existed to survive one-step contact-churn equality
  spikes (~700 N transients from the anchor-vs-contact fight). With
  ``disable_contacts_while_latched`` the fight is gone at the source —
  latched anchor loads are smooth (measured: zero margin sign flips over
  held sequences) — so violations release immediately; a lingering debounce
  would only re-open the oscillating-overload loophole it once enabled
  (motionref6's 90-deg feet-hang). Raise above 1 only for legacy setups
  that keep contacts active while latched."""

  release_mode: str = "binary"
  """"binary" releases on any violation. "slip_drag" (anchor dragging /
  stick-slip) is reserved for a later milestone and currently rejected."""


@dataclasses.dataclass(kw_only=True)
class MicrospineCfg:
  """Configuration for constraint-based microspine wall contact.

  This config owns the physical layout, ordinary sphere-wall contact pairs,
  equality anchors, limit surface, and affordance engage/release thresholds.
  """

  ee_body_names: tuple[str, ...]
  """End-effector body names that have microspine contact surfaces."""

  spheres_per_ee_rows: int
  """Rows in the per-EE contact-sphere grid."""

  spheres_per_ee_cols: int
  """Columns in the per-EE contact-sphere grid."""

  grip_enabled: bool = True
  """Enable physical latch engagement. False is the explicit ablation mode."""

  affordance_name: str | None = None
  """Command-manager term name that provides the per-EE engage state."""

  wall_geom_name: str = "wall/wall_geom"
  """Scoped wall geom name used by generated sphere-wall contact pairs."""

  contact_friction: tuple[float, float, float, float, float] = (
    0.8,
    0.8,
    0.005,
    1.0e-4,
    1.0e-4,
  )
  """Ordinary sphere-wall contact friction; grip comes from the anchor."""

  contact_condim: int = 3
  """Sphere-wall contact dimension (normal plus two sliding directions)."""

  contact_solref: tuple[float, float] = (0.02, 1.0)
  """MuJoCo solref for ordinary microspine sphere-wall contacts."""

  contact_solimp: tuple[float, float, float, float, float] = (
    0.9,
    0.99,
    0.001,
    0.5,
    2.0,
  )
  """MuJoCo solimp for ordinary microspine sphere-wall contacts."""

  contact_margin: float = 0.01
  """MuJoCo pair margin for microspine sphere-to-wall contacts."""

  contact_gap: float = 0.01
  """MuJoCo pair gap for microspine sphere-to-wall contacts."""

  disable_contacts_while_latched: bool = False
  """Switch a latched pad's sphere-wall contacts off (per env, per EE).

  While latched the bilateral anchor already pins the pad, so the sphere
  contacts are a redundant second load path; the two fight over penetration
  corrections and the contact set flickers at near-zero normal force, which
  makes the measured anchor load oscillate frame-to-frame (single-step ~700 N
  equality transients; the 90-deg margin 'barcode'). Disabling the contacts
  while latched removes the fight at the source: compression and tension both
  ride the anchor, and contacts return the instant the pad releases so
  seating, press shaping and push-off are untouched. Implemented by
  retracting the sphere geoms into the pad (``geom_pos`` per world) — the
  collision pipeline ignores runtime ``pair_*`` writes but kinematics reads
  ``geom_pos`` every step."""

  latched_contact_retract: float = 0.05
  """Retraction depth [m] of a latched pad's contact spheres along the face
  axis (into the pad). Must exceed sphere radius + contact margin so no
  sphere-wall contact can be generated while latched."""

  debug: bool = False
  """Print engage-state diagnostics. This causes CPU-GPU sync and is slow."""

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

  reengage_cooldown_steps: int = 30
  """Steps after release during which re-engagement is blocked."""

  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
  """Local EE axis that should face the wall inward normal."""

  wall_entity_name: str = "wall"
  """Scene entity name used for wall orientation checks."""

  initial_on: bool = True
  """Initial per-EE engaged state after reset."""

  initial_engaged_ee_names: tuple[str, ...] | None = None
  """Optional per-EE override for the engaged state after reset.

  When set, only the listed entries from ``ee_body_names`` start engaged and
  ``initial_on`` is ignored. This is an environment initialization choice,
  not an action: the policy still has no engage/disengage control channel.
  ``None`` preserves the legacy all-on/all-off behavior from ``initial_on``.
  """

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

  anchor_lock_rotation: bool = False
  """Lock pad orientation while engaged (weld equality instead of connect).

  A real spine array grips asperities across the whole pad face: once
  engaged the pad cannot pivot about the anchor point — the wall holds its
  orientation, not the wrist servo. A connect leaves rotation free (single
  point = ball joint); measured under push DR that allowed up to ~20 deg of
  engaged-pad tilt and let policies latch pads at arbitrary yaw for free.
  With this flag each EE's equality is a weld whose relpose is captured at
  engage time and re-seated during the grace window exactly like the
  position anchor; releasing and re-engaging is the only way to reorient an
  engaged pad."""

  anchor_torquescale: float = 1.0
  """Weld torquescale (eq_data[10]): scales rotational residuals into the
  constraint impedance. Used only when ``anchor_lock_rotation``."""

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
  policy could in principle ratchet by repeatedly causing passive release and
  re-engagement: each fresh grace window lets the reference follow the pad
  again, with a gain of roughly millimeters per engage cycle."""

  limit_surface: MicrospineLimitSurfaceCfg = dataclasses.field(
    default_factory=MicrospineLimitSurfaceCfg
  )
  """Hold envelope evaluated by the affordance in constraint mode."""

  def resolve_initial_engaged_mask(self) -> tuple[bool, ...]:
    """Resolve the reset latch state in ``ee_body_names`` order."""
    if self.initial_engaged_ee_names is None:
      return (bool(self.initial_on),) * len(self.ee_body_names)

    selected = set(self.initial_engaged_ee_names)
    unknown = selected.difference(self.ee_body_names)
    if unknown:
      unknown_names = ", ".join(sorted(unknown))
      raise ValueError(
        "initial_engaged_ee_names contains names outside ee_body_names: "
        f"{unknown_names}"
      )
    return tuple(name in selected for name in self.ee_body_names)


@dataclasses.dataclass(kw_only=True)
class MicrospineAffordanceCfg:
  """Configuration for the environment-side microspine state machine."""

  microspine_cfg: MicrospineCfg
  asset_name: str = "robot"
  contact_sensor_name: str
  physics_event_name: str = "microspine_grip"
  """Event-term name of the constraint executor that exposes anchor loads."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv):
    """Build the runtime command term lazily to avoid registration-time cycles."""
    from scr_phybot_mjab.microspine.affordance import MicrospineAffordance

    return MicrospineAffordance(self, env)
