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

  hold_judge_guard_steps: int = 2
  """Policy steps after an engage transition before hold checks apply — ONLY
  the constraint-force readout settling (the equality produces force one
  decimation late, so the load reads zero on the engage step itself).

  2 (was hold_grace_steps=14 doing this job): the unjudged window is gone —
  a young anchor is judged from its first real load reading, it is simply
  WEAK until seated (``capacity_ramp_*`` below). The old 14-step immunity
  let policies yank fresh anchors for free, which reality does not."""

  capacity_ramp_floor: float = 0.3
  """Fresh-anchor capacity fraction at the engage step. Spine recruitment
  starts partial — only the spines that caught during the press are loaded;
  a just-latched pad is real but weak (Wang 2017: capacity is the sum of
  CAUGHT spines, and catching continues through the loading phase)."""

  capacity_ramp_time_s: float = 0.25
  """Rake-shear-loaded seconds to full capacity. Recruitment progresses only
  while the anchor carries rake shear (>= ``capacity_ramp_shear_n``):
  dragging the loaded array down its asperities seats more spines (the
  paper's loading phase). An anchor never loaded in shear stays partially
  recruited indefinitely — as it should. Seeded (RSI) latches start fully
  charged (the reference stance is a mature grip). 0 disables the ramp."""

  capacity_ramp_shear_n: float = 10.0
  """Rake-shear threshold [N] above which recruitment time accumulates."""

  redraw_f_t_max_per_latch: bool = True
  """Resample F0 from ``f_t_max_range`` at every engage transition. Each
  grip lands on different asperities — capacity is a per-GRIP lottery
  (paper Fig 6), not a per-episode trait the policy can memorize. The
  reset-time draw still seeds initial latches; a collapsed range (bench)
  redraws the same value, keeping bench determinism."""

  hold_grace_steps: int = 14
  """Re-seat window length in policy steps (the judging role moved to
  ``hold_judge_guard_steps``). During this window the anchor may follow the
  pad per ``MicrospineCfg.anchor_reseat_mode`` ("down": rake-direction
  seating creep only). Historical note — the free-follow era sized this by
  engage survival (grace 6 → 31%, 10 → 93%, 14 → 100%); under retraction
  the 9-grid recalibration (07-27) shows survival 8/8 at every window, so
  the length now only bounds the seating creep: 14 steps = 0.28 s, the
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

  engage_min_spheres: int = 1
  """If > 0, overrides engage_min_fraction with an absolute sphere count.

  1 (was 2): a rigid pad on a flat wall generically touches with ONE sphere —
  measured on motionref8, hands pressed 78 N through a single sphere for 99%
  of touching steps and the 2-sphere gate never passed once, making every
  release an irreversible loss (the mechanical root of four generations of
  welded-limb policies). Seating quality is now judged by the virtual spine
  field (``engage_min_spine_fraction``), which is the honest representation:
  per Wang et al. 2017 the real pad's normal suspension is ~5 mN/mm per spine
  (force-free conformity over ``le_max`` = 5 mm of travel), so which spines
  can reach the wall is a kinematic question, not a force question."""

  engage_parallel_cos_threshold: float = 0.25
  """Minimum dot(face axis, wall inward normal) required to engage."""

  virtual_spine_grid: tuple[int, int] = (5, 5)
  """Virtual spine field resolution (rows, cols) across the pad face.

  The four contact-sphere centers span the pad; a bilinear grid over that
  quad stands in for the real pad's hundreds of individually-suspended
  spines. Purely kinematic — the paper's normal suspension (kn = 5 mN/mm)
  carries negligible force, so 'can this spine reach the wall' is geometry."""

  virtual_spine_travel: float = 0.005
  """Reach [m] of a virtual spine beyond the sphere contact surface.

  Wang et al. 2017 Sec 2.3: le_max = 5 mm maximum spine extension. A virtual
  spine within this distance of the wall counts as engaged/hooked. Over the
  pad span this sets the seating tilt tolerance (~7 deg for a 40 mm pad)."""

  engage_min_spine_fraction: float = 0.5
  """Minimum engaged fraction of the virtual spine field required to latch.

  Replaces the unpassable 2-rigid-sphere gate as the 'is the pad actually
  seated' judgement: a corner poke reaches ~0.2, a flat seat ~1.0. 0 disables
  the gate."""

  engage_max_antirake_vel: float = 0.02
  """Maximum pad velocity AGAINST the rake (m/s) allowed to latch.

  Wang 2017 spines hook only under along-rake loading: a pad sliding
  up-wall (backing off its rake) physically cannot catch. Without this
  gate, a foot the body still stands on during a scheduled swing keeps
  re-latching while it slides upward — and an RL policy can farm anchors
  mid-scramble the hardware would never grant. Downward sliding and
  stationary pads latch normally, so a falling pad still catches (the
  microspine's whole point). ≤ 0 disables the gate."""

  seat_quality_scales_capacity: bool = True
  """Scale the wang2017 hold capacity F0 by the engaged spine fraction
  captured at the latch step. Wang et al. 2017 Sec 2.4: pad capacity is the
  sum of per-spine contributions, so a half-seated pad holds half the load.
  Gives seating quality a continuous physical payoff instead of a binary
  gate. Wedge (legacy) mode ignores this."""

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

  anchor_reseat_mode: str = "down"
  """Grace-window re-seat behavior when ``reanchor_after_grace`` is on.

  "free" (legacy): the anchor follows the settled pad in all six DOF each
  grace step. Tuned when latched pads still had live wall contacts to fight
  (grace 6 -> 31% engage survival, 14 -> 100%); with contacts now retracted
  at latch that standoff is gone, and free following became a drift license
  — the press->hang load flip dragged hand anchors 12-44 mm outward before
  the weld froze (motionref12 probe), something real hooked spines cannot
  do.

  "down": the anchor follows only the pad's down-the-wall rake component,
  ``anchor_reseat_step_cap`` per step, orientation frozen at the engage
  frame. This is the honest Wang 2017 seating motion — spines seat by
  dragging down into asperities; they never relocate outward, upward or
  sideways. The anchor also stays in the wall plane (the rake direction is
  in-plane), killing the hovering-pad visual."""

  anchor_reseat_step_cap: float = 0.001
  """Per-step cap [m] on the "down" re-seat follow (~real spine seating
  travel per control step; total drift bounded by cap * grace steps)."""

  anchor_project_to_wall: bool = False
  """Project the engage-time anchor point onto the wall surface plane.

  False (was True): hooked spines hold the pad exactly where it gripped —
  projection moved the anchor onto the plane while the pad sat compressed
  a few mm INTO it, so the moment the contacts retracted the weld snapped
  the pad outward (the 4-11 mm post-engage pop measured on motionref13).
  The original phantom-tension rationale died with retraction: there is no
  live contact left to push the pad against the anchor while latched. The
  pad simply stays at its (mm-compressed) grip pose; on release the
  restored contacts see the same latch-era compression they always did."""

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
