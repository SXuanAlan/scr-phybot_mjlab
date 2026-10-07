"""Microspine configuration dataclasses."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclasses.dataclass(kw_only=True)
class MicrospineLimitSurfaceCfg:
  """Hold-envelope (limit-surface) parameters of a latched pad.

  The anchor load is decomposed in the engage frame (t̂ = rake direction down
  the wall, l̂ = lateral, n̂ = outward wall normal) and the pad wrench is
  distributed onto the active units (equal-stiffness bolt-group statics).
  Each unit is judged by the empirical model: a conditional inverse CDF
  fitted to single-unit bench observations gives its normal-load threshold
  at its current shear/lateral load. Two independent quantiles per unit are
  held for an entire engagement (strength and early loading failure). Unit
  independence, cross-direction rank coupling and first-unit pad release are
  unvalidated array assumptions.
  """

  enable: bool = True
  """Evaluate the hold envelope. False keeps engaged spines latched forever
  (diagnostic benches only)."""

  empirical_table_path: str | None = None
  """Single-unit conditional inverse-CDF NPZ; None selects the packaged table."""

  ee_rake_local_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """EE-local rake axis. The empirical table assumes the pad's -z."""

  hold_judge_guard_steps: int = 2
  """Policy steps after an engage transition before hold checks apply: the
  constraint produces force one decimation late, so the load reads zero on
  the engage step itself. There is no immunity window beyond that."""

  unit_layout: str = "all"
  """Active unit sites: "all" or the four outer "corners".

  Disabled units contribute neither preload nor holding capacity. This
  changes the load-bearing array, not the four ordinary contact proxies. The
  unit grid comes from the selected CAD sites.
  """

  seed_hold_judge_guard_steps: int | None = None
  """Guard for latches seeded at reset (RSI); None uses ``hold_judge_guard_steps``.

  A whole-body RSI settle can put a one-shot shear and tension transient on
  seeded welds exactly at the first judged step; a longer guard lets it pass.
  Seeds stay seeds only while continuously engaged since birth, so the guard
  never shields in-play engagements."""

  hold_grace_steps: int = 14
  """Re-seat window in policy steps (see ``MicrospineCfg.reanchor_after_grace``).
  14 steps = 0.28 s, the physical spine engagement time. The window bounds
  the seating creep; cyclic re-engagement can ratchet the reference along
  the wall."""

  violation_consecutive_steps: int = 1
  """Violation debounce (1 = release on the first violating step). Latched
  pads' contact proxies are retracted, so anchor loads are smooth and a
  longer debounce would only let oscillating overloads flicker through."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class MicrospineSurfaceCfg:
  """A finite rectangular grasp face, expressed in its target body's frame.

  ``normal`` points out of the object; ``tangent`` spans the first half-extent.
  The second tangent is normal cross tangent. Names are entity-local.
  """

  ee_body_name: str
  entity_name: str
  body_name: str
  geom_name: str
  point: tuple[float, float, float]
  normal: tuple[float, float, float]
  tangent: tuple[float, float, float]
  half_extents: tuple[float, float]

  def __post_init__(self):
    import math

    if any(len(axis) != 3 for axis in (self.point, self.normal, self.tangent)) or len(self.half_extents) != 2:
      raise ValueError("Surface point and axes must be 3D; half-extents must have two entries")
    if not all(math.isfinite(v) for row in (self.point, self.normal, self.tangent, self.half_extents) for v in row):
      raise ValueError("Surface geometry must be finite")
    if any(v <= 0 for v in self.half_extents):
      raise ValueError("Surface half-extents must be positive")
    if any(abs(sum(v * v for v in axis) - 1.0) > 1e-6 for axis in (self.normal, self.tangent)):
      raise ValueError("Surface axes must be unit vectors")
    if abs(sum(a * b for a, b in zip(self.normal, self.tangent))) > 1e-6:
      raise ValueError("Surface normal and tangent must be orthogonal")

  @property
  def scoped_body(self) -> str:
    return f"{self.entity_name}/{self.body_name}"

  @property
  def scoped_geom(self) -> str:
    return f"{self.entity_name}/{self.geom_name}"


@dataclasses.dataclass(kw_only=True)
class MicrospineCfg:
  """Configuration for constraint-based microspine wall contact.

  This config owns the physical layout, ordinary corner-wall contact pairs,
  equality anchors, limit surface, and affordance engage/release thresholds.
  """

  ee_body_names: tuple[str, ...]
  """End-effector body names that have microspine contact surfaces."""

  spheres_per_ee_rows: int
  """Rows in the per-EE contact-proxy grid (legacy field name)."""

  spheres_per_ee_cols: int
  """Columns in the per-EE contact-proxy grid (legacy field name)."""

  grip_enabled: bool = True
  """Enable physical latch engagement. False is the explicit ablation mode."""

  affordance_name: str | None = None
  """Command-manager term name that provides the per-EE engage state."""

  wall_geom_name: str = "wall/wall_geom"
  """Scoped wall geom name used by generated proxy-wall contact pairs."""

  surfaces: tuple[MicrospineSurfaceCfg, ...] = ()
  """Optional per-EE moving faces. Empty anchors the pads to the world wall."""

  arm_wall_contact_geoms: tuple[str, ...] = ()
  """Arm collision geoms given ordinary wall contact through explicit pairs
  (``add_arm_wall_contact_pairs``)."""

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

  latched_contact_retract: float = 0.05
  """Retraction depth [m] of a latched pad's contact proxies along the face
  axis (into the pad). Must clear penetration and the contact margin over
  the supported pad-tilt range; box edges must also clear the wall."""

  engage_unit_preload_n: float = 10.0
  """Minimum estimated normal compression [N] on EVERY active unit before latching.

  Uses the C-referenced contact wrench and the equal-stiffness unit allocator,
  including bending moments. This is a model estimate from corner contacts,
  not a measurement of individual units, and needs a netforce sensor with
  synchronized force, torque and position histories. Only gates OFF -> ON;
  unloading after latching is judged by the hold surface. Zero disables the
  gate.
  """

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
  """Minimum engaged fraction of the virtual spine field required to latch:
  the seating judgement (a corner poke reaches ~0.2, a flat seat ~1.0).
  0 disables the gate."""

  engage_max_antirake_vel: float = 0.02
  """Maximum pad velocity AGAINST the rake (m/s) allowed to latch.

  Wang 2017 spines hook only under along-rake loading: a pad sliding
  up-wall (backing off its rake) physically cannot catch. Without this
  gate, a foot the body still stands on during a scheduled swing keeps
  re-latching while it slides upward — and an RL policy can farm anchors
  mid-scramble the hardware would never grant. Downward sliding and
  stationary pads latch normally, so a falling pad still catches (the
  microspine's whole point). ≤ 0 disables the gate."""

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
  ``None`` applies ``initial_on`` to every EE.
  """

  anchor_solref: tuple[float, float] = (0.01, 1.0)
  """Anchor weld solref. timeconst = 5 * dt keeps implicitfast stable."""

  anchor_solimp: tuple[float, float, float, float, float] = (
    0.9,
    0.95,
    0.001,
    0.5,
    2.0,
  )
  """Anchor weld solimp."""

  anchor_torquescale: float = 1.0
  """Weld torquescale (eq_data[10]): the pad's rotational compliance at the anchor.

  It scales rotational residuals into the constraint impedance, so in a
  redundant multi-weld stance it decides how much parasitic moment a
  strained pad traps versus sheds into force redistribution across the
  other pads. The same value converts the weld's rotational multipliers
  back into physical moments (``weld_rotational_force_to_world_moment``),
  so measurement and solver behavior stay consistent."""

  anchor_reseat_step_cap: float = 0.001
  """Per-step cap [m] on the re-seat follow, about the real spine seating
  travel per control step (total drift is bounded by cap * grace steps)."""

  reanchor_after_grace: bool = True
  """Re-seat a fresh anchor during the grace window ("re-grip").

  A spine array seats into the asperities under its initial load: for
  ``limit_surface.hold_grace_steps`` after a latch the anchor follows only
  the pad's down-the-wall rake motion (``anchor_reseat_step_cap`` per step)
  and the weld orientation conforms to the pad, then both freeze. Hooked
  spines never relocate outward, upward or sideways. RSI-seeded anchors
  skip this lifecycle. Repeated release and re-engagement can ratchet the
  reference along the wall by about a millimeter per cycle."""

  limit_surface: MicrospineLimitSurfaceCfg = dataclasses.field(default_factory=MicrospineLimitSurfaceCfg)
  """Hold envelope evaluated by the affordance."""

  def resolve_initial_engaged_mask(self) -> tuple[bool, ...]:
    """Resolve the reset latch state in ``ee_body_names`` order."""
    if self.initial_engaged_ee_names is None:
      return (bool(self.initial_on),) * len(self.ee_body_names)

    selected = set(self.initial_engaged_ee_names)
    unknown = selected.difference(self.ee_body_names)
    if unknown:
      unknown_names = ", ".join(sorted(unknown))
      raise ValueError(f"initial_engaged_ee_names contains names outside ee_body_names: {unknown_names}")
    return tuple(name in selected for name in self.ee_body_names)

  def ordered_surfaces(self) -> tuple[MicrospineSurfaceCfg, ...]:
    if not self.surfaces:
      return ()
    by_name = {s.ee_body_name: s for s in self.surfaces}
    if len(by_name) != len(self.surfaces) or set(by_name) != set(self.ee_body_names):
      raise ValueError("surfaces must bind every configured EE exactly once")
    return tuple(by_name[name] for name in self.ee_body_names)


@dataclasses.dataclass(kw_only=True)
class MicrospineAffordanceCfg:
  """Configuration for the environment-side microspine state machine."""

  microspine_cfg: MicrospineCfg
  skip_zero_time_update: bool = False
  """Do not advance contact debouncing or hold ages during reset compute(dt=0)."""
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
