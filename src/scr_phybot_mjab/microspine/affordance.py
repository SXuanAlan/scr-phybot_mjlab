"""Environment-side affordance model for microspine grip."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import warp as wp

from mjlab.managers.command_manager import CommandTerm
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg
from scr_phybot_mjab.microspine.wang_limit_surface import (
  build_capacity_lut,
  capacity_ratio,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# release_reason code for releases commanded from outside the limit-surface
# judgement (DR random-release event, scripted diagnostics). Physical faces
# use 1-4, tilt-peel uses 5.
RELEASE_REASON_DR_FORCED = 6


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quaternion (wxyz convention). Batched."""
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  vx, vy, vz = vec[0], vec[1], vec[2]
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return torch.stack([rx, ry, rz], dim=-1)


def moment_adjusted_loads(
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
  m_t: torch.Tensor,
  m_l: torch.Tensor,
  m_n: torch.Tensor,
  *,
  pad_span_rake_m: float,
  pad_span_lateral_m: float,
  moment_arm_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Fold pad moments into the worst-edge force seen by the hold envelope."""
  arm_rake = float(pad_span_rake_m) * float(moment_arm_fraction)
  arm_lateral = float(pad_span_lateral_m) * float(moment_arm_fraction)
  if arm_rake <= 0.0 or arm_lateral <= 0.0:
    raise ValueError(
      "Microspine moment arms must be positive; got "
      f"rake={arm_rake:g} m, lateral={arm_lateral:g} m."
    )
  f_n_eff = f_n + m_l.abs() / arm_rake + m_t.abs() / arm_lateral
  f_l_eff = f_l.abs() + m_n.abs() / arm_rake
  return f_t, f_l_eff, f_n_eff


def distribute_wrench_to_units(
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
  m_t: torch.Tensor,
  m_l: torch.Tensor,
  m_n: torch.Tensor,
  unit_x: torch.Tensor,
  unit_y: torch.Tensor,
  sxx: float,
  syy: float,
  srr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Distribute a pad wrench onto spine units (rigid pad, equal stiffness).

  Unit i sits at ``r_i = x_i t̂ + y_i l̂`` in the pad plane and carries
  ``F_i = a_i t̂ + b_i l̂ + c_i n̂``. The equal-stiffness (least-norm)
  bolt-group solution reproducing ``sum F_i = F`` and ``sum r_i x F_i = M``
  is, with ``sxx = sum x², syy = sum y², srr = sxx + syy``:

    a_i = f_t/N − m_n·y_i/srr          (twist shares tangentially, ∝ radius)
    b_i = f_l/N + m_n·x_i/srr
    c_i = f_n/N − m_l·x_i/sxx + m_t·y_i/syy   (bending: linear in position)

  (cross products in the engage frame: m_t = Σ y c, m_l = −Σ x c,
  m_n = Σ (x b − y a)). Inputs (..., ) broadcast against unit_x/unit_y of
  shape (..., N) or (N,); returns per-unit (a, b, c), each (..., N).
  """
  n_units = unit_x.shape[-1]
  ft_u = f_t.unsqueeze(-1) / n_units - m_n.unsqueeze(-1) * unit_y / srr
  fl_u = f_l.unsqueeze(-1) / n_units + m_n.unsqueeze(-1) * unit_x / srr
  fn_u = (
    f_n.unsqueeze(-1) / n_units
    - m_l.unsqueeze(-1) * unit_x / sxx
    + m_t.unsqueeze(-1) * unit_y / syy
  )
  return ft_u, fl_u, fn_u


class MicrospineAffordance(CommandTerm):
  """Stateful world affordance for per-EE microspine grip.

  The affordance is intentionally outside the robot action space. It reads the
  world's contact features and owns the per-EE engage/disengage state machine.
  The constraint executor consumes the resulting on/off signal and toggles the
  per-EE equality anchors.

  Engagement requires sphere compression and optional face alignment. Once
  latched, release is governed exclusively by the anisotropic load limit
  surface; losing contact alone does not silently drop the anchor.
  """

  cfg: MicrospineAffordanceCfg

  def __init__(self, cfg: MicrospineAffordanceCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._mcfg = cfg.microspine_cfg
    self._num_ee = len(self._mcfg.ee_body_names)
    # This latched state is consumed by MicrospineConstraint. It is separate
    # from robot actions, so policy outputs cannot command grip directly.
    initial_mask = self._mcfg.resolve_initial_engaged_mask()
    if not self._mcfg.grip_enabled:
      initial_mask = (False,) * self._num_ee
    self._initial_engaged = torch.tensor(
      initial_mask,
      dtype=torch.bool,
      device=self.device,
    )
    self._engaged = self._initial_engaged.expand(
      self.num_envs, self._num_ee
    ).clone()
    # Consecutive-step counters implement hysteresis/debouncing for noisy contact
    # signals. Cooldown prevents immediate re-engage after a forced release.
    self._engage_count = torch.zeros(
      self.num_envs, self._num_ee, dtype=torch.long, device=self.device
    )
    self._cooldown_steps = torch.zeros(
      self.num_envs, self._num_ee, dtype=torch.long, device=self.device
    )

    self._asset = env.scene[cfg.asset_name]
    self._ee_body_ids: list[int] = []
    for bname in self._mcfg.ee_body_names:
      ids, _ = self._asset.find_bodies([bname])
      self._ee_body_ids.append(ids[0])

    # Latched-contact disable (see MicrospineCfg): per-EE sphere geom indices
    # and the compiled default geom offsets, resolved lazily on first sync
    # because the per-world expansion re-captures CUDA graphs and must run on
    # the live simulation.
    self._sphere_geom_ids: torch.Tensor | None = None
    self._sphere_pos_default: torch.Tensor | None = None
    self._sphere_pos_retracted: torch.Tensor | None = None

    # ``face_local_axis`` is the end-effector direction expected to point into
    # the wall. Normalizing once keeps the per-step dot product numerically clean.
    face_axis = torch.tensor(
      self._mcfg.face_local_axis, dtype=torch.float32, device=self.device
    )
    face_norm = torch.linalg.norm(face_axis).clamp_min(1.0e-6)
    self._face_local_axis = face_axis / face_norm

    # --- limit-surface state ---------------------------------------------
    # The hold envelope releases an engaged spine when its anchor load leaves
    # the configured cone. Parameters are per-env tensors resampled from cfg
    # ranges at reset — built-in domain randomization, calibratable against
    # bench pull-off tests.
    self._ls_cfg = self._mcfg.limit_surface
    if self._ls_cfg.model not in ("wedge", "wang2017"):
      raise ValueError(
        f"Unknown limit-surface model '{self._ls_cfg.model}' "
        "(expected 'wedge' or 'wang2017')."
      )
    # Normalized closed capacity surface R(theta, phi), built once. None in
    # legacy wedge mode.
    self._wang_lut = (
      build_capacity_lut(self._ls_cfg, self.device)
      if self._ls_cfg.model == "wang2017"
      else None
    )
    self._executor = None
    self._gravity_view: torch.Tensor | None = None
    # Sensor-row permutation: ContactSensor expands its primary pattern via
    # find_geoms, which returns MODEL order (kinematic tree: feet before
    # hands), NOT the pattern order this affordance's EE-major slicing
    # assumes. Resolved by name on first read; identity when they agree.
    # Without it the hand blocks read the FOOT sensors (retracted-to-zero
    # while latched) and the press-force engage gate is judged on the
    # wrong limb.
    self._sensor_row_perm: torch.Tensor | None = None
    shape = (self.num_envs, self._num_ee)
    self.limit_f_t_max = torch.zeros(shape, device=self.device)
    self.limit_f_rev_max = torch.zeros(shape, device=self.device)
    self.limit_mu_adhesion = torch.zeros(shape, device=self.device)
    self.limit_f_n0 = torch.zeros(shape, device=self.device)
    self.limit_mu_lateral = torch.zeros(shape, device=self.device)
    self.limit_f_l0 = torch.zeros(shape, device=self.device)
    # Per-grip tilt-peel threshold [rad] (0 = disabled); see
    # MicrospineLimitSurfaceCfg.tilt_peel_max_deg_range.
    self.limit_tilt_peel_max = torch.zeros(shape, device=self.device)
    # Engaged-pad rotation from the engage-capture orientation [rad];
    # 0 where not engaged. The suspension-travel usage the tilt-peel check
    # and the engaged-tilt reward read.
    self.tilt_from_capture = torch.zeros(shape, device=self.device)
    self._violation_count = torch.zeros(
      shape, dtype=torch.long, device=self.device
    )
    self._steps_since_engage = torch.zeros(
      shape, dtype=torch.long, device=self.device
    )
    self.anchor_load_tnl = torch.zeros(
      self.num_envs, self._num_ee, 3, device=self.device
    )
    self.anchor_moment_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.anchor_effective_load_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.hold_margin = torch.ones(shape, device=self.device)
    # Force-only margin (moment billing excluded); see _limit_surface_release.
    self.hold_margin_raw = torch.ones(shape, device=self.device)
    self.forced_release_this_step = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )
    # True only for a debounced OFF -> ON affordance transition. Unlike the
    # constraint executor's rising edge, this deliberately excludes reset-time
    # placement of initially engaged anchors.
    self.engaged_this_step = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )
    # HUD/debug readouts (pure GPU writes; no host syncs in the step path).
    # release_reason: which envelope face fired on a forced release —
    # 0 none, 1 reverse-shear, 2 tension-cone, 3 lateral, 4 ultimate-shear.
    # Like anchor_load_tnl it holds the violating value on the release step
    # and resets the step after.
    self.release_reason = torch.zeros(shape, dtype=torch.int8, device=self.device)
    # R4 reseat hold (2026-08-03): a dr-forced release (reason 6, report=True)
    # arms the pending bit until that EE re-latches, the hold times out
    # (dr_reseat_hold_timeout_steps), or the env resets. The motion command
    # freezes its phase clock while any bit is set (``reseat_hold``) and the
    # engage-mismatch bill exempts pending EEs — unplanned-release recovery
    # trains as a first-class skill instead of a race against the schedule.
    self.dr_release_pending = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._reseat_hold_steps = torch.zeros(
      shape[0], dtype=torch.long, device=self.device
    )
    self.face_cos = torch.zeros(shape, device=self.device)
    self.contact_count = torch.zeros(shape, device=self.device)
    # Per-EE total pad compression against the wall (N). A contact fact, not a
    # latch state: nonzero whenever the sphere grid presses the wall, engaged
    # or not. Consumed by the seat-phase press shaping reward.
    self.pad_press_force = torch.zeros(shape, device=self.device)
    # Virtual spine field (MicrospineCfg.virtual_spine_*): engaged fraction of
    # a kinematic spine grid spanning the pad face — the seating judgement —
    # and the per-latch capacity scale captured from it at the engage step.
    self.spine_fraction = torch.zeros(shape, device=self.device)
    self._seat_scale = torch.ones(shape, device=self.device)
    # Spine recruitment charge [s]: rake-shear-loaded time since this latch
    # engaged. Capacity ramps capacity_ramp_floor -> 1 as it fills (young
    # anchors are weak, not immune). Seeded latches start fully charged.
    self._seat_charge = torch.zeros(shape, device=self.device)
    self._spine_corner_body_ids: torch.Tensor | None = None
    self._spine_sphere_radius: float | None = None
    self._wall_geom_id: int | None = None
    rake_axis = torch.tensor(
      self._ls_cfg.ee_rake_local_axis, dtype=torch.float32, device=self.device
    )
    self._rake_local_axis = rake_axis / torch.linalg.norm(rake_axis).clamp_min(
      1.0e-6
    )
    # Per-unit moment billing: unit grid spanning the pad, precomputed once.
    if self._ls_cfg.moment_billing_mode not in ("worst_edge", "per_unit"):
      raise ValueError(
        "Unknown moment_billing_mode "
        f"'{self._ls_cfg.moment_billing_mode}' (expected 'worst_edge' or "
        "'per_unit')."
      )
    self._unit_x: torch.Tensor | None = None
    self._unit_y: torch.Tensor | None = None
    if (
      self._ls_cfg.model == "wang2017"
      and self._ls_cfg.moment_billing_mode == "per_unit"
    ):
      rows = self._ls_cfg.unit_rows_rake
      cols = self._ls_cfg.unit_cols_lateral
      if rows < 2 or cols < 2:
        raise ValueError(
          "per_unit moment billing needs a >=2x2 unit grid to carry "
          f"moments; got {rows}x{cols}."
        )
      xs = (
        torch.linspace(-0.5, 0.5, rows, device=self.device)
        * self._ls_cfg.pad_span_rake_m
      )
      ys = (
        torch.linspace(-0.5, 0.5, cols, device=self.device)
        * self._ls_cfg.pad_span_lateral_m
      )
      gx, gy = torch.meshgrid(xs, ys, indexing="ij")
      self._unit_x = gx.reshape(-1)
      self._unit_y = gy.reshape(-1)
      self._unit_sxx = float((self._unit_x**2).sum())
      self._unit_syy = float((self._unit_y**2).sum())
      self._unit_srr = self._unit_sxx + self._unit_syy
    self._sample_limit_params(torch.arange(self.num_envs, device=self.device))

  def _sample_limit_params(self, env_ids: torch.Tensor) -> None:
    """Resample per-env limit-surface parameters from the cfg ranges."""
    ls = self._ls_cfg

    def sample(value_range: tuple[float, float]) -> torch.Tensor:
      low, high = value_range
      return torch.empty(
        (len(env_ids), self._num_ee), device=self.device
      ).uniform_(low, high)

    self.limit_f_t_max[env_ids] = sample(ls.f_t_max_range)
    self.limit_f_rev_max[env_ids] = sample(ls.f_rev_max_range)
    self.limit_mu_adhesion[env_ids] = sample(ls.mu_adhesion_range)
    self.limit_f_n0[env_ids] = sample(ls.f_n0_range)
    self.limit_mu_lateral[env_ids] = sample(ls.mu_lateral_range)
    self.limit_f_l0[env_ids] = sample(ls.f_l0_range)
    self.limit_tilt_peel_max[env_ids] = torch.deg2rad(
      sample(ls.tilt_peel_max_deg_range)
    )

  def _get_executor(self):
    """Lazily fetch the physics executor (built after command terms)."""
    if self._executor is None:
      term_cfg = self._env.event_manager.get_term_cfg(
        self.cfg.physics_event_name
      )
      self._executor = term_cfg.func
      if not hasattr(self._executor, "compute_anchor_loads"):
        raise ValueError(
          "Limit surface requires the constraint executor "
          "(MicrospineConstraint) registered as event term "
          f"'{self.cfg.physics_event_name}'."
        )
    return self._executor

  @property
  def command(self) -> torch.Tensor:
    return self._engaged.float()

  @property
  def engaged(self) -> torch.Tensor:
    return self._engaged

  @property
  def cooldown_steps(self) -> torch.Tensor:
    """Remaining per-EE re-engagement cooldown in policy steps."""
    return self._cooldown_steps

  def set_initial_engaged_mask(self, mask: tuple[bool, ...]) -> None:
    """Set the latch mask used by subsequent episode resets."""
    if len(mask) != self._num_ee:
      raise ValueError(
        f"Initial mask has {len(mask)} entries; expected {self._num_ee}."
      )
    self._initial_engaged.copy_(
      torch.tensor(mask, dtype=torch.bool, device=self.device)
    )

  def set_initial_engaged_ee_names(self, names: tuple[str, ...]) -> None:
    """Select reset latches by configured EE name."""
    selected = set(names)
    unknown = selected.difference(self._mcfg.ee_body_names)
    if unknown:
      raise ValueError(
        "Unknown initial microspine EEs: " + ", ".join(sorted(unknown))
      )
    self.set_initial_engaged_mask(
      tuple(name in selected for name in self._mcfg.ee_body_names)
    )

  def seed_engaged(self, env_ids: torch.Tensor, engaged: torch.Tensor) -> None:
    """Overwrite per-env latch state during a reset (motion-clip RSI).

    Must run AFTER this term's own ``_resample_command`` in the same reset
    pass (register the caller later in the command dict), or the seeded mask
    is clobbered by the global initial mask. The latch bookkeeping is zeroed
    here as well so correctness does not depend on that reset having run.
    """
    if not self._mcfg.grip_enabled:
      return
    if engaged.shape[-1] != self._num_ee:
      raise ValueError(
        f"Seed mask has {engaged.shape[-1]} EEs; expected {self._num_ee}."
      )
    self._engaged[env_ids] = engaged.to(device=self.device, dtype=torch.bool)
    self.dr_release_pending[env_ids] = False
    self._reseat_hold_steps[env_ids] = 0
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    # Seeded latches start with the SPAWN judge guard: the RSI teleport
    # resolves written velocities against the fresh welds as an impulse
    # reality never produces (see spawn_judge_guard_steps). The negative
    # start extends the guard runway without a separate counter.
    self._steps_since_engage[env_ids] = self._ls_cfg.hold_judge_guard_steps - max(
      self._ls_cfg.spawn_judge_guard_steps, self._ls_cfg.hold_judge_guard_steps
    )
    # Seeded (RSI) latches get full capacity: the reference stance seats the
    # pads flat by construction, and kinematics may not have refreshed the
    # just-written reset pose yet, so a live fraction here could be stale.
    self._seat_scale[env_ids] = 1.0
    self._seat_charge[env_ids] = self._ls_cfg.capacity_ramp_time_s
    self._sync_latched_contact_margins()

  @property
  def reseat_hold(self) -> torch.Tensor:
    """(E,) bool — clock-freeze request: a dr-forced release awaits reseat."""
    return self.dr_release_pending.any(dim=-1)

  def force_release(
    self,
    env_ids: torch.Tensor,
    ee_mask: torch.Tensor,
    reason: int = RELEASE_REASON_DR_FORCED,
    report: bool = True,
  ) -> None:
    """Release latched EEs outside the limit-surface judgement.

    Serves the DR random-release event and scripted diagnostics. Callers run
    after this step's ``_update_command``, so ``forced_release_this_step`` is
    OR-ed (a same-step limit-surface release must survive) while the latch
    bookkeeping mirrors a real disengage transition: cooldown starts, seat
    scale/charge reset, and ``_sync_latched_contact_margins`` restores the
    pad's contact spheres so it can press the wall again. The equality
    constraint follows the cleared latch bit on the next step's constraint
    event, same one-step lag as every latch transition.

    ``report=False`` skips the ``forced_release_this_step``/``release_reason``
    writes: a scheduled scripted release is not a fault and must not pollute
    forced-release statistics or the reason-billed penalty.
    """
    if not self._mcfg.grip_enabled:
      return
    if ee_mask.shape[-1] != self._num_ee:
      raise ValueError(
        f"Release mask has {ee_mask.shape[-1]} EEs; expected {self._num_ee}."
      )
    mask = ee_mask.to(device=self.device, dtype=torch.bool)
    mask = mask & self._engaged[env_ids]
    if not bool(mask.any()):
      return
    self._engaged[env_ids] = self._engaged[env_ids] & ~mask
    if report and reason == RELEASE_REASON_DR_FORCED:
      # Arm the reseat hold (see ``reseat_hold``). Scripted no-fault
      # releases (report=False) never freeze the clock.
      self.dr_release_pending[env_ids] = self.dr_release_pending[env_ids] | mask
      self._reseat_hold_steps[env_ids] = torch.where(
        mask.any(dim=-1),
        torch.zeros_like(self._reseat_hold_steps[env_ids]),
        self._reseat_hold_steps[env_ids],
      )
    cooldown_set = max(0, self._mcfg.reengage_cooldown_steps)
    self._cooldown_steps[env_ids] = torch.where(
      mask,
      torch.full_like(self._cooldown_steps[env_ids], cooldown_set),
      self._cooldown_steps[env_ids],
    )
    self._engage_count[env_ids] = torch.where(
      mask, torch.zeros_like(self._engage_count[env_ids]),
      self._engage_count[env_ids],
    )
    self._violation_count[env_ids] = torch.where(
      mask, torch.zeros_like(self._violation_count[env_ids]),
      self._violation_count[env_ids],
    )
    self._steps_since_engage[env_ids] = torch.where(
      mask, torch.zeros_like(self._steps_since_engage[env_ids]),
      self._steps_since_engage[env_ids],
    )
    if report:
      self.forced_release_this_step[env_ids] |= mask
      self.release_reason[env_ids] = torch.where(
        mask,
        torch.full_like(self.release_reason[env_ids], reason),
        self.release_reason[env_ids],
      )
    self._seat_scale[env_ids] = torch.where(
      mask, torch.ones_like(self._seat_scale[env_ids]),
      self._seat_scale[env_ids],
    )
    self._seat_charge[env_ids] = torch.where(
      mask, torch.zeros_like(self._seat_charge[env_ids]),
      self._seat_charge[env_ids],
    )
    self._sync_latched_contact_margins()

  def _sync_latched_contact_margins(self) -> None:
    """Retract latched pads' contact spheres; restore them on release.

    Contacts model seating only; while latched, the bilateral anchor pins the
    pad and the sphere contacts are a redundant second load path that fights
    it (penetration-correction transients, near-zero-normal contact flicker —
    the oscillating-load 'barcode'). A latched pad's sphere geoms are moved
    ``latched_contact_retract`` meters INTO the pad along the face axis so
    they geometrically cannot touch the wall; released pads get the compiled
    offsets back the same step. Implemented through ``geom_pos`` because the
    kinematics kernel reads it per world every step (runtime ``pair_*``
    writes are not honored by this mjwarp's collision pipeline — measured).
    First call expands ``geom_pos`` per world via the official mjlab path
    (which re-captures the CUDA graphs).
    """
    if not self._mcfg.disable_contacts_while_latched:
      return
    import warp as wp

    from scr_phybot_mjab.microspine.contact import resolve_sphere_geom_ids

    if self._sphere_geom_ids is None:
      geom_lists = resolve_sphere_geom_ids(self._env.sim.mj_model, self._mcfg)
      self._sphere_geom_ids = torch.tensor(
        geom_lists, dtype=torch.long, device=self.device
      ).reshape(-1)  # (H*S,), EE-major
      pos = wp.to_torch(self._env.sim.wp_model.geom_pos)
      self._sphere_pos_default = pos[0, self._sphere_geom_ids].clone()
      # Sphere child bodies share the pad's orientation, so the pad-frame
      # face axis retracts them straight away from the wall face.
      self._sphere_pos_retracted = (
        self._sphere_pos_default
        - self._face_local_axis.unsqueeze(0)
        * self._mcfg.latched_contact_retract
      )
      if pos.shape[0] != self.num_envs:
        self._env.sim.expand_model_fields(("geom_pos",))

    view = wp.to_torch(self._env.sim.wp_model.geom_pos)  # (E, ngeom, 3)
    spheres = (
      self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    )
    latched = (
      self._engaged.unsqueeze(-1)
      .expand(-1, self._num_ee, spheres)
      .reshape(self.num_envs, -1, 1)
    )
    defaults = self._sphere_pos_default.unsqueeze(0).expand(
      self.num_envs, -1, -1
    )
    retracted = self._sphere_pos_retracted.unsqueeze(0).expand(
      self.num_envs, -1, -1
    )
    view[:, self._sphere_geom_ids] = torch.where(latched, retracted, defaults)

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Command terms are resampled on reset. Reset all latch bookkeeping alongside
    # the command so old contact history cannot leak into the new episode.
    self._engaged[env_ids] = (
      self._initial_engaged if self._mcfg.grip_enabled else False
    )
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    # Reset-time latches (initial_on / about-to-be-RSI-seeded) get the spawn
    # judge guard — the reset writes state as a teleport (see
    # spawn_judge_guard_steps).
    self._steps_since_engage[env_ids] = self._ls_cfg.hold_judge_guard_steps - max(
      self._ls_cfg.spawn_judge_guard_steps, self._ls_cfg.hold_judge_guard_steps
    )
    self.anchor_load_tnl[env_ids] = 0.0
    self.anchor_moment_tnl[env_ids] = 0.0
    self.anchor_effective_load_tnl[env_ids] = 0.0
    self.hold_margin[env_ids] = 1.0
    self.hold_margin_raw[env_ids] = 1.0
    self.dr_release_pending[env_ids] = False
    self._reseat_hold_steps[env_ids] = 0
    self.face_cos[env_ids] = 0.0
    self.contact_count[env_ids] = 0.0
    self.pad_press_force[env_ids] = 0.0
    self.spine_fraction[env_ids] = 0.0
    self._seat_scale[env_ids] = 1.0
    # Initially-engaged (reset-mask) latches are mature grips; anything else
    # starts uncharged.
    self._seat_charge[env_ids] = torch.where(
      self._engaged[env_ids],
      torch.full_like(self._seat_charge[env_ids], self._ls_cfg.capacity_ramp_time_s),
      torch.zeros_like(self._seat_charge[env_ids]),
    )
    self.forced_release_this_step[env_ids] = False
    self.engaged_this_step[env_ids] = False
    self.release_reason[env_ids] = 0
    self._sample_limit_params(env_ids)
    self._sync_latched_contact_margins()

  def _wall_inward_normal(self) -> torch.Tensor:
    wall = self._env.scene[self._mcfg.wall_entity_name]
    wquat = wall.data.root_link_quat_w
    # Wall local +x is the inward normal.
    ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
    inward_x = 1.0 - 2.0 * (wy * wy + wz * wz)
    inward_y = 2.0 * (wx * wy + ww * wz)
    inward_z = 2.0 * (wx * wz - ww * wy)
    return torch.stack([inward_x, inward_y, inward_z], dim=-1)

  def _engage_frame(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-frame engage basis (t̂, l̂, n̂_out), each (E, H, 3).

    t̂ is the rake/engage direction the spines load along ("down the wall"):
    either gravity projected onto the wall plane (default — follows the
    gravity-tilt curriculum) or a per-EE local axis projected onto the wall.
    n̂_out is the outward wall normal, so positive f_n means pull-off tension.
    """
    n_out = -self._wall_inward_normal()  # (E, 3)
    n = n_out.unsqueeze(1)  # (E, 1, 3)
    if self._ls_cfg.rake_mode == "ee_local":
      axes = []
      for body_id in self._ee_body_ids:
        hquat = self._asset.data.body_link_quat_w[:, body_id, :]
        axes.append(_quat_rotate(hquat, self._rake_local_axis))
      raw = torch.stack(axes, dim=1)  # (E, H, 3)
    else:
      if self._gravity_view is None:
        # In-place view of the warp gravity array (the curriculum writes it
        # in place too, so caching the view once is safe), shape (N, 3).
        self._gravity_view = wp.to_torch(self._env.sim.wp_model.opt.gravity)
      raw = self._gravity_view.unsqueeze(1).expand(
        self.num_envs, self._num_ee, 3
      )
    t = raw - (raw * n).sum(dim=-1, keepdim=True) * n
    t_norm = torch.linalg.norm(t, dim=-1, keepdim=True)
    # Degenerate rake (load axis ⟂ wall plane): fall back to world -z projected.
    fallback = torch.tensor(
      [0.0, 0.0, -1.0], device=self.device
    ).expand_as(t)
    fb = fallback - (fallback * n).sum(dim=-1, keepdim=True) * n
    fb = fb / torch.linalg.norm(fb, dim=-1, keepdim=True).clamp_min(1.0e-6)
    t_hat = torch.where(t_norm > 1.0e-6, t / t_norm.clamp_min(1.0e-9), fb)
    n_hat = n.expand_as(t_hat)
    l_hat = torch.cross(n_hat, t_hat, dim=-1)
    return t_hat, l_hat, n_hat

  def _limit_surface_release(self) -> torch.Tensor:
    """Evaluate the hold envelope on currently engaged EEs, (E, H) bool.

    The anchor load comes from the constraint efc rows — a smooth function of
    the pose error, unlike contact friction impulses — which is what makes
    the release condition detectable at all. Wedge mode: conditions 1-3
    (reverse shear, tension cone, lateral cone) are debounced and condition 4
    (ultimate shear) releases instantly. wang2017 mode: every violation is
    debounced (see _wang_envelope).
    """
    executor = self._get_executor()
    loads = executor.compute_anchor_loads()  # (E, H, 3) world
    t_hat, l_hat, n_hat = self._engage_frame()
    f_t = (loads * t_hat).sum(dim=-1)
    f_l = (loads * l_hat).sum(dim=-1)
    f_n = (loads * n_hat).sum(dim=-1)
    self.anchor_load_tnl = torch.stack([f_t, f_l, f_n], dim=-1)

    per_unit_moments = None
    if self._ls_cfg.moment_capacity_enable and self._mcfg.anchor_lock_rotation:
      moment = executor.anchor_moment_w
      m_t = (moment * t_hat).sum(dim=-1)
      m_l = (moment * l_hat).sum(dim=-1)
      m_n = (moment * n_hat).sum(dim=-1)
      self.anchor_moment_tnl = torch.stack([m_t, m_l, m_n], dim=-1)
      if self._unit_x is not None:
        # per_unit billing distributes the raw wrench inside the envelope;
        # no aggregate fold. The effective-load diagnostic becomes the
        # worst unit's load at pad scale (x N), written below.
        f_t_eff, f_l_eff, f_n_eff = f_t, f_l, f_n
        per_unit_moments = (m_t, m_l, m_n)
      else:
        f_t_eff, f_l_eff, f_n_eff = moment_adjusted_loads(
          f_t,
          f_l,
          f_n,
          m_t,
          m_l,
          m_n,
          pad_span_rake_m=self._ls_cfg.pad_span_rake_m,
          pad_span_lateral_m=self._ls_cfg.pad_span_lateral_m,
          moment_arm_fraction=self._ls_cfg.moment_arm_fraction,
        )
    else:
      f_t_eff, f_l_eff, f_n_eff = f_t, f_l, f_n
      self.anchor_moment_tnl.zero_()

    if per_unit_moments is not None:
      margin, soft_cond, instant_cond, reason, worst_pad_scale = (
        self._wang_envelope_per_unit(f_t, f_l, f_n, *per_unit_moments)
      )
      self.anchor_effective_load_tnl = worst_pad_scale
      # Force-only margin (moment billing excluded): with moments zeroed
      # the equal-stiffness distribution is uniform and the per-unit test
      # collapses to the aggregate radial test, so the aggregate margin IS
      # the moment-free per-unit margin. Same moment-tax semantics as the
      # worst_edge path (engage_raster diagnostics).
      margin_raw = self._wang_margin(f_t, f_l, f_n)
    elif self._wang_lut is not None:
      self.anchor_effective_load_tnl = torch.stack(
        [f_t_eff, f_l_eff, f_n_eff], dim=-1
      )
      margin, soft_cond, instant_cond, reason = self._wang_envelope(
        f_t_eff, f_l_eff, f_n_eff
      )
      # Force-only margin (moment billing excluded): the gap to the billed
      # margin is the moment tax, and a release with margin_raw > 0 was
      # caused by the moment fold alone. Diagnostics (engage_raster) only.
      margin_raw = self._wang_margin(f_t, f_l, f_n)
    else:
      self.anchor_effective_load_tnl = torch.stack(
        [f_t_eff, f_l_eff, f_n_eff], dim=-1
      )
      margin, soft_cond, instant_cond, reason = self._wedge_envelope(
        f_t_eff, f_l_eff, f_n_eff
      )
      margin_raw = margin
    self.hold_margin = torch.where(
      self._engaged, margin, torch.ones_like(margin)
    )
    self.hold_margin_raw = torch.where(
      self._engaged, margin_raw, torch.ones_like(margin_raw)
    )

    # The constraint only produces force one decimation after engaging, so
    # hold checks wait exactly the readout settling. There is no immunity
    # window beyond that: young anchors are judged weak (capacity ramp), not
    # exempt.
    in_grace = self._steps_since_engage < self._ls_cfg.hold_judge_guard_steps
    active = self._engaged & ~in_grace
    soft_violation = active & soft_cond
    self._violation_count = torch.where(
      soft_violation,
      self._violation_count + 1,
      torch.zeros_like(self._violation_count),
    )
    release = (active & instant_cond) | (
      self._violation_count
      >= max(1, self._ls_cfg.violation_consecutive_steps)
    )
    self.release_reason = torch.where(
      release, reason, torch.zeros_like(reason)
    )

    # Tilt-peel (reason 5): engaged-pad rotation from the engage-capture
    # orientation beyond the per-grip suspension-travel draw. Geometric and
    # instant — no debounce: once the tension-side row's travel is
    # exhausted, further rotation is structural pull-out, not a load
    # transient. Judged from the same guard window as the force envelope.
    if self._ls_cfg.tilt_peel_max_deg_range[1] > 0.0:
      pad_quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids, :]
      dot = (
        (pad_quat * executor.anchor_quat_w).sum(dim=-1).abs().clamp(max=1.0)
      )
      tilt = 2.0 * torch.acos(dot)
      self.tilt_from_capture = torch.where(
        self._engaged, tilt, torch.zeros_like(tilt)
      )
      tilt_cond = active & (tilt > self.limit_tilt_peel_max)
      release = release | tilt_cond
      self.release_reason = torch.where(
        tilt_cond,
        torch.full_like(self.release_reason, 5),
        self.release_reason,
      )
    return release

  def _wedge_envelope(
    self, f_t: torch.Tensor, f_l: torch.Tensor, f_n: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy piecewise-linear cone: (margin, soft, instant, reason)."""
    ft_pos = f_t.clamp(min=0.0)
    cap_n = self.limit_mu_adhesion * ft_pos + self.limit_f_n0
    cap_l = self.limit_mu_lateral * ft_pos + self.limit_f_l0
    c_rev = f_t >= -self.limit_f_rev_max
    c_tension = f_n <= cap_n
    c_lateral = f_l.abs() <= cap_l
    c_ultimate = f_t <= self.limit_f_t_max

    # Normalized distance to each envelope face; min is the overall margin.
    eps = 1.0e-6
    m_rev = (f_t + self.limit_f_rev_max) / (self.limit_f_rev_max + eps)
    m_tension = (cap_n - f_n) / (cap_n + eps)
    m_lateral = (cap_l - f_l.abs()) / (cap_l + eps)
    m_ultimate = (self.limit_f_t_max - f_t) / (self.limit_f_t_max + eps)
    margin = torch.minimum(
      torch.minimum(m_rev, m_tension), torch.minimum(m_lateral, m_ultimate)
    )

    # Which face fired (HUD/debug). Priority by overwrite order: lateral <
    # reverse < tension < ultimate, masked to actual releases by the caller.
    reason = torch.zeros_like(self.release_reason)
    reason = torch.where(~c_lateral, torch.full_like(reason, 3), reason)
    reason = torch.where(~c_rev, torch.full_like(reason, 1), reason)
    reason = torch.where(~c_tension, torch.full_like(reason, 2), reason)
    reason = torch.where(~c_ultimate, torch.full_like(reason, 4), reason)

    soft_cond = ~(c_rev & c_tension & c_lateral)
    instant_cond = ~c_ultimate
    return margin, soft_cond, instant_cond, reason

  def _wang_capacity_f0(self) -> torch.Tensor:
    """Per-latch F0 with seat quality and recruitment scaling applied."""
    ls = self._ls_cfg
    f0 = self.limit_f_t_max
    if self._mcfg.seat_quality_scales_capacity:
      # Wang 2017 Sec 2.4: pad capacity sums per-spine contributions, so a
      # partially seated latch scales the whole closed surface down by the
      # spine fraction captured at its engage step.
      f0 = f0 * self._seat_scale
    if ls.capacity_ramp_time_s > 0.0:
      # Recruitment ramp: a fresh latch holds capacity_ramp_floor of its
      # seated capacity and charges to 1 with shear-loaded time. Replaces
      # the unjudged grace window — young anchors are weak, not immune.
      ramp = ls.capacity_ramp_floor + (1.0 - ls.capacity_ramp_floor) * (
        self._seat_charge / ls.capacity_ramp_time_s
      ).clamp(0.0, 1.0)
      f0 = f0 * ramp
    return f0

  def _wang_margin(
    self, f_t: torch.Tensor, f_l: torch.Tensor, f_n: torch.Tensor
  ) -> torch.Tensor:
    """Normalized signed envelope distance for one load triple."""
    ls = self._ls_cfg
    tension = f_n.clamp_min(0.0)
    f0 = self._wang_capacity_f0()
    ratio = capacity_ratio(self._wang_lut, f_t, f_l, f_n)
    cap = f0 * ratio.clamp_min(ls.wang_r_floor)
    load_mag = torch.sqrt(f_t**2 + f_l**2 + tension**2)
    # Normalized signed distance: ~1 unloaded, 0 on the surface, < 0 outside.
    return (cap - load_mag) / (cap + ls.wang_margin_eps * f0)

  def _wang_envelope(
    self, f_t: torch.Tensor, f_l: torch.Tensor, f_n: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed Wang-2017 surface: (margin, soft, instant, reason).

    Radial test: the load magnitude |(f_t, f_l, f_n+)| must stay inside the
    direction-dependent capacity F0 * max(R(theta, phi), r_floor), F0 drawn
    from ``f_t_max_range``. Compression rides the contacts and is clamped
    out of the tension component.
    """
    tension = f_n.clamp_min(0.0)
    margin = self._wang_margin(f_t, f_l, f_n)
    violation = margin < 0.0

    # Reason codes keep the wedge semantics used by rewards/HUD: 1 = reverse
    # shear (the intended cheap unhook path), 2 = tension-dominant pull-off,
    # 3 = lateral peel, 4 = ultimate shear along the rake.
    in_plane = torch.sqrt(f_t**2 + f_l**2)
    tension_dominant = tension >= in_plane  # phi >= 45 deg
    reverse = f_t < 0.0
    lateral_dominant = f_l.abs() > f_t
    reason = torch.full_like(self.release_reason, 4)
    reason = torch.where(
      lateral_dominant, torch.full_like(reason, 3), reason
    )
    reason = torch.where(
      tension_dominant, torch.full_like(reason, 2), reason
    )
    reason = torch.where(reverse, torch.full_like(reason, 1), reason)

    # All violations go through the counted path; the count threshold is
    # violation_consecutive_steps (1 since latched-contact retraction: the
    # ~700 N one-step contact-churn transients that once forced a 2-step
    # debounce cannot occur — the latched anchor is the only load path and
    # its force is smooth — so the first genuine violation releases, and
    # oscillating overloads can no longer flicker through a counter reset).
    instant_cond = torch.zeros_like(violation)
    soft_cond = violation
    return margin, soft_cond, instant_cond, reason

  def _wang_envelope_per_unit(
    self,
    f_t: torch.Tensor,
    f_l: torch.Tensor,
    f_n: torch.Tensor,
    m_t: torch.Tensor,
    m_l: torch.Tensor,
    m_n: torch.Tensor,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Per-unit Wang billing: weakest unit of the spine-unit grid rules.

    The pad's 6D wrench is distributed onto the unit grid
    (``distribute_wrench_to_units``: rigid pad, equal-stiffness bolt-group
    statics), and every unit's 3D load is judged against its own share of
    the closed surface — capacity F0_eff/N with the SAME normalized
    R(θ, φ): Wang capacity is a per-spine sum and the surface shape is
    single-unit physics, so the unit test is the paper's Fig 8 object. A
    unit in compression is judged at φ = 0 like the aggregate (its
    compression rides the anchor for free). The pad margin is the minimum
    unit margin — the tension-side edge pulling out unzips the array, so
    one unit out of envelope releases the whole pad.

    Returns (margin, soft_cond, instant_cond, reason, worst_load_tnl); the
    last is the worst unit's load scaled back to pad scale (x N) for the
    ``anchor_effective_load_tnl`` diagnostic.
    """
    ls = self._ls_cfg
    ft_u, fl_u, fn_u = distribute_wrench_to_units(
      f_t,
      f_l,
      f_n,
      m_t,
      m_l,
      m_n,
      self._unit_x,
      self._unit_y,
      self._unit_sxx,
      self._unit_syy,
      self._unit_srr,
    )
    n_units = self._unit_x.shape[-1]
    f0_unit = (self._wang_capacity_f0() / n_units).unsqueeze(-1)
    ratio = capacity_ratio(self._wang_lut, ft_u, fl_u, fn_u)
    cap = f0_unit * ratio.clamp_min(ls.wang_r_floor)
    tension = fn_u.clamp_min(0.0)
    load_mag = torch.sqrt(ft_u**2 + fl_u**2 + tension**2)
    margins = (cap - load_mag) / (cap + ls.wang_margin_eps * f0_unit)
    margin, worst = margins.min(dim=-1)  # (E, H) each.
    violation = margin < 0.0

    # Reason taxonomy from the worst unit's own load direction (same
    # semantics as the aggregate path: 1 reverse, 2 tension, 3 lateral,
    # 4 ultimate shear).
    widx = worst.unsqueeze(-1)
    a_w = ft_u.gather(-1, widx).squeeze(-1)
    b_w = fl_u.gather(-1, widx).squeeze(-1)
    c_w = fn_u.gather(-1, widx).squeeze(-1)
    tension_w = c_w.clamp_min(0.0)
    in_plane = torch.sqrt(a_w**2 + b_w**2)
    reason = torch.full_like(self.release_reason, 4)
    reason = torch.where(
      b_w.abs() > a_w, torch.full_like(reason, 3), reason
    )
    reason = torch.where(
      tension_w >= in_plane, torch.full_like(reason, 2), reason
    )
    reason = torch.where(a_w < 0.0, torch.full_like(reason, 1), reason)

    # Same counted-release semantics as the aggregate wang path.
    instant_cond = torch.zeros_like(violation)
    worst_load_tnl = (
      torch.stack([a_w, b_w, c_w], dim=-1) * float(n_units)
    )
    return margin, violation, instant_cond, reason, worst_load_tnl

  def _face_wall_cos(self) -> torch.Tensor:
    wall_inward = self._wall_inward_normal()
    result = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    for h, body_id in enumerate(self._ee_body_ids):
      hquat = self._asset.data.body_link_quat_w[:, body_id, :]
      # Dot product of the EE face axis and wall inward normal. Values near 1
      # mean the spine is facing into the wall; values near 0 are side-on.
      face = _quat_rotate(hquat, self._face_local_axis)
      result[:, h] = (face * wall_inward).sum(dim=-1)
    return result

  def _virtual_spine_fraction(self) -> torch.Tensor:
    """Engaged fraction of the kinematic virtual spine field, per EE (E, H).

    Corner positions come from the corner sphere child BODIES — their world
    poses are latch-invariant (retraction moves the sphere geoms within those
    bodies, never the bodies) — and the wall surface point from the wall
    plane geom's live world position, with the same inward normal the engage
    frame trusts. Purely kinematic: per Wang et al. 2017 the spine normal
    suspension is force-free over its travel, so reach is the whole question.
    """
    if self._spine_corner_body_ids is None:
      import mujoco

      from scr_phybot_mjab.microspine.contact import (
        resolve_sphere_geom_ids,
        sphere_names,
      )

      rows = self._mcfg.spheres_per_ee_rows
      cols = self._mcfg.spheres_per_ee_cols
      if rows < 2 or cols < 2:
        raise ValueError(
          "The virtual spine field needs a >=2x2 contact-sphere grid to span "
          f"the pad face; got {rows}x{cols}."
        )
      corner_ids: list[int] = []
      for ee in self._mcfg.ee_body_names:
        names = sphere_names(ee, rows, cols)
        for name in (
          names[0],  # r0c0
          names[cols - 1],  # r0c(C-1)
          names[(rows - 1) * cols],  # r(R-1)c0
          names[rows * cols - 1],  # r(R-1)c(C-1)
        ):
          ids, _ = self._asset.find_bodies([name])
          if len(ids) != 1:
            raise ValueError(f"Corner sphere body '{name}' not found.")
          corner_ids.append(ids[0])
      self._spine_corner_body_ids = torch.tensor(
        corner_ids, dtype=torch.long, device=self.device
      )
      mj_model = self._env.sim.mj_model
      geom_lists = resolve_sphere_geom_ids(mj_model, self._mcfg)
      self._spine_sphere_radius = float(mj_model.geom_size[geom_lists[0][0]][0])
      wall_gid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, self._mcfg.wall_geom_name
      )
      if wall_gid < 0:
        raise ValueError(
          f"Wall geom '{self._mcfg.wall_geom_name}' not found for the "
          "virtual spine field."
        )
      self._wall_geom_id = wall_gid
      # The climbable surface: a plane's surface passes through its frame
      # origin; a box's is the face half-extent out along the wall normal
      # (the teleop bench wall is a box; the climb wall is a plane). The
      # offset is resolved once from the live rotation — the face-to-center
      # relation is rigid.
      gtype = int(mj_model.geom_type[wall_gid])
      if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        self._wall_surface_offset = 0.0
      elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        rot = (
          wp.to_torch(self._env.sim.wp_data.geom_xmat)[0, wall_gid]
          .reshape(3, 3)
        )
        n0 = -self._wall_inward_normal()[0]
        local = rot.T @ n0
        axis = int(torch.argmax(torch.abs(local)).item())
        self._wall_surface_offset = float(mj_model.geom_size[wall_gid][axis])
      else:
        raise ValueError(
          f"Wall geom '{self._mcfg.wall_geom_name}' has unsupported type "
          f"{gtype} for the virtual spine field (plane or box)."
        )

    from scr_phybot_mjab.microspine.contact import virtual_spine_fraction

    corners = self._asset.data.body_link_pos_w[
      :, self._spine_corner_body_ids
    ].reshape(self.num_envs, self._num_ee, 4, 3)
    # Off-surface direction toward the robot is the engage frame's n_out; the
    # surface point is the wall geom's origin pushed out by the type-resolved
    # face offset (0 for a plane, the half-extent for a box).
    n_out = -self._wall_inward_normal()
    wall_point = (
      wp.to_torch(self._env.sim.wp_data.geom_xpos)[:, self._wall_geom_id]
      + self._wall_surface_offset * n_out
    )
    return virtual_spine_fraction(
      corners,
      wall_point.unsqueeze(1).expand(-1, self._num_ee, -1),
      n_out.unsqueeze(1).expand(-1, self._num_ee, -1),
      self._spine_sphere_radius,
      self._mcfg.virtual_spine_grid,
      self._mcfg.virtual_spine_travel,
    )

  def _normal_force_condition(self, normal_forces: torch.Tensor) -> torch.Tensor:
    # Optional compression gate. The z-force gate decides support direction;
    # this gate requires enough actual wall contact on the sphere grid.
    if self._mcfg.engage_normal_force_threshold <= 0.0:
      return torch.ones_like(self._engaged)

    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    expected_spheres = self._num_ee * spheres_per_ee
    if normal_forces.shape != (self.num_envs, expected_spheres):
      raise ValueError(
        f"Microspine normal force shape {tuple(normal_forces.shape)} is incompatible "
        f"with {self._num_ee} EEs and {spheres_per_ee} spheres per EE."
      )

    compression = normal_forces.clamp(min=0.0)
    result = torch.zeros_like(self._engaged)
    min_count = self._mcfg.engage_min_spheres
    min_fraction = self._mcfg.engage_min_fraction
    for h in range(self._num_ee):
      s_start = h * spheres_per_ee
      s_end = s_start + spheres_per_ee
      above = (
        compression[:, s_start:s_end] >= self._mcfg.engage_normal_force_threshold
      )
      # Use either an absolute sphere count or a fraction of the EE grid. Count
      # mode is useful for small grids where a single sphere matters.
      if min_count > 0:
        result[:, h] = above.sum(dim=1) >= min_count
      else:
        result[:, h] = above.float().mean(dim=1) >= min_fraction
    return result

  def _validate_sensor_shapes(
    self,
    found: torch.Tensor,
    force: torch.Tensor,
  ) -> None:
    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    expected_spheres = self._num_ee * spheres_per_ee
    # Sensor rows must match get_sphere_geom_names(): EE-major order, then rows,
    # then columns. If this changes, force aggregation by EE would be wrong.
    if found.shape != (self.num_envs, expected_spheres):
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' found shape "
        f"{tuple(found.shape)} does not match expected "
        f"({self.num_envs}, {expected_spheres})."
      )
    if force.shape != (self.num_envs, expected_spheres, 3):
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' force shape "
        f"{tuple(force.shape)} does not match expected "
        f"({self.num_envs}, {expected_spheres}, 3)."
      )
  def _sensor_row_permutation(self, sensor) -> torch.Tensor:
    """Map the sensor's actual row order onto the EE-major sphere order."""
    if self._sensor_row_perm is None:
      from scr_phybot_mjab.microspine.contact import get_sphere_geom_names

      names = getattr(sensor, "primary_names", None)
      if names is None:
        # Older ContactSensor API (this mjlab): row order lives in the
        # slot list — one row per unique primary, in slot order
        # (num_slots=1 for this sensor). Dedup preserves that order.
        names = list(
          dict.fromkeys(slot.primary_name for slot in sensor._slots)
        )
      actual = [name.split("/")[-1] for name in names]
      expected = list(get_sphere_geom_names(self._mcfg))
      try:
        perm = [actual.index(name) for name in expected]
      except ValueError as exc:
        raise ValueError(
          f"Contact sensor '{self.cfg.contact_sensor_name}' rows "
          f"{actual} cannot be mapped onto the expected sphere order "
          f"{expected}."
        ) from exc
      self._sensor_row_perm = torch.tensor(
        perm, dtype=torch.long, device=self.device
      )
    return self._sensor_row_perm

  def _sensor_contact_features(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    data = sensor.data
    if data.found is None or data.force is None:
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' must provide "
        "'found' and 'force' fields."
      )

    perm = self._sensor_row_permutation(sensor)
    found = data.found[:, perm]
    found_mask = found > 0
    engage_force = data.force[:, perm]
    if data.force_history is not None:
      # Average force history so one compression impulse cannot latch a spine.
      force_history = data.force_history[:, perm]
      engage_force = force_history.mean(dim=2)
      # MuJoCo 3.10 contacts can be created and resolved within a decimation
      # window. In that case the instantaneous ``found`` value is zero even
      # though the force history contains the compression that should be used
      # by the latch gate. The history length is configured to one policy step,
      # so treating any historical normal force as contact cannot leak across
      # steps.
      found_mask |= (force_history[..., 0] > 0.0).any(dim=2)

    self._validate_sensor_shapes(found, engage_force)

    # CONTACT sensor force is expressed as [normal, tangent1, tangent2].
    normal_force = engage_force[..., 0].clamp(min=0.0)
    normal_force = torch.where(
      found_mask, normal_force, torch.zeros_like(normal_force)
    )

    found_count = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    for h in range(self._num_ee):
      s_start = h * spheres_per_ee
      s_end = s_start + spheres_per_ee
      found_count[:, h] = found_mask[:, s_start:s_end].sum(dim=1)

    return normal_force, found_count

  def _log_transition_sensor_data(
    self,
    was_engaged: torch.Tensor,
    new_engaged: torch.Tensor,
    engage_condition: torch.Tensor,
    normal_forces: torch.Tensor,
    found_count: torch.Tensor,
    face_cos: torch.Tensor,
    engage_count: torch.Tensor,
  ) -> None:
    if not self._mcfg.debug:
      return

    changed = was_engaged[0] != new_engaged[0]
    if not changed.any():
      return

    step = self._env.episode_length_buf[0].item()
    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    normal_threshold = self._mcfg.engage_normal_force_threshold
    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      if not changed[ee_index].item():
        continue

      s_start = ee_index * spheres_per_ee
      s_end = s_start + spheres_per_ee
      ee_normal = normal_forces[0, s_start:s_end]
      normal_max = ee_normal.max().item()
      normal_mean = ee_normal.mean().item()
      normal_above = (ee_normal >= normal_threshold).sum().item()
      old = "ON" if was_engaged[0, ee_index].item() else "OFF"
      new = "ON" if new_engaged[0, ee_index].item() else "OFF"
      print(
        f"[Microspine] SENSOR step={step} {bname} {old}->{new} "
        f"condition={int(engage_condition[0, ee_index].item())} "
        f"found={int(found_count[0, ee_index].item())}/{spheres_per_ee} "
        f"normal_above={int(normal_above)}/{spheres_per_ee} "
        f"normal_max={normal_max:.3f}N normal_mean={normal_mean:.3f}N "
        f"face_cos={face_cos[0, ee_index].item():.3f} "
        f"engage_count={int(engage_count[0, ee_index].item())} "
        f"cooldown={int(self._cooldown_steps[0, ee_index].item())}"
      )

  def _update_command(self) -> None:
    normal_forces, found_count = self._sensor_contact_features()
    if self._mcfg.disable_contacts_while_latched:
      # Latched pads' spheres are retracted — they physically have no wall
      # contact. The substep-history recovery (meant to bridge sampling
      # aliasing of real contacts) must not replay stale pre-latch forces.
      spheres_per_ee_mask = (
        self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
      )
      latched_spheres = (
        self._engaged.unsqueeze(-1)
        .expand(-1, self._num_ee, spheres_per_ee_mask)
        .reshape(self.num_envs, -1)
      )
      normal_forces = torch.where(
        latched_spheres, torch.zeros_like(normal_forces), normal_forces
      )
      found_count = torch.where(
        self._engaged, torch.zeros_like(found_count), found_count
      )
    self.contact_count = found_count
    spheres_per_ee = (
      self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    )
    self.pad_press_force = (
      normal_forces.clamp(min=0.0)
      .view(self.num_envs, self._num_ee, spheres_per_ee)
      .sum(dim=2)
    )
    self.engaged_this_step.zero_()

    # Engage from OFF requires compression on the sphere grid and optional
    # face alignment.
    engage_condition = self._normal_force_condition(normal_forces)

    # Always computed (cheap, a few quat rotates) so HUD/debug readers see a
    # live value even when the alignment gate is disabled.
    face_cos = self._face_wall_cos()
    self.face_cos = face_cos
    # Kinematic seating judgement; also an observation channel, so it stays
    # live on every path (grip-disabled ablation included).
    self.spine_fraction = self._virtual_spine_fraction()

    if not self._mcfg.grip_enabled:
      self._engaged.zero_()
      self._engage_count.zero_()
      self._cooldown_steps.zero_()
      self._violation_count.zero_()
      self._steps_since_engage.zero_()
      self.anchor_load_tnl.zero_()
      self.anchor_moment_tnl.zero_()
      self.anchor_effective_load_tnl.zero_()
      self.hold_margin.fill_(1.0)
      self.hold_margin_raw.fill_(1.0)
      self.forced_release_this_step.zero_()
      self.release_reason.zero_()
      self._seat_scale.fill_(1.0)
      self._sync_latched_contact_margins()
      return

    if self._mcfg.engage_parallel_cos_threshold > 0.0:
      engage_condition &= (
        face_cos >= self._mcfg.engage_parallel_cos_threshold
      )
    if self._mcfg.engage_min_spine_fraction > 0.0:
      # The seating gate: enough of the virtual spine field must be able to
      # reach the wall. Replaces the 2-rigid-sphere count as the 'actually
      # seated' judgement (a rigid pad generically touches a flat wall with
      # one sphere; demanding two demanded sub-degree parallelism).
      engage_condition &= (
        self.spine_fraction >= self._mcfg.engage_min_spine_fraction
      )
    if self._mcfg.engage_max_antirake_vel > 0.0:
      # Rake-direction gate (Wang 2017 §2): spines hook under along-rake
      # loading only. A pad sliding against its rake (up-wall) cannot
      # catch — pressing feet legitimately skid upward during a body
      # pull-up, and must not farm anchors doing so. Stationary or
      # down-sliding pads (a catch during a fall) latch normally.
      t_hat, _, _ = self._engage_frame()
      vel_w = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]
      v_rake = (vel_w * t_hat).sum(dim=-1)
      engage_condition &= v_rake >= -self._mcfg.engage_max_antirake_vel

    was_engaged = self._engaged.clone()
    cooldown_active = self._cooldown_steps > 0
    engage_candidate = engage_condition & ~self._engaged & ~cooldown_active
    # Count only consecutive eligible steps. Any failed step resets the counter,
    # which avoids flickering around the threshold.
    engage_count = torch.where(
      engage_candidate,
      self._engage_count + 1,
      torch.zeros_like(self._engage_count),
    )
    engage_transition = engage_count >= max(
      1, self._mcfg.engage_consecutive_steps
    )
    self.engaged_this_step.copy_(engage_transition)

    # The limit surface owns release. It runs on the pre-transition latch state,
    # so a spine engaged this very step is never judged on stale loads.
    if self._ls_cfg.enable:
      self.forced_release_this_step = self._limit_surface_release()
      if self._mcfg.debug and bool(self.forced_release_this_step[0].any()):
        load = self.anchor_load_tnl[0].tolist()
        print(
          "[Microspine] LIMIT-SURFACE RELEASE env0 "
          f"released={self.forced_release_this_step[0].tolist()} "
          f"load_tnl={[[round(v, 1) for v in ee] for ee in load]}"
        )
    else:
      self.forced_release_this_step = torch.zeros_like(self._engaged)
    disengage_transition = self.forced_release_this_step

    new_engaged = torch.where(
      engage_transition,
      torch.ones_like(self._engaged),
      torch.where(
        disengage_transition,
        torch.zeros_like(self._engaged),
        self._engaged,
      ),
    )
    self._log_transition_sensor_data(
      was_engaged,
      new_engaged,
      engage_condition,
      normal_forces,
      found_count,
      face_cos,
      engage_count,
    )

    self._engaged[:] = new_engaged
    # Reseat-hold maintenance: a re-latch clears the pending bit; a hold that
    # outlives its timeout is abandoned (clock resumes, normal billing).
    self.dr_release_pending &= ~new_engaged
    hold_active = self.dr_release_pending.any(dim=-1)
    self._reseat_hold_steps = torch.where(
      hold_active,
      self._reseat_hold_steps + 1,
      torch.zeros_like(self._reseat_hold_steps),
    )
    expired = self._reseat_hold_steps > self._mcfg.dr_reseat_hold_timeout_steps
    self.dr_release_pending = torch.where(
      expired.unsqueeze(-1),
      torch.zeros_like(self.dr_release_pending),
      self.dr_release_pending,
    )
    self._reseat_hold_steps = torch.where(
      expired, torch.zeros_like(self._reseat_hold_steps), self._reseat_hold_steps
    )
    if self._mcfg.seat_quality_scales_capacity:
      # This latch's capacity is set by how much of the spine field could
      # reach at the seat step (Wang 2017 Sec 2.4: pad capacity is the sum of
      # engaged spines' contributions). Release restores full scale so the
      # next latch is judged on its own seating. Engage and disengage are
      # mutually exclusive per EE within a step (release runs on the
      # pre-transition latch state).
      self._seat_scale = torch.where(
        engage_transition,
        self.spine_fraction.clamp(0.0, 1.0),
        self._seat_scale,
      )
      self._seat_scale = torch.where(
        disengage_transition,
        torch.ones_like(self._seat_scale),
        self._seat_scale,
      )
    if self._ls_cfg.capacity_ramp_time_s > 0.0:
      # Recruitment: rake-shear-loaded time charges the young anchor toward
      # full capacity (dragging the loaded array seats more spines). Fresh
      # transitions restart from zero; releases clear.
      shear_loaded = self._engaged & (
        self.anchor_load_tnl[..., 0] >= self._ls_cfg.capacity_ramp_shear_n
      )
      self._seat_charge = torch.where(
        shear_loaded,
        self._seat_charge + self._env.step_dt,
        self._seat_charge,
      )
      self._seat_charge = torch.where(
        engage_transition | disengage_transition,
        torch.zeros_like(self._seat_charge),
        self._seat_charge,
      )
    if self._ls_cfg.redraw_f_t_max_per_latch:
      # Every grip is a fresh asperity lottery: redraw F0 for the just-
      # latched pads (paper Fig 6; a collapsed DR range redraws the same
      # value, keeping the bench deterministic).
      lo, hi = self._ls_cfg.f_t_max_range
      draw = lo + (hi - lo) * torch.rand_like(self.limit_f_t_max)
      self.limit_f_t_max = torch.where(
        engage_transition, draw, self.limit_f_t_max
      )
      # Same lottery for the tilt-peel travel headroom: which asperities the
      # spines caught decides how much chassis rotation the patch tolerates.
      t_lo, t_hi = self._ls_cfg.tilt_peel_max_deg_range
      if t_hi > 0.0:
        t_draw = torch.deg2rad(
          t_lo + (t_hi - t_lo) * torch.rand_like(self.limit_tilt_peel_max)
        )
        self.limit_tilt_peel_max = torch.where(
          engage_transition, t_draw, self.limit_tilt_peel_max
        )
    self._engage_count = engage_count
    # After an engage transition, clear its counter.
    self._engage_count = torch.where(
      engage_transition, torch.zeros_like(self._engage_count), self._engage_count
    )

    self._cooldown_steps = (self._cooldown_steps - 1).clamp(min=0)
    cooldown_set = max(0, self._mcfg.reengage_cooldown_steps)
    if cooldown_set > 0:
      # Cooldown starts only after a disengage transition. It blocks engage while
      # still allowing the release state to be reported immediately.
      self._cooldown_steps = torch.where(
        disengage_transition,
        torch.full_like(self._cooldown_steps, cooldown_set),
        self._cooldown_steps,
      )

    # Once engaged, stale engage counts are irrelevant.
    self._engage_count = torch.where(
      self._engaged, torch.zeros_like(self._engage_count), self._engage_count
    )
    # Contact pairs follow the (possibly just-changed) latch mask: latched
    # pads hand their load path to the anchor, released pads get their
    # contacts back for seating and push-off.
    self._sync_latched_contact_margins()

    # Engage age drives the limit-surface grace window. Resets to 0 on the
    # engage step (covers initial_on latches too, which never see a
    # transition: they start counting from the reset value of 0).
    self._steps_since_engage = torch.where(
      self._engaged,
      torch.where(
        engage_transition,
        torch.zeros_like(self._steps_since_engage),
        self._steps_since_engage + 1,
      ),
      torch.zeros_like(self._steps_since_engage),
    )
    self._violation_count = torch.where(
      self._engaged, self._violation_count, torch.zeros_like(self._violation_count)
    )
