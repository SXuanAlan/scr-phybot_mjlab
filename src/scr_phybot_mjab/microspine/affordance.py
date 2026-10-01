"""Environment-side affordance model for microspine grip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import quat_apply

from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg
from scr_phybot_mjab.microspine.empirical_limit_surface import (
  DEFAULT_TABLE,
  EmpiricalLimitSurface,
)
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry
from scr_phybot_mjab.microspine.surfaces import MicrospineSurfaces, pad_face_vertices, rotate
from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# release_reason code for releases commanded from outside the limit-surface
# judgement (DR random-release event, scripted diagnostics). Physical faces
# use 1-4.
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


class MicrospineAffordance(CommandTerm):
  """Stateful world affordance for per-EE microspine grip.

  The affordance is intentionally outside the robot action space. It reads the
  world's contact features and owns the per-EE engage/disengage state machine.
  The constraint executor consumes the resulting on/off signal and toggles the
  per-EE equality anchors.

  Engagement requires unit preload (or legacy contact gates) and seating. Once
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
    self._engaged = self._initial_engaged.expand(self.num_envs, self._num_ee).clone()
    # Consecutive-step counters implement hysteresis/debouncing for noisy contact
    # signals. Cooldown prevents immediate re-engage after a forced release.
    self._engage_count = torch.zeros(self.num_envs, self._num_ee, dtype=torch.long, device=self.device)
    self._cooldown_steps = torch.zeros(self.num_envs, self._num_ee, dtype=torch.long, device=self.device)

    self._asset = env.scene[cfg.asset_name]
    self._surfaces = MicrospineSurfaces(env, self._mcfg) if self._mcfg.surfaces else None
    self._surface_corners = None
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
    face_axis = torch.tensor(self._mcfg.face_local_axis, dtype=torch.float32, device=self.device)
    face_norm = torch.linalg.norm(face_axis).clamp_min(1.0e-6)
    self._face_local_axis = face_axis / face_norm

    # --- limit-surface state ---------------------------------------------
    # The hold envelope releases an engaged spine when its anchor load leaves
    # the envelope. Capacity uses persistent per-unit draws from the bench CDF.
    self._ls_cfg = self._mcfg.limit_surface
    if self._ls_cfg.model != "empirical_unit":
      raise ValueError(f"Unknown limit-surface model '{self._ls_cfg.model}' (only 'empirical_unit' is supported).")
    if (
      self._ls_cfg.moment_billing_mode != "per_unit"
      or self._ls_cfg.rake_mode != "ee_local"
      or tuple(self._ls_cfg.ee_rake_local_axis) != (0.0, 0.0, -1.0)
      or tuple(self._mcfg.face_local_axis) != (0.0, -1.0, 0.0)
      or self._mcfg.spring_field_enabled
      or self._mcfg.seat_quality_scales_capacity
      or self._ls_cfg.capacity_ramp_time_s > 0
      or self._ls_cfg.redraw_f_t_max_per_latch
    ):
      raise ValueError(
        "empirical_unit requires per_unit billing, ee_local rake -z, pad face -y, and no spring field, "
        "seat scaling, capacity ramp or F0 redraw"
      )
    self._empirical = EmpiricalLimitSurface(self._ls_cfg.empirical_table_path or DEFAULT_TABLE, self.device)
    self._executor = None
    # Sensor-row permutation: ContactSensor expands its primary pattern via
    # find_geoms, which returns MODEL order (kinematic tree: feet before
    # hands), NOT the pattern order this affordance's EE-major slicing
    # assumes. Resolved by name on first read; identity when they agree.
    # Without it the hand blocks read the FOOT sensors (retracted-to-zero
    # while latched) and the press-force engage gate is judged on the
    # wrong limb.
    self._sensor_row_perm: torch.Tensor | None = None
    shape = (self.num_envs, self._num_ee)
    self._violation_count = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._steps_since_engage = torch.zeros(shape, dtype=torch.long, device=self.device)
    self.anchor_load_tnl = torch.zeros(self.num_envs, self._num_ee, 3, device=self.device)
    self.anchor_moment_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.anchor_effective_load_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.hold_margin = torch.ones(shape, device=self.device)
    # Force-only margin (moment billing excluded); see _limit_surface_release.
    self.hold_margin_raw = torch.ones(shape, device=self.device)
    self.forced_release_this_step = torch.zeros(shape, dtype=torch.bool, device=self.device)
    # True only for a debounced OFF -> ON affordance transition. Unlike the
    # constraint executor's rising edge, this deliberately excludes reset-time
    # placement of initially engaged anchors.
    self.engaged_this_step = torch.zeros(shape, dtype=torch.bool, device=self.device)
    # HUD/debug readouts (pure GPU writes; no host syncs in the step path).
    # release_reason: which envelope face fired on a forced release —
    # 0 none, 1 reverse-shear, 2 normal threshold, 3 lateral, 4 ultimate-shear,
    # 6 externally forced, 7 empirical domain exceeded,
    # 8 empirical early-loading failure.
    # Like anchor_load_tnl it holds the violating value on the release step
    # and resets the step after.
    self.release_reason = torch.zeros(shape, dtype=torch.int8, device=self.device)
    # Seed provenance (2026-08-15 seed economy): True while an RSI-SEEDED
    # anchor is the one holding — cleared the moment it releases; a weld the
    # policy makes itself is never seeded. ``released_was_seeded`` latches
    # (for one step, like forced_release_this_step) the falling edges of
    # seeded anchors so the transition can bill demolishing a gifted anchor
    # at full price while the rip-bill holiday keeps subsidizing the
    # policy's own experimental welds.
    self._seeded = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.released_was_seeded = torch.zeros(shape, dtype=torch.bool, device=self.device)
    # R4 reseat hold (2026-08-03): a dr-forced release (reason 6, report=True)
    # arms the pending bit until that EE re-latches, the hold times out
    # (dr_reseat_hold_timeout_steps), or the env resets. The motion command
    # freezes its phase clock while any bit is set (``reseat_hold``) and the
    # engage-mismatch bill exempts pending EEs — unplanned-release recovery
    # trains as a first-class skill instead of a race against the schedule.
    self.dr_release_pending = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._reseat_hold_steps = torch.zeros(shape[0], dtype=torch.long, device=self.device)
    self.face_cos = torch.zeros(shape, device=self.device)
    self.contact_count = torch.zeros(shape, device=self.device)
    # Per-EE total pad compression against the wall (N). A contact fact, not a
    # latch state: nonzero whenever the sphere grid presses the wall, engaged
    # or not. Consumed by the seat-phase press shaping reward.
    self.pad_press_force = torch.zeros(shape, device=self.device)
    # Virtual spine field (MicrospineCfg.virtual_spine_*): engaged fraction of
    # a kinematic spine grid spanning the pad face — the seating judgement.
    self.spine_fraction = torch.zeros(shape, device=self.device)
    self._spine_corner_body_ids: torch.Tensor | None = None
    self._spine_corner_offsets: torch.Tensor | None = None
    self._spine_sphere_offsets: torch.Tensor | None = None
    self._spine_corner_radii: torch.Tensor | None = None
    self._wall_geom_id: int | None = None
    rake_axis = torch.tensor(self._ls_cfg.ee_rake_local_axis, dtype=torch.float32, device=self.device)
    self._rake_local_axis = rake_axis / torch.linalg.norm(rake_axis).clamp_min(1.0e-6)
    # Per-unit moment billing: the active CAD unit sites, resolved once.
    rows = self._ls_cfg.unit_rows_rake
    cols = self._ls_cfg.unit_cols_lateral
    if rows < 2 or cols < 2:
      raise ValueError(f"per_unit moment billing needs a >=2x2 unit grid to carry moments; got {rows}x{cols}.")
    geometry = matching_pad_geometry(
      env.sim.mj_model,
      self._mcfg.ee_body_names,
      cfg.asset_name,
      rake_local_axis=self._ls_cfg.ee_rake_local_axis,
      face_local_axis=self._mcfg.face_local_axis,
      unit_layout=self._ls_cfg.unit_layout,
    )
    if (rows, cols) != (geometry.rows, geometry.cols):
      raise ValueError("Empirical unit rows/cols disagree with the asset sites")
    self._unit_x = torch.tensor(geometry.x.copy(), dtype=torch.float32, device=self.device)
    self._unit_y = torch.tensor(geometry.y.copy(), dtype=torch.float32, device=self.device)
    self.unit_site_names: tuple[str, ...] = geometry.unit_names
    self._unit_sxx = float((self._unit_x**2).sum())
    self._unit_syy = float((self._unit_y**2).sum())
    self._unit_srr = self._unit_sxx + self._unit_syy
    if not np.isfinite(self._mcfg.engage_unit_preload_n) or self._mcfg.engage_unit_preload_n < 0:
      raise ValueError("engage_unit_preload_n must be finite and nonnegative")
    # Pre-latch contact estimates, separate from the latched anchor loads.
    self.unit_preload_n = torch.zeros((*shape, self._unit_x.numel()), device=self.device)
    self.pad_contact_wrench_tln = torch.zeros((*shape, 6), device=self.device)
    empirical_shape = (*shape, int(self._unit_x.numel()))
    self.unit_strength_quantile = torch.zeros(empirical_shape, device=self.device)
    self.unit_loading_quantile = torch.zeros_like(self.unit_strength_quantile)
    self.unit_load_tln = torch.zeros((*empirical_shape, 3), device=self.device)
    self.unit_normal_capacity = torch.zeros_like(self.unit_strength_quantile)
    self.unit_empirical_supported = torch.ones(empirical_shape, dtype=torch.bool, device=self.device)
    self.unit_empirical_early_failed = torch.zeros_like(self.unit_empirical_supported)
    self.unit_empirical_tail_capped = torch.zeros_like(self.unit_empirical_supported)
    self._sample_limit_params(torch.arange(self.num_envs, device=self.device))

  def _sample_limit_params(self, env_ids: torch.Tensor) -> None:
    """Draw each pad's persistent per-unit quantiles and reset its unit readouts."""
    shape = (len(env_ids), self._num_ee, self._unit_x.numel())
    self.unit_strength_quantile[env_ids] = torch.rand(shape, device=self.device)
    self.unit_loading_quantile[env_ids] = torch.rand(shape, device=self.device)
    self.unit_load_tln[env_ids] = 0
    self.unit_normal_capacity[env_ids] = 0
    self.unit_empirical_supported[env_ids] = True
    self.unit_empirical_early_failed[env_ids] = False
    self.unit_empirical_tail_capped[env_ids] = False

  def _get_executor(self):
    """Lazily fetch the physics executor (built after command terms)."""
    if self._executor is None:
      term_cfg = self._env.event_manager.get_term_cfg(self.cfg.physics_event_name)
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
  def seeded(self) -> torch.Tensor:
    """(E, EE) bool: this EE's current hold is an RSI-seeded anchor."""
    return self._seeded

  @property
  def cooldown_steps(self) -> torch.Tensor:
    """Remaining per-EE re-engagement cooldown in policy steps."""
    return self._cooldown_steps

  def set_initial_engaged_mask(self, mask: tuple[bool, ...]) -> None:
    """Set the latch mask used by subsequent episode resets."""
    if len(mask) != self._num_ee:
      raise ValueError(f"Initial mask has {len(mask)} entries; expected {self._num_ee}.")
    self._initial_engaged.copy_(torch.tensor(mask, dtype=torch.bool, device=self.device))

  def set_initial_engaged_ee_names(self, names: tuple[str, ...]) -> None:
    """Select reset latches by configured EE name."""
    selected = set(names)
    unknown = selected.difference(self._mcfg.ee_body_names)
    if unknown:
      raise ValueError("Unknown initial microspine EEs: " + ", ".join(sorted(unknown)))
    self.set_initial_engaged_mask(tuple(name in selected for name in self._mcfg.ee_body_names))

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
      raise ValueError(f"Seed mask has {engaged.shape[-1]} EEs; expected {self._num_ee}.")
    self._engaged[env_ids] = engaged.to(device=self.device, dtype=torch.bool)
    self._seeded[env_ids] = self._engaged[env_ids]
    self.released_was_seeded[env_ids] = False
    self.dr_release_pending[env_ids] = False
    self._reseat_hold_steps[env_ids] = 0
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
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
    bookkeeping mirrors a real disengage transition: cooldown starts and
    ``_sync_latched_contact_margins`` restores the
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
      raise ValueError(f"Release mask has {ee_mask.shape[-1]} EEs; expected {self._num_ee}.")
    mask = ee_mask.to(device=self.device, dtype=torch.bool)
    mask = mask & self._engaged[env_ids]
    if not bool(mask.any()):
      return
    if report:
      # Scripted no-fault releases (report=False) never count as seed
      # demolition either — same statistics discipline as release_reason.
      self.released_was_seeded[env_ids] |= mask & self._seeded[env_ids]
    self._seeded[env_ids] = self._seeded[env_ids] & ~mask
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
      mask,
      torch.zeros_like(self._engage_count[env_ids]),
      self._engage_count[env_ids],
    )
    self._violation_count[env_ids] = torch.where(
      mask,
      torch.zeros_like(self._violation_count[env_ids]),
      self._violation_count[env_ids],
    )
    self._steps_since_engage[env_ids] = torch.where(
      mask,
      torch.zeros_like(self._steps_since_engage[env_ids]),
      self._steps_since_engage[env_ids],
    )
    if report:
      self.forced_release_this_step[env_ids] |= mask
      self.release_reason[env_ids] = torch.where(
        mask,
        torch.full_like(self.release_reason[env_ids], reason),
        self.release_reason[env_ids],
      )
    self._sync_latched_contact_margins()

  def _sync_latched_contact_margins(self) -> None:
    """Retract latched pads' contact proxies; restore them on release.

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
      self._sphere_geom_ids = torch.tensor(geom_lists, dtype=torch.long, device=self.device).reshape(
        -1
      )  # (H*S,), EE-major
      pos = wp.to_torch(self._env.sim.wp_model.geom_pos)
      self._sphere_pos_default = pos[0, self._sphere_geom_ids].clone()
      # Contact child bodies share the pad's orientation, so the pad-frame
      # face axis retracts them straight away from the wall face.
      self._sphere_pos_retracted = (
        self._sphere_pos_default - self._face_local_axis.unsqueeze(0) * self._mcfg.latched_contact_retract
      )
      if pos.shape[0] != self.num_envs:
        self._env.sim.expand_model_fields(("geom_pos",))

    view = wp.to_torch(self._env.sim.wp_model.geom_pos)  # (E, ngeom, 3)
    spheres = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    latched = self._engaged.unsqueeze(-1).expand(-1, self._num_ee, spheres).reshape(self.num_envs, -1, 1)
    defaults = self._sphere_pos_default.unsqueeze(0).expand(self.num_envs, -1, -1)
    retracted = self._sphere_pos_retracted.unsqueeze(0).expand(self.num_envs, -1, -1)
    view[:, self._sphere_geom_ids] = torch.where(latched, retracted, defaults)

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Command terms are resampled on reset. Reset all latch bookkeeping alongside
    # the command so old contact history cannot leak into the new episode.
    self._engaged[env_ids] = self._initial_engaged if self._mcfg.grip_enabled else False
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
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
    self.unit_preload_n[env_ids] = 0.0
    self.pad_contact_wrench_tln[env_ids] = 0.0
    self.spine_fraction[env_ids] = 0.0
    self.forced_release_this_step[env_ids] = False
    self.engaged_this_step[env_ids] = False
    self.release_reason[env_ids] = 0
    # Reset-mask engagements are not RSI seeds; seed_engaged (which runs
    # after this in the same reset pass) marks the actual seeded pattern.
    self._seeded[env_ids] = False
    self.released_was_seeded[env_ids] = False
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
    the pad-local rake axis projected onto the wall plane. n̂_out is the
    outward wall normal, so positive f_n means pull-off tension.
    """
    if getattr(self, "_surfaces", None) is not None:
      _, n, _, _ = self._surfaces.frame()
    else:
      n = -self._wall_inward_normal().unsqueeze(1)  # (E, 1, 3)
    axes = []
    for body_id in self._ee_body_ids:
      hquat = self._asset.data.body_link_quat_w[:, body_id, :]
      axes.append(_quat_rotate(hquat, self._rake_local_axis))
    raw = torch.stack(axes, dim=1)  # (E, H, 3)
    t = raw - (raw * n).sum(dim=-1, keepdim=True) * n
    t_norm = torch.linalg.norm(t, dim=-1, keepdim=True)
    # Degenerate rake (load axis ⟂ wall plane): fall back to world -z projected.
    fallback = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(t)
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
    the release condition detectable at all. The pad wrench is distributed
    onto the active units and each unit is judged against its empirical
    draw; every violation goes through the debounce counter.
    """
    executor = self._get_executor()
    loads = executor.compute_anchor_loads()  # (E, H, 3) world
    t_hat, l_hat, n_hat = self._engage_frame()
    f_t = (loads * t_hat).sum(dim=-1)
    f_l = (loads * l_hat).sum(dim=-1)
    f_n = (loads * n_hat).sum(dim=-1)
    self.anchor_load_tnl = torch.stack([f_t, f_l, f_n], dim=-1)

    if self._ls_cfg.moment_capacity_enable and self._mcfg.anchor_lock_rotation:
      moment = executor.anchor_moment_w
      m_t = (moment * t_hat).sum(dim=-1)
      m_l = (moment * l_hat).sum(dim=-1)
      m_n = (moment * n_hat).sum(dim=-1)
      self.anchor_moment_tnl = torch.stack([m_t, m_l, m_n], dim=-1)
      moments = (m_t, m_l, m_n)
    else:
      self.anchor_moment_tnl.zero_()
      zeros = torch.zeros_like(f_t)
      moments = (zeros, zeros, zeros)

    margin, soft_cond, instant_cond, reason, worst_pad_scale = self._empirical_envelope_per_unit(
      f_t, f_l, f_n, *moments
    )
    # The worst unit's load at pad scale (x N).
    self.anchor_effective_load_tnl = worst_pad_scale
    # Force-only margin with the same unit draws (moment billing excluded);
    # diagnostics only, never resampled here.
    count = self._unit_x.numel()
    margin_raw = self._empirical.margins(
      f_t.unsqueeze(-1).expand_as(self.unit_strength_quantile) / count,
      f_l.unsqueeze(-1).expand_as(self.unit_strength_quantile) / count,
      f_n.unsqueeze(-1).expand_as(self.unit_strength_quantile) / count,
      self.unit_strength_quantile,
      self.unit_loading_quantile,
    )[0].amin(-1)
    self.hold_margin = torch.where(self._engaged, margin, torch.ones_like(margin))
    self.hold_margin_raw = torch.where(self._engaged, margin_raw, torch.ones_like(margin_raw))

    # The constraint only produces force one decimation after engaging, so
    # hold checks wait exactly the readout settling; there is no immunity
    # window beyond that.
    guard = self._ls_cfg.hold_judge_guard_steps
    seed_guard = self._ls_cfg.seed_hold_judge_guard_steps
    if seed_guard is None:
      in_grace = self._steps_since_engage < guard
    else:
      # Seeded latches wait out the RSI whole-body settle transient too
      # (see the cfg field); _seeded survives only while continuously
      # engaged since birth, so this never loosens in-play engagements.
      in_grace = self._steps_since_engage < torch.where(
        self._seeded,
        torch.full_like(self._steps_since_engage, seed_guard),
        torch.full_like(self._steps_since_engage, guard),
      )
    active = self._engaged & ~in_grace
    soft_violation = active & soft_cond
    self._violation_count = torch.where(
      soft_violation,
      self._violation_count + 1,
      torch.zeros_like(self._violation_count),
    )
    release = (active & instant_cond) | (self._violation_count >= max(1, self._ls_cfg.violation_consecutive_steps))
    self.release_reason = torch.where(release, reason, torch.zeros_like(reason))
    return release

  def _redraw_empirical_latches(self, engage_transition: torch.Tensor) -> None:
    """Only newly engaged pads change; every retained unit surface is stable."""
    on = engage_transition.unsqueeze(-1)
    self.unit_strength_quantile = torch.where(
      on, torch.rand_like(self.unit_strength_quantile), self.unit_strength_quantile
    )
    self.unit_loading_quantile = torch.where(
      on, torch.rand_like(self.unit_loading_quantile), self.unit_loading_quantile
    )

  def _empirical_envelope_per_unit(self, ft, fl, fn, mt, ml, mn):
    a, b, c = distribute_wrench_to_units(
      ft, fl, fn, mt, ml, mn, self._unit_x, self._unit_y, self._unit_sxx, self._unit_syy, self._unit_srr
    )
    self.unit_load_tln = torch.stack((a, b, c), dim=-1)
    margins, cap, domain, early, tail = self._empirical.margins(
      a, b, c, self.unit_strength_quantile, self.unit_loading_quantile
    )
    self.unit_normal_capacity = cap
    self.unit_empirical_supported = domain
    self.unit_empirical_early_failed = early
    self.unit_empirical_tail_capped = tail
    margin, worst = margins.min(-1)
    idx = worst.unsqueeze(-1)
    worst_load = self.unit_load_tln.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, 3)).squeeze(-2)
    reason = torch.full_like(self.release_reason, 2)  # conditional normal release
    reason = torch.where(early.gather(-1, idx).squeeze(-1), 8, reason)
    # Domain rejection is explicitly distinct from a measured strength failure.
    reason = torch.where(~domain.gather(-1, idx).squeeze(-1), 7, reason)
    violation = margin < 0
    return margin, violation, torch.zeros_like(violation), reason, worst_load * self._unit_x.numel()

  def _face_wall_cos(self) -> torch.Tensor:
    moving = getattr(self, "_surfaces", None) is not None
    wall_inward = -self._surfaces.frame()[1] if moving else self._wall_inward_normal()
    result = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    for h, body_id in enumerate(self._ee_body_ids):
      hquat = self._asset.data.body_link_quat_w[:, body_id, :]
      # Dot product of the EE face axis and wall inward normal. Values near 1
      # mean the spine is facing into the wall; values near 0 are side-on.
      face = _quat_rotate(hquat, self._face_local_axis)
      result[:, h] = (face * (wall_inward[:, h] if moving else wall_inward)).sum(dim=-1)
    return result

  def _moving_surface_fraction(self):
    from scr_phybot_mjab.microspine.contact import pad_contact_center, pad_contact_face_centers, virtual_spine_fraction

    if self._surface_corners is None:
      model = self._env.sim.mj_model
      tensor = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self.device)
      self._surface_corners = tensor(
        [pad_contact_face_centers(model, self._mcfg, ee, self.cfg.asset_name) for ee in self._mcfg.ee_body_names]
      )
      self._surface_vertices = tensor(
        [pad_face_vertices(model, self._mcfg, ee, self.cfg.asset_name) for ee in self._mcfg.ee_body_names]
      )
      self._surface_centers = tensor(
        [pad_contact_center(model, self._mcfg, ee, self.cfg.asset_name) for ee in self._mcfg.ee_body_names]
      )
    pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids]
    quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
    corners = pos.unsqueeze(2) + rotate(quat.unsqueeze(2), self._surface_corners)
    vertices = pos.unsqueeze(2) + rotate(quat.unsqueeze(2), self._surface_vertices)
    point, normal, _, _ = self._surfaces.frame()
    self.surface_covered = self._surfaces.contains(vertices)
    fraction = virtual_spine_fraction(
      corners, point, normal, 0.0, self._mcfg.virtual_spine_grid, self._mcfg.virtual_spine_travel
    )
    return fraction * self.surface_covered

  def _virtual_spine_fraction(self) -> torch.Tensor:
    """Engaged fraction of the kinematic virtual spine field, per EE (E, H).

    Corner positions come from the corner sphere child BODIES — their world
    poses are latch-invariant (retraction moves the sphere geoms within those
    bodies, never the bodies) — and the wall surface point from the wall
    plane geom's live world position, with the same inward normal the engage
    frame trusts. Purely kinematic: per Wang et al. 2017 the spine normal
    suspension is force-free over its travel, so reach is the whole question.
    """
    if getattr(self, "_surfaces", None) is not None:
      return self._moving_surface_fraction()
    if self._spine_corner_body_ids is None:
      import mujoco

      from scr_phybot_mjab.microspine.contact import (
        contact_face_offset,
        resolve_sphere_geom_ids,
        sphere_names,
      )

      rows = self._mcfg.spheres_per_ee_rows
      cols = self._mcfg.spheres_per_ee_cols
      if rows < 2 or cols < 2:
        raise ValueError(
          f"The virtual spine field needs a >=2x2 contact-sphere grid to span the pad face; got {rows}x{cols}."
        )
      corner_ids: list[int] = []
      offsets, sphere_offsets, radii = [], [], []
      mj_model = self._env.sim.mj_model
      geom_lists = resolve_sphere_geom_ids(mj_model, self._mcfg)
      for ee_index, ee in enumerate(self._mcfg.ee_body_names):
        names = sphere_names(ee, rows, cols)
        for corner in (0, cols - 1, (rows - 1) * cols, rows * cols - 1):
          name = names[corner]
          ids, _ = self._asset.find_bodies([name])
          if len(ids) != 1:
            raise ValueError(f"Corner sphere body '{name}' not found.")
          corner_ids.append(ids[0])
          gid = geom_lists[ee_index][corner]
          rotation = np.empty(9)
          mujoco.mju_quat2Mat(rotation, mj_model.body_quat[mj_model.geom_bodyid[gid]])
          face_local = rotation.reshape(3, 3).T @ np.asarray(self._mcfg.face_local_axis)
          offset = contact_face_offset(
            mj_model.geom_type[gid], mj_model.geom_size[gid], face_local, mj_model.geom_quat[gid]
          )
          offsets.append(mj_model.geom_pos[gid] + offset)
          is_sphere = mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
          sphere_offsets.append(offset if is_sphere else np.zeros(3))
          radii.append(float(mj_model.geom_size[gid, 0]) if is_sphere else 0.0)
      self._spine_corner_body_ids = torch.tensor(corner_ids, dtype=torch.long, device=self.device)
      self._spine_corner_offsets = torch.tensor(np.asarray(offsets), dtype=torch.float32, device=self.device)
      self._spine_sphere_offsets = torch.tensor(np.asarray(sphere_offsets), dtype=torch.float32, device=self.device)
      self._spine_corner_radii = torch.tensor(radii, dtype=torch.float32, device=self.device)
      wall_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, self._mcfg.wall_geom_name)
      if wall_gid < 0:
        raise ValueError(f"Wall geom '{self._mcfg.wall_geom_name}' not found for the virtual spine field.")
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
        rot = wp.to_torch(self._env.sim.wp_data.geom_xmat)[0, wall_gid].reshape(3, 3)
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

    corners = self._contact_face_corners_world(for_reach=True)
    # Off-surface direction toward the robot is the engage frame's n_out; the
    # surface point is the wall geom's origin pushed out by the type-resolved
    # face offset (0 for a plane, the half-extent for a box).
    n_out = -self._wall_inward_normal()
    wall_point = wp.to_torch(self._env.sim.wp_data.geom_xpos)[:, self._wall_geom_id] + self._wall_surface_offset * n_out
    return virtual_spine_fraction(
      corners,
      wall_point.unsqueeze(1).expand(-1, self._num_ee, -1),
      n_out.unsqueeze(1).expand(-1, self._num_ee, -1),
      0.0,  # Corners already lie on the contact surface, including for boxes.
      self._mcfg.virtual_spine_grid,
      self._mcfg.virtual_spine_travel,
    )

  def _contact_face_corners_world(self, *, for_reach: bool = False) -> torch.Tensor:
    """Nominal surface centers, unaffected by latched geom retraction.

    Boxes use their face centers. Sphere proxies use their closest point to
    the wall for reach checks (``for_reach``) and their pad-facing poles
    otherwise.
    """
    ids = self._spine_corner_body_ids
    offsets = self._spine_corner_offsets
    if for_reach:
      offsets = offsets - self._spine_sphere_offsets
    points = self._asset.data.body_link_pos_w[:, ids] + quat_apply(
      self._asset.data.body_link_quat_w[:, ids],
      offsets.unsqueeze(0).expand(self.num_envs, -1, -1),
    )
    if for_reach:
      points = points + self._spine_corner_radii[None, :, None] * self._wall_inward_normal()[:, None, :]
    return points.reshape(self.num_envs, self._num_ee, 4, 3)

  def _unit_preload_condition(self) -> torch.Tensor:
    """Require compression on every unit in the equal-stiffness contact model.

    Netforce reports pad-on-wall force and torque about each contact centroid
    P. Shift each simultaneous sample to its array centre C:
    M_C = torque_P + (P-C) x force. Average the resulting wrenches over one
    policy step, then use the same unit geometry/allocation as the hold judge.
    An outward-normal unit load is tensile, so preload = max(-f_n_unit, 0).
    """
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    data = sensor.data
    if sensor.cfg.reduce != "netforce" or any(
      getattr(data, field, None) is None for field in ("force", "torque", "pos", "reference_pos")
    ):
      raise ValueError("Unit preload engagement needs netforce contact force, torque, pos and reference_pos")
    values = [data.force, data.torque, data.pos, data.reference_pos]
    if data.force_history is not None:
      values = [
        data.force_history,
        data.torque_history,
        getattr(data, "pos_history", None),
        getattr(data, "reference_pos_history", None),
      ]
      if any(v is None or v.shape != values[0].shape for v in values):
        raise ValueError("Unit preload engagement needs synchronized force/torque/pos/reference_pos histories")
    else:
      values = [v.unsqueeze(2) for v in values]
    perm = self._sensor_row_permutation(sensor)
    proxies = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    force, torque, position, center = (v[:, perm].reshape(self.num_envs, self._num_ee, proxies, -1, 3) for v in values)
    moment = torque + torch.linalg.cross(position - center, force, dim=-1)
    force = force.sum(dim=2).mean(dim=2)
    moment = moment.sum(dim=2).mean(dim=2)
    if self._mcfg.disable_contacts_while_latched:
      # Do not report stale pre-latch contact history after retraction.
      force = torch.where(self._engaged[..., None], 0.0, force)
      moment = torch.where(self._engaged[..., None], 0.0, moment)
    basis = self._engage_frame()
    components = [(vector * axis).sum(-1) for vector in (force, moment) for axis in basis]
    self.pad_contact_wrench_tln = torch.stack(components, dim=-1)
    _, _, normal = distribute_wrench_to_units(
      *components,
      self._unit_x,
      self._unit_y,
      self._unit_sxx,
      self._unit_syy,
      self._unit_srr,
    )
    self.unit_preload_n = (-normal).clamp_min(0.0)
    return (
      torch.isfinite(self.pad_contact_wrench_tln).all(dim=-1)
      & torch.isfinite(self.unit_preload_n).all(dim=-1)
      & (self.unit_preload_n >= self._mcfg.engage_unit_preload_n).all(dim=-1)
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
      above = compression[:, s_start:s_end] >= self._mcfg.engage_normal_force_threshold
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
        names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
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
      self._sensor_row_perm = torch.tensor(perm, dtype=torch.long, device=self.device)
    return self._sensor_row_perm

  def _sensor_contact_features(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    data = sensor.data
    if data.found is None or data.force is None:
      raise ValueError(f"Contact sensor '{self.cfg.contact_sensor_name}' must provide 'found' and 'force' fields.")
    sensor_cfg = getattr(sensor, "cfg", None)
    world_force = getattr(sensor_cfg, "reduce", None) == "netforce" or getattr(sensor_cfg, "global_frame", False)

    perm = self._sensor_row_permutation(sensor)
    found = data.found[:, perm]
    found_mask = found > 0
    engage_force = data.force[:, perm]
    inward = None
    if world_force:
      if getattr(self, "_surfaces", None) is not None:
        inward = -self._surfaces.frame()[1].repeat_interleave(
          self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols, dim=1
        )
      else:
        inward = self._wall_inward_normal()[:, None, :]
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
      if world_force:
        normal_history = (force_history * inward.unsqueeze(2)).sum(-1)
      else:
        normal_history = force_history[..., 0]
      found_mask |= (normal_history > 0.0).any(dim=2)

    self._validate_sensor_shapes(found, engage_force)

    # Netforce sums every box-wall contact and returns a world-frame vector
    # directed primary -> secondary. Legacy sensors use contact coordinates.
    normal_component = (engage_force * inward).sum(-1) if world_force else engage_force[..., 0]
    normal_force = normal_component.clamp(min=0.0)
    normal_force = torch.where(found_mask, normal_force, torch.zeros_like(normal_force))

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

  def compute(self, dt: float) -> None:
    if self.cfg.skip_zero_time_update and dt <= 0.0:
      # A partial reset also calls compute for untouched environments. No
      # physics elapsed: preserve their release events, debounce and ages.
      self._update_metrics()
      return
    super().compute(dt)

  def _update_command(self) -> None:
    normal_forces, found_count = self._sensor_contact_features()
    if self._mcfg.disable_contacts_while_latched:
      # Latched pads' spheres are retracted — they physically have no wall
      # contact. The substep-history recovery (meant to bridge sampling
      # aliasing of real contacts) must not replay stale pre-latch forces.
      spheres_per_ee_mask = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
      latched_spheres = (
        self._engaged.unsqueeze(-1).expand(-1, self._num_ee, spheres_per_ee_mask).reshape(self.num_envs, -1)
      )
      normal_forces = torch.where(latched_spheres, torch.zeros_like(normal_forces), normal_forces)
      found_count = torch.where(self._engaged, torch.zeros_like(found_count), found_count)
    self.contact_count = found_count
    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    self.pad_press_force = normal_forces.clamp(min=0.0).view(self.num_envs, self._num_ee, spheres_per_ee).sum(dim=2)
    self.engaged_this_step.zero_()
    self.released_was_seeded.zero_()

    # Unit preload replaces the legacy proxy-force gate. It is evaluated after
    # the grip-disabled early-out below.
    if self._mcfg.engage_unit_preload_n > 0:
      engage_condition = torch.ones_like(self._engaged)
    else:
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
      self._seeded.zero_()
      self.unit_preload_n.zero_()
      self.pad_contact_wrench_tln.zero_()
      self._sync_latched_contact_margins()
      return

    if self._mcfg.engage_unit_preload_n > 0:
      engage_condition &= self._unit_preload_condition()
    if self._mcfg.engage_parallel_cos_threshold > 0.0:
      engage_condition &= face_cos >= self._mcfg.engage_parallel_cos_threshold
    if self._mcfg.engage_min_total_normal_force > 0.0:
      engage_condition &= self.pad_press_force >= self._mcfg.engage_min_total_normal_force
    if self._mcfg.engage_min_spine_fraction > 0.0:
      # The seating gate: enough of the virtual spine field must be able to
      # reach the wall. Replaces the 2-rigid-sphere count as the 'actually
      # seated' judgement (a rigid pad generically touches a flat wall with
      # one sphere; demanding two demanded sub-degree parallelism).
      engage_condition &= self.spine_fraction >= self._mcfg.engage_min_spine_fraction
    if self._mcfg.engage_max_antirake_vel > 0.0:
      # Rake-direction gate (Wang 2017 §2): spines hook under along-rake
      # loading only. A pad sliding against its rake (up-wall) cannot
      # catch — pressing feet legitimately skid upward during a body
      # pull-up, and must not farm anchors doing so. Stationary or
      # down-sliding pads (a catch during a fall) latch normally.
      t_hat, _, _ = self._engage_frame()
      vel_w = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]
      if getattr(self, "_surfaces", None) is not None:
        quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
        offset = rotate(quat, self._surface_centers)
        center = self._asset.data.body_link_pos_w[:, self._ee_body_ids] + offset
        omega = self._asset.data.body_link_ang_vel_w[:, self._ee_body_ids]
        vel_w = vel_w + torch.cross(omega, offset, dim=-1) - self._surfaces.velocity_at(center)
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
    engage_transition = engage_count >= max(1, self._mcfg.engage_consecutive_steps)
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
    if getattr(self, "_surfaces", None) is not None:
      outside = self._engaged & ~self.surface_covered
      self.forced_release_this_step |= outside
      self.release_reason = torch.where(outside, torch.full_like(self.release_reason, 9), self.release_reason)
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

    # Seed provenance: a falling edge on a seeded anchor is a demolished
    # gift (the transition's seed economy bills failure-reason cases at
    # full price); rising edges are the policy's own welds — never seeded.
    falling = was_engaged & ~new_engaged
    self.released_was_seeded |= falling & self._seeded
    self._seeded &= new_engaged

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
    self._reseat_hold_steps = torch.where(expired, torch.zeros_like(self._reseat_hold_steps), self._reseat_hold_steps)
    self._redraw_empirical_latches(engage_transition)
    self._engage_count = engage_count
    # After an engage transition, clear its counter.
    self._engage_count = torch.where(engage_transition, torch.zeros_like(self._engage_count), self._engage_count)

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
    self._engage_count = torch.where(self._engaged, torch.zeros_like(self._engage_count), self._engage_count)
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
    self._violation_count = torch.where(self._engaged, self._violation_count, torch.zeros_like(self._violation_count))
