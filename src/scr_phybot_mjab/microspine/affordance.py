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
    shape = (self.num_envs, self._num_ee)
    self.limit_f_t_max = torch.zeros(shape, device=self.device)
    self.limit_f_rev_max = torch.zeros(shape, device=self.device)
    self.limit_mu_adhesion = torch.zeros(shape, device=self.device)
    self.limit_f_n0 = torch.zeros(shape, device=self.device)
    self.limit_mu_lateral = torch.zeros(shape, device=self.device)
    self.limit_f_l0 = torch.zeros(shape, device=self.device)
    self._violation_count = torch.zeros(
      shape, dtype=torch.long, device=self.device
    )
    self._steps_since_engage = torch.zeros(
      shape, dtype=torch.long, device=self.device
    )
    self.anchor_load_tnl = torch.zeros(
      self.num_envs, self._num_ee, 3, device=self.device
    )
    self.hold_margin = torch.ones(shape, device=self.device)
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
    self._spine_corner_body_ids: torch.Tensor | None = None
    self._spine_sphere_radius: float | None = None
    self._wall_geom_id: int | None = None
    rake_axis = torch.tensor(
      self._ls_cfg.ee_rake_local_axis, dtype=torch.float32, device=self.device
    )
    self._rake_local_axis = rake_axis / torch.linalg.norm(rake_axis).clamp_min(
      1.0e-6
    )
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
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
    # Seeded (RSI) latches get full capacity: the reference stance seats the
    # pads flat by construction, and kinematics may not have refreshed the
    # just-written reset pose yet, so a live fraction here could be stale.
    self._seat_scale[env_ids] = 1.0
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
    self._steps_since_engage[env_ids] = 0
    self.anchor_load_tnl[env_ids] = 0.0
    self.hold_margin[env_ids] = 1.0
    self.face_cos[env_ids] = 0.0
    self.contact_count[env_ids] = 0.0
    self.pad_press_force[env_ids] = 0.0
    self.spine_fraction[env_ids] = 0.0
    self._seat_scale[env_ids] = 1.0
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
    loads = self._get_executor().compute_anchor_loads()  # (E, H, 3) world
    t_hat, l_hat, n_hat = self._engage_frame()
    f_t = (loads * t_hat).sum(dim=-1)
    f_l = (loads * l_hat).sum(dim=-1)
    f_n = (loads * n_hat).sum(dim=-1)
    self.anchor_load_tnl = torch.stack([f_t, f_l, f_n], dim=-1)

    if self._wang_lut is not None:
      margin, soft_cond, instant_cond, reason = self._wang_envelope(
        f_t, f_l, f_n
      )
    else:
      margin, soft_cond, instant_cond, reason = self._wedge_envelope(
        f_t, f_l, f_n
      )
    self.hold_margin = torch.where(
      self._engaged, margin, torch.ones_like(margin)
    )

    # The constraint only produces force one decimation after engaging, so
    # give fresh engagements a few steps of grace before hold checks apply.
    in_grace = self._steps_since_engage < self._ls_cfg.hold_grace_steps
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

  def _wang_envelope(
    self, f_t: torch.Tensor, f_l: torch.Tensor, f_n: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed Wang-2017 surface: (margin, soft, instant, reason).

    Radial test: the load magnitude |(f_t, f_l, f_n+)| must stay inside the
    direction-dependent capacity F0 * max(R(theta, phi), r_floor), F0 drawn
    from ``f_t_max_range``. Compression rides the contacts and is clamped
    out of the tension component.
    """
    ls = self._ls_cfg
    tension = f_n.clamp_min(0.0)
    f0 = self.limit_f_t_max
    if self._mcfg.seat_quality_scales_capacity:
      # Wang 2017 Sec 2.4: pad capacity sums per-spine contributions, so a
      # partially seated latch scales the whole closed surface down by the
      # spine fraction captured at its engage step.
      f0 = f0 * self._seat_scale
    ratio = capacity_ratio(self._wang_lut, f_t, f_l, f_n)
    cap = f0 * ratio.clamp_min(ls.wang_r_floor)
    load_mag = torch.sqrt(f_t**2 + f_l**2 + tension**2)
    # Normalized signed distance: ~1 unloaded, 0 on the surface, < 0 outside.
    margin = (cap - load_mag) / (cap + ls.wang_margin_eps * f0)
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

    found = data.found
    found_mask = found > 0
    engage_force = data.force
    if data.force_history is not None:
      # Average force history so one compression impulse cannot latch a spine.
      engage_force = data.force_history.mean(dim=2)
      # MuJoCo 3.10 contacts can be created and resolved within a decimation
      # window. In that case the instantaneous ``found`` value is zero even
      # though the force history contains the compression that should be used
      # by the latch gate. The history length is configured to one policy step,
      # so treating any historical normal force as contact cannot leak across
      # steps.
      found_mask |= (data.force_history[..., 0] > 0.0).any(dim=2)

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
      self.hold_margin.fill_(1.0)
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
