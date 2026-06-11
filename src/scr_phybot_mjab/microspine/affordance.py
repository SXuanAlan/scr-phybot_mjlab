"""Environment-side affordance model for microspine adhesion."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import warp as wp

from mjlab.managers.command_manager import CommandTerm
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg

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
  """Stateful world affordance for per-EE microspine adhesion.

  The affordance is intentionally outside the robot action space. It reads the
  world's contact features and owns the per-EE engage/disengage state machine.
  The physics term only consumes the resulting on/off signal and applies
  external sphere-body forces.

  Force sign convention used here:
  - ``force_z > 0`` means the wall pushes the spine in world +z.
  - ``force_z < 0`` means the wall pushes the spine in world -z.
  Engage requires enough +z support; an already-engaged spine only releases on
  enough -z force, so losing the engage condition alone does not drop the latch.
  """

  cfg: MicrospineAffordanceCfg

  def __init__(self, cfg: MicrospineAffordanceCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._mcfg = cfg.microspine_cfg
    self._num_ee = len(self._mcfg.ee_body_names)
    # This is the latched affordance state consumed by MicrospinePhysics. It is
    # intentionally separate from robot actions, so policy outputs cannot command
    # adhesion directly.
    self._engaged = torch.full(
      (self.num_envs, self._num_ee),
      bool(self._mcfg.initial_on),
      dtype=torch.bool,
      device=self.device,
    )
    # Consecutive-step counters implement hysteresis/debouncing for noisy contact
    # signals. Cooldown prevents immediate re-engage after a forced release.
    self._engage_count = torch.zeros(
      self.num_envs, self._num_ee, dtype=torch.long, device=self.device
    )
    self._disengage_count = torch.zeros(
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

    # ``face_local_axis`` is the end-effector direction expected to point into
    # the wall. Normalizing once keeps the per-step dot product numerically clean.
    face_axis = torch.tensor(
      self._mcfg.face_local_axis, dtype=torch.float32, device=self.device
    )
    face_norm = torch.linalg.norm(face_axis).clamp_min(1.0e-6)
    self._face_local_axis = face_axis / face_norm

    # --- limit-surface state (constraint attachment mode) ---------------
    # The hold envelope releases an engaged spine when its anchor load leaves
    # the configured cone. Parameters are per-env tensors resampled from cfg
    # ranges at reset — built-in domain randomization, calibratable against
    # bench pull-off tests.
    self._ls_cfg = self._mcfg.limit_surface
    self._constraint_mode = self._mcfg.attachment_mode == "constraint"
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
    # HUD/debug readouts (pure GPU writes; no host syncs in the step path).
    # release_reason: which envelope face fired on a forced release —
    # 0 none, 1 reverse-shear, 2 tension-cone, 3 lateral, 4 ultimate-shear.
    # Like anchor_load_tnl it holds the violating value on the release step
    # and resets the step after.
    self.release_reason = torch.zeros(shape, dtype=torch.int8, device=self.device)
    self.face_cos = torch.zeros(shape, device=self.device)
    self.contact_count = torch.zeros(shape, device=self.device)
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

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Command terms are resampled on reset. Reset all latch bookkeeping alongside
    # the command so old contact history cannot leak into the new episode.
    self._engaged[env_ids] = bool(self._mcfg.initial_on)
    self._engage_count[env_ids] = 0
    self._disengage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
    self.anchor_load_tnl[env_ids] = 0.0
    self.hold_margin[env_ids] = 1.0
    self.forced_release_this_step[env_ids] = False
    self.release_reason[env_ids] = 0
    self._sample_limit_params(env_ids)

  def _wall_inward_normal(self) -> torch.Tensor:
    wall = self._env.scene[self._mcfg.wall_entity_name]
    wquat = wall.data.root_link_quat_w
    # Same convention as MicrospinePhysics: wall local +x is the inward normal.
    # Keeping this calculation identical makes the sensing gate and applied
    # adhesion force agree when the wall pose changes.
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
    the release condition detectable at all. Conditions 1-3 (reverse shear,
    tension cone, lateral cone) are debounced; condition 4 (ultimate shear)
    releases instantly.
    """
    loads = self._get_executor().compute_anchor_loads()  # (E, H, 3) world
    t_hat, l_hat, n_hat = self._engage_frame()
    f_t = (loads * t_hat).sum(dim=-1)
    f_l = (loads * l_hat).sum(dim=-1)
    f_n = (loads * n_hat).sum(dim=-1)
    self.anchor_load_tnl = torch.stack([f_t, f_l, f_n], dim=-1)

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
    self.hold_margin = torch.where(
      self._engaged, margin, torch.ones_like(margin)
    )

    # The constraint only produces force one decimation after engaging, so
    # give fresh engagements a few steps of grace before hold checks apply.
    in_grace = self._steps_since_engage < self._ls_cfg.hold_grace_steps
    active = self._engaged & ~in_grace
    soft_violation = active & ~(c_rev & c_tension & c_lateral)
    self._violation_count = torch.where(
      soft_violation,
      self._violation_count + 1,
      torch.zeros_like(self._violation_count),
    )
    release = (active & ~c_ultimate) | (
      self._violation_count
      >= max(1, self._ls_cfg.violation_consecutive_steps)
    )

    # Which face fired (HUD/debug). Priority by overwrite order: lateral <
    # reverse < tension < ultimate, masked to actual releases.
    reason = torch.zeros_like(self.release_reason)
    reason = torch.where(~c_lateral, torch.full_like(reason, 3), reason)
    reason = torch.where(~c_rev, torch.full_like(reason, 1), reason)
    reason = torch.where(~c_tension, torch.full_like(reason, 2), reason)
    reason = torch.where(~c_ultimate, torch.full_like(reason, 4), reason)
    self.release_reason = torch.where(
      release, reason, torch.zeros_like(reason)
    )
    return release

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
    normal: torch.Tensor,
    tangent: torch.Tensor,
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
    if normal.shape != (self.num_envs, expected_spheres, 3):
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' normal shape "
        f"{tuple(normal.shape)} does not match expected "
        f"({self.num_envs}, {expected_spheres}, 3)."
      )
    if tangent.shape != (self.num_envs, expected_spheres, 3):
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' tangent shape "
        f"{tuple(tangent.shape)} does not match expected "
        f"({self.num_envs}, {expected_spheres}, 3)."
      )

  def _contact_force_features(
    self,
    force: torch.Tensor,
    normal: torch.Tensor,
    tangent: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    normal_unit = normal / torch.linalg.norm(
      normal, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    tangent_unit = tangent / torch.linalg.norm(
      tangent, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    tangent2_unit = torch.cross(normal_unit, tangent_unit, dim=-1)

    # MuJoCo Warp CONTACT sensor stores force as
    # [normal, tangent, second_tangent] in the primary->secondary contact frame.
    # Therefore force[..., 2] is not world z. Reconstruct the contact basis in
    # world coordinates and project the full contact-frame force onto world z.
    normal_force = force[..., 0]
    primary_on_secondary_force_z = (
      force[..., 0] * normal_unit[..., 2]
      + force[..., 1] * tangent_unit[..., 2]
      + force[..., 2] * tangent2_unit[..., 2]
    )
    return normal_force, primary_on_secondary_force_z

  def _sensor_contact_features(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    data = sensor.data
    if (
      data.found is None
      or data.force is None
      or data.normal is None
      or data.tangent is None
    ):
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' must provide "
        "'found', 'force', 'normal', and 'tangent' fields."
      )

    found = data.found
    instant_force = data.force
    engage_force = instant_force
    if data.force_history is not None:
      # Average the force history for engage so a single supporting impulse does
      # not latch the spine. Disengage uses the instantaneous force below; using
      # the same history average there can hide a real pull-off load behind older
      # +z support frames.
      engage_force = data.force_history.mean(dim=2)
    normal = data.normal
    tangent = data.tangent

    self._validate_sensor_shapes(found, engage_force, normal, tangent)

    found_mask = found > 0
    normal_force, primary_on_secondary_force_z = self._contact_force_features(
      engage_force, normal, tangent
    )
    normal_force = normal_force.clamp(min=0.0)
    normal_force = torch.where(
      found_mask, normal_force, torch.zeros_like(normal_force)
    )

    # Primary is the robot sphere and secondary is the wall. The sensor reports
    # robot-on-wall force, so negate to get the wall-on-robot support force.
    # This makes +z mean the wall supports/pushes the spine upward.
    force_z_raw = -primary_on_secondary_force_z
    force_z_per_sphere = torch.where(
      found_mask,
      force_z_raw,
      torch.zeros_like(force_z_raw),
    )
    _, instant_primary_on_secondary_force_z = self._contact_force_features(
      instant_force, normal, tangent
    )
    instant_force_z_per_sphere = torch.where(
      found_mask,
      -instant_primary_on_secondary_force_z,
      torch.zeros_like(instant_primary_on_secondary_force_z),
    )
    force_z = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    instant_force_z = torch.zeros_like(force_z)
    found_count = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    spheres_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    for h in range(self._num_ee):
      s_start = h * spheres_per_ee
      s_end = s_start + spheres_per_ee
      # Aggregate z support over all spheres on the same end-effector. The state
      # machine works per EE, not per individual sphere.
      force_z[:, h] = force_z_per_sphere[:, s_start:s_end].sum(dim=1)
      instant_force_z[:, h] = instant_force_z_per_sphere[
        :, s_start:s_end
      ].sum(dim=1)
      found_count[:, h] = found_mask[:, s_start:s_end].sum(dim=1)

    return normal_force, force_z, instant_force_z, found_count

  def _log_transition_sensor_data(
    self,
    was_engaged: torch.Tensor,
    new_engaged: torch.Tensor,
    engage_condition: torch.Tensor,
    normal_forces: torch.Tensor,
    force_z: torch.Tensor,
    instant_force_z: torch.Tensor,
    found_count: torch.Tensor,
    face_cos: torch.Tensor,
    engage_count: torch.Tensor,
    disengage_count: torch.Tensor,
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
        f"wall_on_spine_z={force_z[0, ee_index].item():.3f}N "
        f"instant_wall_on_spine_z={instant_force_z[0, ee_index].item():.3f}N "
        f"face_cos={face_cos[0, ee_index].item():.3f} "
        f"engage_count={int(engage_count[0, ee_index].item())} "
        f"disengage_count={int(disengage_count[0, ee_index].item())} "
        f"cooldown={int(self._cooldown_steps[0, ee_index].item())}"
      )

  def _update_command(self) -> None:
    normal_forces, force_z, instant_force_z, found_count = (
      self._sensor_contact_features()
    )
    self.contact_count = found_count

    # Engage from OFF requires all enabled gates: upward wall-on-spine force,
    # optional compression on the sphere grid, and optional face alignment.
    engage_condition = torch.ones_like(self._engaged)
    if self._mcfg.engage_force_z_threshold > 0.0:
      engage_condition &= force_z > self._mcfg.engage_force_z_threshold
    engage_condition &= self._normal_force_condition(normal_forces)

    # Always computed (cheap, a few quat rotates) so HUD/debug readers see a
    # live value even when the alignment gate is disabled.
    face_cos = self._face_wall_cos()
    self.face_cos = face_cos

    if self._mcfg.engage_parallel_cos_threshold > 0.0:
      engage_condition &= (
        face_cos >= self._mcfg.engage_parallel_cos_threshold
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

    # Once latched, release only when the wall pushes the EE downward strongly
    # enough. Losing the +z engage condition alone does not disengage.
    disengage_candidate = torch.zeros_like(self._engaged)
    if self._mcfg.disengage_force_z_threshold > 0.0:
      disengage_candidate |= self._engaged & (
        instant_force_z < -self._mcfg.disengage_force_z_threshold
      )
    # Disengage also uses consecutive steps, so a transient downward impulse does
    # not immediately release the spine unless configured to one step.
    disengage_count = torch.where(
      disengage_candidate,
      self._disengage_count + 1,
      torch.zeros_like(self._disengage_count),
    )
    disengage_transition = disengage_count >= max(
      1, self._mcfg.disengage_consecutive_steps
    )

    # Constraint mode: the limit surface owns forced release. It runs on the
    # pre-transition latch state, so a spine engaged this very step is never
    # judged on stale loads.
    if self._constraint_mode and self._ls_cfg.enable:
      self.forced_release_this_step = self._limit_surface_release()
      disengage_transition = disengage_transition | self.forced_release_this_step
      if self._mcfg.debug and bool(self.forced_release_this_step[0].any()):
        load = self.anchor_load_tnl[0].tolist()
        print(
          "[Microspine] LIMIT-SURFACE RELEASE env0 "
          f"released={self.forced_release_this_step[0].tolist()} "
          f"load_tnl={[[round(v, 1) for v in ee] for ee in load]}"
        )
    else:
      self.forced_release_this_step = torch.zeros_like(self._engaged)

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
      force_z,
      instant_force_z,
      found_count,
      face_cos,
      engage_count,
      disengage_count,
    )

    self._engaged[:] = new_engaged
    self._engage_count = engage_count
    self._disengage_count = disengage_count
    # After a transition, clear that transition's counter. Without this, an old
    # satisfied counter could immediately retrigger after the latch changes.
    self._engage_count = torch.where(
      engage_transition, torch.zeros_like(self._engage_count), self._engage_count
    )
    self._disengage_count = torch.where(
      disengage_transition,
      torch.zeros_like(self._disengage_count),
      self._disengage_count,
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

    # Once engaged, stale engage counts are irrelevant; once disengaged, stale
    # disengage counts are irrelevant. Keep only the counter that can matter next.
    self._engage_count = torch.where(
      self._engaged, torch.zeros_like(self._engage_count), self._engage_count
    )

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
