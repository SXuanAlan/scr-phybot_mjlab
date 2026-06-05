"""Environment-side affordance model for microspine adhesion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg


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
  actuator forces.
  """

  cfg: MicrospineAffordanceCfg

  def __init__(self, cfg: MicrospineAffordanceCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._num_ee = len(cfg.ee_body_names)
    self._engaged = torch.full(
      (self.num_envs, self._num_ee),
      bool(cfg.initial_on),
      dtype=torch.bool,
      device=self.device,
    )
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
    for bname in cfg.ee_body_names:
      ids, _ = self._asset.find_bodies([bname])
      self._ee_body_ids.append(ids[0])

    face_axis = torch.tensor(
      cfg.face_local_axis, dtype=torch.float32, device=self.device
    )
    face_norm = torch.linalg.norm(face_axis).clamp_min(1.0e-6)
    self._face_local_axis = face_axis / face_norm

  @property
  def command(self) -> torch.Tensor:
    return self._engaged.float()

  @property
  def engaged(self) -> torch.Tensor:
    return self._engaged

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._engaged[env_ids] = bool(self.cfg.initial_on)
    self._engage_count[env_ids] = 0
    self._disengage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0

  def _wall_inward_normal(self) -> torch.Tensor:
    wall = self._env.scene[self.cfg.wall_entity_name]
    wquat = wall.data.root_link_quat_w
    ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
    inward_x = 1.0 - 2.0 * (wy * wy + wz * wz)
    inward_y = 2.0 * (wx * wy + ww * wz)
    inward_z = 2.0 * (wx * wz - ww * wy)
    return torch.stack([inward_x, inward_y, inward_z], dim=-1)

  def _face_wall_cos(self) -> torch.Tensor:
    wall_inward = self._wall_inward_normal()
    result = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    for h, body_id in enumerate(self._ee_body_ids):
      hquat = self._asset.data.body_link_quat_w[:, body_id, :]
      face = _quat_rotate(hquat, self._face_local_axis)
      result[:, h] = (face * wall_inward).sum(dim=-1)
    return result

  def _normal_force_condition(self, normal_forces: torch.Tensor) -> torch.Tensor:
    if self.cfg.engage_normal_force_threshold <= 0.0:
      return torch.ones_like(self._engaged)

    spheres_per_ee = self.cfg.spheres_per_ee
    expected_spheres = self._num_ee * spheres_per_ee
    if normal_forces.shape != (self.num_envs, expected_spheres):
      raise ValueError(
        f"Microspine normal force shape {tuple(normal_forces.shape)} is incompatible "
        f"with {self._num_ee} EEs and {spheres_per_ee} spheres per EE."
      )

    compression = normal_forces.clamp(min=0.0)
    result = torch.zeros_like(self._engaged)
    min_count = self.cfg.engage_min_spheres
    min_fraction = self.cfg.engage_min_fraction
    for h in range(self._num_ee):
      s_start = h * spheres_per_ee
      s_end = s_start + spheres_per_ee
      above = (
        compression[:, s_start:s_end] >= self.cfg.engage_normal_force_threshold
      )
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
  ) -> None:
    expected_spheres = self._num_ee * self.cfg.spheres_per_ee
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

  def _sensor_contact_features(self) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    data = sensor.data
    if data.found is None or data.force is None or data.normal is None:
      raise ValueError(
        f"Contact sensor '{self.cfg.contact_sensor_name}' must provide "
        "'found', 'force', and 'normal' fields."
      )

    found = data.found
    force = data.force
    if data.force_history is not None:
      force = data.force_history.mean(dim=2)
    normal = data.normal

    self._validate_sensor_shapes(found, force, normal)

    found_mask = found > 0
    normal_unit = normal / torch.linalg.norm(
      normal, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    normal_force = torch.abs((force * normal_unit).sum(dim=-1))
    normal_force = torch.where(
      found_mask, normal_force, torch.zeros_like(normal_force)
    )

    force_z_per_sphere = torch.where(
      found_mask,
      force[..., 2],
      torch.zeros_like(force[..., 2]),
    )
    force_z = torch.zeros(self.num_envs, self._num_ee, device=self.device)
    for h in range(self._num_ee):
      s_start = h * self.cfg.spheres_per_ee
      s_end = s_start + self.cfg.spheres_per_ee
      force_z[:, h] = force_z_per_sphere[:, s_start:s_end].sum(dim=1)

    return normal_force, force_z

  def _update_command(self) -> None:
    normal_forces, force_z = self._sensor_contact_features()

    engage_condition = torch.ones_like(self._engaged)
    if self.cfg.engage_force_z_threshold > 0.0:
      engage_condition &= force_z > self.cfg.engage_force_z_threshold
    engage_condition &= self._normal_force_condition(normal_forces)

    if self.cfg.engage_parallel_cos_threshold > 0.0:
      engage_condition &= (
        self._face_wall_cos() >= self.cfg.engage_parallel_cos_threshold
      )

    cooldown_active = self._cooldown_steps > 0
    engage_candidate = engage_condition & ~self._engaged & ~cooldown_active
    self._engage_count = torch.where(
      engage_candidate,
      self._engage_count + 1,
      torch.zeros_like(self._engage_count),
    )
    engage_transition = self._engage_count >= max(
      1, self.cfg.engage_consecutive_steps
    )

    disengage_candidate = self._engaged & ~engage_condition
    if self.cfg.disengage_force_z_threshold > 0.0:
      disengage_candidate |= self._engaged & (
        force_z < -self.cfg.disengage_force_z_threshold
      )
    self._disengage_count = torch.where(
      disengage_candidate,
      self._disengage_count + 1,
      torch.zeros_like(self._disengage_count),
    )
    disengage_transition = self._disengage_count >= max(
      1, self.cfg.disengage_consecutive_steps
    )

    self._engaged[:] = torch.where(
      engage_transition,
      torch.ones_like(self._engaged),
      torch.where(
        disengage_transition,
        torch.zeros_like(self._engaged),
        self._engaged,
      ),
    )
    self._engage_count = torch.where(
      engage_transition, torch.zeros_like(self._engage_count), self._engage_count
    )
    self._disengage_count = torch.where(
      disengage_transition,
      torch.zeros_like(self._disengage_count),
      self._disengage_count,
    )

    self._cooldown_steps = (self._cooldown_steps - 1).clamp(min=0)
    cooldown_set = max(0, self.cfg.reengage_cooldown_steps)
    if cooldown_set > 0:
      self._cooldown_steps = torch.where(
        disengage_transition,
        torch.full_like(self._cooldown_steps, cooldown_set),
        self._cooldown_steps,
      )

    self._engage_count = torch.where(
      self._engaged, torch.zeros_like(self._engage_count), self._engage_count
    )
