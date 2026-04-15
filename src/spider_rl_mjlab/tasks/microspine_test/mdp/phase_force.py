"""Three-phase test force event for the microspine adhesion test.

Phase 1 [0, phase1_end):        push force perpendicular to hand contact face
                                (local +x rotated to world via hand quaternion)
                                to engage hand against wall.
Phase 2 [phase1_end, phase2_end): shear/pull force with randomized direction.
Phase 3 [phase2_end, inf):      upward shear (+z) to test disengage.

Phase 2 force is composed of:
  - A constant -z (downward) shear component (gravity-like pull).
  - Randomized ±x component (push into / pull off wall).
  - Randomized ±y component (lateral shear).
All three components evolve via independent OU processes, biased toward
large magnitudes so the adhesion is thoroughly tested.

Phase 3 applies a constant +z (upward) force to generate friction along
the hand's "up" direction, testing the disengage condition.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quaternion (wxyz convention). Batched.
  quat: (E, 4), vec: (3,) or (E, 3) → (E, 3).
  """
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  if vec.dim() == 1:
    vx, vy, vz = vec[0], vec[1], vec[2]
  else:
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return torch.stack([rx, ry, rz], dim=-1)


class PhaseForce:
  """Step-mode event that applies a three-phase external force to a body.

  Phase 2 uses independent OU processes for each axis force component:
    fz: always negative (downward shear), magnitude from phase2_mag_range.
    fx: ±x (push/pull relative to wall), magnitude up to fz * tan(max_angle).
    fy: ±y (lateral shear), magnitude up to fz * tan(max_angle).

  Phase 3 applies a constant +z (upward) force to test disengage.

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the entity containing the target body.
    body_name: Name of the body to apply forces to.
    phase1_duration: Duration of phase 1 [s].
    phase1_mag_range: (min, max) magnitude of phase 1 +x push [N].
    phase2_duration: Duration of phase 2 [s]. If 0 or absent, phase 2 lasts forever.
    phase2_mag_range: (min, max) magnitude of phase 2 -z shear [N].
    phase2_angle_range: (min, max) deviation angle from -z [degrees].
    phase2_ou_theta: OU mean-reversion rate [1/s]. Default 1.0.
    phase2_ou_sigma: OU noise std for all components [fraction/sqrt(s)]. Default 0.5.
    phase3_mag_range: (min, max) magnitude of phase 3 +z force [N]. If absent, no phase 3.
  """

  def __init__(self, cfg, env):
    from mjlab.managers.event_manager import EventTermCfg

    assert isinstance(cfg, EventTermCfg)
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._body_name: str = cfg.params["body_name"]
    self._phase1_duration: float = cfg.params["phase1_duration"]

    self._phase1_mag_range = cfg.params["phase1_mag_range"]
    self._phase2_mag_range = cfg.params["phase2_mag_range"]
    max_angle_deg = cfg.params["phase2_angle_range"][1]
    self._tan_max_angle = math.tan(math.radians(max_angle_deg))

    self._ou_theta = cfg.params.get("phase2_ou_theta", 1.0)
    self._ou_sigma = cfg.params.get("phase2_ou_sigma", 0.5)

    # Phase 2 duration (0 = infinite, no phase 3).
    self._phase2_duration: float = cfg.params.get("phase2_duration", 0.0)

    # Phase 3 (optional).
    self._phase3_mag_range = cfg.params.get("phase3_mag_range", None)
    self._has_phase3 = self._phase3_mag_range is not None and self._phase2_duration > 0

    self._asset: Entity = env.scene[self._asset_cfg.name]
    body_ids, _ = self._asset.find_bodies([self._body_name])
    self._body_ids = body_ids

    num_envs = env.num_envs
    device = env.device
    self._dt = env.step_dt

    # Phase 1: store magnitude only; direction = hand local +x rotated to world per step.
    self._phase1_mag = torch.zeros(num_envs, device=device)
    self._phase1_local_dir = torch.tensor(
      cfg.params.get("phase1_local_dir", (1.0, 0.0, 0.0)),
      dtype=torch.float32, device=device,
    )
    self._phase1_forces = torch.zeros(num_envs, 1, 3, device=device)

    # Phase 1 torque: small torque in hand local frame, rotated to world per step.
    self._phase1_local_torque = torch.tensor(
      cfg.params.get("phase1_local_torque", (0.0, 0.0, 0.0)),
      dtype=torch.float32, device=device,
    )
    self._phase1_torques = torch.zeros(num_envs, 1, 3, device=device)

    # Phase 2 OU state: independent fz, fx_ratio, fy_ratio
    self._p2_fz = torch.zeros(num_envs, device=device)
    self._p2_fx_ratio = torch.zeros(num_envs, device=device)
    self._p2_fy_ratio = torch.zeros(num_envs, device=device)
    self._p2_fz_mu = torch.zeros(num_envs, device=device)
    self._p2_fx_ratio_mu = torch.zeros(num_envs, device=device)
    self._p2_fy_ratio_mu = torch.zeros(num_envs, device=device)
    self._phase2_forces = torch.zeros(num_envs, 1, 3, device=device)

    # Phase 3: fixed per episode
    self._phase3_forces = torch.zeros(num_envs, 1, 3, device=device)

    self._sample_reset(torch.arange(num_envs, device=device))

  def _sample_reset(self, env_ids: torch.Tensor) -> None:
    n = env_ids.shape[0]
    device = env_ids.device

    # Phase 1: sample magnitude; direction is hand-local, computed each step.
    mag1_min, mag1_max = self._phase1_mag_range
    mag1 = torch.empty(n, device=device).uniform_(mag1_min, mag1_max)
    self._phase1_mag[env_ids] = mag1

    # Phase 2: biased toward large values via Beta(2,1)
    mag2_min, mag2_max = self._phase2_mag_range
    fz_beta = torch.distributions.Beta(2.0, 1.0).sample((n,)).to(device)
    fz_mu = mag2_min + (mag2_max - mag2_min) * fz_beta

    # fx ratio: biased toward -1 (pull off wall, -x direction) via Beta(1, 3)
    fx_beta = torch.distributions.Beta(1.0, 3.0).sample((n,)).to(device)
    fx_ratio_mu = fx_beta * 2.0 - 1.0
    # fy ratio: concentrated near 0 via Beta(3, 3)
    fy_beta = torch.distributions.Beta(3.0, 3.0).sample((n,)).to(device)
    fy_ratio_mu = fy_beta * 2.0 - 1.0

    self._p2_fz_mu[env_ids] = fz_mu
    self._p2_fx_ratio_mu[env_ids] = fx_ratio_mu
    self._p2_fy_ratio_mu[env_ids] = fy_ratio_mu

    self._p2_fz[env_ids] = fz_mu
    self._p2_fx_ratio[env_ids] = fx_ratio_mu
    self._p2_fy_ratio[env_ids] = fy_ratio_mu

    self._update_phase2_forces(env_ids)

    # Phase 3: +z upward force
    if self._has_phase3:
      mag3_min, mag3_max = self._phase3_mag_range
      mag3 = torch.empty(n, device=device).uniform_(mag3_min, mag3_max)
      self._phase3_forces[env_ids, 0, 0] = 0.0
      self._phase3_forces[env_ids, 0, 1] = 0.0
      self._phase3_forces[env_ids, 0, 2] = mag3

  def _ou_step(self) -> None:
    dt = self._dt
    theta = self._ou_theta
    sqrt_dt = math.sqrt(dt)
    sigma = self._ou_sigma

    mag_min, mag_max = self._phase2_mag_range

    self._p2_fz += theta * (self._p2_fz_mu - self._p2_fz) * dt \
        + sigma * (mag_max - mag_min) * sqrt_dt * torch.randn_like(self._p2_fz)
    self._p2_fz.clamp_(mag_min, mag_max)

    self._p2_fx_ratio += theta * (self._p2_fx_ratio_mu - self._p2_fx_ratio) * dt \
        + sigma * sqrt_dt * torch.randn_like(self._p2_fx_ratio)
    self._p2_fx_ratio.clamp_(-1.0, 1.0)

    self._p2_fy_ratio += theta * (self._p2_fy_ratio_mu - self._p2_fy_ratio) * dt \
        + sigma * sqrt_dt * torch.randn_like(self._p2_fy_ratio)
    self._p2_fy_ratio.clamp_(-1.0, 1.0)

  def _update_phase2_forces(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      fz = self._p2_fz
      fx_ratio = self._p2_fx_ratio
      fy_ratio = self._p2_fy_ratio
      lateral_max = fz * self._tan_max_angle
      self._phase2_forces[:, 0, 0] = fx_ratio * lateral_max
      self._phase2_forces[:, 0, 1] = fy_ratio * lateral_max
      self._phase2_forces[:, 0, 2] = -fz
    else:
      fz = self._p2_fz[env_ids]
      fx_ratio = self._p2_fx_ratio[env_ids]
      fy_ratio = self._p2_fy_ratio[env_ids]
      lateral_max = fz * self._tan_max_angle
      self._phase2_forces[env_ids, 0, 0] = fx_ratio * lateral_max
      self._phase2_forces[env_ids, 0, 1] = fy_ratio * lateral_max
      self._phase2_forces[env_ids, 0, 2] = -fz

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    self._ou_step()
    self._update_phase2_forces()

    # Phase 1 force: world frame (always toward wall in +x).
    self._phase1_forces[:, 0, 0] = self._phase1_mag
    self._phase1_forces[:, 0, 1] = 0.0
    self._phase1_forces[:, 0, 2] = 0.0
    # Phase 1 torque: still in hand local frame, rotated to world (corrects tilt).
    body_id = self._body_ids[0]
    hand_quat = self._asset.data.body_link_quat_w[:, body_id, :]
    hand_local_torque_world = _quat_rotate(hand_quat, self._phase1_local_torque)
    self._phase1_torques[:, 0, :] = hand_local_torque_world

    num_envs = env.num_envs
    device = env.device
    elapsed = env.episode_length_buf.float() * env.step_dt

    phase1_end = self._phase1_duration
    phase2_end = phase1_end + self._phase2_duration if self._has_phase3 else float('inf')

    in_phase1 = (elapsed < phase1_end).view(num_envs, 1, 1)
    in_phase2 = ((elapsed >= phase1_end) & (elapsed < phase2_end)).view(num_envs, 1, 1)
    # Phase 3: everything after phase2_end (only if has_phase3).
    in_phase3 = (elapsed >= phase2_end).view(num_envs, 1, 1) if self._has_phase3 else torch.zeros_like(in_phase1)

    forces = torch.where(in_phase1, self._phase1_forces,
             torch.where(in_phase2, self._phase2_forces,
                         self._phase3_forces))
    zeros3 = torch.zeros(num_envs, 1, 3, device=device)
    # Torque only during phase 1; phase 2/3 apply zero torque.
    torques = torch.where(in_phase1, self._phase1_torques, zeros3)

    self._last_forces = forces
    self._asset.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

  def reset(self, env_ids: torch.Tensor) -> None:
    self._sample_reset(env_ids)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not hasattr(self, "_last_forces"):
      return

    body_id = self._body_ids[0]
    body_pos = self._asset.data.body_link_pos_w[0, body_id].cpu().numpy()
    force = self._last_forces[0, 0].cpu().numpy()
    force_mag = np.linalg.norm(force)
    if force_mag < 0.1:
      return

    scale = 0.003
    visualizer.add_arrow(
      start=body_pos,
      end=body_pos + force * scale,
      color=(1.0, 0.7, 0.0, 1.0),
      width=0.008,
    )

    if hasattr(visualizer, "add_text"):
      elapsed = self._env.episode_length_buf[0].item() * self._dt
      phase1_end = self._phase1_duration
      phase2_end = phase1_end + self._phase2_duration if self._has_phase3 else float('inf')
      if elapsed < phase1_end:
        phase = "P1"
      elif elapsed < phase2_end:
        phase = "P2"
      else:
        phase = "P3"
      visualizer.add_text(
        position=body_pos + force * scale,
        text=f"{phase}: {force_mag:.0f}N",
        color=(1.0, 0.7, 0.0, 1.0),
      )
