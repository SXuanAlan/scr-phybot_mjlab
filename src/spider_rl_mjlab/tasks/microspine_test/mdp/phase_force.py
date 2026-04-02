"""Two-phase test force event for the microspine adhesion test.

Phase 1 [0, duration):     push force (e.g. 20-100N +x) to engage hand against wall.
Phase 2 [duration, inf):   shear/pull force with randomized direction in xyz.

Phase 2 force is composed of:
  - A constant -z (downward) shear component (gravity-like pull).
  - Randomized ±x component (push into / pull off wall).
  - Randomized ±y component (lateral shear).
All three components evolve via independent OU processes, biased toward
large magnitudes so the adhesion is thoroughly tested.
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


class PhaseForce:
  """Step-mode event that applies a two-phase external force to a body.

  Phase 2 uses independent OU processes for each axis force component:
    fz: always negative (downward shear), magnitude from phase2_mag_range.
    fx: ±x (push/pull relative to wall), magnitude up to fz * tan(max_angle).
    fy: ±y (lateral shear), magnitude up to fz * tan(max_angle).

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the entity containing the target body.
    body_name: Name of the body to apply forces to.
    phase1_duration: Duration of phase 1 [s].
    phase1_mag_range: (min, max) magnitude of phase 1 +x push [N].
    phase2_mag_range: (min, max) magnitude of phase 2 -z shear [N].
    phase2_angle_range: (min, max) deviation angle from -z [degrees].
        Controls max ±x and ±y force relative to z: fx,fy_max = fz * tan(angle).
    phase2_ou_theta: OU mean-reversion rate [1/s]. Default 1.0.
    phase2_ou_sigma: OU noise std for all components [fraction/sqrt(s)]. Default 0.5.
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

    self._asset: Entity = env.scene[self._asset_cfg.name]
    body_ids, _ = self._asset.find_bodies([self._body_name])
    self._body_ids = body_ids

    num_envs = env.num_envs
    device = env.device
    self._dt = env.step_dt

    # Phase 1: fixed per episode
    self._phase1_forces = torch.zeros(num_envs, 1, 3, device=device)

    # Phase 2 OU state: independent fz, fx_ratio, fy_ratio
    # fz: magnitude of -z force (always negative z)
    # fx_ratio, fy_ratio: in [-1, 1], multiplied by fz * tan(max_angle) for fx, fy
    self._p2_fz = torch.zeros(num_envs, device=device)
    self._p2_fx_ratio = torch.zeros(num_envs, device=device)
    self._p2_fy_ratio = torch.zeros(num_envs, device=device)
    # OU targets
    self._p2_fz_mu = torch.zeros(num_envs, device=device)
    self._p2_fx_ratio_mu = torch.zeros(num_envs, device=device)
    self._p2_fy_ratio_mu = torch.zeros(num_envs, device=device)

    self._phase2_forces = torch.zeros(num_envs, 1, 3, device=device)

    self._sample_reset(torch.arange(num_envs, device=device))

  def _sample_reset(self, env_ids: torch.Tensor) -> None:
    n = env_ids.shape[0]
    device = env_ids.device

    # Phase 1: +x push
    mag1_min, mag1_max = self._phase1_mag_range
    mag1 = torch.empty(n, device=device).uniform_(mag1_min, mag1_max)
    self._phase1_forces[env_ids, 0, 0] = mag1
    self._phase1_forces[env_ids, 0, 1] = 0.0
    self._phase1_forces[env_ids, 0, 2] = 0.0

    # Phase 2: biased toward large values via Beta(2,1)
    mag2_min, mag2_max = self._phase2_mag_range
    fz_beta = torch.distributions.Beta(2.0, 1.0).sample((n,)).to(device)
    fz_mu = mag2_min + (mag2_max - mag2_min) * fz_beta

    # fx ratio: biased toward -1 (pull off wall, -x direction) via Beta(1, 3)
    # Beta(1,3) has mean=0.25, mapped to [-1,1] gives mean=-0.5 (biased negative)
    fx_beta = torch.distributions.Beta(1.0, 3.0).sample((n,)).to(device)
    fx_ratio_mu = fx_beta * 2.0 - 1.0  # map [0,1] to [-1,1], mean ≈ -0.5
    # fy ratio: concentrated near 0 via Beta(3, 3) (bell-shaped around 0.5)
    fy_beta = torch.distributions.Beta(3.0, 3.0).sample((n,)).to(device)
    fy_ratio_mu = fy_beta * 2.0 - 1.0  # map [0,1] to [-1,1], mean = 0, concentrated

    self._p2_fz_mu[env_ids] = fz_mu
    self._p2_fx_ratio_mu[env_ids] = fx_ratio_mu
    self._p2_fy_ratio_mu[env_ids] = fy_ratio_mu

    self._p2_fz[env_ids] = fz_mu
    self._p2_fx_ratio[env_ids] = fx_ratio_mu
    self._p2_fy_ratio[env_ids] = fy_ratio_mu

    self._update_phase2_forces(env_ids)

  def _ou_step(self) -> None:
    dt = self._dt
    theta = self._ou_theta
    sqrt_dt = math.sqrt(dt)
    sigma = self._ou_sigma

    mag_min, mag_max = self._phase2_mag_range

    # fz magnitude
    self._p2_fz += theta * (self._p2_fz_mu - self._p2_fz) * dt \
        + sigma * (mag_max - mag_min) * sqrt_dt * torch.randn_like(self._p2_fz)
    self._p2_fz.clamp_(mag_min, mag_max)

    # fx/fy ratios
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

    num_envs = env.num_envs
    device = env.device
    elapsed = env.episode_length_buf.float() * env.step_dt
    in_phase1 = (elapsed < self._phase1_duration).view(num_envs, 1, 1)

    forces = torch.where(in_phase1, self._phase1_forces, self._phase2_forces)
    torques = torch.zeros(num_envs, 1, 3, device=device)

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
      phase = "P1" if elapsed < self._phase1_duration else "P2"
      visualizer.add_text(
        position=body_pos + force * scale,
        text=f"{phase}: {force_mag:.0f}N",
        color=(1.0, 0.7, 0.0, 1.0),
      )
