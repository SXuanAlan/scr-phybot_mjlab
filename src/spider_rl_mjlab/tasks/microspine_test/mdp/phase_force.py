"""Two-phase test force event for the microspine adhesion test.

Phase 1 [0, duration):     push force (e.g. 20-100N +x) to engage hand against wall.
Phase 2 [duration, inf):   shear force (e.g. 50-300N, 0-20° from -z) to test adhesion.

Supports domain randomization:
  - Phase 1: magnitude sampled per env on reset.
  - Phase 2: magnitude and direction evolve slowly via Ornstein-Uhlenbeck process.
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

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the entity containing the target body.
    body_name: Name of the body to apply forces to.
    phase1_duration: Duration of phase 1 [s].
    phase1_mag_range: (min, max) magnitude of phase 1 +x push [N].
    phase2_mag_range: (min, max) magnitude of phase 2 shear [N].
    phase2_angle_range: (min, max) angle from -z axis [degrees].
    phase2_ou_theta: OU mean-reversion rate [1/s]. Higher = faster reversion. Default 1.0.
    phase2_ou_sigma_mag: OU noise std for magnitude [N/sqrt(s)]. Default 30.0.
    phase2_ou_sigma_angle: OU noise std for polar angle [rad/sqrt(s)]. Default 0.05.
    phase2_ou_sigma_azimuth: OU noise std for azimuth [rad/sqrt(s)]. Default 0.3.
  """

  def __init__(self, cfg, env):
    from mjlab.managers.event_manager import EventTermCfg

    assert isinstance(cfg, EventTermCfg)
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._body_name: str = cfg.params["body_name"]
    self._phase1_duration: float = cfg.params["phase1_duration"]

    # Randomization ranges
    self._phase1_mag_range = cfg.params["phase1_mag_range"]
    self._phase2_mag_range = cfg.params["phase2_mag_range"]
    self._phase2_angle_min = math.radians(cfg.params["phase2_angle_range"][0])
    self._phase2_angle_max = math.radians(cfg.params["phase2_angle_range"][1])

    # OU process parameters
    self._ou_theta = cfg.params.get("phase2_ou_theta", 1.0)
    self._ou_sigma_mag = cfg.params.get("phase2_ou_sigma_mag", 30.0)
    self._ou_sigma_angle = cfg.params.get("phase2_ou_sigma_angle", 0.05)
    self._ou_sigma_azimuth = cfg.params.get("phase2_ou_sigma_azimuth", 0.3)

    self._asset: Entity = env.scene[self._asset_cfg.name]
    body_ids, _ = self._asset.find_bodies([self._body_name])
    self._body_ids = body_ids

    num_envs = env.num_envs
    device = env.device
    self._dt = env.step_dt

    # Phase 1: fixed per episode
    self._phase1_forces = torch.zeros(num_envs, 1, 3, device=device)

    # Phase 2 OU state: magnitude, polar angle (from -z), azimuth
    self._p2_mag = torch.zeros(num_envs, device=device)
    self._p2_theta = torch.zeros(num_envs, device=device)
    self._p2_azimuth = torch.zeros(num_envs, device=device)
    # Per-env mean-reversion targets (sampled on reset)
    self._p2_mag_mu = torch.zeros(num_envs, device=device)
    self._p2_theta_mu = torch.zeros(num_envs, device=device)
    self._p2_azimuth_mu = torch.zeros(num_envs, device=device)

    # Output buffer
    self._phase2_forces = torch.zeros(num_envs, 1, 3, device=device)

    # Initialize all envs
    self._sample_reset(torch.arange(num_envs, device=device))

  def _sample_reset(self, env_ids: torch.Tensor) -> None:
    """Sample initial OU state and phase 1 forces for given env indices."""
    n = env_ids.shape[0]
    device = env_ids.device

    # Phase 1: random magnitude in +x
    mag1_min, mag1_max = self._phase1_mag_range
    mag1 = torch.empty(n, device=device).uniform_(mag1_min, mag1_max)
    self._phase1_forces[env_ids, 0, 0] = mag1
    self._phase1_forces[env_ids, 0, 1] = 0.0
    self._phase1_forces[env_ids, 0, 2] = 0.0

    # Phase 2: sample OU targets and init state
    mag2_min, mag2_max = self._phase2_mag_range
    mag_mu = torch.empty(n, device=device).uniform_(mag2_min, mag2_max)
    theta_mu = torch.empty(n, device=device).uniform_(self._phase2_angle_min, self._phase2_angle_max)
    azimuth_mu = torch.empty(n, device=device).uniform_(0, 2 * math.pi)

    self._p2_mag_mu[env_ids] = mag_mu
    self._p2_theta_mu[env_ids] = theta_mu
    self._p2_azimuth_mu[env_ids] = azimuth_mu

    # Start at the mean
    self._p2_mag[env_ids] = mag_mu
    self._p2_theta[env_ids] = theta_mu
    self._p2_azimuth[env_ids] = azimuth_mu

    self._update_phase2_forces(env_ids)

  def _ou_step(self) -> None:
    """Advance OU process one step for all envs."""
    dt = self._dt
    theta = self._ou_theta
    sqrt_dt = math.sqrt(dt)

    # mag
    self._p2_mag += theta * (self._p2_mag_mu - self._p2_mag) * dt \
        + self._ou_sigma_mag * sqrt_dt * torch.randn_like(self._p2_mag)
    self._p2_mag.clamp_(self._phase2_mag_range[0], self._phase2_mag_range[1])

    # polar angle
    self._p2_theta += theta * (self._p2_theta_mu - self._p2_theta) * dt \
        + self._ou_sigma_angle * sqrt_dt * torch.randn_like(self._p2_theta)
    self._p2_theta.clamp_(self._phase2_angle_min, self._phase2_angle_max)

    # azimuth (free to wrap)
    self._p2_azimuth += theta * (self._p2_azimuth_mu - self._p2_azimuth) * dt \
        + self._ou_sigma_azimuth * sqrt_dt * torch.randn_like(self._p2_azimuth)

  def _update_phase2_forces(self, env_ids: torch.Tensor | None = None) -> None:
    """Convert OU state (mag, theta, azimuth) to Cartesian force."""
    if env_ids is None:
      mag = self._p2_mag
      theta = self._p2_theta
      azimuth = self._p2_azimuth
      self._phase2_forces[:, 0, 0] = mag * torch.sin(theta) * torch.cos(azimuth)
      self._phase2_forces[:, 0, 1] = mag * torch.sin(theta) * torch.sin(azimuth)
      self._phase2_forces[:, 0, 2] = -mag * torch.cos(theta)
    else:
      mag = self._p2_mag[env_ids]
      theta = self._p2_theta[env_ids]
      azimuth = self._p2_azimuth[env_ids]
      self._phase2_forces[env_ids, 0, 0] = mag * torch.sin(theta) * torch.cos(azimuth)
      self._phase2_forces[env_ids, 0, 1] = mag * torch.sin(theta) * torch.sin(azimuth)
      self._phase2_forces[env_ids, 0, 2] = -mag * torch.cos(theta)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Apply phase-dependent force every step."""
    num_envs = env.num_envs
    device = env.device

    # Advance OU and recompute phase 2 forces
    self._ou_step()
    self._update_phase2_forces()

    elapsed = env.episode_length_buf.float() * env.step_dt  # (num_envs,)
    in_phase1 = (elapsed < self._phase1_duration).view(num_envs, 1, 1)

    forces = torch.where(in_phase1, self._phase1_forces, self._phase2_forces)
    torques = torch.zeros(num_envs, 1, 3, device=device)

    # Cache current applied force for env 0 visualization
    self._last_forces = forces

    self._asset.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

  def reset(self, env_ids: torch.Tensor) -> None:
    """Re-sample OU targets and state for reset environments."""
    self._sample_reset(env_ids)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw applied phase force arrow on env 0."""
    if not hasattr(self, "_last_forces"):
      return

    body_id = self._body_ids[0]
    body_pos = self._asset.data.body_link_pos_w[0, body_id].cpu().numpy()
    force = self._last_forces[0, 0].cpu().numpy()
    force_mag = np.linalg.norm(force)
    if force_mag < 0.1:
      return

    scale = 0.003  # m/N, same as ContactForceVis
    visualizer.add_arrow(
      start=body_pos,
      end=body_pos + force * scale,
      color=(1.0, 0.7, 0.0, 1.0),  # orange for applied test force
      width=0.008,
    )

    # Text label with magnitude and phase info (skip if viewer doesn't support it).
    if hasattr(visualizer, "add_text"):
      elapsed = self._env.episode_length_buf[0].item() * self._dt
      phase = "P1" if elapsed < self._phase1_duration else "P2"
      visualizer.add_text(
        position=body_pos + force * scale,
        text=f"{phase}: {force_mag:.0f}N",
        color=(1.0, 0.7, 0.0, 1.0),
      )
