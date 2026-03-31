"""Microspine controller -- Path B: Custom force model via xfrc_applied.

Implements a spring-damper-based adhesion model that simulates microspine
engagement. Forces are computed analytically each timestep and applied via
xfrc_applied, accumulating on top of the PhaseForce test harness forces.

Contact model:
  - Detect proximity between hand contact surface and wall face.
  - Once in contact, engage immediately: record anchor position.
  - When engaged, PD controller targets anchor position in all 3 axes.
  - Anisotropic: z-direction uses higher stiffness/damping (grasp direction).
  - Force magnitudes clamped to prevent oscillation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

# State machine constants
STATE_FREE = 0
STATE_ENGAGED = 1


class MicrospineController:
  """Step-mode event for microspine adhesion physics."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._mcfg = cfg.params["microspine_cfg"]
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._wall_cfg: SceneEntityCfg = cfg.params["wall_cfg"]

    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt

    # Resolve robot asset and body IDs
    self._asset = env.scene[self._asset_cfg.name]
    body_ids, _ = self._asset.find_bodies(list(self._mcfg.ee_body_names))
    self._hand_body_id = body_ids[0]

    # Resolve wall entity
    self._wall_entity = env.scene[self._wall_cfg.name]

    # State buffers (per-env)
    self._state = torch.full(
      (self._num_envs,), STATE_FREE, device=self._device, dtype=torch.long
    )
    self._anchor_pos = torch.zeros(
      (self._num_envs, 3), device=self._device, dtype=torch.float32
    )
    self._prev_hand_pos = None

    # Wall geometry
    self._wall_half_x = 0.5
    # Sub-geom contact surface: center at local x=0.08, box half-size x=0.002
    self._contact_offset_x = self._mcfg.hand_contact_offset_x + 0.002

    self._initialized = False

  def _lazy_init(self):
    """Resolve wall geometry on first call."""
    if self._initialized:
      return
    self._initialized = True
    wall_pos = self._wall_entity.data.root_link_pos_w
    self._wall_face_x = wall_pos[:, 0] - self._wall_half_x
    hand_pos = self._asset.data.body_link_pos_w[:, self._hand_body_id, :]
    self._prev_hand_pos = hand_pos.clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Per-step callback: compute and apply adhesion forces."""
    self._lazy_init()

    cfg = self._mcfg
    num_envs = self._num_envs
    dt = self._step_dt
    device = self._device

    # -- 1. Read current hand state --
    hand_pos = self._asset.data.body_link_pos_w[:, self._hand_body_id, :].clone()

    if self._prev_hand_pos is None:
      hand_vel = torch.zeros_like(hand_pos)
    else:
      hand_vel = (hand_pos - self._prev_hand_pos) / dt
    self._prev_hand_pos = hand_pos.clone()

    # -- 2. Contact detection --
    contact_x = hand_pos[:, 0] + self._contact_offset_x
    wall_x = self._wall_face_x
    penetration = contact_x - wall_x  # positive = penetrating

    # -- 3. State transitions --
    old_state = self._state.clone()

    # FREE -> ENGAGED: contact surface within 1mm of wall
    free_mask = (old_state == STATE_FREE)
    to_engaged = free_mask & (penetration > -0.001)
    self._state[to_engaged] = STATE_ENGAGED
    self._anchor_pos[to_engaged] = hand_pos[to_engaged].clone()

    # ENGAGED -> FREE: extreme separation (hand moved way too far from wall)
    engaged_mask = (self._state == STATE_ENGAGED)
    separation_x = self._anchor_pos[:, 0] - hand_pos[:, 0]
    to_free = engaged_mask & (separation_x > 0.2)  # 20cm separation = disengage
    self._state[to_free] = STATE_FREE

    # -- 4. Compute forces --
    force = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)

    is_engaged = (self._state == STATE_ENGAGED)
    if is_engaged.any():
      # Displacement from anchor: positive = moved away from anchor
      disp = hand_pos - self._anchor_pos  # (num_envs, 3)

      # --- X-direction (normal to wall): keep hand pressed against wall ---
      # Spring: pulls hand back toward anchor x-position
      # Use moderate stiffness to avoid oscillation
      f_x = -cfg.k_normal * disp[:, 0] - cfg.c_normal * hand_vel[:, 0]
      # Clamp to prevent extreme forces
      f_x = f_x.clamp(-500.0, 500.0)
      force[is_engaged, 0] += f_x[is_engaged]

      # --- Z-direction (along shear): resist 300N downward force ---
      # This is the critical force. Must counteract 300N -z from PhaseForce.
      # Use anisotropic stiffness: higher when resisting downward motion
      dz = disp[:, 2]
      vz = hand_vel[:, 2]

      # Anisotropic: higher effective stiffness for downward displacement
      # dz < 0 means hand moved downward -> high resistance (microspine hooks digging in)
      k_z = torch.where(
        dz < 0,
        torch.tensor(cfg.k_tangential, device=device),
        torch.tensor(cfg.k_tangential * 0.3, device=device),  # weaker upward
      )
      c_z = torch.where(
        dz < 0,
        torch.tensor(cfg.c_tangential, device=device),
        torch.tensor(cfg.c_tangential * 0.3, device=device),
      )

      f_z = -k_z * dz - c_z * vz
      # Clamp to physical maximum (microspine holding capacity)
      f_z = f_z.clamp(-500.0, 500.0)
      force[is_engaged, 2] += f_z[is_engaged]

      # --- Y-direction (lateral): moderate resistance ---
      dy = disp[:, 1]
      vy = hand_vel[:, 1]
      f_y = -cfg.k_tangential * dy - cfg.c_tangential * vy
      f_y = f_y.clamp(-200.0, 200.0)
      force[is_engaged, 1] += f_y[is_engaged]

    # -- 5. Accumulate forces on top of PhaseForce --
    current_wrench = self._asset.data.body_external_wrench
    current_force = current_wrench[:, self._hand_body_id, 0:3].clone()
    current_torque = current_wrench[:, self._hand_body_id, 3:6].clone()

    combined_force = current_force + force

    self._asset.write_external_wrench_to_sim(
      combined_force.unsqueeze(1),
      current_torque.unsqueeze(1),
      body_ids=[self._hand_body_id],
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset internal state on episode reset."""
    self._state[env_ids] = STATE_FREE
    self._anchor_pos[env_ids] = 0.0
    if self._prev_hand_pos is not None:
      hand_pos = self._asset.data.body_link_pos_w[:, self._hand_body_id, :]
      self._prev_hand_pos[env_ids] = hand_pos[env_ids]
