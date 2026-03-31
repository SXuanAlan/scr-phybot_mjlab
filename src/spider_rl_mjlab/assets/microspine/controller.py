"""Microspine controller -- Custom force model via xfrc_applied.

Uses MuJoCo contact data for engagement detection (works regardless of EE
orientation), then applies PD restoring forces to resist shear/gravity.

Supports multiple end-effectors (hands and/or feet) for wall-climbing robots.
Forces are accumulated on top of any existing xfrc_applied values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

STATE_FREE = 0
STATE_ENGAGED = 1


class MicrospineController:
  """Step-mode event for microspine adhesion physics.

  Per end-effector, per environment:
  - Detect wall contact using MuJoCo contact data (geom pair matching).
  - On contact: record anchor position, transition to ENGAGED.
  - While ENGAGED: apply PD restoring forces in all 3 axes.
  - On large separation: transition back to FREE.

  Forces are added on top of existing xfrc_applied (read-add-write).

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the robot.
    microspine_cfg: MicrospineCfg instance.
    wall_cfg: SceneEntityCfg for the wall.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._mcfg = cfg.params["microspine_cfg"]
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._wall_cfg: SceneEntityCfg = cfg.params["wall_cfg"]

    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt

    # Resolve robot asset and EE body IDs.
    self._asset = env.scene[self._asset_cfg.name]
    ee_names = list(self._mcfg.ee_body_names)
    body_ids, _ = self._asset.find_bodies(ee_names)
    self._ee_body_ids = body_ids  # list of int
    self._n_ee = len(self._ee_body_ids)

    # Resolve wall geom ID for contact matching.
    mj_model = env.sim.mj_model
    self._wall_gid = mj_model.geom("wall/wall_geom").id

    # Build mapping: geom_id -> ee_index.
    # Match all geoms that belong to EE bodies (collision geoms with contype>0).
    self._geom_to_ee: dict[int, int] = {}
    entity_name = self._asset_cfg.name
    for ee_idx, body_name in enumerate(ee_names):
      # Find the global body ID.
      full_body_name = f"{entity_name}/{body_name}"
      body_id_global = mj_model.body(full_body_name).id
      # Scan all geoms belonging to this body.
      for gi in range(mj_model.ngeom):
        geom = mj_model.geom(gi)
        if geom.bodyid == body_id_global:
          # Only include collision geoms (contype > 0 or conaffinity > 0).
          if geom.contype > 0 or geom.conaffinity > 0:
            self._geom_to_ee[gi] = ee_idx

    # Per-env, per-EE state buffers.
    self._state = torch.full(
      (self._num_envs, self._n_ee), STATE_FREE, device=self._device, dtype=torch.long
    )
    self._anchor_pos = torch.zeros(
      (self._num_envs, self._n_ee, 3), device=self._device, dtype=torch.float32
    )
    self._prev_ee_pos = None

    # Warp contact data cache.
    self._warp_cached = False

  def _cache_warp_refs(self) -> None:
    """Cache warp array -> torch tensor references for contact data."""
    raw = self._env.sim.data.struct
    self._c_geom = wp.to_torch(raw.contact.geom)
    self._c_wid = wp.to_torch(raw.contact.worldid)
    self._nacon = wp.to_torch(raw.nacon)
    self._warp_cached = True

  def _get_ee_positions(self) -> torch.Tensor:
    """Get world positions of all EE bodies. Returns (num_envs, n_ee, 3)."""
    body_pos = self._asset.data.body_link_pos_w
    return body_pos[:, self._ee_body_ids, :]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Per-step callback: detect contacts and apply adhesion forces."""
    if not self._warp_cached:
      self._cache_warp_refs()

    cfg = self._mcfg
    num_envs = self._num_envs
    device = self._device
    dt = self._step_dt

    # -- 1. Read current EE positions and estimate velocity --
    ee_pos = self._get_ee_positions()  # (num_envs, n_ee, 3)

    if self._prev_ee_pos is None:
      ee_vel = torch.zeros_like(ee_pos)
    else:
      ee_vel = (ee_pos - self._prev_ee_pos) / dt
    self._prev_ee_pos = ee_pos.clone()

    # -- 2. Contact detection via MuJoCo contact data --
    contact_detected = torch.zeros(
      (num_envs, self._n_ee), dtype=torch.bool, device=device
    )

    nacon = int(self._nacon[0].item())
    if nacon > 0:
      max_contacts = min(nacon, self._c_geom.shape[0])
      wall_gid = self._wall_gid
      geom_to_ee = self._geom_to_ee

      for ci in range(max_contacts):
        wid = int(self._c_wid[ci].item())
        if wid >= num_envs:
          continue

        g0 = int(self._c_geom[ci, 0].item())
        g1 = int(self._c_geom[ci, 1].item())

        # Check if this is an EE-wall contact.
        ee_idx = None
        if g0 == wall_gid and g1 in geom_to_ee:
          ee_idx = geom_to_ee[g1]
        elif g1 == wall_gid and g0 in geom_to_ee:
          ee_idx = geom_to_ee[g0]

        if ee_idx is not None:
          contact_detected[wid, ee_idx] = True

    # -- 3. State transitions --
    old_state = self._state.clone()

    # FREE -> ENGAGED: MuJoCo reports contact with wall.
    free_mask = (old_state == STATE_FREE)
    to_engaged = free_mask & contact_detected
    self._state[to_engaged] = STATE_ENGAGED
    self._anchor_pos[to_engaged] = ee_pos[to_engaged].clone()

    # ENGAGED -> FREE: large separation from anchor.
    engaged_mask = (self._state == STATE_ENGAGED)
    if engaged_mask.any():
      separation = (ee_pos - self._anchor_pos).norm(dim=-1)
      to_free = engaged_mask & (separation > cfg.disengage_distance)
      self._state[to_free] = STATE_FREE

    # -- 4. Compute forces per EE --
    total_force = torch.zeros(
      (num_envs, self._n_ee, 3), device=device, dtype=torch.float32
    )

    engaged_mask = (self._state == STATE_ENGAGED)
    if engaged_mask.any():
      disp = ee_pos - self._anchor_pos  # (num_envs, n_ee, 3)

      # --- X-direction (normal to wall): keep EE pressed ---
      f_x = -cfg.k_normal * disp[:, :, 0] - cfg.c_normal * ee_vel[:, :, 0]
      f_x = f_x.clamp(-cfg.f_normal_max, cfg.f_normal_max)

      # --- Z-direction (shear / gravity): resist downward sliding ---
      dz = disp[:, :, 2]
      vz = ee_vel[:, :, 2]

      # Anisotropic: full stiffness downward, reduced upward.
      k_z = torch.where(
        dz < 0,
        torch.tensor(cfg.k_tangential, device=device),
        torch.tensor(cfg.k_tangential * cfg.aniso_ratio, device=device),
      )
      c_z = torch.where(
        dz < 0,
        torch.tensor(cfg.c_tangential, device=device),
        torch.tensor(cfg.c_tangential * cfg.aniso_ratio, device=device),
      )
      f_z = -k_z * dz - c_z * vz
      f_z = f_z.clamp(-cfg.f_tangential_max, cfg.f_tangential_max)

      # --- Y-direction (lateral) ---
      f_y = -cfg.k_tangential * disp[:, :, 1] - cfg.c_tangential * ee_vel[:, :, 1]
      f_y = f_y.clamp(-cfg.f_lateral_max, cfg.f_lateral_max)

      total_force[engaged_mask, 0] = f_x[engaged_mask]
      total_force[engaged_mask, 1] = f_y[engaged_mask]
      total_force[engaged_mask, 2] = f_z[engaged_mask]

    # -- Debug: env 0 status --
    env0_states = self._state[0].tolist()
    ee_names = list(self._mcfg.ee_body_names)
    parts = []
    for ei in range(self._n_ee):
      s = "ENGAGED" if env0_states[ei] == STATE_ENGAGED else "FREE"
      f = total_force[0, ei].tolist()
      parts.append(f"{ee_names[ei]}:{s} F=({f[0]:.1f},{f[1]:.1f},{f[2]:.1f})")
    print(f"[ms] {' | '.join(parts)}")

    # -- 5. Apply forces: read current xfrc, add ours, write back --
    current_wrench = self._asset.data.body_external_wrench

    for i, body_id in enumerate(self._ee_body_ids):
      cur_f = current_wrench[:, body_id, 0:3].clone()
      cur_t = current_wrench[:, body_id, 3:6].clone()
      combined_f = cur_f + total_force[:, i, :]

      self._asset.write_external_wrench_to_sim(
        combined_f.unsqueeze(1),
        cur_t.unsqueeze(1),
        body_ids=[body_id],
      )

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset internal state on episode reset."""
    self._state[env_ids] = STATE_FREE
    self._anchor_pos[env_ids] = 0.0
    if self._prev_ee_pos is not None:
      ee_pos = self._get_ee_positions()
      self._prev_ee_pos[env_ids] = ee_pos[env_ids]
