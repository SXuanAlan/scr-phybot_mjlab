"""Microspine controller -- Path A: Contact-triggered equality constraint.

Implements a state machine that dynamically activates/deactivates MuJoCo
equality constraints (connect type) based on contact state. When hand-wall
contact is detected with sufficient normal force, a connect constraint is
activated to anchor the hand to the wall at the contact point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg


# State constants
STATE_FREE = 0
STATE_ENGAGED = 1


class MicrospineController:
  """Step-mode event for microspine adhesion via equality constraints.

  Uses MuJoCo connect constraints to lock the hand to the wall when
  contact is detected. The constraint solver handles force computation
  implicitly, supporting large shear loads (300N).

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the robot.
    microspine_cfg: MicrospineCfg instance.
    wall_cfg: SceneEntityCfg for the wall.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._microspine_cfg = cfg.params["microspine_cfg"]
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._wall_cfg: SceneEntityCfg = cfg.params["wall_cfg"]
    self._num_envs = env.num_envs
    self._device = env.device

    mcfg = self._microspine_cfg
    mj_model = env.sim.mj_model

    # Resolve wall geom ID for contact matching.
    self._wall_gid = mj_model.geom(mcfg.wall_geom_name).id

    # Resolve hand sub-geom IDs.
    self._hand_gids: list[int] = []
    self._hand_gid_set: set[int] = set()
    for i in range(mj_model.ngeom):
      name = mj_model.geom(i).name
      if mcfg.hand_sub_geom_prefix in name:
        self._hand_gids.append(i)
        self._hand_gid_set.add(i)
    self._n_hand_geoms = len(self._hand_gids)

    # Resolve body IDs for anchor computation.
    asset = env.scene[self._asset_cfg.name]
    hand_body_ids, _ = asset.find_bodies(list(mcfg.ee_body_names))
    self._hand_body_id = hand_body_ids[0]  # local body index in entity

    # Wall body ID (in global MuJoCo model).
    self._wall_body_id_global = mj_model.body(mcfg.wall_body_name).id

    # Hand body ID in global MuJoCo model.
    entity_name = self._asset_cfg.name
    self._hand_body_id_global = mj_model.body(f"{entity_name}/right_hand").id

    # Equality constraint indices: one per hand sub-geom.
    # These are created by the spec_fn in env_cfgs.py.
    self._eq_ids: list[int] = []
    for i in range(mj_model.neq):
      eq_type = int(mj_model.eq_type[i])
      if eq_type == 0:  # mjEQ_CONNECT
        self._eq_ids.append(i)
    self._n_constraints = len(self._eq_ids)

    # Map from hand geom index (in self._hand_gids) to constraint index.
    # We have one constraint per hand sub-geom.
    self._geom_to_eq = {}
    for idx, gid in enumerate(self._hand_gids):
      if idx < self._n_constraints:
        self._geom_to_eq[gid] = self._eq_ids[idx]

    # Cache warp data references for contact reading.
    self._warp_cached = False

    # Per-env state: (num_envs,) tensor of state constants.
    self._state = torch.full(
      (self._num_envs,), STATE_FREE, dtype=torch.int32, device=self._device
    )

    # Per-env, per-constraint engaged flag.
    self._engaged = torch.zeros(
      (self._num_envs, self._n_constraints), dtype=torch.bool, device=self._device
    )

    # Per-env engaged anchor positions (world frame, for displacement check).
    self._anchor_pos_world = torch.zeros(
      (self._num_envs, self._n_constraints, 3), dtype=torch.float32, device=self._device
    )

    # Expand eq_data for per-env anchor points if needed.
    if self._n_constraints > 0:
      try:
        env.sim.expand_model_fields(("eq_data",))
      except Exception:
        pass  # May already be expanded or not needed

    self._asset = asset

  def _cache_warp_refs(self) -> None:
    """Cache warp array -> torch tensor references for contact data."""
    raw = self._env.sim.data.struct
    self._c_geom = wp.to_torch(raw.contact.geom)
    self._c_wid = wp.to_torch(raw.contact.worldid)
    self._c_pos = wp.to_torch(raw.contact.pos)
    self._c_frame = wp.to_torch(raw.contact.frame)
    self._c_dim = wp.to_torch(raw.contact.dim)
    self._c_efc = wp.to_torch(raw.contact.efc_address)
    self._efc_force = wp.to_torch(raw.efc.force)
    self._nacon = wp.to_torch(raw.nacon)
    self._warp_cached = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Per-step callback: detect contacts and toggle constraints."""
    if self._n_constraints == 0:
      return

    if not self._warp_cached:
      self._cache_warp_refs()

    nacon = int(self._nacon[0].item())

    # Track which constraints have active contacts this step.
    # Shape: (num_envs, n_constraints) -- True if contact found.
    contact_detected = torch.zeros(
      (self._num_envs, self._n_constraints), dtype=torch.bool, device=self._device
    )

    # Track contact positions and normal forces per constraint per env.
    contact_positions = torch.zeros(
      (self._num_envs, self._n_constraints, 3), dtype=torch.float32, device=self._device
    )
    contact_normal_force = torch.zeros(
      (self._num_envs, self._n_constraints), dtype=torch.float32, device=self._device
    )

    # Scan active contacts.
    if nacon > 0:
      max_contacts = min(nacon, self._c_geom.shape[0])

      for ci in range(max_contacts):
        wid = int(self._c_wid[ci].item())
        if wid >= self._num_envs:
          continue

        g0 = int(self._c_geom[ci, 0].item())
        g1 = int(self._c_geom[ci, 1].item())

        # Check if this is a hand-wall contact.
        hand_gid = None
        if g0 == self._wall_gid and g1 in self._hand_gid_set:
          hand_gid = g1
          sign = 1.0
        elif g1 == self._wall_gid and g0 in self._hand_gid_set:
          hand_gid = g0
          sign = -1.0
        else:
          continue

        if hand_gid not in self._geom_to_eq:
          continue

        eq_id = self._geom_to_eq[hand_gid]
        eq_local_idx = self._eq_ids.index(eq_id)

        # Extract contact force.
        frame = self._c_frame[ci].reshape(3, 3)
        condim = int(self._c_dim[ci].item())

        f_contact = torch.zeros(3, device=self._device)
        for k in range(min(condim, 3)):
          addr = int(self._c_efc[ci, k].item())
          if 0 <= addr < self._efc_force.shape[1]:
            f_contact[k] = self._efc_force[wid, addr]

        # Normal force is f_contact[0] (in contact frame).
        normal_force = float(f_contact[0].item())

        # Contact position in world frame.
        pos = self._c_pos[ci].clone()

        contact_detected[wid, eq_local_idx] = True
        contact_positions[wid, eq_local_idx] = pos
        contact_normal_force[wid, eq_local_idx] = normal_force

    # State machine logic (vectorized across envs).
    mcfg = self._microspine_cfg

    # --- FREE -> ENGAGED transition ---
    # Engage when contact is detected with sufficient normal force.
    should_engage = (
      contact_detected
      & (contact_normal_force > mcfg.engage_threshold)
      & ~self._engaged
    )

    if should_engage.any():
      self._engage_constraints(should_engage, contact_positions)

    # --- ENGAGED -> FREE transition (displacement check) ---
    if self._engaged.any():
      self._check_disengage()

    # Update overall state per env.
    any_engaged = self._engaged.any(dim=1)
    self._state[any_engaged] = STATE_ENGAGED
    self._state[~any_engaged] = STATE_FREE

  def _engage_constraints(
    self,
    should_engage: torch.Tensor,
    contact_positions: torch.Tensor,
  ) -> None:
    """Activate constraints for newly engaged contact points.

    Args:
      should_engage: (num_envs, n_constraints) bool mask of constraints to engage.
      contact_positions: (num_envs, n_constraints, 3) world positions of contacts.
    """
    mcfg = self._microspine_cfg

    # Get body positions and rotation matrices for coordinate transforms.
    # Using the GPU data from mujoco_warp.
    raw = self._env.sim.data.struct
    xpos = wp.to_torch(raw.xpos)   # (nworld, nbody, 3)
    xmat = wp.to_torch(raw.xmat)   # (nworld, nbody, 3, 3)

    hand_bid = self._hand_body_id_global
    wall_bid = self._wall_body_id_global

    for env_i in range(self._num_envs):
      for c_idx in range(self._n_constraints):
        if not should_engage[env_i, c_idx]:
          continue

        eq_id = self._eq_ids[c_idx]
        world_pos = contact_positions[env_i, c_idx]  # (3,)

        # Convert to hand body local frame.
        hand_pos = xpos[env_i, hand_bid]    # (3,)
        hand_rot = xmat[env_i, hand_bid]    # (3, 3)
        local_hand = hand_rot.T @ (world_pos - hand_pos)

        # Convert to wall body local frame.
        wall_pos = xpos[env_i, wall_bid]    # (3,)
        wall_rot = xmat[env_i, wall_bid]    # (3, 3)
        local_wall = wall_rot.T @ (world_pos - wall_pos)

        # Write anchor points to eq_data.
        # eq_data layout for connect: [body1_anchor(3), body2_anchor(3), ...]
        eq_data = self._env.sim.model.eq_data
        eq_data[env_i, eq_id, 0] = local_hand[0]
        eq_data[env_i, eq_id, 1] = local_hand[1]
        eq_data[env_i, eq_id, 2] = local_hand[2]
        eq_data[env_i, eq_id, 3] = local_wall[0]
        eq_data[env_i, eq_id, 4] = local_wall[1]
        eq_data[env_i, eq_id, 5] = local_wall[2]

        # Activate constraint.
        self._env.sim.data.eq_active[env_i, eq_id] = True

        # Record anchor world position for displacement check.
        self._anchor_pos_world[env_i, c_idx] = world_pos

        self._engaged[env_i, c_idx] = True

  def _check_disengage(self) -> None:
    """Check if engaged constraints should be released due to displacement."""
    mcfg = self._microspine_cfg

    # Get current hand body position.
    raw = self._env.sim.data.struct
    xpos = wp.to_torch(raw.xpos)
    hand_bid = self._hand_body_id_global

    for env_i in range(self._num_envs):
      for c_idx in range(self._n_constraints):
        if not self._engaged[env_i, c_idx]:
          continue

        # Check normal displacement (x direction, since wall faces -x).
        current_hand_pos = xpos[env_i, hand_bid]
        anchor_pos = self._anchor_pos_world[env_i, c_idx]

        # Normal direction is +x (away from wall).
        normal_disp = (current_hand_pos[0] - anchor_pos[0]).item()

        if normal_disp > mcfg.disengage_normal_disp:
          eq_id = self._eq_ids[c_idx]
          self._env.sim.data.eq_active[env_i, eq_id] = False
          self._engaged[env_i, c_idx] = False

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset internal state on episode reset."""
    if self._n_constraints == 0:
      return

    for env_i in env_ids.tolist():
      self._state[env_i] = STATE_FREE
      self._engaged[env_i] = False
      self._anchor_pos_world[env_i] = 0.0

      # Deactivate all constraints for this env.
      for eq_id in self._eq_ids:
        self._env.sim.data.eq_active[env_i, eq_id] = False
