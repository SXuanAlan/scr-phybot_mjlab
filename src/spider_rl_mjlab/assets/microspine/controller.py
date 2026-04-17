"""Microspine controller — per-sphere adhesion via MuJoCo adhesion actuators.

Each hand has 8 contact spheres (2x4 grid), each in its own sub-body.
MuJoCo adhesion actuators are added to each sub-body via spec_fn.

Engage logic (when engage_force_threshold > 0):
  - Per-hand latch state, initially OFF.
  - Each step, compute per-sphere normal force against wall.
  - When ALL spheres on a hand have normal force >= threshold,
    that hand latches to ENGAGED.
  - Once engaged, adhesion stays on until:
    (a) episode reset, OR
    (b) friction force along the hand's up direction exceeds
        disengage_friction_threshold (if > 0).

All operations are batched across envs — GPU friendly (no CPU-GPU sync
in the hot path).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets.microspine.config import MicrospineCfg

_GEOM_ROWS = 2
_GEOM_COLS = 4
_GEOMS_PER_HAND = _GEOM_ROWS * _GEOM_COLS


def _sphere_body_names(side: str) -> list[str]:
  return [
    f"{side}_hand_c_{i}_{j}"
    for i in range(_GEOM_ROWS)
    for j in range(_GEOM_COLS)
  ]


def _sphere_geom_names(side: str) -> list[str]:
  return [
    f"{side}_hand_c_{i}_{j}"
    for i in range(_GEOM_ROWS)
    for j in range(_GEOM_COLS)
  ]


def add_adhesion_actuators(
  spec: mujoco.MjSpec,
  ee_body_names: tuple[str, ...],
  adhesion_force: float,
  robot_prefix: str = "robot",
) -> None:
  """Add MuJoCo adhesion actuators for each contact sphere sub-body.

  Args:
    spec: MuJoCo spec to modify.
    ee_body_names: End-effector body names (e.g. ("left_hand", "right_hand")).
    adhesion_force: Total adhesion force [N] distributed across all spheres.
    robot_prefix: Robot entity name prefix in the spec.
  """
  num_spheres = len(ee_body_names) * _GEOMS_PER_HAND
  force_per_sphere = adhesion_force / num_spheres

  for bname in ee_body_names:
    side = bname.split("_")[0]
    for sname in _sphere_body_names(side):
      act = spec.add_actuator()
      act.name = f"{robot_prefix}/adh_{sname}"
      act.target = f"{robot_prefix}/{sname}"
      act.trntype = mujoco.mjtTrn.mjTRN_BODY
      act.set_to_adhesion(gain=force_per_sphere)
      act.ctrlrange[0] = 0.0
      act.ctrlrange[1] = 1.0
      act.ctrllimited = True
      act.forcelimited = True
      act.forcerange[0] = 0.0
      act.forcerange[1] = force_per_sphere


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quaternion (wxyz convention). Batched.

  Args:
    quat: (E, 4) quaternions in wxyz order.
    vec: (3,) or (E, 3) vectors.

  Returns:
    (E, 3) rotated vectors.
  """
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  if vec.dim() == 1:
    vx, vy, vz = vec[0], vec[1], vec[2]
  else:
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
  # q * v * q^-1 expanded:
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return torch.stack([rx, ry, rz], dim=-1)


class MicrospineController:
  """Per-hand latching adhesion via MuJoCo adhesion actuators (GPU-batched).

  When engage_force_threshold <= 0 (default): sets ctrl=1 for all adhesion
  actuators every step.

  When engage_force_threshold > 0: per-hand latch. A hand's adhesion
  engages only when ALL its spheres simultaneously exceed the threshold.
  Disengages when friction force along hand-up exceeds disengage threshold,
  or on episode reset.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    mj_model = env.sim.mj_model
    robot_prefix = cfg.params["asset_cfg"].name
    device = env.device
    self._device = device
    self._num_envs = env.num_envs
    self._env = env  # cached for debug_vis access to sim.wp_model.opt.gravity
    self._threshold = self._mcfg.engage_force_threshold
    self._disengage_threshold = self._mcfg.disengage_friction_threshold
    self._num_hands = len(self._mcfg.ee_body_names)
    self._debug = self._mcfg.debug

    # Disengage threshold curriculum: linearly ramp the threshold from
    # ``disengage_threshold_start`` to ``disengage_friction_threshold`` over
    # the first ``disengage_threshold_ramp_steps`` global env-step calls.
    self._disengage_curriculum_enabled = (
      self._mcfg.disengage_threshold_ramp_steps > 0
      and self._mcfg.disengage_threshold_start > 0
    )
    self._disengage_global_step = 0

    # Per-hand engage-time ramp of effective adhesion force.
    self._ramp_enabled = (
      self._mcfg.adhesion_force_start > 0
      and self._mcfg.adhesion_force_end > 0
      and self._mcfg.adhesion_ramp_duration_s > 0
    )
    if self._ramp_enabled:
      assert self._mcfg.adhesion_force_end <= self._mcfg.adhesion_force, (
        "adhesion_force_end must be <= adhesion_force (the actuator ceiling)"
      )
      self._log_ratio = math.log(
        self._mcfg.adhesion_force_end / self._mcfg.adhesion_force_start
      )
      self._ramp_steps = max(
        1, int(round(self._mcfg.adhesion_ramp_duration_s / env.step_dt))
      )
      self._start_ctrl_scale = (
        self._mcfg.adhesion_force_start / self._mcfg.adhesion_force
      )

    # Hand up direction in local frame.
    self._hand_up_local = torch.tensor(
      self._mcfg.hand_up_local, dtype=torch.float32, device=device
    )

    # Resolve hand body IDs for orientation queries.
    self._asset_name = robot_prefix
    asset = env.scene[robot_prefix]
    self._asset = asset
    self._hand_body_ids: list[int] = []
    for bname in self._mcfg.ee_body_names:
      ids, _ = asset.find_bodies([bname])
      self._hand_body_ids.append(ids[0])

    # Collect actuator IDs and geom IDs, grouped by hand.
    all_actuator_ids: list[int] = []
    self._hand_actuator_slices: list[tuple[int, int]] = []
    sphere_geom_ids_flat: list[int] = []
    # Flat list of entity-local body IDs for every contact sphere, same order
    # as ``_actuator_ids`` so the sphere-index → hand-index mapping is shared.
    self._sphere_body_ids: list[int] = []

    offset = 0
    for bname in self._mcfg.ee_body_names:
      side = bname.split("_")[0]
      hand_act_ids: list[int] = []
      hand_geom_ids: list[int] = []
      for sname in _sphere_geom_names(side):
        act_name = f"{robot_prefix}/adh_{sname}"
        hand_act_ids.append(mj_model.actuator(act_name).id)
        hand_geom_ids.append(mj_model.geom(f"{robot_prefix}/{sname}").id)
        body_ids, _ = asset.find_bodies([sname])
        self._sphere_body_ids.append(body_ids[0])
      all_actuator_ids.extend(hand_act_ids)
      sphere_geom_ids_flat.extend(hand_geom_ids)
      self._hand_actuator_slices.append((offset, offset + len(hand_act_ids)))
      offset += len(hand_act_ids)

    self._actuator_ids = torch.tensor(all_actuator_ids, dtype=torch.long, device=device)
    self._num_spheres = len(all_actuator_ids)

    # Reverse lookup: sphere_index -> hand_index.
    self._sphere_to_hand = torch.zeros(self._num_spheres, dtype=torch.long, device=device)
    for h, (s_start, s_end) in enumerate(self._hand_actuator_slices):
      self._sphere_to_hand[s_start:s_end] = h

    # For threshold-based engagement.
    if self._threshold > 0:
      self._sphere_geom_ids = torch.tensor(sphere_geom_ids_flat, dtype=torch.long, device=device)
      max_gid = max(sphere_geom_ids_flat) + 1
      self._gid_to_sphere = torch.full((max_gid,), -1, dtype=torch.long, device=device)
      for idx, gid in enumerate(sphere_geom_ids_flat):
        self._gid_to_sphere[gid] = idx

      self._wall_gid = mj_model.geom(self._mcfg.wall_geom_name).id
      self._warp_cached = False

      # Per-(env, hand) latch state. Public so reward/obs functions can read.
      self.engaged = torch.zeros(
        self._num_envs, self._num_hands, dtype=torch.bool, device=device
      )
      # Steps each hand has been continuously engaged (for force ramp).
      self._engage_steps = torch.zeros(
        self._num_envs, self._num_hands, dtype=torch.long, device=device
      )
      # Public: latest per-(env, hand) friction along hand-up direction.
      # Used by reward functions to softly penalize approaching the disengage
      # cliff (instead of waiting for the discrete disengage event).
      self.last_friction_up = torch.zeros(
        self._num_envs, self._num_hands, device=device
      )
      # EMA-smoothed version of last_friction_up. Same units, but heavily
      # filtered so per-step contact noise doesn't blow up the reward variance.
      self.last_friction_up_smoothed = torch.zeros(
        self._num_envs, self._num_hands, device=device
      )
      self._fric_up_ema_alpha: float = 0.95  # ~20-step half life
      # Public: per-sphere streak length at the moment contact was lost on
      # the current step. 0 on non-loss steps; equals previous streak on the
      # step the streak resets to 0. Used by contact_lost_penalty reward.
      self.last_streak_at_loss = torch.zeros(
        self._num_envs, self._num_spheres, dtype=torch.long, device=device
      )
      # Public: current continuous-contact step counter per (env, sphere).
      # Each sphere has its own streak — counts every step that sphere is in
      # physical contact (force >= contact_streak_floor). Has a small grace
      # period: brief one- or two-step gaps from numerical noise do not reset
      # the streak. Independent of the engage latch.
      self.contact_current_streak = torch.zeros(
        self._num_envs, self._num_spheres, dtype=torch.long, device=device
      )
      # Per-(env, sphere) consecutive-miss counter, used for the grace period.
      self._contact_miss_count = torch.zeros(
        self._num_envs, self._num_spheres, dtype=torch.long, device=device
      )
      # Floor force [N] under which a sphere counts as "not in contact".
      self._contact_streak_floor: float = 0.5
      # How many consecutive miss steps before the streak is allowed to reset.
      self._contact_streak_grace: int = 10
      # Latest per-(env, sphere) normal force, refreshed each step. Public.
      self.last_normal_forces = torch.zeros(
        self._num_envs, self._num_spheres, device=device
      )
      # Running max of soft contact count per env across the episode.
      # soft_count = mean(clip(f / contact_floor, 0, 1)) across spheres.
      # Tracks the best simultaneous multi-sphere contact achieved so far,
      # so early low-contact steps don't dilute the engage_progress reward.
      self.peak_soft_count = torch.zeros(self._num_envs, device=device)
      self._peak_contact_floor: float = 0.5
      # Latest per-(env, hand) total contact reaction force in WORLD frame
      # (sum of normal + friction over all spheres on the hand). Used by
      # reward functions that want to score the direction of the reaction
      # force — e.g. "out-of-wall + upward" for microspine engagement.
      self.last_hand_contact_force_world = torch.zeros(
        self._num_envs, self._num_hands, 3, device=device
      )
      # Per-(env, hand) flag: True on the single step an engaged hand disengages.
      self.disengaged_this_step = torch.zeros(
        self._num_envs, self._num_hands, dtype=torch.bool, device=device
      )
      # Per-env counter for steps since reset. Used to skip disengage check
      # for the first few steps (contact data may be stale after reset).
      self._steps_since_reset = torch.zeros(
        self._num_envs, dtype=torch.long, device=device
      )
      # Per-(env, hand) disengage hysteresis: consecutive steps fric_up has
      # exceeded the disengage threshold. Only release when streak reaches
      # ``disengage_consecutive_steps``.
      self._disengage_streak = torch.zeros(
        self._num_envs, self._num_hands, dtype=torch.long, device=device
      )
      # Per-(env, hand) post-disengage cooldown: steps remaining during which
      # re-engagement is blocked. Prevents single-step engage/disengage
      # oscillation when CCD jitter spikes friction.
      self._reengage_cooldown = torch.zeros(
        self._num_envs, self._num_hands, dtype=torch.long, device=device
      )
      # Reusable zero-torque buffer for the per-sphere body wrench write.
      self._zero_sphere_torque = torch.zeros(
        self._num_envs, self._num_spheres, 3, device=device
      )

  def _cache_warp_tensors(self, env: ManagerBasedRlEnv) -> None:
    """Cache warp→torch tensor views for contact data (once)."""
    import warp as wp
    raw = env.sim.data.struct
    self._c_geom = wp.to_torch(raw.contact.geom)       # (nconmax, 2)
    self._c_wid = wp.to_torch(raw.contact.worldid)      # (nconmax,)
    self._c_efc = wp.to_torch(raw.contact.efc_address)  # (nconmax, condim)
    self._c_frame = wp.to_torch(raw.contact.frame)       # (nconmax, 9)
    self._efc_force = wp.to_torch(raw.efc.force)         # (num_envs, nefc)
    self._nacon = wp.to_torch(raw.nacon)                  # (1,)
    self._nconmax = self._c_geom.shape[0]
    self._nefc = self._efc_force.shape[1]
    self._warp_cached = True

  def _get_valid_contact_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify valid wall-sphere contacts and return mask, sphere indices, sign.

    Uses efc_address >= 0 to filter out inactive/stale contact slots
    (MuJoCo sets efc_address to -1 for inactive contacts).

    Returns:
      valid_mask: (nconmax,) bool — True for valid wall-sphere contacts.
      sphere_idx: (nconmax,) long — sphere index for each contact (0 if invalid).
      sign: (nconmax,) float — +1 if sphere is g1, -1 if sphere is g0.
    """
    wall_gid = self._wall_gid
    g0 = self._c_geom[:, 0]
    g1 = self._c_geom[:, 1]

    wall_is_g0 = (g0 == wall_gid)
    wall_is_g1 = (g1 == wall_gid)
    is_wall_contact = wall_is_g0 | wall_is_g1

    sphere_gid = torch.where(wall_is_g0, g1, g0).long()
    safe_gid = sphere_gid.clamp(max=self._gid_to_sphere.shape[0] - 1)
    sphere_idx = self._gid_to_sphere[safe_gid]

    # efc_address[0] >= 0 ensures the contact slot is active, not stale.
    is_active = self._c_efc[:, 0] >= 0

    valid_mask = is_wall_contact & (sphere_gid < self._gid_to_sphere.shape[0]) & (sphere_idx >= 0) & is_active
    safe_sphere_idx = sphere_idx.clamp(min=0)

    # Sign: contact normal points g0→g1.
    # Force on g1 = +frame.T @ f, force on g0 = -frame.T @ f.
    # If sphere is g1 (wall_is_g0): sign = +1
    # If sphere is g0 (wall_is_g1): sign = -1
    sign = torch.where(wall_is_g0, 1.0, -1.0)

    return valid_mask, safe_sphere_idx, sign

  def _compute_sphere_normal_forces(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute per-(env, sphere) normal contact force against wall.

    Fully GPU — no .item() or .any() calls.

    Returns:
      Tensor of shape (num_envs, num_spheres) with normal force magnitudes.
    """
    if not self._warp_cached:
      self._cache_warp_tensors(env)

    num_envs = self._num_envs
    num_spheres = self._num_spheres

    normal_forces = torch.zeros(num_envs, num_spheres, device=self._device)

    valid_mask, safe_sphere_idx, _ = self._get_valid_contact_mask()

    wids = self._c_wid.long()
    efc_addrs = self._c_efc[:, 0].long().clamp(min=0, max=self._nefc - 1)
    fn = self._efc_force[wids, efc_addrs] * valid_mask.float()

    flat_idx = wids * num_spheres + safe_sphere_idx
    flat_idx = flat_idx.clamp(max=normal_forces.numel() - 1)
    normal_forces.view(-1).scatter_add_(0, flat_idx, fn)

    return normal_forces

  def _compute_hand_friction_along_up(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute per-(env, hand) friction force projected onto hand's up direction.

    For each contact, friction force in world frame =
      sign * (frame[1,:] * f_tan1 + frame[2,:] * f_tan2)
    Then sum per hand and project onto that hand's up-in-world.

    Returns:
      (num_envs, num_hands) friction force component along hand-up [N].
      Positive = friction pushing in the hand-up direction.
    """
    if not self._warp_cached:
      self._cache_warp_tensors(env)

    num_envs = self._num_envs
    num_hands = self._num_hands
    device = self._device

    result = torch.zeros(num_envs, num_hands, device=device)

    valid_mask, safe_sphere_idx, sign = self._get_valid_contact_mask()
    valid_f = valid_mask.float()  # (nconmax,)

    wids = self._c_wid.long()

    # Friction efc addresses: tangent1 = efc_address[:,1], tangent2 = efc_address[:,2].
    efc_addr1 = self._c_efc[:, 1].long().clamp(min=0, max=self._nefc - 1)
    efc_addr2 = self._c_efc[:, 2].long().clamp(min=0, max=self._nefc - 1)
    f_tan1 = self._efc_force[wids, efc_addr1] * valid_f  # (nconmax,)
    f_tan2 = self._efc_force[wids, efc_addr2] * valid_f  # (nconmax,)

    # Contact frame: (nconmax, 9) → (nconmax, 3, 3), rows = [normal, tan1, tan2].
    frame = self._c_frame.reshape(-1, 3, 3)
    tan1_dir = frame[:, 1, :]  # (nconmax, 3)
    tan2_dir = frame[:, 2, :]  # (nconmax, 3)

    # Friction force on sphere in world frame.
    # sign accounts for which geom is g0/g1.
    f_friction_world = sign.unsqueeze(1) * (
      f_tan1.unsqueeze(1) * tan1_dir + f_tan2.unsqueeze(1) * tan2_dir
    )  # (nconmax, 3)

    # Map sphere_idx → hand_idx.
    hand_idx = self._sphere_to_hand[safe_sphere_idx]  # (nconmax,)

    # Compute hand-up direction in world frame for each hand.
    # hand_up_world[h] = rotate(hand_quat[h], hand_up_local)
    for h in range(num_hands):
      body_id = self._hand_body_ids[h]
      hand_quat = self._asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz
      hand_up_world = _quat_rotate(hand_quat, self._hand_up_local)  # (E, 3)

      # Mask contacts belonging to this hand.
      h_mask = ((hand_idx == h) & valid_mask).float()  # (nconmax,)

      # Per-contact friction dot hand_up: need per-env hand_up.
      # f_friction_world is (nconmax, 3), hand_up_world is (E, 3).
      # For each contact ci with world_id=wid, dot with hand_up_world[wid].
      hand_up_per_contact = hand_up_world[wids]  # (nconmax, 3)
      dot = (f_friction_world * hand_up_per_contact).sum(dim=1)  # (nconmax,)
      dot = dot * h_mask

      # Scatter-add per env.
      result[:, h].scatter_add_(0, wids, dot)

    return result

  def _compute_hand_contact_force_world(
    self, env: ManagerBasedRlEnv
  ) -> torch.Tensor:
    """Compute per-(env, hand) total contact reaction force in world frame.

    Returns the sum of normal + friction contact force (as felt by the sphere,
    so the force the wall applies ON the hand) across all spheres on each
    hand, in world coordinates.

    Returns:
      (num_envs, num_hands, 3) force vectors [N].
    """
    if not self._warp_cached:
      self._cache_warp_tensors(env)

    num_envs = self._num_envs
    num_hands = self._num_hands
    device = self._device

    result = torch.zeros(num_envs, num_hands, 3, device=device)

    valid_mask, safe_sphere_idx, sign = self._get_valid_contact_mask()
    valid_f = valid_mask.float()

    wids = self._c_wid.long()
    efc_addr_n = self._c_efc[:, 0].long().clamp(min=0, max=self._nefc - 1)
    efc_addr1 = self._c_efc[:, 1].long().clamp(min=0, max=self._nefc - 1)
    efc_addr2 = self._c_efc[:, 2].long().clamp(min=0, max=self._nefc - 1)

    f_n = self._efc_force[wids, efc_addr_n] * valid_f
    f_tan1 = self._efc_force[wids, efc_addr1] * valid_f
    f_tan2 = self._efc_force[wids, efc_addr2] * valid_f

    frame = self._c_frame.reshape(-1, 3, 3)  # [normal, tan1, tan2] rows
    n_dir = frame[:, 0, :]
    tan1_dir = frame[:, 1, :]
    tan2_dir = frame[:, 2, :]

    # Force on sphere in world frame. ``sign`` flips for the g0/g1 case so
    # the result is always the force applied ON the sphere.
    f_world = sign.unsqueeze(1) * (
      f_n.unsqueeze(1) * n_dir
      + f_tan1.unsqueeze(1) * tan1_dir
      + f_tan2.unsqueeze(1) * tan2_dir
    )  # (nconmax, 3)

    hand_idx = self._sphere_to_hand[safe_sphere_idx]  # (nconmax,)

    # Scatter sum per env per hand for each of the 3 components.
    for h in range(num_hands):
      h_mask = ((hand_idx == h) & valid_mask).float().unsqueeze(1)
      f_masked = f_world * h_mask  # (nconmax, 3)
      result[:, h, 0].scatter_add_(0, wids, f_masked[:, 0])
      result[:, h, 1].scatter_add_(0, wids, f_masked[:, 1])
      result[:, h, 2].scatter_add_(0, wids, f_masked[:, 2])

    return result

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    if self._threshold <= 0:
      env.sim.data.ctrl[:, self._actuator_ids] = self._mcfg.ctrl_value
      return

    self._steps_since_reset += 1

    normal_forces = self._compute_sphere_normal_forces(env)  # (E, S)
    self.last_normal_forces = normal_forces

    # Update running max of soft contact count.
    soft_count = (normal_forces / self._peak_contact_floor).clamp(0.0, 1.0).mean(dim=1)
    self.peak_soft_count = torch.max(self.peak_soft_count, soft_count)

    # World-frame total contact reaction force per hand (sum of normal +
    # friction across spheres). Cached for reward functions that want to
    # score the direction of the reaction force.
    self.last_hand_contact_force_world = self._compute_hand_contact_force_world(env)

    # --- Engage: minimum number/fraction of spheres above force threshold,
    # optionally gated by a hand-parallel-to-wall check ---
    # Use compression-only forces (clamp negatives to 0): adhesion pulls the
    # hand into the wall and registers as a large negative constraint force,
    # which is NOT evidence the hand is actively pushing the wall. Only
    # positive (compressive) normal force qualifies for latching.
    was_engaged = self.engaged.clone()
    compression_forces = normal_forces.clamp(min=0.0)
    min_frac = self._mcfg.engage_min_fraction
    min_count = self._mcfg.engage_min_spheres
    cos_thresh = self._mcfg.engage_parallel_cos_threshold

    # Wall inward normal in world (per env). Used both by the parallel-engage
    # check and by the body-level adhesion force application below, so always
    # compute it.
    wall = env.scene[self._mcfg.wall_entity_name]
    wquat = wall.data.root_link_quat_w  # (E, 4) wxyz
    ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
    inward_x = (1.0 - 2.0 * (wy * wy + wz * wz))
    inward_y = 2.0 * (wx * wy + ww * wz)
    inward_z = 2.0 * (wx * wz - ww * wy)
    wall_inward = torch.stack([inward_x, inward_y, inward_z], dim=-1)  # (E, 3)

    # Cooldown mask: engagement blocked while any cooldown steps remain.
    cooldown_active = self._reengage_cooldown > 0  # (E, H)

    for h in range(self._num_hands):
      s_start, s_end = self._hand_actuator_slices[h]
      hand_forces = compression_forces[:, s_start:s_end]  # (E, spheres_per_hand)
      above = (hand_forces >= self._threshold)
      if min_count > 0:
        engaged_now = above.sum(dim=1) >= min_count  # (E,)
      else:
        engaged_now = above.float().mean(dim=1) >= min_frac  # (E,)

      if cos_thresh > 0:
        body_id = self._hand_body_ids[h]
        hquat = self._asset.data.body_link_quat_w[:, body_id, :]
        hw, hx, hy, hz = hquat[:, 0], hquat[:, 1], hquat[:, 2], hquat[:, 3]
        # Hand contact face normal = R_hand · (-1,0,0) in world.
        fx = -(1.0 - 2.0 * (hy * hy + hz * hz))
        fy = -2.0 * (hx * hy + hw * hz)
        fz = -2.0 * (hx * hz - hw * hy)
        face = torch.stack([fx, fy, fz], dim=-1)
        parallel = (face * wall_inward).sum(dim=-1) >= cos_thresh
        engaged_now = engaged_now & parallel

      # Block latching during post-disengage cooldown.
      engaged_now = engaged_now & ~cooldown_active[:, h]
      self.engaged[:, h] |= engaged_now

    # Tick cooldown down (saturating at 0).
    self._reengage_cooldown = (self._reengage_cooldown - 1).clamp(min=0)
    newly_engaged = self.engaged & ~was_engaged  # (E, H)

    # --- Disengage: friction force along hand-up exceeds threshold ---
    # Skip:
    #   (1) the engage step itself (contact data may be stale before adhesion),
    #   (2) the first 10 steps after reset (sim needs time to settle).
    self._disengage_global_step += 1
    if self._disengage_curriculum_enabled:
      ramp_steps = self._mcfg.disengage_threshold_ramp_steps
      progress = min(self._disengage_global_step / ramp_steps, 1.0)
      effective_disengage = (
        self._mcfg.disengage_threshold_start
        + (self._disengage_threshold - self._mcfg.disengage_threshold_start) * progress
      )
    else:
      effective_disengage = self._disengage_threshold

    friction_up = None
    if effective_disengage > 0:
      friction_up = self._compute_hand_friction_along_up(env)  # (E, H)
      self.last_friction_up = friction_up
      a = self._fric_up_ema_alpha
      self.last_friction_up_smoothed = (
        a * self.last_friction_up_smoothed + (1.0 - a) * friction_up
      )
      grace = (self._steps_since_reset < 10).unsqueeze(1)  # (E, 1)
      need_streak = max(1, self._mcfg.disengage_consecutive_steps)
      cooldown_set = max(0, self._mcfg.reengage_cooldown_steps)
      for h in range(self._num_hands):
        exceeded = friction_up[:, h] > effective_disengage  # (E,)
        exceeded = exceeded & ~newly_engaged[:, h] & ~grace[:, 0] & self.engaged[:, h]
        # Hysteresis: increment streak while exceeding, reset otherwise.
        self._disengage_streak[:, h] = torch.where(
          exceeded,
          self._disengage_streak[:, h] + 1,
          torch.zeros_like(self._disengage_streak[:, h]),
        )
        release = self._disengage_streak[:, h] >= need_streak  # (E,)
        self.engaged[:, h] &= ~release
        # Clear the streak for hands that actually released.
        self._disengage_streak[:, h] = torch.where(
          release,
          torch.zeros_like(self._disengage_streak[:, h]),
          self._disengage_streak[:, h],
        )
        # Arm re-engage cooldown on actual release.
        if cooldown_set > 0:
          self._reengage_cooldown[:, h] = torch.where(
            release,
            torch.full_like(self._reengage_cooldown[:, h], cooldown_set),
            self._reengage_cooldown[:, h],
          )

    # Track disengage transitions (latched-ON → OFF) for reward signal.
    self.disengaged_this_step = was_engaged & ~self.engaged  # (E, H)

    # Update per-sphere continuous-contact streak — each sphere tracks its
    # own streak independently. Grows whenever that sphere has physical
    # contact (force >= floor); brief gaps (< grace) hold the value; gaps
    # >= grace reset to 0. Independent of the engage latch, so the policy
    # gets a signal from the first step any sphere touches the wall.
    floor = self._contact_streak_floor
    grace = self._contact_streak_grace
    prev_streak = self.contact_current_streak.clone()

    sphere_in_contact = (normal_forces >= floor)  # (E, S)
    miss_inc = self._contact_miss_count + 1
    next_miss = torch.where(
      sphere_in_contact, torch.zeros_like(miss_inc), miss_inc
    )
    should_reset = next_miss >= grace
    # In contact: streak += 1. In grace: streak holds. Beyond grace: reset.
    held = torch.where(
      sphere_in_contact,
      self.contact_current_streak + 1,
      self.contact_current_streak,
    )
    next_streak = torch.where(
      should_reset, torch.zeros_like(self.contact_current_streak), held
    )
    self.contact_current_streak = next_streak
    self._contact_miss_count = next_miss
    # Detect contact-loss transitions: streak just reset to 0 from positive.
    contact_lost = (next_streak == 0) & (prev_streak > 0)
    self.last_streak_at_loss = torch.where(
      contact_lost, prev_streak, torch.zeros_like(prev_streak)
    )

    # --- Log state transitions (env 0 only) ---
    if self._debug:
      step = env.episode_length_buf[0].item()
      changed = (self.engaged[0] != was_engaged[0])
      if changed.any():
        for h in range(self._num_hands):
          if changed[h].item():
            old = "ON" if was_engaged[0, h].item() else "OFF"
            new = "ON" if self.engaged[0, h].item() else "OFF"
            hf = normal_forces[0, self._hand_actuator_slices[h][0]:self._hand_actuator_slices[h][1]].cpu().tolist()
            min_f = min(hf)
            fu = friction_up[0, h].item() if friction_up is not None else 0.0
            bname = self._mcfg.ee_body_names[h]
            print(f"[Microspine] TRANSITION step={step} {bname} {old}->{new} "
                  f"min_n={min_f:.1f}N fric_up={fu:.1f}N")

    # --- Per-hand engage counter and force ramp ---
    # Reset counter on hands that just engaged; zero it for disengaged hands;
    # otherwise increment.
    self._engage_steps = torch.where(
      newly_engaged, torch.zeros_like(self._engage_steps), self._engage_steps + 1
    )
    self._engage_steps = torch.where(
      self.engaged, self._engage_steps, torch.zeros_like(self._engage_steps)
    )

    if self._ramp_enabled:
      progress = (self._engage_steps.float() / self._ramp_steps).clamp_(0.0, 1.0)
      # force = start * exp(log(end/start) * progress); normalize by ceiling.
      ctrl_scale = self._start_ctrl_scale * torch.exp(self._log_ratio * progress)
    else:
      ctrl_scale = torch.ones_like(self._engage_steps, dtype=torch.float32)

    # --- Adhesion via per-sphere body-level external wrench ---
    # Apply the force on each contact-sphere sub-body rather than on the hand
    # CoM. Distributing the pull across the 8 spheres on a hand produces the
    # restoring moment that keeps the hand flat against the wall: a hand-CoM
    # force lets the top edge peel off while the bottom row pivots in contact,
    # whereas per-sphere force pulls every sphere (including the non-contacting
    # upper row) into the wall.
    # ``adhesion_force`` is the per-hand maximum: every latched hand can pull
    # up to this many newtons independently. Within a hand it is split evenly
    # across the 8 contact spheres.
    spheres_per_hand = self._num_spheres // self._num_hands
    force_per_sphere = self._mcfg.adhesion_force / spheres_per_hand
    # Broadcast per-hand engage + ramp onto each sphere.
    engaged_per_sphere = self.engaged.float()[:, self._sphere_to_hand]  # (E, S)
    scale_per_sphere = ctrl_scale[:, self._sphere_to_hand]              # (E, S)
    sphere_mag = engaged_per_sphere * scale_per_sphere * force_per_sphere  # (E, S)
    sphere_force = sphere_mag.unsqueeze(-1) * wall_inward.unsqueeze(1)     # (E, S, 3)
    self._asset.write_external_wrench_to_sim(
      forces=sphere_force,
      torques=self._zero_sphere_torque,
      body_ids=self._sphere_body_ids,
    )

    # Per-sphere adhesion actuators are now redundant: zero their ctrl so
    # they don't double-apply on whichever spheres happen to be in contact.
    env.sim.data.ctrl[:, self._actuator_ids] = 0.0

    if self._debug:
      step = env.episode_length_buf[0].item()
      if step % 50 == 0:
        for h in range(self._num_hands):
          if bool(self.engaged[0, h].item()):
            bname = self._mcfg.ee_body_names[h]
            fx = wall_inward[0, 0].item()
            fy = wall_inward[0, 1].item()
            fz = wall_inward[0, 2].item()
            s_start, s_end = self._hand_actuator_slices[h]
            mag_sum = sphere_mag[0, s_start:s_end].sum().item()
            n_contact = int((normal_forces[0, s_start:s_end] >= 0.5).sum().item())
            print(f"[Microspine] APPLIED step={step} {bname} eng=1 "
                  f"|F|={mag_sum:.1f}N dir=({fx:.2f},{fy:.2f},{fz:.2f}) "
                  f"contact_spheres={n_contact}/{s_end - s_start}")

    # Periodic status print removed — only transitions (engage/disengage) log.

  def reset(self, env_ids: torch.Tensor) -> None:
    if self._threshold > 0:
      self.engaged[env_ids] = False
      self.disengaged_this_step[env_ids] = False
      self.last_normal_forces[env_ids] = 0.0
      self.peak_soft_count[env_ids] = 0.0
      self.last_hand_contact_force_world[env_ids] = 0.0
      self.last_friction_up[env_ids] = 0.0
      self.last_friction_up_smoothed[env_ids] = 0.0
      self.last_streak_at_loss[env_ids] = 0
      self._engage_steps[env_ids] = 0
      self.contact_current_streak[env_ids] = 0
      self._contact_miss_count[env_ids] = 0
      self._steps_since_reset[env_ids] = 0
      self._disengage_streak[env_ids] = 0
      self._reengage_cooldown[env_ids] = 0

  def debug_vis(self, visualizer) -> None:
    """Draw a gravity-direction arrow over the tracked environment.

    Only runs when the viewer is active (train has no viewer, so no cost).
    Arrow is anchored just above env 0's origin and points along current
    gravity so you can see the curriculum's gravity tilt live.
    """
    import warp as wp

    env_idx = getattr(visualizer, "env_idx", 0)
    env = self._env
    if env_idx >= env.num_envs:
      return

    grav = wp.to_torch(env.sim.wp_model.opt.gravity)[0].cpu().numpy()
    mag = float((grav * grav).sum() ** 0.5)
    if mag < 1e-6:
      return
    direction = grav / mag  # unit vector along gravity

    origin = env.scene.env_origins[env_idx].cpu().numpy()
    # Anchor the arrow a few meters above the env origin so it doesn't
    # overlap with the robot. Length scales with viewer mean-size for
    # comfortable visibility.
    scale = max(1.5, visualizer.meansize * 4.0)
    start = origin + [0.0, 0.0, 3.5]
    end = start + direction * scale
    visualizer.add_arrow(
      start=start,
      end=end,
      color=(1.0, 0.3, 0.1, 1.0),  # orange
      width=0.04,
      label="gravity",
    )
