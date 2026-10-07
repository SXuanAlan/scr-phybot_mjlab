"""The climb reference command: phase clock, reference-state initialization and tracking targets.

The command owns three things:

  * the per-env phase index into one regrip cycle (advances every policy
    step and wraps: the reference is a loop, not a finite take);
  * reference-state initialization (RSI): on reset each env is placed at a
    sampled clip frame (root pose, joint state and finite-difference
    velocities) and the microspine latches are seeded from the frame's
    ``ee_engaged`` flags;
  * the reference quantities the tracking rewards compare against:
    BeyondMimic body tracking with XY/yaw pose alignment and recorded world
    velocities, plus the along-wall height and anchor ladder that keep
    absolute climbing progress.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_mul,
  sample_uniform,
)

from scr_phybot_mjab.assets.wall import WALL_PIVOT_X

from .body_tracking import align_body_reference, body_velocities_from_poses, build_body_pose_tables
from .clip import ClimbMotionClip
from .phase_sampling import AdaptivePhaseSampler

__all__ = ("ClimbMotionCommand", "ClimbMotionCommandCfg", "vertical_height_spawn_offset")

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_GHOST_COLOR = (0.5, 0.7, 0.5, 0.5)


def vertical_height_spawn_offset(
  clip_root_pos: torch.Tensor,
  wall_quat: torch.Tensor,
  pivot_x: float,
) -> torch.Tensor:
  """Along-wall shift restoring each frame's VERTICAL-wall spawn height.

  The pure base-line rotation moves a spawn's world z with the wall (down at
  shallow angles — ever closer to the floor). Sliding the whole RSI reference
  along the tilted surface by this offset puts the root back at the world
  height the same clip frame has on the vertical wall, while pads, anchors
  and pose keep conforming to the surface (on-wall points recover their 90°
  world heights to within the base-standoff correction). Solves
  ``-rel_x sin(a) + (z + h) cos(a) = z`` for ``h``; exactly zero at identity.
  """
  alpha = 2.0 * torch.atan2(wall_quat[:, 2], wall_quat[:, 0])
  ca = torch.cos(alpha)
  sa = torch.sin(alpha)
  rel_x = clip_root_pos[:, 0] - pivot_x
  target_z = clip_root_pos[:, 2]
  shift = (target_z + rel_x * sa) / ca - target_z
  vertical = wall_quat[:, 0] >= 1.0 - 1.0e-9
  return torch.where(vertical, torch.zeros_like(shift), shift)


class ClimbMotionCommand(CommandTerm):
  """Phase-indexed climb-cycle reference with reference-state initialization.

  Reset places the robot AT a sampled clip frame and seeds the microspine
  latches from that frame, so the reward landscape at t=0 is exactly "keep
  doing the reference". Requires the microspine affordance command to be
  registered BEFORE this term (its own reset would otherwise clobber the
  seeded latch state); the ordering is asserted on first reset.
  """

  cfg: ClimbMotionCommandCfg

  def __init__(self, cfg: ClimbMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.clip = ClimbMotionClip.from_file(cfg.motion_file, step_dt=env.step_dt, device=str(self.device))
    if tuple(cfg.ee_body_names) != self.clip.ee_names:
      raise ValueError(
        f"Clip EE order {self.clip.ee_names} does not match the configured microspine order {tuple(cfg.ee_body_names)}."
      )

    joint_ids, joint_names = self.robot.find_joints(list(self.clip.joint_names), preserve_order=True)
    if tuple(joint_names) != self.clip.joint_names:
      raise ValueError(
        f"Clip joints could not be mapped onto the robot in order: {joint_names} vs {self.clip.joint_names}."
      )
    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    passive_ids = []
    if self.clip.passive_joint_names:
      passive_ids, passive_names = self.robot.find_joints(list(self.clip.passive_joint_names), preserve_order=True)
      if tuple(passive_names) != self.clip.passive_joint_names:
        raise ValueError("Passive reference joints do not match this robot")
    self._passive_joint_ids = torch.tensor(passive_ids, dtype=torch.long, device=self.device)
    ee_body_ids = [self.robot.find_bodies([name])[0][0] for name in cfg.ee_body_names]
    self._ee_body_ids = torch.tensor(ee_body_ids, dtype=torch.long, device=self.device)

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Offset from clip-frame root position to this env's reference position
    # in the CANONICAL clip frame (vertical wall, no env origin): reset
    # height jitter plus one step height per completed loop. World placement
    # applies the per-env wall frame + env origin via ``_to_world_point``.
    # Also used by body/height tracking and the scheduled anchor ladder.
    self.root_offset = torch.zeros(self.num_envs, 3, device=self.device)
    # Per-env wall frame (wxyz; identity = vertical wall), cached at reset
    # from the wall entity's default root state — the wall_angle curriculum
    # writes that earlier in the same reset pass, and the wall never moves
    # mid-episode, so reset-time caching is exact. All world-facing outputs
    # (RSI write, ref_* properties, ghost) rotate through this frame about
    # the wall base line; at identity the transform is a bit-exact no-op.
    self.wall_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.wall_quat[:, 0] = 1.0
    self._pivot_local = torch.tensor([WALL_PIVOT_X, 0.0, 0.0], device=self.device)
    self.spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._order_checked = False
    self._ghost_model = None
    self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._advance_time = True
    self.phase_sampler = None
    if cfg.sampling_mode == "adaptive":
      self.phase_sampler = AdaptivePhaseSampler(
        self.clip.cycle_frames,
        self.clip.fps,
        self.device,
        bin_time_s=cfg.adaptive_bin_time_s,
        alpha=cfg.adaptive_alpha,
        uniform_ratio=cfg.adaptive_uniform_ratio,
      )
      self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.tracking_body_names:
      self._build_body_tables()
      for name in ("error_body_pos", "error_body_ori", "error_body_lin_vel", "error_body_ang_vel"):
        self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ee_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["engage_match"] = torch.zeros(self.num_envs, device=self.device)

  # -- reference (clip) side --------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Phase clock + engage schedule chart, shape (E, 2 + 3H + 2).

    Layout: ``[cos, sin, engaged(H), (t_release, t_regrip) x H,
    toe_braced(2)]``. The countdown pairs are the schedule's upcoming edges
    per EE (normalized by the cycle, sliding linearly with the clock), so
    the policy reads each raster row's whole future — when this pad must
    let go and when it must re-latch — instead of only the bar under the
    playhead. The trailing pair flags the foot-toe wall-brace windows (the
    released foot's support during its own regrip).
    """
    phase = self.time_steps.float() / self.clip.cycle_frames * (2.0 * torch.pi)
    edges = self.clip.ee_edge_times[self.time_steps]
    return torch.cat(
      [
        torch.cos(phase).unsqueeze(-1),
        torch.sin(phase).unsqueeze(-1),
        self.ref_engaged.float(),
        edges.flatten(1, 2),
        self.ref_toe_braced.float(),
      ],
      dim=-1,
    )

  @property
  def ref_joint_pos(self) -> torch.Tensor:
    return self.clip.joint_pos[self.time_steps]

  @property
  def ref_joint_vel(self) -> torch.Tensor:
    return self.clip.joint_vel[self.time_steps]

  def _build_body_tables(self) -> None:
    ids, names = self.robot.find_bodies(list(self.cfg.tracking_body_names), preserve_order=True)
    if tuple(names) != self.cfg.tracking_body_names:
      raise ValueError("Tracking bodies do not match the robot")
    self._tracking_body_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
    model = self._env.sim.mj_model
    indexing = self.robot.indexing
    joint_qadr = indexing.joint_q_adr.cpu().numpy()
    base = np.array(model.qpos0, copy=True)
    base[joint_qadr] = self.robot.data.default_joint_pos[0].cpu().numpy()
    clip_ids = self._joint_ids.cpu().numpy()
    passive_ids = self._passive_joint_ids.cpu().numpy()
    global_ids = [model.body(f"{self.cfg.entity_name}/{name}").id for name in names]

    def with_tail(values, tail):
      return torch.cat((values, tail)).cpu().numpy().astype(np.float64)

    pos, quat = build_body_pose_tables(
      model,
      base,
      indexing.free_joint_q_adr.cpu().numpy(),
      joint_qadr[clip_ids],
      global_ids,
      with_tail(self.clip.root_pos, self.clip.tail_root_pos),
      with_tail(self.clip.root_quat, self.clip.tail_root_quat),
      with_tail(self.clip.joint_pos, self.clip.tail_joint_pos),
      joint_qadr[passive_ids],
      with_tail(self.clip.passive_joint_pos, self.clip.tail_passive_joint_pos),
    )
    pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
    quat = torch.tensor(quat, device=self.device, dtype=torch.float32)
    self._body_lin_vel, self._body_ang_vel = body_velocities_from_poses(pos, quat, self.clip.fps)
    self._body_pos, self._body_quat = pos[:-1], quat[:-1]

  @property
  def ref_body_pos_w(self):
    return self._to_world_point(self._body_pos[self.time_steps] + self.root_offset[:, None])

  @property
  def ref_body_quat_w(self):
    quats = self._body_quat[self.time_steps]
    return quat_mul(self.wall_quat[:, None].expand_as(quats), quats)

  @property
  def ref_body_lin_vel_w(self):
    vecs = self._body_lin_vel[self.time_steps]
    return quat_apply(self.wall_quat[:, None].expand(-1, vecs.shape[1], -1), vecs)

  @property
  def ref_body_ang_vel_w(self):
    vecs = self._body_ang_vel[self.time_steps]
    return quat_apply(self.wall_quat[:, None].expand(-1, vecs.shape[1], -1), vecs)

  def aligned_body_reference(self):
    return align_body_reference(
      self.ref_body_pos_w,
      self.ref_body_quat_w,
      self.ref_root_pos_w,
      self.ref_root_quat_w,
      self.robot.data.root_link_pos_w,
      self.robot_root_quat_w,
    )

  @property
  def robot_body_pos_w(self):
    return self.robot.data.body_link_pos_w[:, self._tracking_body_ids]

  @property
  def robot_body_quat_w(self):
    return self.robot.data.body_link_quat_w[:, self._tracking_body_ids]

  @property
  def robot_body_lin_vel_w(self):
    return self.robot.data.body_link_lin_vel_w[:, self._tracking_body_ids]

  @property
  def robot_body_ang_vel_w(self):
    return self.robot.data.body_link_ang_vel_w[:, self._tracking_body_ids]

  @property
  def ref_root_quat_w(self) -> torch.Tensor:
    return self._to_world_quat(self.clip.root_quat[self.time_steps])

  @property
  def ref_ee_pos_r(self) -> torch.Tensor:
    return self.clip.ee_pos_r[self.time_steps]

  @property
  def ref_engaged(self) -> torch.Tensor:
    return self.clip.ee_engaged[self.time_steps]

  @property
  def ref_toe_braced(self) -> torch.Tensor:
    """Scheduled foot-toe wall-brace flags, (E, 2) in PHYBOT_C2_FOOT_TOES
    order (left foot, right foot)."""
    return self.clip.toe_braced[self.time_steps]

  @property
  def ref_anchor_pos_w(self) -> torch.Tensor:
    """Scheduled anchor positions in sim world coordinates, (E, H, 3).

    Clip-frame anchors shifted by the accumulated per-loop step height and
    mapped through the per-env wall frame (+ env origin), i.e. exactly the
    ghost's anchor ladder. While an EE swings this is its upcoming landing
    target."""
    return self._to_world_point(self.clip.ee_anchor_pos[self.time_steps] + self.root_offset.unsqueeze(1))

  @property
  def ref_edge_distance(self) -> torch.Tensor:
    """Normalized time to the NEAREST schedule edge per EE, (E, H).

    Consumed by the edge-graced engage mismatch bill, which thereby stays a
    piecewise-linear function of the countdown channels the policy observes.
    """
    return self.clip.ee_edge_distance[self.time_steps]

  @property
  def ref_run_time(self) -> torch.Tensor:
    """Normalized length of each EE's CURRENT scheduled ON/OFF run, (E, H).

    Multiplied by ``cycle_time_s`` this is the containing window's duration
    in seconds — the denominator of the window-proportional mismatch grace.
    """
    return self.clip.ee_run_time[self.time_steps]

  @property
  def ref_edge_countdowns(self) -> torch.Tensor:
    """Normalized countdowns to the next release/regrip edge, (E, H, 2).

    The same table the ``motion_command`` observation exposes; reward terms
    gated on it (seat-phase press shaping) therefore pay along signals the
    policy can see.
    """
    return self.clip.ee_edge_times[self.time_steps]

  @property
  def cycle_time_s(self) -> float:
    """Duration of one reference cycle in seconds."""
    return self.clip.cycle_frames / self.clip.fps

  # -- per-env wall frame -----------------------------------------------------
  # The clip is canonical (vertical wall, gravity down). Under the wall-angle
  # representation each env's wall is tilted about the base line
  # (x = WALL_PIVOT_X, z = 0); every world-facing quantity applies that
  # rotation. Envs at exactly 90° take a bit-exact identity fast path so the
  # terminal rung reproduces the retired gravity representation verbatim.

  @property
  def wall_up_w(self) -> torch.Tensor:
    """Along-wall climb direction per env, (E, 3) — ẑ on a vertical wall."""
    z_axis = torch.zeros(self.num_envs, 3, device=self.device)
    z_axis[:, 2] = 1.0
    return quat_apply(self.wall_quat, z_axis)

  @property
  def wall_pivot_w(self) -> torch.Tensor:
    """World position of the wall base (hinge) line per env, (E, 3)."""
    return self._env.scene.env_origins + self._pivot_local

  def _vertical_mask(self, env_ids: torch.Tensor | None) -> torch.Tensor:
    quat = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    return quat[:, 0] >= 1.0 - 1.0e-9

  def _to_world_point(self, points: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Canonical clip-frame points -> world: rotate about the base line, add
    the env origin. Accepts (N, 3) or (N, H, 3)."""
    quat = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    origins = self._env.scene.env_origins
    if env_ids is not None:
      origins = origins[env_ids]
    vertical = self._vertical_mask(env_ids)
    pivot = self._pivot_local
    if points.dim() == 3:
      quat_b = quat.unsqueeze(1).expand(-1, points.shape[1], -1)
      rotated = pivot + quat_apply(quat_b, points - pivot)
      out = torch.where(vertical.view(-1, 1, 1), points, rotated)
      return out + origins.unsqueeze(1)
    rotated = pivot + quat_apply(quat, points - pivot)
    out = torch.where(vertical.view(-1, 1), points, rotated)
    return out + origins

  def _to_world_quat(self, quats: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    wall = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    vertical = self._vertical_mask(env_ids)
    return torch.where(vertical.view(-1, 1), quats, quat_mul(wall, quats))

  def _to_world_vec(self, vecs: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    wall = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    # quat_apply with an identity quaternion is exact — no fast path needed.
    return quat_apply(wall, vecs)

  @property
  def ref_root_pos_w(self) -> torch.Tensor:
    """Ghost/RSI-only global reference position (untracked by rewards)."""
    return self._to_world_point(self.clip.root_pos[self.time_steps] + self.root_offset)

  # -- robot side -------------------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self._joint_ids]

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self._joint_ids]

  @property
  def robot_root_quat_w(self) -> torch.Tensor:
    return self.robot.data.root_link_quat_w

  @property
  def robot_ee_pos_r(self) -> torch.Tensor:
    """EE body origins in the robot's root frame, (E, H, 3)."""
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    ee_pos = self.robot.data.body_link_pos_w[:, self._ee_body_ids]
    delta = ee_pos - root_pos.unsqueeze(1)
    quat = root_quat.unsqueeze(1).expand(-1, len(self.cfg.ee_body_names), -1)
    return quat_apply_inverse(quat, delta)

  @property
  def robot_engaged(self) -> torch.Tensor:
    return self._affordance().engaged

  @property
  def height_gained(self) -> torch.Tensor:
    """Along-wall base rise since the episode's RSI placement, (E,).

    Curriculum input. Projection onto ``wall_up_w`` — world z on a vertical
    wall (bit-identical to the retired z-difference there)."""
    delta = self.robot.data.root_link_pos_w - self.spawn_pos
    return (delta * self.wall_up_w).sum(dim=-1)

  def _affordance(self):
    return self._env.command_manager.get_term(self.cfg.affordance_command_name)

  # -- lifecycle --------------------------------------------------------------

  def _check_term_order(self) -> None:
    """The affordance must reset before this term seeds it (dict order)."""
    if self._order_checked:
      return
    names = self._env.command_manager.active_terms
    own = next(name for name in names if self._env.command_manager.get_term(name) is self)
    if names.index(self.cfg.affordance_command_name) > names.index(own):
      raise RuntimeError(
        f"Command '{self.cfg.affordance_command_name}' must be registered "
        f"before '{own}': its reset would clobber the RSI-seeded latches."
      )
    self._order_checked = True

  def _update_metrics(self) -> None:
    if self.cfg.tracking_body_names:
      position, orientation = self.aligned_body_reference()
      self.metrics["error_body_pos"] = (position - self.robot_body_pos_w).norm(dim=-1).mean(-1)
      self.metrics["error_body_ori"] = quat_error_magnitude(orientation, self.robot_body_quat_w).mean(-1)
      self.metrics["error_body_lin_vel"] = (self.ref_body_lin_vel_w - self.robot_body_lin_vel_w).norm(dim=-1).mean(-1)
      self.metrics["error_body_ang_vel"] = (self.ref_body_ang_vel_w - self.robot_body_ang_vel_w).norm(dim=-1).mean(-1)
    self.metrics["error_joint_pos"] = (self.ref_joint_pos - self.robot_joint_pos).abs().mean(dim=-1)
    self.metrics["error_ee_pos"] = (self.ref_ee_pos_r - self.robot_ee_pos_r).norm(dim=-1).mean(dim=-1)
    self.metrics["engage_match"] = (self.ref_engaged == self.robot_engaged).float().mean(dim=-1)

  def _record_phase_failures(self, env_ids: torch.Tensor) -> None:
    if self.phase_sampler is not None:
      manager = getattr(self._env, "termination_manager", None)
      if manager is not None:
        failed = manager.terminated[env_ids] & (self._env.episode_length_buf[env_ids] > 0)
        self.phase_sampler.record(self.time_steps[env_ids][failed])

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._check_term_order()
    count = len(env_ids)
    self._record_phase_failures(env_ids)
    if self.cfg.sampling_mode == "start":
      frames = torch.zeros(count, dtype=torch.long, device=self.device)
    elif self.phase_sampler is not None:
      frames = self.phase_sampler.sample(count)
    else:
      frames = torch.randint(0, self.clip.cycle_frames, (count,), device=self.device)
    self.time_steps[env_ids] = frames
    self._fresh[env_ids] = True

    # --- reference state initialization -------------------------------------
    # Per-env wall frame first: the wall_angle curriculum wrote this env's
    # pose into the wall's default_root_state earlier in this reset pass
    # (identity when no wall_angle term is installed — vertical wall).
    wall: Entity = self._env.scene[self.cfg.wall_entity_name]
    self.wall_quat[env_ids] = wall.data.default_root_state[env_ids, 3:7].clone()
    # Canonical-frame offset: spawn-height jitter (along the wall) plus the
    # vertical-height compensation; env origins are applied by the world
    # transform.
    offset = torch.zeros(count, 3, device=self.device)
    offset[:, 2] = sample_uniform(
      self.cfg.height_offset_range[0],
      self.cfg.height_offset_range[1],
      (count,),
      device=self.device,
    )
    if self.cfg.spawn_match_vertical_height:
      offset[:, 2] += vertical_height_spawn_offset(
        self.clip.root_pos[frames],
        self.wall_quat[env_ids],
        float(self._pivot_local[0].item()),
      )
    self.root_offset[env_ids] = offset
    root_pos = self._to_world_point(self.clip.root_pos[frames] + offset, env_ids)
    self.spawn_pos[env_ids] = root_pos

    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, self._joint_ids] = self.clip.joint_pos[frames]
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, self._joint_ids] = self.clip.joint_vel[frames]
    if self.clip.passive_joint_names:
      # Passive suspension geometry is part of RSI. Leaving gimbals at zero
      # would move C off the wall despite a correctly retargeted reference.
      joint_pos[:, self._passive_joint_ids] = self.clip.passive_joint_pos[frames]
      joint_vel[:, self._passive_joint_ids] = self.clip.passive_joint_vel[frames]

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat(
      [
        root_pos,
        self._to_world_quat(self.clip.root_quat[frames], env_ids),
        self._to_world_vec(self.clip.root_lin_vel[frames], env_ids),
        self._to_world_vec(self.clip.root_ang_vel[frames], env_ids),
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

    # Seed the microspine latches from the frame's engage schedule. The pads
    # of engaged EEs sit on the wall in every clip frame. Empirical tasks
    # install the physical anchors after reset FK, before physics advances.
    self._affordance().seed_engaged(env_ids, self.clip.ee_engaged[frames])

  def compute(self, dt: float) -> None:
    self._advance_time = dt > 0.0
    super().compute(dt)

  def _update_command(self) -> None:
    if self.cfg.synchronize_reset_phase:
      # A freshly reset env holds its reference frame for one step, and no
      # frame advances while no physics time elapsed.
      hold = self._fresh | (not self._advance_time)
      next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    else:
      next_steps = self.time_steps + 1
    wrapped = next_steps >= self.clip.cycle_frames
    self.time_steps = torch.where(wrapped, torch.zeros_like(next_steps), next_steps)
    # The loop's state-5-equals-state-1 closure is one step height up; keep
    # the ghost/RSI reference continuous across the wrap.
    self.root_offset[wrapped, 2] += self.clip.step_height
    if self.cfg.initialize_seeded_anchors:
      fresh_ids = self._fresh.nonzero().flatten()
      if len(fresh_ids):
        # env.reset() and the auto-reset path both refresh FK before command
        # compute. Step events alone would leave the first manual-reset
        # physics interval without anchors. Birth events must not pay rewards.
        affordance = self._affordance()
        event = self._env.event_manager.get_term_cfg(affordance.cfg.physics_event_name).func
        event.initialize_seeded_anchors(fresh_ids)
        affordance._steps_since_engage[fresh_ids] = 0
        # Auto-reset computes dt>0 for the whole batch although these rows
        # were just born. Only elapsed physics counts as preload debounce.
        affordance._engage_count[fresh_ids] = 0
    self._fresh.zero_()
    if self.phase_sampler is not None:
      if self._advance_time:
        self.phase_sampler.update()
      probabilities = self.phase_sampler.probabilities()
      self.metrics["sampling_entropy"][:] = -(probabilities * probabilities.log()).sum()
      self.metrics["sampling_top1_prob"][:] = probabilities.max()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._ghost_model.geom_rgba[:] = np.array(_GHOST_COLOR, dtype=np.float32)

    indexing = self.robot.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()
    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.ref_root_pos_w[batch].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = self.ref_root_quat_w[batch].cpu().numpy()
      full_joint_pos = self.robot.data.default_joint_pos[batch].clone()
      full_joint_pos[self._joint_ids] = self.ref_joint_pos[batch]
      if self.clip.passive_joint_names:
        full_joint_pos[self._passive_joint_ids] = self.clip.passive_joint_pos[self.time_steps[batch]]
      qpos[joint_q_adr] = full_joint_pos.cpu().numpy()
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class ClimbMotionCommandCfg(CommandTermCfg):
  """Configuration for :class:`ClimbMotionCommand`."""

  entity_name: str = "robot"
  motion_file: str
  affordance_command_name: str = "microspine_affordance"
  # Wall entity whose per-env pose defines the wall frame (tilt about the
  # base line); read once per reset AFTER the wall_angle curriculum writes it.
  wall_entity_name: str = "wall"
  ee_body_names: tuple[str, ...]
  # "uniform" = DeepMimic RSI (sample a random cycle phase per episode);
  # "adaptive" = failure-weighted full-cycle sampling with a uniform floor;
  # "start" = always begin at frame 0 (evaluation/play).
  sampling_mode: Literal["uniform", "start", "adaptive"] = "uniform"
  tracking_body_names: tuple[str, ...] = ()
  adaptive_bin_time_s: float = 1.0
  adaptive_alpha: float = 0.001
  adaptive_uniform_ratio: float = 0.1
  synchronize_reset_phase: bool = False
  initialize_seeded_anchors: bool = False
  # Uniform extra spawn height (m) on top of the clip's root z, decoupling
  # absolute altitude from cycle phase.
  height_offset_range: tuple[float, float] = (0.0, 0.0)
  # Slide the RSI reference along a TILTED wall so the spawn keeps the same
  # WORLD height as the vertical-wall spawn (floor clearance stops shrinking
  # with the angle). Exactly zero at 90° — the bit-identity path holds.
  # Disable to reproduce the pure-rotation world (equivalence audits).
  spawn_match_vertical_height: bool = True
  # The reference loops; it is never resampled mid-episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ClimbMotionCommand:
    return ClimbMotionCommand(self, env)
