"""Measured-pose phase controller and reference-state training initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.microspine.surfaces import conjugate, multiply, rotate

from .geometry import DEFAULT_UNIT_LAYOUT
from .reference import REFERENCE_PATH, load_reference


def quaternion_error(q, target):
  return 2 * torch.acos((q * target).sum(-1).abs().clamp(max=1.0))


def interpolate_reference(values, phase, progress):
  frame = progress.clamp(0, 1) * (values.shape[1] - 1)
  left = frame.long().clamp(max=values.shape[1] - 2)
  mix = frame - left
  while mix.ndim < values.ndim - 1:
    mix = mix.unsqueeze(-1)
  return torch.lerp(values[phase, left], values[phase, left + 1], mix)


def measured_phase_gates(
  *, hand_error, lift, speed, goal_distance, ground_gap, orientation_error, hand_clearance, stand_error, upright, valid
):
  """Deployment-available gates; no contact, capacity or simulator latch input."""
  near = hand_error < 0.025
  stable = speed < 0.05
  at_goal = (goal_distance < 0.03) & (orientation_error < 0.174533)
  return (
    torch.stack(
      (
        near,
        near,
        (lift >= 0.10) & stable,
        at_goal & (lift >= 0.08),
        at_goal & (ground_gap.abs() < 0.015) & stable,
        at_goal & (ground_gap.abs() < 0.015) & stable & (hand_clearance > 0.02),
        (stand_error < 0.18) & (upright < 0.174533),
      ),
      -1,
    )
    & valid[:, None]
  )


@dataclass(kw_only=True)
class BrickTaskCommandCfg(CommandTermCfg):
  brick: BrickCfg = field(default_factory=BrickCfg)
  reference_file: str = str(REFERENCE_PATH)
  unit_layout: str = DEFAULT_UNIT_LAYOUT
  training_stage: Literal["full", "hold", "engage", "place"] = "full"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  randomize: bool = False
  pose_position_noise_m: float = 0.003
  pose_rotation_noise_rad: float = 0.01
  max_delay_steps: int = 2
  dropout_probability: float = 0.0
  dropout_steps: int = 5
  reset_xy_noise_m: float = 0.0
  reset_yaw_noise_rad: float = 0.0

  def build(self, env):
    return BrickTaskCommand(self, env)


class BrickTaskCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot, self.brick = env.scene["robot"], env.scene["brick"]
    self.joint_ids, _ = self.robot.find_joints(PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True)
    self.gimbal_ids, _ = self.robot.find_joints(
      tuple(f"{s}_hand_spine_gimbal_{a}" for s in ("left", "right") for a in "xyz"), preserve_order=True
    )
    if len(self.gimbal_ids) != 6:
      raise ValueError("Brick reference requires the hand pad gimbal joints")
    self.foot_ids, _ = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    self.arm_ids, _ = self.robot.find_bodies(("left_elbow_pitch", "right_elbow_pitch"), preserve_order=True)
    archive = load_reference(cfg.reference_file)
    if str(archive["unit_layout"]) != cfg.unit_layout or float(archive["unit_preload_n"]) != 10.0:
      raise ValueError("Reference unit layout and preload must match the grasp configuration")
    if not np.allclose(archive["brick_size"], cfg.brick.size) or not np.isclose(archive["brick_mass"], cfg.brick.mass):
      raise ValueError("Regenerate the reference when changing nominal brick geometry or mass")
    start = np.r_[cfg.brick.start_xy, cfg.brick.size[2] / 2 + 0.001]
    goal = start + np.r_[cfg.brick.goal_offset_xy, 0.0]
    if (
      not np.allclose(archive["brick_pos"][0, 0], start)
      or not np.allclose(archive["brick_pos"][-1, -1], goal)
      or not np.isclose(archive["brick_pos"][2, -1, 2] - start[2], cfg.brick.lift_height)
    ):
      raise ValueError("Regenerate the reference when changing the pickup, goal or lift height")
    self.reference = {
      k: torch.tensor(archive[k], dtype=torch.float32, device=self.device)
      for k in ("joint_pos", "root_pose", "gimbal_pos", "brick_pos", "hand_pos", "durations")
    }
    self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.dwell = self.elapsed.clone()
    self.lift_dwell = self.elapsed.clone()
    self.placed_dwell = self.elapsed.clone()
    self.peak_foot_drift = self.elapsed.clone()
    self.lifted = torch.zeros_like(self.phase, dtype=torch.bool)
    self.carried = self.lifted.clone()
    self.success = self.lifted.clone()
    self.training_complete = self.lifted.clone()
    self.fresh = torch.ones_like(self.lifted)
    self.milestone = self.elapsed.clone()
    self.goal = torch.zeros(self.num_envs, 7, device=self.device)
    self.goal[:, 3] = 1.0
    self.initial_brick = torch.zeros(self.num_envs, 3, device=self.device)
    self.foot_anchors = torch.zeros(self.num_envs, 2, 3, device=self.device)
    self.measurement = self.goal.clone()
    self.measurement_velocity = torch.zeros_like(self.initial_brick)
    self.measurement_valid = torch.ones_like(self.lifted)
    self.measurement_age = self.elapsed.clone()
    self.dropout_remaining = self.phase.clone()
    self.delay = self.phase.clone()
    self.pose_history = self.goal[:, None].expand(-1, cfg.max_delay_steps + 1, -1).clone()
    # Nominal pad FK from the active forearm, excluding uninstrumented gimbals.
    model = env.sim.mj_model
    offsets = []
    import mujoco

    for side in ("left", "right"):
      body = model.body(f"robot/{side}_hand_spine")
      center = model.site(f"robot/{side}_hand_spine_contact_center").pos
      rotation = np.empty(9)
      mujoco.mju_quat2Mat(rotation, body.quat)
      offsets.append(body.pos + rotation.reshape(3, 3) @ center)
    self.hand_offsets = torch.tensor(np.asarray(offsets), dtype=torch.float32, device=self.device)
    for key in ("success", "stage_complete", "lifted", "carried", "phase", "goal_error", "foot_drift", "pose_valid"):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  @property
  def progress(self):
    return (self.elapsed / self.reference["durations"][self.phase]).clamp(max=1.0)

  def ref(self, key):
    return interpolate_reference(self.reference[key], self.phase, self.progress)

  @property
  def joint_pos_ref(self):
    return self.ref("joint_pos")

  def nominal_hands_w(self):
    return self.robot.data.body_link_pos_w[:, self.arm_ids] + rotate(
      self.robot.data.body_link_quat_w[:, self.arm_ids], self.hand_offsets
    )

  def relative_pose(self, pose):
    q = conjugate(self.robot.data.root_link_quat_w)
    return torch.cat((rotate(q, pose[:, :3] - self.robot.data.root_link_pos_w), multiply(q, pose[:, 3:])), -1)

  @property
  def command(self):
    # Position/velocity packets are synthetic external measurements in sim.
    # A deployment adapter must supply the same world-calibrated packets.
    return torch.cat(
      (
        self.relative_pose(self.measurement),
        self.relative_pose(self.goal),
        rotate(conjugate(self.robot.data.root_link_quat_w), self.measurement_velocity),
        self.measurement_valid.float()[:, None],
        self.measurement_age[:, None],
        torch.nn.functional.one_hot(self.phase, 7).float(),
        self.progress[:, None],
        self.joint_pos_ref - self.robot.data.joint_pos[:, self.joint_ids],
      ),
      -1,
    )

  def _resample_command(self, env_ids):
    start_phase = {"full": 0, "engage": 1, "hold": 3, "place": 4}[self.cfg.training_stage]
    self.phase[env_ids] = start_phase
    for tensor in (
      self.elapsed,
      self.dwell,
      self.lift_dwell,
      self.placed_dwell,
      self.milestone,
      self.dropout_remaining,
      self.peak_foot_drift,
    ):
      tensor[env_ids] = 0
    for tensor in (self.lifted, self.carried, self.success, self.training_complete):
      tensor[env_ids] = False
    self.fresh[env_ids] = True
    self.delay[env_ids] = torch.randint(self.cfg.max_delay_steps + 1, (len(env_ids),), device=self.device)
    q = self.robot.data.default_joint_pos[env_ids].clone()
    q[:, self.joint_ids] = self.reference["joint_pos"][start_phase, 0]
    q[:, self.gimbal_ids] = self.reference["gimbal_pos"][start_phase, 0]
    self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    root = self.reference["root_pose"][start_phase, 0].expand(len(env_ids), -1).clone()
    root[:, :3] += self._env.scene.env_origins[env_ids]
    self.robot.write_root_state_to_sim(torch.cat((root, torch.zeros_like(root[:, :6])), -1), env_ids=env_ids)
    block = torch.zeros(len(env_ids), 13, device=self.device)
    block[:, :3] = self.reference["brick_pos"][start_phase, 0] + self._env.scene.env_origins[env_ids]
    block[:, 3] = 1.0
    # Seeded curricula must preserve both captured surfaces exactly. Pose DR
    # starts from real unlatched pickups, never from displaced gifted welds.
    if start_phase < 2 and self.cfg.randomize:
      block[:, :2] += (2 * torch.rand(len(env_ids), 2, device=self.device) - 1) * self.cfg.reset_xy_noise_m
      yaw = (2 * torch.rand(len(env_ids), device=self.device) - 1) * self.cfg.reset_yaw_noise_rad
      block[:, 3], block[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    self.brick.write_root_state_to_sim(block, env_ids=env_ids)
    self.initial_brick[env_ids] = (
      torch.tensor([*self.cfg.brick.start_xy, self.cfg.brick.size[2] / 2 + 0.001], device=self.device)
      + self._env.scene.env_origins[env_ids]
    )
    self.goal[env_ids, :3] = self.initial_brick[env_ids] + torch.tensor(
      [*self.cfg.brick.goal_offset_xy, 0.0], device=self.device
    )
    self.goal[env_ids, 3:] = torch.tensor([1.0, 0, 0, 0], device=self.device)

  def _measure(self, dt):
    pose = torch.cat((self.brick.data.root_link_pos_w, self.brick.data.root_link_quat_w), -1).clone()
    pose[:, :3] += torch.randn_like(pose[:, :3]) * self.cfg.pose_position_noise_m
    noise = torch.cat(
      (torch.ones_like(pose[:, :1]), torch.randn_like(pose[:, :3]) * self.cfg.pose_rotation_noise_rad / 2), -1
    )
    noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    pose[:, 3:] = multiply(noise, pose[:, 3:])
    if dt <= 0:
      # Partial reset is not another camera frame for untouched environments.
      self.pose_history[self.fresh] = pose[self.fresh, None]
      self.measurement[self.fresh] = pose[self.fresh]
      self.measurement_velocity[self.fresh] = 0
      self.measurement_age[self.fresh] = 0
      self.measurement_valid[self.fresh] = True
      return
    self.pose_history = self.pose_history.roll(1, 1)
    self.pose_history[:, 0] = pose
    self.pose_history[self.fresh] = pose[self.fresh, None]
    delayed = self.pose_history[torch.arange(self.num_envs, device=self.device), self.delay]
    drop = torch.rand(self.num_envs, device=self.device) < self.cfg.dropout_probability
    self.dropout_remaining = torch.where(drop, self.cfg.dropout_steps, (self.dropout_remaining - 1).clamp_min(0))
    self.dropout_remaining[self.fresh] = 0
    valid = self.dropout_remaining == 0
    interval = dt + (self.measurement_age - self.delay * self._env.step_dt).clamp_min(0)
    velocity = (delayed[:, :3] - self.measurement[:, :3]) / interval.clamp_min(0.001)[:, None]
    velocity[self.fresh] = 0
    self.measurement_velocity = torch.where(
      valid[:, None], 0.95 * self.measurement_velocity + 0.05 * velocity.clamp(-2, 2), self.measurement_velocity
    )
    self.measurement_velocity[self.fresh] = 0
    self.measurement = torch.where(valid[:, None], delayed, self.measurement)
    self.measurement_age = torch.where(valid, self.delay * self._env.step_dt, self.measurement_age + dt)
    self.measurement_valid = valid

  def compute(self, dt):
    self._dt = dt
    # Command manager reset explicitly resamples; no mid-episode resampling.
    self._update_command()
    self._update_metrics()

  def _update_command(self):
    dt = getattr(self, "_dt", self._env.step_dt)
    fresh_ids = self.fresh.nonzero().flatten()
    if len(fresh_ids):
      self.foot_anchors[fresh_ids] = self.robot.data.site_pos_w[fresh_ids][:, self.foot_ids]
      aff = self._env.command_manager.get_term("microspine_affordance")
      mask = torch.full((len(fresh_ids), 2), self.cfg.training_stage in ("hold", "place"), device=self.device)
      aff.seed_engaged(fresh_ids, mask)
      self._env.event_manager.get_term_cfg("microspine_grip").func.initialize_seeded_anchors(fresh_ids)
    self._measure(dt)
    if dt <= 0:
      self.fresh.zero_()
      return
    active = ~self.fresh & ~self.training_complete & self.measurement_valid
    self.elapsed += active * dt
    hands = self.nominal_hands_w()
    offsets = torch.tensor(
      [[0, self.cfg.brick.size[1] / 2, 0], [0, -self.cfg.brick.size[1] / 2, 0]], device=self.device
    )
    grip = self.measurement[:, None, :3] + rotate(self.measurement[:, None, 3:], offsets)
    # Reach ends 25 mm outside each face; engage ends at the face.
    normals = rotate(
      self.measurement[:, None, 3:], torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], device=self.device)
    )
    target = grip + (self.phase == 0)[:, None, None] * 0.025 * normals
    hand_error = torch.linalg.vector_norm(hands - target, dim=-1).amax(-1)
    clearance = ((hands - grip) * normals).sum(-1).amin(-1)
    speed = torch.linalg.vector_norm(self.measurement_velocity, dim=-1)
    lift = self.measurement[:, 2] - self.initial_brick[:, 2]
    goal_dist = torch.linalg.vector_norm(self.measurement[:, :2] - self.goal[:, :2], dim=-1)
    ori = quaternion_error(self.measurement[:, 3:], self.goal[:, 3:])
    stand_q = self.reference["joint_pos"][0, 0]
    stand_error = ((self.robot.data.joint_pos[:, self.joint_ids] - stand_q) ** 2).mean(-1).sqrt()
    up = rotate(self.robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device))
    upright = torch.acos(up[:, 2].clamp(-1, 1))
    gates = measured_phase_gates(
      hand_error=hand_error,
      lift=lift,
      speed=speed,
      goal_distance=goal_dist,
      ground_gap=self.measurement[:, 2] - self.goal[:, 2],
      orientation_error=ori,
      hand_clearance=clearance,
      stand_error=stand_error,
      upright=upright,
      valid=self.measurement_valid,
    )
    selected = gates.gather(1, self.phase[:, None]).squeeze(1) & active & (self.progress >= 1.0)
    self.dwell = torch.where(selected, self.dwell + dt, torch.zeros_like(self.dwell))
    required = torch.tensor([0.1, 0.2, 1.0, 0.2, 1.0, 0.3, 1.0], device=self.device)[self.phase]
    advance = self.dwell >= required
    self.milestone = advance.float()
    last = advance & (self.phase == 6)
    stage_end = {"full": 6, "engage": 2, "hold": 3, "place": 5}[self.cfg.training_stage]
    self.training_complete |= advance & (self.phase == stage_end)
    self.phase = torch.where(advance, (self.phase + 1).clamp(max=6), self.phase)
    self.elapsed = torch.where(advance & ~last, 0.0, self.elapsed)
    self.dwell = torch.where(advance, 0.0, self.dwell)
    # Privileged checks score physical completion, never drive the phase clock.
    true_lift = self.brick.data.root_link_pos_w[:, 2] - self.initial_brick[:, 2]
    stable = torch.linalg.vector_norm(self.brick.data.root_link_lin_vel_w, dim=-1) < 0.05
    self.lift_dwell = torch.where((true_lift >= 0.10) & stable & ~self.fresh, self.lift_dwell + dt, 0.0)
    self.lifted |= self.lift_dwell >= 1.0
    self.carried |= (
      self.lifted
      & (true_lift >= 0.08)
      & (torch.linalg.vector_norm(self.brick.data.root_link_pos_w[:, :2] - self.goal[:, :2], dim=-1) <= 0.03)
    )
    aff = self._env.command_manager.get_term("microspine_affordance")
    true_goal = torch.linalg.vector_norm(self.brick.data.root_link_pos_w - self.goal[:, :3], dim=-1) <= 0.03
    true_goal &= quaternion_error(self.brick.data.root_link_quat_w, self.goal[:, 3:]) <= 0.174533
    grounded = (self.brick.data.root_link_pos_w[:, 2] - self.goal[:, 2]).abs() < 0.008
    resting = true_goal & grounded & stable & ~aff.engaged.any(-1)
    self.placed_dwell = torch.where(resting & ~self.fresh, self.placed_dwell + dt, 0.0)
    drift = torch.linalg.vector_norm(self.robot.data.site_pos_w[:, self.foot_ids] - self.foot_anchors, dim=-1).amax(-1)
    self.peak_foot_drift = torch.maximum(self.peak_foot_drift, drift)
    feet_ok = self.peak_foot_drift <= 0.03
    self.success |= last & self.lifted & self.carried & feet_ok & (self.placed_dwell >= 1.0)
    if self.cfg.training_stage != "full":
      self.success.zero_()
    self.fresh.zero_()

  def _update_metrics(self):
    self.metrics["success"] = self.success.float()
    self.metrics["stage_complete"] = self.training_complete.float()
    self.metrics["lifted"] = self.lifted.float()
    self.metrics["carried"] = self.carried.float()
    self.metrics["phase"] = self.phase.float()
    self.metrics["goal_error"] = torch.linalg.vector_norm(self.brick.data.root_link_pos_w - self.goal[:, :3], dim=-1)
    self.metrics["foot_drift"] = torch.linalg.vector_norm(
      self.robot.data.site_pos_w[:, self.foot_ids] - self.foot_anchors, dim=-1
    ).amax(-1)
    self.metrics["pose_valid"] = self.measurement_valid.float()
