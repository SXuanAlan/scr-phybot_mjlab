"""Phase-driven full-body squat reference command.

Loads the standalone ``squat.npy`` produced by ``scripts/gen_squat_clip.py`` and
exposes its joint/base targets.  Start-mode playback is synchronized directly to
the episode step, so one episode consumes exactly one motion frame per control
step without skipping frame zero after a timeout reset.

The policy observation is the phase clock ``(sin, cos)``.  All motion joints are
written on reset; the neck is absent from both the policy and the motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class SquatMotionLoader:
  """Loads one standalone ``.npy`` squat clip and derives velocities.

  The clip uses the AMP/motion schema (see ``scripts/gen_squat_clip.py``): a dict
  with ``joints_list``, ``joint_positions`` (T, N), ``root_position`` (T, 3),
  ``root_quaternion`` (T, 4) in xyzw, and ``fps``.
  """

  def __init__(
    self, motion_files: tuple[str, ...], joint_names: tuple[str, ...], device: str
  ) -> None:
    if not motion_files:
      raise ValueError("SquatMotionCommandCfg.motion_files must not be empty.")
    if len(motion_files) != 1:
      raise ValueError(
        "The squat task requires exactly one standalone motion clip; "
        f"received {len(motion_files)}."
      )

    joint_pos_parts: list[np.ndarray] = []
    root_pos_parts: list[np.ndarray] = []
    root_quat_parts: list[np.ndarray] = []
    fps: float | None = None
    for motion_file in motion_files:
      path = Path(motion_file)
      if not path.is_file():
        raise FileNotFoundError(
          f"Squat motion file not found: {path}. Generate it with "
          "`python scripts/gen_squat_clip.py` (or gen_stand_clip.py)."
        )
      data = np.load(path, allow_pickle=True).item()
      clip_joints = list(data["joints_list"])
      if clip_joints != list(joint_names):
        raise ValueError(
          f"Clip {path.name} joints_list does not match the task's tracked joints.\n"
          f"  clip: {clip_joints}\n  task: {list(joint_names)}"
        )
      clip_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      if fps is None:
        fps = clip_fps
      elif abs(clip_fps - fps) > 1e-3:
        raise ValueError(
          f"All squat clips must share one fps; {path.name} has {clip_fps} != {fps}."
        )
      joint_pos_parts.append(np.asarray(data["joint_positions"], dtype=np.float32))
      root_pos_parts.append(np.asarray(data["root_position"], dtype=np.float32))
      root_quat_parts.append(np.asarray(data["root_quaternion"], dtype=np.float32))

    assert fps is not None
    self.fps = fps
    joint_pos = np.concatenate(joint_pos_parts, axis=0)
    root_pos = np.concatenate(root_pos_parts, axis=0)
    # Store quaternions wxyz (mjlab/MuJoCo convention).
    root_quat = np.concatenate(root_quat_parts, axis=0)[:, [3, 0, 1, 2]]

    joint_vel = self._derivative(joint_pos, self.fps)
    root_lin_vel = self._derivative(root_pos, self.fps)

    self.joint_pos = torch.tensor(joint_pos, device=device)
    self.joint_vel = torch.tensor(joint_vel, device=device)
    self.root_pos = torch.tensor(root_pos, device=device)
    self.root_quat = torch.tensor(root_quat, device=device)
    self.root_lin_vel = torch.tensor(root_lin_vel, device=device)
    self.time_step_total = self.joint_pos.shape[0]

  @staticmethod
  def _derivative(x: np.ndarray, fps: float) -> np.ndarray:
    # The standalone clip starts and ends at the same zero-command pose with a
    # smooth stop. Use centered differences internally and make that endpoint
    # contract explicit so reset does not inject a non-zero standing velocity.
    d = np.zeros_like(x)
    d[1:-1] = 0.5 * (x[2:] - x[:-2]) * fps
    return d


class SquatMotionCommand(CommandTerm):
  cfg: SquatMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: SquatMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.joint_ids = torch.tensor(
      self.robot.find_joints(cfg.joint_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = SquatMotionLoader(cfg.motion_files, cfg.joint_names, self.device)
    self._check_playback_rate()

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_base_height"] = torch.zeros(self.num_envs, device=self.device)

  def _check_playback_rate(self) -> None:
    control_fps = 1.0 / self._env.step_dt
    if abs(self.motion.fps - control_fps) > 0.5:
      raise ValueError(
        f"Squat clip fps ({self.motion.fps:.1f}) must match the control rate "
        f"({control_fps:.1f} = 1 / (sim_dt * decimation)). Regenerate the clip "
        "with a matching --fps."
      )

  # Reference (indexed by the per-env phase clock).

  @property
  def phase(self) -> torch.Tensor:
    return self.time_steps.float() / self.motion.time_step_total

  @property
  def command(self) -> torch.Tensor:
    angle = 2.0 * torch.pi * self.phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)

  @property
  def joint_pos_ref(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel_ref(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def base_height_ref(self) -> torch.Tensor:
    return self.motion.root_pos[self.time_steps, 2]

  @property
  def root_position_ref(self) -> torch.Tensor:
    return self.motion.root_pos[self.time_steps]

  @property
  def root_lin_vel_ref(self) -> torch.Tensor:
    return self.motion.root_lin_vel[self.time_steps]

  @property
  def root_quat_ref(self) -> torch.Tensor:
    return self.motion.root_quat[self.time_steps]

  # CommandTerm hooks.

  def _update_metrics(self) -> None:
    robot_joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos_ref - robot_joint_pos, dim=-1
    )
    self.metrics["error_base_height"] = torch.abs(
      self.base_height_ref - self.robot.data.root_link_pos_w[:, 2]
    )

  def _update_command(self) -> None:
    if self.cfg.sampling_mode == "start":
      # The command manager runs after timeout resets.  Reading the reset-aware
      # episode counter avoids advancing a freshly reset env from frame 0 to 1.
      self.time_steps = self._env.episode_length_buf % self.motion.time_step_total
    else:
      self.time_steps = (self.time_steps + 1) % self.motion.time_step_total

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    else:
      self.time_steps[env_ids] = torch.randint(
        0, self.motion.time_step_total, (len(env_ids),), device=self.device
      )

    # Reference State Initialization: teleport the robot to the sampled phase.
    joint_pos = self.joint_pos_ref[env_ids].clone()
    joint_vel = self.joint_vel_ref[env_ids].clone()
    joint_pos += sample_uniform(
      *self.cfg.joint_position_range, joint_pos.shape, device=self.device
    )
    limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
    joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
    self.robot.write_joint_state_to_sim(
      joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.joint_ids
    )

    root_pos = self.motion.root_pos[self.time_steps][env_ids].clone()
    root_pos += self._env.scene.env_origins[env_ids]
    root_quat = self.root_quat_ref[env_ids].clone()
    root_lin_vel = self.motion.root_lin_vel[self.time_steps][env_ids].clone()
    root_ang_vel = torch.zeros_like(root_lin_vel)

    pose_ranges = torch.tensor(
      [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
      device=self.device,
    )
    pose_delta = sample_uniform(
      pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += pose_delta[:, 0:3]
    root_quat = quat_mul(
      quat_from_euler_xyz(pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]), root_quat
    )
    vel_ranges = torch.tensor(
      [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
      device=self.device,
    )
    vel_delta = sample_uniform(
      vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += vel_delta[:, 0:3]
    root_ang_vel += vel_delta[:, 3:6]

    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)


@dataclass(kw_only=True)
class SquatMotionCommandCfg(CommandTermCfg):
  entity_name: str
  # Exactly one standalone clip; a tuple keeps the CommandTermCfg serialization stable.
  motion_files: tuple[str, ...]
  joint_names: tuple[str, ...]
  sampling_mode: str = "start"
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)

  def build(self, env: ManagerBasedRlEnv) -> SquatMotionCommand:
    return SquatMotionCommand(self, env)
