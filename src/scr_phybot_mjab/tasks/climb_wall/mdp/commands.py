"""Command generators for the wall-climbing task.

The commanded-gait state machine (``ClimbGaitCommand``) was removed in the
2026-07-21 DeepMimic redesign: limb sequencing now comes from the motion
reference (see ``motion.py``); only the world-frame velocity command remains
here to drive the height/velocity progress rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "ClimbVelocityCommand",
  "ClimbVelocityCommandCfg",
)


def _sample_uniform(count: int, value_range: tuple[float, float], device: torch.device) -> torch.Tensor:
  lo, hi = value_range
  if lo == hi:
    return torch.full((count,), float(lo), device=device)
  return torch.empty(count, device=device).uniform_(lo, hi)


class ClimbVelocityCommand(CommandTerm):
  """Wall-frame linear velocity command for climbing.

  The command triplet is (x, y, along-wall): the z channel is the climb rate
  ALONG the wall surface (world vz on a vertical wall — bit-identical to the
  retired world-frame reading there, and the policy-visible observation is
  unchanged at every angle). x/y stay world axes; this task always commands
  them zero.
  """

  cfg: ClimbVelocityCommandCfg

  def __init__(self, cfg: ClimbVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["error_lin_vel_w"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vz"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_w

  def _wall_up(self) -> torch.Tensor:
    """Along-wall direction from the live wall pose (static per episode)."""
    wall: Entity = self._env.scene["wall"]
    z_axis = torch.zeros(self.num_envs, 3, device=self.device)
    z_axis[:, 2] = 1.0
    return quat_apply(wall.data.root_link_quat_w, z_axis)

  def _update_metrics(self) -> None:
    max_command_time = max(self.cfg.resampling_time_range[1], self._env.step_dt)
    max_command_step = max_command_time / self._env.step_dt
    lin_vel_w = self.robot.data.root_link_lin_vel_w
    up = self._wall_up()
    # World image of the wall-frame command: z channel along the wall.
    cmd_w = self.vel_command_w.clone()
    cmd_w[:, 2] = 0.0
    cmd_w = cmd_w + self.vel_command_w[:, 2:3] * up
    err = cmd_w - lin_vel_w
    self.metrics["error_lin_vel_w"] += torch.norm(err, dim=-1) / max_command_step
    err_along = self.vel_command_w[:, 2] - (lin_vel_w * up).sum(dim=-1)
    self.metrics["error_vz"] += torch.abs(err_along) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    ranges = self.cfg.ranges
    self.vel_command_w[env_ids, 0] = _sample_uniform(count, ranges.lin_vel_x, self.device)
    self.vel_command_w[env_ids, 1] = _sample_uniform(count, ranges.lin_vel_y, self.device)
    self.vel_command_w[env_ids, 2] = _sample_uniform(count, ranges.lin_vel_z, self.device)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class ClimbVelocityCommandCfg(CommandTermCfg):
  """Config for :class:`ClimbVelocityCommand`."""

  entity_name: str = "robot"

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    lin_vel_z: tuple[float, float] = (0.08, 0.08)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> ClimbVelocityCommand:
    return ClimbVelocityCommand(self, env)
