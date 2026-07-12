"""Reward terms for the squat motion-tracking task.

Only the lower body and the base are tracked; the upper body (arms/waist) is left
to the policy (regularized by the generic energy / action-rate / joint-limit terms
in the env config), matching the DeepMimic-style "track legs, free arms" design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .motion_command import SquatMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Track the reference joint angles (optionally a subset, e.g. the legs)."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  if joint_names is None:
    ref_ids = slice(None)
    robot_ids = command.joint_ids
  else:
    ref_ids = torch.tensor(
      [command.cfg.joint_names.index(n) for n in joint_names],
      dtype=torch.long,
      device=env.device,
    )
    robot_ids = command.joint_ids[ref_ids]
  error = torch.sum(
    torch.square(command.joint_pos_ref[:, ref_ids] - command.robot.data.joint_pos[:, robot_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def track_base_height(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the reference base height (the squat's primary degree of freedom)."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  height = command.robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  error = torch.square(command.base_height_ref - height)
  return torch.exp(-error / std**2)


def track_base_orientation(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the reference torso tilt (forward lean), yaw-invariant.

  Compares gravity expressed in the reference vs. robot base frames, so it rewards
  matching the roll+pitch lean while ignoring absolute heading -- a squat can face
  any yaw. This is what lets RSI randomize yaw (like walk) without collapsing the
  reward: yaw-rotating the base leaves the projected gravity unchanged.
  """
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  gravity_w = command.robot.data.gravity_vec_w
  ref_gravity_b = quat_apply_inverse(command.root_quat_ref, gravity_w)
  robot_gravity_b = quat_apply_inverse(command.robot.data.root_link_quat_w, gravity_w)
  error = torch.sum(torch.square(ref_gravity_b - robot_gravity_b), dim=-1)
  return torch.exp(-error / std**2)
