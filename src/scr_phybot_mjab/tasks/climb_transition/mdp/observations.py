"""Transition-only observation lanes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import _motion

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "motion_anchor_relative",
  "root_pose_world",
  "wall_base_position",
)


def root_pose_world(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
) -> torch.Tensor:
  """Robot root pose in the ENV-LOCAL world frame, (E, 7) = pos(3) + quat(4).

  v62 contract extension (user call, 08-19 night): the policy reads its
  absolute stance in the task frame instead of inferring it from proprio
  history. Env-grid origins are subtracted — leaking the env's grid slot
  would be nonsense input. Deploy: base state estimation (odometry/mocap)
  plus the calibrated task origin.
  """
  robot = env.scene[entity_name]
  pos = robot.data.root_link_pos_w - env.scene.env_origins
  quat = robot.data.root_link_quat_w
  return torch.cat([pos, quat], dim=-1)


def wall_base_position(
  env: ManagerBasedRlEnv,
  entity_name: str = "wall",
) -> torch.Tensor:
  """Wall-entity root position in the ENV-LOCAL world frame, (E, 3).

  The wall base ("qiang gen") the mount is built against. Constant per
  episode at 90 deg; read live so tilted band variants stay honest.
  Deploy: the calibrated wall pose.
  """
  wall = env.scene[entity_name]
  return wall.data.root_link_pos_w - env.scene.env_origins


def motion_anchor_relative(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
) -> torch.Tensor:
  """Per-EE pad -> scheduled-anchor vectors in the BASE frame, (E, 12).

  v3 contract extension (user call, 08-18): the policy was placing welds
  blind — motion_command carries engaged bits and edge countdowns but
  never WHERE the anchors are, so the standing-spawn scatter turned the
  memorized map wrong by the accumulated offset and first touches were
  probing moves. The mount anchors are task-frame constants at deploy
  (C++ computes them from the calibrated wall pose + FK), so this lane
  needs no exteroception. Order: (LH, RH, LF, RF) x (x, y, z), meters.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  executor = affordance._get_executor()
  robot = env.scene["robot"]
  rel_w = motion.ref_anchor_pos_w - executor.pad_center_world()  # (E, 4, 3)
  quat = robot.data.root_link_quat_w
  e = rel_w.shape[0]
  rel_b = quat_apply_inverse(
    quat.unsqueeze(1).expand(-1, rel_w.shape[1], -1).reshape(-1, 4),
    rel_w.reshape(-1, 3),
  ).reshape(e, -1)
  return rel_b
