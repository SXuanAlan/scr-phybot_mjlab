"""Transition observation lanes.

Actor lanes added here (all deploy-legal: clip lookups, FK off the base
state estimate, and the calibrated wall pose — no force sensing):

* BeyondMimic reference lanes — ``motion_ref_joint_pos_rel`` /
  ``motion_ref_joint_vel`` (the clip's joint targets) and
  ``motion_anchor_pos_b`` / ``motion_anchor_ori_b`` (the reference root pose
  expressed in the robot root frame; mjlab ``tasks/tracking`` verbatim);
* ``motion_pad_center_relative`` — reference pad FACE centers minus the
  robot's, in the base frame (the swing-phase trajectory target; the same
  point the pad-face reward and termination judge);
* ``motion_anchor_relative`` — pad face -> scheduled anchor (the landing
  target while a pad swings);
* ``microspine_pad_center_world`` — the four pad faces in the env-local
  task frame;
* ``root_pose_world`` and ``wall_face_corners`` — the absolute stance and
  the calibrated climbing face (four corners, env-local).

Critic-only: ``robot_body_pos_b`` / ``robot_body_ori_b`` (mjlab tracking
critic lanes) and ``microspine_unit_preload_fraction`` (the engage gate's
preload progress).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from scr_phybot_mjab.assets.wall.wall_constants import WALL_HALF_SIZE

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "microspine_pad_center_world",
  "microspine_unit_preload_fraction",
  "motion_anchor_ori_b",
  "motion_anchor_pos_b",
  "motion_anchor_relative",
  "motion_pad_center_relative",
  "motion_ref_joint_pos_rel",
  "motion_ref_joint_vel",
  "robot_body_ori_b",
  "robot_body_pos_b",
  "root_pose_world",
  "wall_face_corners",
)

# Climbing face = the wall box's local -x face; corners in the wall body
# frame (== the mocap root frame: the box is centred on it), order
# bottom-left, bottom-right, top-left, top-right (y = lateral, z = along the
# wall). Read from the asset constants, NOT from the compiled collision
# geom: the climb scene's ``wall_geom`` is an analytic PLANE whose size
# lanes are visual-only (half-height reads 0.1 there).
_HX, _HY, _HZ = (float(v) for v in WALL_HALF_SIZE)
_WALL_FACE_CORNERS_LOCAL: tuple[tuple[float, float, float], ...] = (
  (-_HX, -_HY, -_HZ),
  (-_HX, +_HY, -_HZ),
  (-_HX, -_HY, +_HZ),
  (-_HX, +_HY, +_HZ),
)


def _to_base_frame(quat_w: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
  """Rotate world vectors (E, H, 3) into the frame of ``quat_w`` (E, 4),
  flattened to (E, 3H)."""
  num_envs, num_vecs = vec_w.shape[0], vec_w.shape[1]
  return quat_apply_inverse(
    quat_w.unsqueeze(1).expand(-1, num_vecs, -1).reshape(-1, 4),
    vec_w.reshape(-1, 3),
  ).reshape(num_envs, -1)


# -- absolute stance / scene ------------------------------------------------


def root_pose_world(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """Robot root pose in the ENV-LOCAL world frame, (E, 7) = pos + quat.
  Deploy: base state estimation plus the calibrated task origin."""
  robot = env.scene[entity_name]
  pos = robot.data.root_link_pos_w - env.scene.env_origins
  return torch.cat([pos, robot.data.root_link_quat_w], dim=-1)


def wall_face_corners(env: ManagerBasedRlEnv, entity_name: str = "wall") -> torch.Tensor:
  """The climbing face's four corners in the ENV-LOCAL world frame, (E, 12)
  = (BL, BR, TL, TR) x (x, y, z).

  Replaces the wall root position (a single constant at a pinned angle):
  the corners carry the face plane AND its extent, and stay honest under
  any wall tilt because the live mocap pose is read. Deploy: the
  calibrated wall pose + the panel dimensions."""
  wall = env.scene[entity_name]
  pos = wall.data.root_link_pos_w - env.scene.env_origins  # (E, 3)
  quat = wall.data.root_link_quat_w  # (E, 4) wxyz
  num_envs = pos.shape[0]
  local = torch.tensor(_WALL_FACE_CORNERS_LOCAL, device=pos.device, dtype=pos.dtype)
  rotated = quat_apply(
    quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
    local.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3),
  ).reshape(num_envs, 4, 3)
  return (rotated + pos.unsqueeze(1)).reshape(num_envs, -1)


def microspine_pad_center_world(env: ManagerBasedRlEnv, command_name: str = "microspine_affordance") -> torch.Tensor:
  """The four pad FACE centers in the ENV-LOCAL world frame, (E, 12), order
  (LH, RH, LF, RF) x (x, y, z) — the contact law's own point. Deploy: FK
  off the base state estimate."""
  affordance = env.command_manager.get_term(command_name)
  centers = affordance._get_executor().pad_center_world()  # (E, 4, 3)
  local = centers - env.scene.env_origins.unsqueeze(1)
  return local.reshape(centers.shape[0], -1)


# -- reference (BeyondMimic) ------------------------------------------------


def motion_ref_joint_pos_rel(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Reference joint positions minus the default pose, (E, 21) — the same
  basis as ``joint_pos_rel`` so the residual is a subtraction away."""
  return env.command_manager.get_term(command_name).ref_joint_pos_rel


def motion_ref_joint_vel(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Reference joint velocities (clip finite differences), (E, 21)."""
  return env.command_manager.get_term(command_name).ref_joint_vel


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Reference root position in the robot ROOT frame, (E, 3)
  (mjlab tracking ``motion_anchor_pos_b`` verbatim)."""
  command = env.command_manager.get_term(command_name)
  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return pos.view(pos.shape[0], -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Reference root orientation relative to the robot root, (E, 6) = the
  first two columns of the relative rotation matrix (mjlab verbatim)."""
  command = env.command_manager.get_term(command_name)
  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_pad_center_relative(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Reference pad FACE center minus the robot's, per EE, in the BASE frame,
  (E, 12), order (LH, RH, LF, RF) x (x, y, z). While a pad is scheduled
  engaged this equals the anchor vector; while it swings it is the
  trajectory target the pad-face kernel pays on."""
  motion = env.command_manager.get_term(command_name)
  rel_w = motion.ref_pad_center_w - motion.robot_pad_center_w  # (E, 4, 3)
  return _to_base_frame(motion.robot_anchor_quat_w, rel_w)


def motion_anchor_relative(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Per-EE pad-center -> scheduled-anchor vectors in the BASE frame,
  (E, 12), order (LH, RH, LF, RF) x (x, y, z). The mount anchors are
  task-frame constants at deploy (computed from the calibrated wall pose),
  so the lane needs no exteroception."""
  motion = env.command_manager.get_term(command_name)
  rel_w = motion.ref_anchor_pos_w - motion.robot_pad_center_w  # (E, 4, 3)
  return _to_base_frame(motion.robot_anchor_quat_w, rel_w)


# -- critic only ----------------------------------------------------------------


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Tracked body positions in the robot ROOT frame, (E, 3B) (mjlab
  tracking critic lane verbatim; B = the tracked set)."""
  command = env.command_manager.get_term(command_name)
  num_bodies = len(command.tracking_body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return pos_b.reshape(pos_b.shape[0], -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str = "climb_motion") -> torch.Tensor:
  """Tracked body orientations relative to the robot root, (E, 6B) — two
  rotation-matrix columns per body (mjlab verbatim)."""
  command = env.command_manager.get_term(command_name)
  num_bodies = len(command.tracking_body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def unit_preload_fraction(affordance) -> torch.Tensor:
  """Engage-gate progress per pad in [0, 1], (E, 4).

  A pad may engage only once every active unit carries
  ``engage_unit_preload_n``, so the weakest unit's compression over that
  threshold is the binding quantity.
  """
  threshold = max(float(affordance._mcfg.engage_unit_preload_n), 1.0e-6)
  return (affordance.unit_preload_n.min(dim=-1).values / threshold).clamp(0.0, 1.0)


def microspine_unit_preload_fraction(
  env: ManagerBasedRlEnv, command_name: str = "microspine_affordance"
) -> torch.Tensor:
  """Per-pad engage-gate preload progress in [0, 1], (E, 4). Critic only."""
  return unit_preload_fraction(env.command_manager.get_term(command_name))
