"""Body reference tables and BeyondMimic pose alignment for cyclic climbing.

The final FK sample is the actual frame after the loop. It must include the
climb-height increment; differentiating a zero-height wrap invents a velocity
spike. Positions refer to body origins, matching EntityData.body_link_*.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat


def build_body_pose_tables(
  model, base_qpos, root_qadr, joint_qadr, body_ids, root_pos, root_quat, joint_pos, passive_qadr, passive_pos
):
  """FK of all supplied frames, including the one-frame boundary tail."""
  data = mujoco.MjData(model)
  positions = np.empty((len(root_pos), len(body_ids), 3), dtype=np.float64)
  orientations = np.empty((len(root_pos), len(body_ids), 4), dtype=np.float64)
  for frame in range(len(root_pos)):
    data.qpos[:] = base_qpos
    data.qpos[root_qadr[:3]] = root_pos[frame]
    data.qpos[root_qadr[3:7]] = root_quat[frame]
    data.qpos[joint_qadr] = joint_pos[frame]
    data.qpos[passive_qadr] = passive_pos[frame]
    mujoco.mj_kinematics(model, data)
    positions[frame] = data.xpos[body_ids]
    orientations[frame] = data.xquat[body_ids]
  return positions, orientations


def body_velocities_from_poses(positions, orientations, fps):
  """World-frame forward differences; C+1 input poses produce C velocities."""
  linear = (positions[1:] - positions[:-1]) * fps
  conjugate = orientations[:-1] * orientations.new_tensor([1.0, -1.0, -1.0, -1.0])
  delta = quat_mul(orientations[1:], conjugate)
  delta = torch.where(delta[..., :1] < 0, -delta, delta)
  norm = delta[..., 1:].norm(dim=-1, keepdim=True)
  angle = 2.0 * torch.atan2(norm, delta[..., :1])
  angular = delta[..., 1:] * (angle / norm.clamp_min(1.0e-12)) * fps
  return linear, angular


def align_body_reference(body_pos, body_quat, reference_root_pos, reference_root_quat, robot_root_pos, robot_root_quat):
  """Official BeyondMimic XY/yaw alignment, retaining reference world height.

  Apply after the reference has been rotated onto the actual wall. The
  separate climbing-height reward uses along-wall altitude at every angle.
  Velocities are intentionally not reanchored or differentiated through this
  alignment: their reference is the recorded world motion.
  """
  delta_quat = yaw_quat(quat_mul(robot_root_quat, quat_inv(reference_root_quat)))
  delta_quat = delta_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
  anchor = torch.cat((robot_root_pos[:, :2], reference_root_pos[:, 2:3]), dim=-1)
  pos = anchor.unsqueeze(1) + quat_apply(delta_quat, body_pos - reference_root_pos.unsqueeze(1))
  return pos, quat_mul(delta_quat, body_quat)
