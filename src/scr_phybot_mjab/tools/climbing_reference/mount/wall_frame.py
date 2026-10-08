"""The tilted wall's frame: rotation about its base line and canonical-frame storage."""

from __future__ import annotations

import mujoco
import numpy as np

from scr_phybot_mjab.tools.climbing_reference import kinematics as kin


class WallFrame:
  """Tilted-wall geometry: Rot_y(90 - angle) about the base line (x=1.5, z=0).

  Matches the runtime convention exactly (curriculums.wall_pose_for_angle):
  along-wall up u = (cos a, 0, sin a), outward normal n = (-sin a, 0, cos a),
  quat (wxyz) = (cos(t/2), 0, sin(t/2), 0) with t = radians(90 - a). The
  generator SOLVES in the native world (real gravity, horizontal floor,
  tilted wall) and DE-ROTATES all world-frame outputs to the canonical
  vertical-wall frame at write time, so the runtime's ``_to_world_*``
  machinery re-poses them at the sampled angle.
  """

  def __init__(self, angle_deg: float) -> None:
    self.angle_deg = float(angle_deg)
    self.pivot = np.array([kin.WALL_SURFACE_X, 0.0, 0.0])
    theta = np.radians(90.0 - self.angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    self.R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    self.outward = self.R @ np.array([-1.0, 0.0, 0.0])
    self.up = self.R @ np.array([0.0, 0.0, 1.0])
    half = 0.5 * theta
    self.quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    self.quat_inv = self.quat * np.array([1.0, -1.0, -1.0, -1.0])

  def rotate_point(self, p: np.ndarray) -> np.ndarray:
    return self.pivot + self.R @ (np.asarray(p, dtype=np.float64) - self.pivot)

  def rotate_mat(self, rot: np.ndarray) -> np.ndarray:
    return self.R @ rot

  def rotate_root_qpos(self, qpos: np.ndarray) -> np.ndarray:
    """Rotate a free-root qpos (pos + wxyz quat) about the base line."""
    out = qpos.copy()
    out[0:3] = self.rotate_point(qpos[0:3])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, self.quat, qpos[3:7])
    out[3:7] = rotated
    return out

  def along_wall(self, p: np.ndarray) -> float:
    """Along-wall coordinate (the canonical-frame z) of a native point."""
    return float(self.up @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def wall_distance(self, p: np.ndarray) -> float:
    """Signed clearance to the wall surface (positive on the robot side)."""
    return float(self.outward @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def derotate_points(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 3) array."""
    shape = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float64)
    flat = (flat - self.pivot) @ self.R + self.pivot
    return flat.reshape(shape).astype(arr.dtype)

  def derotate_quats_xyzw(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 4) xyzw quaternion array."""
    shape = arr.shape
    flat = arr.reshape(-1, 4).astype(np.float64)
    out = np.empty_like(flat)
    q = np.zeros(4)
    for i in range(flat.shape[0]):
      wxyz = np.array([flat[i, 3], flat[i, 0], flat[i, 1], flat[i, 2]])
      mujoco.mju_mulQuat(q, self.quat_inv, wxyz)
      out[i] = (q[1], q[2], q[3], q[0])
    return out.reshape(shape).astype(arr.dtype)
