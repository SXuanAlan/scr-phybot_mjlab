"""Loading of cyclic climbing reference clips onto the policy-step grid.

Clips use the AMP/motion ``.npy`` dictionary schema with climb schedules
and, when present, passive gimbal coordinates. One loopable regrip cycle is
kept, with finite-difference velocities and the schedule tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .schedule import schedule_tables

__all__ = ("ClimbMotionClip", "clip_average_climb_rate", "lerp_frames", "np_quat_mul")

# Nominal cycle climb rate (m/s) used only when the clip file is absent at
# config-build time (before the reference generator has written it). Env
# construction still fails loudly without the file — this fallback exists so
# building the *config* (task registration, the gen script itself) does not
# require the asset it is about to generate. 0.06 m / 4.3 s cycle.
_FALLBACK_CLIMB_RATE = 0.06 / 4.3


def np_quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of wxyz quaternion arrays (..., 4)."""
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def _np_quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vectors ``v`` (T, N, 3) by the inverse of wxyz quats ``q`` (T, 4)."""
  w = q[:, None, 0:1]
  im = -q[:, None, 1:4]  # conjugate
  t = 2.0 * np.cross(im, v)
  return v + w * t + np.cross(im, t)


def lerp_frames(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
  """Linear interpolation of a (T, ...) array at fractional frame indices."""
  last = arr.shape[0] - 1
  i0 = np.clip(np.floor(pos).astype(np.int64), 0, last)
  i1 = np.minimum(i0 + 1, last)
  w = (pos - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
  return (1.0 - w) * arr[i0] + w * arr[i1]


def _nearest_frames(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
  last = arr.shape[0] - 1
  return arr[np.clip(np.round(pos).astype(np.int64), 0, last)]


def _cycle_frames(data: dict) -> int:
  """Frames of one loopable cycle: cycle 0 without the ``end_hold`` tail.

  The cycle generator appends ``end_hold`` (== next cycle's ``hold_start``
  shifted one step height) only after the last cycle, so cycle 0 minus
  ``end_hold`` is the loop period regardless of ``--cycles``.
  """
  names = list(data["segment_names"])
  end_id = names.index("end_hold")
  labels = np.asarray(data["phase_segment"])
  cycles = np.asarray(data["cycle_index"])
  return int(np.sum((cycles == 0) & (labels != end_id)))


def clip_average_climb_rate(motion_file: str | Path, fallback: float = _FALLBACK_CLIMB_RATE) -> float:
  """Average upward base speed of one reference cycle (m/s).

  Used at config-build time to point the climb velocity command at the rate
  the clip actually delivers, so the height-progress normalization and the
  commanded-velocity observation agree with the motion being imitated.
  """
  path = Path(motion_file)
  if not path.exists():
    return float(fallback)
  data = np.load(path, allow_pickle=True).item()
  cycle = _cycle_frames(data)
  return float(data["step_height"]) * float(data["fps"]) / cycle


@dataclass
class ClimbMotionClip:
  """One loopable regrip cycle on the policy-step time grid (torch tensors)."""

  joint_names: tuple[str, ...]
  ee_names: tuple[str, ...]
  fps: float
  step_height: float
  cycle_frames: int
  joint_pos: torch.Tensor  # (C, J)
  joint_vel: torch.Tensor  # (C, J)
  root_pos: torch.Tensor  # (C, 3) world, as generated (env origin added later)
  root_quat: torch.Tensor  # (C, 4) wxyz
  root_lin_vel: torch.Tensor  # (C, 3)
  root_ang_vel: torch.Tensor  # (C, 3)
  ee_pos_r: torch.Tensor  # (C, H, 3) EE body origins in the root frame
  ee_engaged: torch.Tensor  # (C, H) bool
  ee_edge_times: torch.Tensor  # (C, H, 2) countdown to next release/regrip
  ee_edge_distance: torch.Tensor  # (C, H) |time| to the NEAREST edge
  ee_run_time: torch.Tensor  # (C, H) length of the containing ON/OFF run
  toe_braced: torch.Tensor  # (C, 2) bool: foot-toe wall brace schedule
  ee_anchor_pos: torch.Tensor  # (C, H, 3) scheduled anchor, clip world frame
  passive_joint_names: tuple[str, ...] = ()
  passive_joint_pos: torch.Tensor | None = None  # FK/RSI only; never action targets
  passive_joint_vel: torch.Tensor | None = None
  # Actual frame C, following the playable [0, C) cycle. FK differences must
  # include its climb increment rather than wrapping back to the old height.
  tail_joint_pos: torch.Tensor | None = None
  tail_passive_joint_pos: torch.Tensor | None = None
  tail_root_pos: torch.Tensor | None = None
  tail_root_quat: torch.Tensor | None = None

  @classmethod
  def from_file(cls, motion_file: str | Path, step_dt: float, device: str) -> ClimbMotionClip:
    data = np.load(Path(motion_file), allow_pickle=True).item()
    return cls.from_dict(data, step_dt=step_dt, device=device)

  @classmethod
  def from_dict(cls, data: dict, step_dt: float, device: str) -> ClimbMotionClip:
    required = (
      "joints_list",
      "joint_positions",
      "root_position",
      "root_quaternion",
      "fps",
      "ee_list",
      "ee_positions",
      "ee_engaged",
      "phase_segment",
      "segment_names",
      "cycle_index",
      "step_height",
    )
    missing = [key for key in required if key not in data]
    if missing:
      raise KeyError(f"Climb motion clip is missing keys: {missing}")

    src_fps = float(data["fps"])
    target_fps = 1.0 / step_dt
    joint_pos = np.asarray(data["joint_positions"], dtype=np.float64)
    passive_names = tuple(data.get("passive_joint_names", ()))
    passive_pos = np.asarray(data.get("passive_joint_positions", np.zeros((len(joint_pos), 0))), dtype=np.float64)
    if passive_pos.shape != (len(joint_pos), len(passive_names)):
      raise ValueError("Passive reference joint names/frames do not match")
    if set(passive_names) & set(data["joints_list"]):
      raise ValueError("A reference joint cannot be both actuated and passive")
    root_pos = np.asarray(data["root_position"], dtype=np.float64)
    # The AMP/motion schema stores xyzw; mjlab is wxyz throughout.
    root_quat = np.asarray(data["root_quaternion"], dtype=np.float64)[:, [3, 0, 1, 2]]
    ee_pos = np.asarray(data["ee_positions"], dtype=np.float64)
    engaged = np.asarray(data["ee_engaged"]) != 0
    # Foot-toe wall-brace schedule; clips without one never schedule a brace.
    toe_braced = (
      np.asarray(data["toe_braced"]) != 0 if "toe_braced" in data else np.zeros((engaged.shape[0], 2), dtype=bool)
    )
    # Scheduled anchor positions (clip world frame): the weld target while
    # engaged, the upcoming landing target while swinging. Consumed by the
    # landing-accuracy reward.
    ee_anchor = np.asarray(data["ee_anchor_positions"], dtype=np.float64)
    labels = np.asarray(data["phase_segment"])
    cycles = np.asarray(data["cycle_index"])

    if abs(src_fps - target_fps) > 1.0e-6:
      # Resample onto the policy-step grid. Adjacent frames are close, so
      # lerp (+ renormalize for quats) is adequate; flags use nearest.
      duration = (joint_pos.shape[0] - 1) / src_fps
      n_new = int(np.floor(duration * target_fps)) + 1
      pos = np.arange(n_new) / target_fps * src_fps
      joint_pos = lerp_frames(joint_pos, pos)
      passive_pos = lerp_frames(passive_pos, pos)
      root_pos = lerp_frames(root_pos, pos)
      root_quat = lerp_frames(root_quat, pos)
      root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
      ee_pos = lerp_frames(ee_pos, pos)
      engaged = _nearest_frames(engaged, pos)
      toe_braced = _nearest_frames(toe_braced, pos)
      ee_anchor = lerp_frames(ee_anchor, pos)
      labels = _nearest_frames(labels, pos)
      cycles = _nearest_frames(cycles, pos)

    cycle = _cycle_frames(
      {
        "segment_names": data["segment_names"],
        "phase_segment": labels,
        "cycle_index": cycles,
      }
    )
    total = joint_pos.shape[0]
    if cycle < 2:
      raise ValueError(f"Climb motion cycle has {cycle} frames; need >= 2.")
    if cycle >= total:
      raise ValueError(
        f"Climb motion clip has no frame beyond the cycle ({cycle} of "
        f"{total}); finite-difference velocities need the end_hold tail."
      )

    # Forward differences over [0, C) use frame C — present by construction
    # (end_hold, or the next cycle), and equal to frame 0 shifted one step
    # height, so the wrap velocity is the true loop velocity.
    joint_vel = (joint_pos[1 : cycle + 1] - joint_pos[:cycle]) * target_fps
    root_lin_vel = (root_pos[1 : cycle + 1] - root_pos[:cycle]) * target_fps
    q_conj = root_quat[:cycle] * np.array([1.0, -1.0, -1.0, -1.0])
    q_rel = np_quat_mul(root_quat[1 : cycle + 1], q_conj)
    q_rel *= np.where(q_rel[:, 0:1] >= 0.0, 1.0, -1.0)
    vec_norm = np.linalg.norm(q_rel[:, 1:4], axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, q_rel[:, 0])
    axis = q_rel[:, 1:4] / np.maximum(vec_norm, 1.0e-12)[:, None]
    root_ang_vel = axis * (angle * target_fps)[:, None]

    ee_pos_r = _np_quat_rotate_inverse(root_quat[:cycle], ee_pos[:cycle] - root_pos[:cycle, None, :])
    engaged_cycle = engaged[:cycle]
    edge_times, edge_distance, run_time = schedule_tables(engaged_cycle)

    def _t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
      return torch.tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

    return cls(
      joint_names=tuple(data["joints_list"]),
      ee_names=tuple(data["ee_list"]),
      fps=target_fps,
      step_height=float(data["step_height"]),
      cycle_frames=cycle,
      joint_pos=_t(joint_pos[:cycle]),
      joint_vel=_t(joint_vel),
      root_pos=_t(root_pos[:cycle]),
      root_quat=_t(root_quat[:cycle]),
      root_lin_vel=_t(root_lin_vel),
      root_ang_vel=_t(root_ang_vel),
      ee_pos_r=_t(ee_pos_r),
      ee_engaged=_t(engaged_cycle, dtype=torch.bool),
      ee_edge_times=_t(edge_times),
      ee_edge_distance=_t(edge_distance),
      ee_run_time=_t(run_time),
      toe_braced=_t(toe_braced[:cycle], dtype=torch.bool),
      ee_anchor_pos=_t(ee_anchor[:cycle]),
      passive_joint_names=passive_names,
      passive_joint_pos=_t(passive_pos[:cycle]),
      passive_joint_vel=_t((passive_pos[1 : cycle + 1] - passive_pos[:cycle]) * target_fps),
      tail_joint_pos=_t(joint_pos[cycle : cycle + 1]),
      tail_passive_joint_pos=_t(passive_pos[cycle : cycle + 1]),
      tail_root_pos=_t(root_pos[cycle : cycle + 1]),
      tail_root_quat=_t(root_quat[cycle : cycle + 1]),
    )
