"""Unit tests for the climb motion clip loader (DeepMimic reference)."""

import numpy as np
import pytest
import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
  ClimbMotionClip,
  clip_average_climb_rate,
)

_YAW90_XYZW = (0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0))


def _synthetic_clip(fps: float = 50.0) -> dict:
  """One 4-frame cycle + 2-frame end_hold tail; constant-rate quantities.

  Joint 0 ramps 0.1 rad/frame (vel = 0.1 * fps); the base rises 0.02 m/frame
  and the end_hold closure lands exactly one step height (0.08 m) above frame
  0. Every frame has a fixed 90° yaw so the xyzw -> wxyz conversion and the
  root-frame EE transform are both exercised.
  """
  frames = 6
  t = np.arange(frames, dtype=np.float32)
  joint_positions = np.stack([0.1 * t, np.full(frames, -0.5)], axis=1)
  root_position = np.zeros((frames, 3), dtype=np.float32)
  root_position[:, 2] = 1.0 + 0.02 * t
  root_quaternion = np.tile(
    np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1)
  )
  # EEs sit one meter along world +x from the root.
  ee_positions = root_position[:, None, :] + np.array([1.0, 0.0, 0.0])
  ee_positions = np.repeat(ee_positions, 4, axis=1)
  ee_engaged = np.ones((frames, 4), dtype=np.uint8)
  ee_engaged[1:3, 2:] = 0  # feet off during the "step".
  return {
    "joints_list": ["j0", "j1"],
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "fps": np.float32(fps),
    "ee_list": ["lh", "rh", "lf", "rf"],
    "ee_positions": ee_positions.astype(np.float32),
    "ee_engaged": ee_engaged,
    "phase_segment": np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
    "segment_names": ["hold", "end_hold"],
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.08),
  }


def test_loader_slices_one_cycle_and_derives_velocities():
  clip = ClimbMotionClip.from_dict(_synthetic_clip(), step_dt=0.02, device="cpu")

  assert clip.cycle_frames == 4  # end_hold tail excluded from the loop.
  assert clip.joint_pos.shape == (4, 2)
  # Finite differences reach into the tail, so the wrap frame has the true
  # loop velocity, not zero.
  assert torch.allclose(clip.joint_vel[:, 0], torch.full((4,), 5.0))
  assert torch.allclose(clip.joint_vel[:, 1], torch.zeros(4))
  assert torch.allclose(clip.root_lin_vel[:, 2], torch.full((4,), 1.0))
  assert torch.allclose(clip.root_ang_vel, torch.zeros(4, 3), atol=1e-5)

  # xyzw -> wxyz conversion.
  expected_wxyz = torch.tensor(
    [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=torch.float32
  )
  assert torch.allclose(clip.root_quat[0], expected_wxyz, atol=1e-6)

  # World +x offset expressed in a 90°-yawed root frame is body -y.
  assert torch.allclose(
    clip.ee_pos_r[0, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5
  )

  assert clip.ee_engaged.dtype == torch.bool
  assert clip.ee_engaged[0].tolist() == [True, True, True, True]
  assert clip.ee_engaged[1].tolist() == [True, True, False, False]


def test_loader_resamples_other_fps_onto_the_policy_grid():
  clip = ClimbMotionClip.from_dict(
    _synthetic_clip(fps=25.0), step_dt=0.02, device="cpu"
  )
  assert clip.fps == 50.0
  # Ramps stay ramps under lerp: 0.1 rad per 25fps frame = 2.5 rad/s, and
  # 0.02 m per 25fps frame = 0.5 m/s.
  assert torch.allclose(clip.joint_vel[:, 0], torch.full((clip.cycle_frames,), 2.5))
  assert torch.allclose(
    clip.root_lin_vel[:, 2], torch.full((clip.cycle_frames,), 0.5)
  )


def test_loader_rejects_missing_climb_keys():
  data = _synthetic_clip()
  del data["ee_engaged"]
  with pytest.raises(KeyError, match="ee_engaged"):
    ClimbMotionClip.from_dict(data, step_dt=0.02, device="cpu")


def test_average_climb_rate_reads_the_cycle_not_the_tail(tmp_path):
  path = tmp_path / "clip.npy"
  np.save(path, _synthetic_clip())
  # 0.08 m per 4-frame cycle at 50 fps = 1.0 m/s.
  assert clip_average_climb_rate(path) == pytest.approx(1.0)
  # Missing file falls back to the documented nominal rate (config builds
  # must not require the asset they are about to generate).
  assert clip_average_climb_rate(tmp_path / "absent.npy") == pytest.approx(
    0.06 / 4.3
  )
