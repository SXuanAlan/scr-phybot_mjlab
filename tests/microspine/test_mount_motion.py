"""Unit tests for the transition task's MountMotionCommand machinery.

The subclass lives in ``tasks/climb_transition`` and must (a) rebuild the
engage tables acyclically, (b) clamp the clock instead of wrapping, and
(c) scatter only disengaged spawns — while the climb parent keeps its
circular/wrapping behavior byte-for-byte (pinned by test_climb_motion.py).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
  MountMotionCommand,
  _apply_spawn_pose_noise,
  rebuild_acyclic_schedule,
  resolve_segment_window,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
  ClimbMotionClip,
  ClimbMotionCommand,
)

_YAW90_XYZW = (0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0))


def _synthetic_clip(fps: float = 50.0) -> dict:
  """Mirror of test_climb_motion._synthetic_clip: 4-frame cycle + 2-frame
  end_hold tail, feet scheduled OFF at frames 1-2."""
  frames = 6
  t = np.arange(frames, dtype=np.float32)
  joint_positions = np.stack([0.1 * t, np.full(frames, -0.5)], axis=1)
  root_position = np.zeros((frames, 3), dtype=np.float32)
  root_position[:, 2] = 1.0 + 0.02 * t
  root_quaternion = np.tile(np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1))
  ee_positions = root_position[:, None, :] + np.array([1.0, 0.0, 0.0])
  ee_positions = np.repeat(ee_positions, 4, axis=1)
  ee_engaged = np.ones((frames, 4), dtype=np.uint8)
  ee_engaged[1:3, 2:] = 0
  return {
    "joints_list": ["j0", "j1"],
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "fps": np.float32(fps),
    "ee_list": ["lh", "rh", "lf", "rf"],
    "ee_positions": ee_positions.astype(np.float32),
    "ee_anchor_positions": ee_positions.astype(np.float32),
    "ee_engaged": ee_engaged,
    "phase_segment": np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
    "segment_names": ["hold", "end_hold"],
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.08),
  }


def test_rebuild_replaces_circular_tables_with_acyclic_ones():
  """No phantom wrap edges at either end of a one-shot clip.

  The circular numbers on this exact clip are pinned by
  ``test_climb_motion.test_loader_derives_schedule_edge_countdowns`` — the
  values here differ exactly where the wrap used to manufacture edges.
  """
  clip = ClimbMotionClip.from_dict(_synthetic_clip(), step_dt=0.02, device="cpu")
  rebuild_acyclic_schedule(clip)

  # Hands never flip: both lanes read the "never" sentinel everywhere.
  assert torch.allclose(clip.ee_edge_times[:, 0], torch.ones(4, 2))
  assert torch.allclose(clip.ee_edge_times[:, 1], torch.ones(4, 2))
  # Feet: after the LAST real edge every lane reads 1.0 instead of
  # counting down to a wrap phantom.
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_edge_times[:, foot, 0], torch.tensor([0.25, 1.0, 1.0, 1.0])
    )
    assert torch.allclose(
      clip.ee_edge_times[:, foot, 1], torch.tensor([0.75, 0.5, 0.25, 1.0])
    )
  # Edge-free hand rows read exactly 1.0 (the both-sentinel case must not
  # leak (cycle-1)/cycle = 0.75).
  assert torch.allclose(clip.ee_edge_distance[:, 0], torch.ones(4))
  assert torch.allclose(clip.ee_edge_distance[:, 1], torch.ones(4))
  # Interior edges are identical under both computations.
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_edge_distance[:, foot], torch.tensor([0.25, 0.0, 0.25, 0.0])
    )
  # Runs do not fuse across the seam: the feet's frame-0 ON run is a single
  # frame (circular reads 0.5 by joining frames 3 and 0).
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_run_time[:, foot], torch.tensor([0.25, 0.5, 0.5, 0.25])
    )


def _make_update_stub(cls, reseat_hold: torch.Tensor):
  cmd = object.__new__(cls)
  cmd.clip = SimpleNamespace(cycle_frames=4, step_height=0.08)
  cmd.time_steps = torch.tensor([2, 3], dtype=torch.long)
  cmd.root_offset = torch.zeros(2, 3)
  cmd._affordance = lambda: SimpleNamespace(reseat_hold=reseat_hold)
  return cmd


def test_mount_command_clamps_where_the_parent_wraps():
  cmd = _make_update_stub(MountMotionCommand, torch.tensor([False, False]))
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 3]
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 3]  # clamped, no wrap
  assert torch.allclose(cmd.root_offset, torch.zeros(2, 3))  # no step bump

  # The R4 reseat freeze still wins over the clamp advance.
  held = _make_update_stub(MountMotionCommand, torch.tensor([True, False]))
  held._update_command()
  assert held.time_steps.tolist() == [2, 3]

  # Parent regression: the cycle command still wraps and tiles upward.
  parent = _make_update_stub(ClimbMotionCommand, torch.tensor([False, False]))
  parent._update_command()
  assert parent.time_steps.tolist() == [3, 0]
  assert parent.root_offset[1, 2].item() == pytest.approx(0.08)


def test_spawn_pose_noise_only_touches_disengaged_rows():
  pos = torch.tensor([[1.27, 0.0, 0.726], [1.327, 0.0, 1.0]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  disengaged = torch.tensor([True, False])

  out_pos, out_quat = _apply_spawn_pose_noise(
    pos, quat, disengaged, xy_noise=(0.05, 0.03), yaw_noise=0.1
  )
  # Engaged row: exact clip pose (welded pads cannot move with the base).
  assert torch.equal(out_pos[1], pos[1])
  assert torch.equal(out_quat[1], quat[1])
  # Disengaged row: bounded offsets, pure z rotation, unit quaternion.
  assert abs(out_pos[0, 0] - pos[0, 0]) <= 0.05 + 1e-6
  assert abs(out_pos[0, 1] - pos[0, 1]) <= 0.03 + 1e-6
  assert out_pos[0, 2] == pos[0, 2]
  assert abs(out_quat[0, 1]) < 1e-6 and abs(out_quat[0, 2]) < 1e-6
  assert out_quat[0].norm().item() == pytest.approx(1.0, abs=1e-5)
  assert abs(out_quat[0, 3]) <= np.sin(0.05) + 1e-6

  # Zero ranges: identity fast path, inputs returned untouched.
  same_pos, same_quat = _apply_spawn_pose_noise(
    pos, quat, disengaged, xy_noise=(0.0, 0.0), yaw_noise=0.0
  )
  assert same_pos is pos and same_quat is quat


def test_rsi_bias_window_draw():
  cmd = object.__new__(MountMotionCommand)
  cmd.cfg = SimpleNamespace(spawn_early_fraction=0.5)
  cmd.clip = SimpleNamespace(cycle_frames=710)
  cmd._env = SimpleNamespace(device="cpu")  # .device property reads this
  cmd._bias_window = (280, 380)  # e.g. the place -> set latch stretch
  torch.manual_seed(0)
  frames = cmd._draw_rsi_frames(4096)
  inside = ((frames >= 280) & (frames < 380)).float().mean().item()
  # 50% forced into the window + the uniform tail's own ~14% share.
  assert 0.5 < inside < 0.68, inside
  assert int(frames.max()) < 710 and int(frames.min()) >= 0

  # Bias off: plain uniform.
  cmd.cfg = SimpleNamespace(spawn_early_fraction=0.0)
  frames = cmd._draw_rsi_frames(4096)
  assert ((frames >= 280) & (frames < 380)).float().mean().item() < 0.25


def test_resolve_segment_window(tmp_path):
  labels = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.uint8)
  data = {
    "segment_names": ["stand", "place", "set", "end_hold"],
    "phase_segment": labels,
    "cycle_index": np.zeros(8, dtype=np.uint16),
  }
  path = tmp_path / "clip.npy"
  np.save(path, data)
  # Cycle = 6 frames (end_hold tail excluded). place+set span frames 2-5.
  lo, hi = resolve_segment_window(str(path), ("place", "set"))
  assert lo == pytest.approx(2 / 6) and hi == pytest.approx(6 / 6)
  # Unknown segment or missing file degrade to None (caller falls back).
  assert resolve_segment_window(str(path), ("nope",)) is None
  assert resolve_segment_window(str(tmp_path / "absent.npy"), ("place",)) is None
