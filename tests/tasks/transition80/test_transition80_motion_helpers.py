"""Acyclic schedule tables and segment windows of the one-shot mount clip."""

import numpy as np
import pytest
import torch

from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import rebuild_acyclic_schedule, resolve_segment_window
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionClip

_YAW90_XYZW = (0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0))


def _synthetic_clip(fps: float = 50.0) -> dict:
  """4-frame cycle plus a 2-frame end_hold tail; feet scheduled OFF at frames 1-2."""
  frames = 6
  t = np.arange(frames, dtype=np.float32)
  root_position = np.zeros((frames, 3), dtype=np.float32)
  root_position[:, 2] = 1.0 + 0.02 * t
  ee_positions = np.repeat(root_position[:, None, :] + np.array([1.0, 0.0, 0.0]), 4, axis=1)
  ee_engaged = np.ones((frames, 4), dtype=np.uint8)
  ee_engaged[1:3, 2:] = 0
  return {
    "joints_list": ["j0", "j1"],
    "joint_positions": np.stack([0.1 * t, np.full(frames, -0.5)], axis=1),
    "root_position": root_position,
    "root_quaternion": np.tile(np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1)),
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
  """A one-shot clip has no wrap edge, so nothing counts down to a phantom one."""
  clip = ClimbMotionClip.from_dict(_synthetic_clip(), step_dt=0.02, device="cpu")
  rebuild_acyclic_schedule(clip)

  # Hands never flip: both lanes read the "never" sentinel everywhere.
  assert torch.allclose(clip.ee_edge_times[:, 0], torch.ones(4, 2))
  assert torch.allclose(clip.ee_edge_times[:, 1], torch.ones(4, 2))
  # Feet: after the last real edge every lane reads 1.0.
  for foot in (2, 3):
    assert torch.allclose(clip.ee_edge_times[:, foot, 0], torch.tensor([0.25, 1.0, 1.0, 1.0]))
    assert torch.allclose(clip.ee_edge_times[:, foot, 1], torch.tensor([0.75, 0.5, 0.25, 1.0]))
  # Edge-free hand rows read exactly 1.0, interior edges are unchanged.
  assert torch.allclose(clip.ee_edge_distance[:, 0], torch.ones(4))
  assert torch.allclose(clip.ee_edge_distance[:, 1], torch.ones(4))
  for foot in (2, 3):
    assert torch.allclose(clip.ee_edge_distance[:, foot], torch.tensor([0.25, 0.0, 0.25, 0.0]))
  # Runs do not fuse across the seam: the feet's frame-0 ON run is one frame.
  for foot in (2, 3):
    assert torch.allclose(clip.ee_run_time[:, foot], torch.tensor([0.25, 0.5, 0.5, 0.25]))


def test_resolve_segment_window(tmp_path):
  labels = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.uint8)
  data = {
    "segment_names": ["stand", "place", "set", "end_hold"],
    "phase_segment": labels,
    "cycle_index": np.zeros(8, dtype=np.uint16),
  }
  path = tmp_path / "clip.npy"
  np.save(path, data)
  # The cycle is 6 frames (end_hold excluded); place+set span frames 2-5.
  lo, hi = resolve_segment_window(str(path), ("place", "set"))
  assert lo == pytest.approx(2 / 6) and hi == pytest.approx(6 / 6)
  # Unknown segments and missing files degrade to None.
  assert resolve_segment_window(str(path), ("nope",)) is None
  assert resolve_segment_window(str(tmp_path / "absent.npy"), ("place",)) is None
