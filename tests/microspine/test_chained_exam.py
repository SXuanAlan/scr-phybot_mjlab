"""Unit tests for the chained mount->climb exam's pure helpers."""

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "chained_mount_exam.py"
_spec = importlib.util.spec_from_file_location("chained_mount_exam", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_offset_grid_enumeration_is_stable():
  grid = _mod.make_offset_grid([-0.05, 0.0, 0.05], [-0.03, 0.03], [0.0, 5.0])
  assert grid.shape == (12, 3)
  # x-major, then y, then yaw — cell indices must be reproducible across
  # runs so reports stay comparable.
  assert torch.allclose(grid[0], torch.tensor([-0.05, -0.03, 0.0]))
  assert grid[1, 2].item() == pytest.approx(5.0 * torch.pi / 180.0)
  assert torch.allclose(grid[-1, :2], torch.tensor([0.05, 0.03]))
  # Single-cell grid (the smoke path).
  assert _mod.make_offset_grid([0.0], [0.0], [0.0]).shape == (1, 3)


def test_apply_offset_grid_matches_spawn_noise_semantics():
  from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

  pos = torch.tensor([[1.27, 0.0, 0.726], [1.27, 0.1, 0.726]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  offsets = torch.tensor([[0.05, -0.03, 0.1], [0.0, 0.0, 0.0]])

  out_pos, out_quat = _mod.apply_offset_grid(pos, quat, offsets)
  # World x/y added, z untouched (floor-plane delivery-error semantics).
  assert torch.allclose(out_pos[0], torch.tensor([1.32, -0.03, 0.726]))
  assert torch.allclose(out_pos[1], pos[1])
  # Yaw premultiplies about world z — byte-parity with the spawn-noise
  # construction the training scatter uses.
  axis = torch.tensor([[0.0, 0.0, 1.0]])
  expected = quat_mul(quat_from_angle_axis(torch.tensor([0.1]), axis), quat[:1])
  assert torch.allclose(out_quat[0], expected[0], atol=1e-6)
  assert torch.allclose(out_quat[1], quat[1])
  assert out_quat.norm(dim=-1).allclose(torch.ones(2), atol=1e-6)
