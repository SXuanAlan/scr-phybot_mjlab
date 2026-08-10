"""Wall-tilt pose math for the wall-angle representation (real gravity)."""

import math

import torch

from scr_phybot_mjab.assets.wall import WALL_HALF_SIZE, WALL_OFFSET
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  wall_pose_for_angle,
)

_PIVOT_X = WALL_OFFSET[0] - WALL_HALF_SIZE[0]  # surface/ground line, x = 1.5


def _rot_y(alpha: float) -> torch.Tensor:
  ca, sa = math.cos(alpha), math.sin(alpha)
  return torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def test_vertical_wall_is_the_compiled_identity_pose():
  pos, quat = wall_pose_for_angle(torch.tensor([90.0]))
  assert torch.allclose(pos[0], torch.tensor(WALL_OFFSET, dtype=pos.dtype), atol=1e-6)
  assert torch.allclose(
    quat[0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype), atol=1e-6
  )


def test_base_line_stays_on_the_ground_at_every_angle():
  # The robot-facing surface's ground intersection must not move when the
  # wall tilts — the infinite collision plane would otherwise sweep through
  # the robot's standing region (and the floor plane, once enabled).
  angles = torch.tensor([90.0, 88.0, 85.0, 75.0, 45.0, 30.0])
  pos, _ = wall_pose_for_angle(angles)
  # Pivot point in the wall BODY frame (body center at WALL_OFFSET).
  pivot_body = torch.tensor([_PIVOT_X - WALL_OFFSET[0], 0.0, -WALL_OFFSET[2]])
  expected = torch.tensor([_PIVOT_X, 0.0, 0.0])
  for i, angle in enumerate(angles):
    alpha = math.radians(90.0 - float(angle))
    world = pos[i] + _rot_y(alpha) @ pivot_body
    assert torch.allclose(world, expected, atol=1e-5), f"angle {angle}"


def test_45_deg_pose_matches_hand_computation():
  pos, quat = wall_pose_for_angle(torch.tensor([45.0]))
  s = math.sqrt(0.5)
  assert torch.allclose(
    pos[0],
    torch.tensor([_PIVOT_X + 8.0 * s, 0.0, 7.0 * s], dtype=pos.dtype),
    atol=1e-5,
  )
  half = math.radians(45.0) / 2.0
  assert torch.allclose(
    quat[0],
    torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0], dtype=quat.dtype),
    atol=1e-6,
  )


def test_vertical_height_spawn_offset_restores_the_90_deg_height():
  import math as _math

  from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
    vertical_height_spawn_offset,
  )

  root = torch.tensor([[1.327, 0.0, 1.0], [1.327, 0.0, 1.0], [1.327, 0.0, 1.0]])
  quats = []
  for angle in (90.0, 45.0, 75.0):
    half = _math.radians(90.0 - angle) / 2.0
    quats.append([_math.cos(half), 0.0, _math.sin(half), 0.0])
  wall_quat = torch.tensor(quats)
  shift = vertical_height_spawn_offset(root, wall_quat, _PIVOT_X)
  # Identity wall: exactly zero (the bit-identity path must hold).
  assert shift[0].item() == 0.0
  # Shifted spawns land back at the vertical-wall world height.
  for i, angle in enumerate((90.0, 45.0, 75.0)):
    alpha = _math.radians(90.0 - angle)
    rel_x = 1.327 - _PIVOT_X
    world_z = -rel_x * _math.sin(alpha) + (1.0 + shift[i].item()) * _math.cos(
      alpha
    )
    assert abs(world_z - 1.0) < 1e-5, f"angle {angle}: world z {world_z}"
  # Shallow wall raises the spawn up-slope; steep is a small correction.
  assert shift[1].item() > 0.2
  assert abs(shift[2].item()) < 0.02


def test_leans_away_from_the_robot():
  # Shallower angle -> wall top tips toward +x (away from the robot at
  # x < 1.5), i.e. the face becomes a ramp the robot lies on.
  pos, _ = wall_pose_for_angle(torch.tensor([45.0]))
  assert pos[0, 0] > WALL_OFFSET[0]
  assert 0.0 < pos[0, 2] < WALL_OFFSET[2]
