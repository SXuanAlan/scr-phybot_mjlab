"""Virtual spine field math: seating fraction over synthetic pad poses.

The field replaces the unpassable 2-rigid-sphere engage gate: hundreds of
force-free suspended spines (Wang 2017: kn = 5 mN/mm, le_max = 5 mm) are
represented by a bilinear grid over the four corner sphere centers, and a
grid point is engaged when it can reach the wall within the spine travel.
"""

import math

import torch

from scr_phybot_mjab.microspine.contact import virtual_spine_fraction

RADIUS = 0.008
TRAVEL = 0.005
GRID = (5, 5)
SPAN = 0.04  # pad corner-to-corner extent along each axis [m]

WALL_POINT = torch.zeros(3)
WALL_NORMAL = torch.tensor([1.0, 0.0, 0.0])  # +x is off-surface toward robot


def _pad_corners(center_gap: float, tilt_deg: float) -> torch.Tensor:
  """Corner sphere centers for a pad facing the x=0 wall.

  ``center_gap`` is the closest corner's center-to-surface distance;
  ``tilt_deg`` rotates the pad about the y axis so the far edge lifts away.
  Row-major order (r0c0, r0c1, r1c0, r1c1): rows along z, cols along y.
  """
  tilt = math.radians(tilt_deg)
  lift = SPAN * math.sin(tilt)
  return torch.tensor(
    [
      [center_gap, 0.0, 0.0],
      [center_gap, SPAN, 0.0],
      [center_gap + lift, 0.0, SPAN],
      [center_gap + lift, SPAN, SPAN],
    ]
  )


def test_flat_touching_pad_is_fully_seated():
  corners = _pad_corners(center_gap=RADIUS, tilt_deg=0.0)
  frac = virtual_spine_fraction(corners, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL)
  assert frac.item() == 1.0


def test_flat_pad_within_travel_is_seated_and_beyond_is_not():
  near = _pad_corners(center_gap=RADIUS + TRAVEL - 1e-4, tilt_deg=0.0)
  far = _pad_corners(center_gap=RADIUS + TRAVEL + 1e-3, tilt_deg=0.0)
  assert virtual_spine_fraction(near, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item() == 1.0
  assert virtual_spine_fraction(far, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item() == 0.0


def test_tilt_grades_the_fraction_and_corner_poke_fails_the_gate():
  # The engaged band spans atan(travel/span) ~ 7.1 deg from the touching
  # edge: a tilt AT that angle still seats nearly the whole field (paper
  # semantics — every spine can reach), which is exactly the tolerance the
  # rigid 2-sphere gate destroyed.
  edge_of_tolerance = _pad_corners(center_gap=RADIUS, tilt_deg=7.2)
  frac_edge = virtual_spine_fraction(edge_of_tolerance, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item()
  assert 0.7 <= frac_edge <= 0.9  # far row just out of reach
  assert frac_edge >= 0.5  # still latches

  marginal = _pad_corners(center_gap=RADIUS, tilt_deg=13.0)
  frac_marginal = virtual_spine_fraction(marginal, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item()
  assert 0.5 <= frac_marginal <= 0.7  # half-reach: latches, weak capacity

  corner_poke = _pad_corners(center_gap=RADIUS, tilt_deg=25.0)
  frac_poke = virtual_spine_fraction(corner_poke, WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item()
  assert frac_poke < 0.5  # fails the engage_min_spine_fraction=0.5 gate
  assert frac_poke > 0.0  # the touching strip itself still reads engaged

  # Monotone: more tilt, less seated.
  shallow = virtual_spine_fraction(_pad_corners(RADIUS, 3.0), WALL_POINT, WALL_NORMAL, RADIUS, GRID, TRAVEL).item()
  assert shallow == 1.0
  assert shallow > frac_edge > frac_marginal > frac_poke


def test_batched_shapes_broadcast_like_the_affordance_call():
  # (E, H, 4, 3) corners with (E, H, 3) wall point/normal -> (E, H).
  corners = torch.stack(
    [
      _pad_corners(RADIUS, 0.0),
      _pad_corners(RADIUS, 25.0),
      _pad_corners(RADIUS + 0.05, 0.0),
      _pad_corners(RADIUS, 7.2),
    ]
  ).reshape(2, 2, 4, 3)
  wall_point = WALL_POINT.expand(2, 2, 3)
  wall_normal = WALL_NORMAL.expand(2, 2, 3)
  frac = virtual_spine_fraction(corners, wall_point, wall_normal, RADIUS, GRID, TRAVEL)
  assert frac.shape == (2, 2)
  assert frac[0, 0].item() == 1.0  # flat touching
  assert frac[0, 1].item() < 0.5  # corner poke
  assert frac[1, 0].item() == 0.0  # hovering far
  assert 0.7 <= frac[1, 1].item() <= 0.9  # tilted to the tolerance edge
