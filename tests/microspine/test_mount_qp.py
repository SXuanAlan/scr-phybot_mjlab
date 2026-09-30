"""Contract tests for the mount scripted-base contact QP (mount_qp.py).

Pure-numpy: no env, no GPU. Geometry mimics the 90° mount prologue —
wall surface at x = 1.5, feet standing near x = 1.31, hand anchors at
x ≈ 1.49 up the wall.
"""

from __future__ import annotations

import numpy as np
import pytest

from scr_phybot_mjab.tasks.climb_transition.mdp.mount_qp import (
  ContactUnit,
  MountQpCfg,
  solve_mount_qp,
)

MASS = 34.4
GRAVITY = np.array([0.0, 0.0, -9.81])
WEIGHT = MASS * 9.81

# Wall basis at 90°: inward = +x, rake (down-wall) = −z.
WANG_FRAME = np.stack(
  [
    np.array([0.0, 0.0, -1.0]),  # t̂ (rake, down-wall)
    np.array([0.0, -1.0, 0.0]),  # l̂ = n̂_out × t̂
    np.array([-1.0, 0.0, 0.0]),  # n̂_out
  ],
  axis=1,
)
GROUND_FRAME = np.eye(3)


def _foot_points(x: float = 1.31) -> list[ContactUnit]:
  units = []
  for side, y in (("L", 0.148), ("R", -0.148)):
    for tag, dx in (("heel", -0.06), ("toe", 0.09)):
      units.append(
        ContactUnit(
          name=f"gnd_{side}_{tag}",
          kind="ground",
          pos_w=np.array([x + dx, y, 0.02]),
          frame_w=GROUND_FRAME.copy(),
        )
      )
  return units


def _com_standing() -> np.ndarray:
  return np.array([1.27, 0.0, 0.75])


def test_standing_splits_weight_evenly():
  sol = solve_mount_qp(_foot_points(), MASS, GRAVITY, _com_standing())
  assert sol.feasible
  f = sol.force_on_robot_w
  # All vertical support, sums to weight, no gratuitous shear.
  assert abs(f[:, 2].sum() - WEIGHT) < 0.02 * WEIGHT
  assert np.abs(f[:, :2]).max() < 0.15 * WEIGHT
  # Left/right symmetric within tolerance.
  left = f[0, 2] + f[1, 2]
  right = f[2, 2] + f[3, 2]
  assert abs(left - right) < 0.1 * WEIGHT
  # CoM stays at the measurement.
  assert np.linalg.norm(sol.com_target - _com_standing()) < 0.02


def test_press_pushback_leans_com_into_wall():
  units = _foot_points()
  # Hand presses the wall with 20 N at shoulder-plus height: the wall pushes
  # the robot BACK along −x at that point.
  press_point = np.array([1.49, 0.18, 1.44])
  f_push = np.array([-20.0, 0.0, 0.0])
  base = solve_mount_qp(units, MASS, GRAVITY, _com_standing())
  sol = solve_mount_qp(
    units,
    MASS,
    GRAVITY,
    _com_standing(),
    extra_force_w=f_push,
    extra_moment_w=np.cross(press_point, f_push),
  )
  assert sol.feasible
  # Feet must supply the +x shear the pushback removes.
  assert sol.force_on_robot_w[:, 0].sum() == pytest.approx(20.0, abs=1.0)
  # The commanded CoM leans toward the wall relative to the no-press solve.
  assert sol.com_target[0] > base.com_target[0] + 0.01


def test_single_anchor_plus_feet_is_feasible():
  units = _foot_points()
  units.append(
    ContactUnit(
      name="wang_LH",
      kind="wang",
      pos_w=np.array([1.488, 0.176, 1.44]),
      frame_w=WANG_FRAME.copy(),
    )
  )
  sol = solve_mount_qp(units, MASS, GRAVITY, _com_standing())
  assert sol.feasible
  lam = sol.lambda_local[-1]
  cfg = MountQpCfg()
  # Anchor honestly engaged: rake load at or above the floor, tension inside
  # the 4:1 line.
  assert lam[0] >= cfg.f_t_min - 1.0e-6
  assert lam[2] <= lam[0] / 4.0 + 1.0e-6
  assert sol.margin > 0.5


def test_full_hang_on_four_anchors():
  anchors = [
    ("LH", [1.488, 0.176, 1.53]),
    ("RH", [1.488, -0.176, 1.53]),
    ("LF", [1.488, 0.148, 0.62]),
    ("RF", [1.488, -0.148, 0.62]),
  ]
  units = [
    ContactUnit(
      name=f"wang_{n}",
      kind="wang",
      pos_w=np.array(p),
      frame_w=WANG_FRAME.copy(),
    )
    for n, p in anchors
  ]
  com = np.array([1.35, 0.0, 1.05])
  sol = solve_mount_qp(units, MASS, GRAVITY, com)
  assert sol.feasible
  # All rake loads carry the body (positive, sum ≈ weight).
  assert sol.lambda_local[:, 0].sum() == pytest.approx(WEIGHT, rel=0.05)
  assert sol.margin > 0.0


def test_share_cap_preunload_shifts_load():
  units = _foot_points()
  # Cap the LEFT foot's two points to 5% weight each: pre-unload before a
  # foot lift.
  for u in units:
    if "_L_" in u.name:
      u.share_cap = 0.05
  sol = solve_mount_qp(units, MASS, GRAVITY, _com_standing())
  assert sol.feasible
  f = sol.force_on_robot_w
  left = f[0, 2] + f[1, 2]
  right = f[2, 2] + f[3, 2]
  assert left < 0.12 * WEIGHT
  assert right > 0.85 * WEIGHT
  # CoM walks toward the loaded (right) side.
  assert sol.com_target[1] < -0.05


def test_empty_support_reports_infeasible():
  sol = solve_mount_qp([], MASS, GRAVITY, _com_standing())
  assert not sol.feasible


def test_warm_start_requires_matching_units():
  units = _foot_points()
  sol1 = solve_mount_qp(units, MASS, GRAVITY, _com_standing())
  # Same set: warm start accepted (no crash, still feasible).
  sol2 = solve_mount_qp(
    units, MASS, GRAVITY, _com_standing(), warm_start=sol1
  )
  assert sol2.feasible
  # Different set: silently falls back to the fresh seed (no index garbage).
  sol3 = solve_mount_qp(
    units[:2], MASS, GRAVITY, _com_standing(), warm_start=sol1
  )
  assert sol3.unit_names == tuple(u.name for u in units[:2])
