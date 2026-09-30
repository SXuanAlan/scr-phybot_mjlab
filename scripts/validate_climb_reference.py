#!/usr/bin/env python3
"""Audit whether a climbing reference has a moment-free static load path.

This is an *existence* check, deliberately independent of a learned policy
and of MuJoCo weld stiffness.  At every selected motion frame it solves for
one world-frame force at the center of each scheduled engaged pad such that:

* the pad forces balance the robot's weight and whole-body moment;
* every engaged-pad force stays inside the Wang-2017 limit surface at
  ``--f0``;
* a scheduled-off pad whose face is physically on the wall (the pre-engage
  rake) may press the wall inside a Coulomb cone, and a scheduled-off FOOT whose
  bracing toe (``PHYBOT_C2_FOOT_TOES``) is on the wall may push through that
  toe — no adhesion, no shear, and no moment in either case;
* a foot the clip schedules as a FLOOR support (mount prologue clips carry a
  ``foot_on_ground`` key) may push on the floor at its heel and toe sole
  points inside a Coulomb friction cone (``--ground-friction``) — unilateral,
  no adhesion, no local moment beyond the heel/toe couple;
* the resulting joint torques stay inside the real deployment limits; and
* every local pad moment is exactly zero by construction.

Clips with a ``foot_on_ground`` key are treated as ACYCLIC for the
schedule-timing audit (a mount prologue does not loop): edge bookkeeping does
not wrap, and OFF windows that touch frame 0 (a pad that simply has not
engaged yet) are exempt from the re-engagement pipeline floor.

A passing frame therefore has a physical force-only equilibrium and cannot be
using the weld as an unlimited flange.  It does not claim that an open-loop PD
replay, or an existing policy trained on different physics, will discover that
load allocation dynamically.

Example:
  python scripts/validate_climb_reference.py \
    src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

import mujoco
import mjlab  # noqa: F401  # Initialize task entry points before PHYBot imports.
import numpy as np
import torch
from scipy.optimize import minimize

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_DEPLOY_TORQUE_LIMITS,
  PHYBOT_C2_FOOT_TOES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  get_spec,
)
from scr_phybot_mjab.microspine.config import MicrospineLimitSurfaceCfg
from scr_phybot_mjab.microspine.contact import (
  ToeContact,
  add_toe_geoms,
  toe_site_name,
)
from scr_phybot_mjab.microspine.wang_limit_surface import build_capacity_lut
from scr_phybot_mjab.tools.climbing_reference.model import (
  WallGeometry, joint_group, build_reference_model, engage_basis,
)

_DEFAULT_CLIP = (
  _REPO
  / "src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy"
)
_WALL_OUTWARD_W = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
_RAKE_LOCAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class _FrameResult:
  frame: int
  segment: str
  accepted: bool
  solver_success: bool
  eq_residual: float
  anchor_util: float
  anchor_name: str
  torque_util: float
  torque_name: str
  compression_force: float
  compression_name: str
  forces: dict[str, np.ndarray]
  ground_normal_max: float = 0.0
  ground_cone_min: float = np.inf


def _skew(v: np.ndarray) -> np.ndarray:
  x, y, z = v
  return np.array(
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
    dtype=np.float64,
  )


def _set_motion_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  clip: dict,
  frame: int,
  joint_qpos_addr: np.ndarray,
  free_qpos_addr: int,
  wall: WallGeometry | None = None,
) -> None:
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  data.qacc[:] = 0.0
  q_xyzw = clip["root_quaternion"][frame]
  if wall is None or wall.identity:
    data.qpos[free_qpos_addr : free_qpos_addr + 3] = clip["root_position"][
      frame
    ]
    data.qpos[free_qpos_addr + 3 : free_qpos_addr + 7] = (
      q_xyzw[3],
      q_xyzw[0],
      q_xyzw[1],
      q_xyzw[2],
    )
  else:
    # Canonical-frame clip -> native tilted world: rotate the root about
    # the wall base line (joints are frame-invariant).
    data.qpos[free_qpos_addr : free_qpos_addr + 3] = wall.rotate_point(
      np.asarray(clip["root_position"][frame], dtype=np.float64)
    )
    q_wxyz = np.array(
      [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
    )
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, wall.quat, q_wxyz)
    data.qpos[free_qpos_addr + 3 : free_qpos_addr + 7] = rotated
  data.qpos[joint_qpos_addr] = clip["joint_positions"][frame]
  for i, name in enumerate(clip.get("passive_joint_names", ())):
    data.qpos[model.joint(name).qposadr[0]] = clip["passive_joint_positions"][frame, i]
  mujoco.mj_forward(model, data)


def _sample_capacity_ratio(
  lut: np.ndarray,
  f_t: np.ndarray,
  f_l: np.ndarray,
  f_n: np.ndarray,
) -> np.ndarray:
  """NumPy equivalent of ``wang_limit_surface.capacity_ratio``."""
  tension = np.maximum(f_n, 0.0)
  in_plane = np.hypot(f_t, f_l)
  theta = np.degrees(np.arctan2(np.abs(f_l), f_t))
  phi = np.degrees(np.arctan2(tension, in_plane))

  n_theta, n_phi = lut.shape
  ti = np.clip(theta / 180.0 * (n_theta - 1), 0.0, n_theta - 1)
  pi = np.clip(phi / 90.0 * (n_phi - 1), 0.0, n_phi - 1)
  t0 = np.minimum(np.floor(ti).astype(np.int64), n_theta - 2)
  p0 = np.minimum(np.floor(pi).astype(np.int64), n_phi - 2)
  tw = ti - t0
  pw = pi - p0
  return (
    lut[t0, p0] * (1.0 - tw) * (1.0 - pw)
    + lut[t0, p0 + 1] * (1.0 - tw) * pw
    + lut[t0 + 1, p0] * tw * (1.0 - pw)
    + lut[t0 + 1, p0 + 1] * tw * pw
  )


def _anchor_utilization(
  forces_flat: np.ndarray,
  bases: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
  lut: np.ndarray,
  f0: float,
  r_floor: float,
) -> np.ndarray:
  forces = forces_flat.reshape(-1, 3)
  t_hat = np.stack([basis[0] for basis in bases])
  l_hat = np.stack([basis[1] for basis in bases])
  n_hat = np.stack([basis[2] for basis in bases])
  f_t = np.einsum("ij,ij->i", forces, t_hat)
  f_l = np.einsum("ij,ij->i", forces, l_hat)
  f_n = np.einsum("ij,ij->i", forces, n_hat)
  # 1 mN smoothing: |f_l| and hypot(f_t, f_l) are non-differentiable at a
  # zero-shear force, and SLSQP's finite-difference gradient vanishes across
  # those kinks — a pure-tension anchor (utilization ~8 at phi=90) is a
  # stationary point of the raw expression even though shading in shear
  # raises capacity much faster than load (measured on the mount clip's
  # right_foot_mount frames: both seeds froze at f_t = f_l = 0 exactly).
  f_l_s = np.sqrt(f_l**2 + 1.0e-6)
  ratio = _sample_capacity_ratio_smooth(lut, f_t, f_l_s, f_n)
  capacity = f0 * np.maximum(ratio, r_floor)
  load = np.sqrt(f_t**2 + f_l**2 + np.maximum(f_n, 0.0) ** 2 + 1.0e-6)
  return load / capacity


def _sample_capacity_ratio_smooth(
  lut: np.ndarray,
  f_t: np.ndarray,
  f_l_abs: np.ndarray,
  f_n: np.ndarray,
) -> np.ndarray:
  """`_sample_capacity_ratio` with the |f_l| already smoothed by the caller
  and the in-plane norm smoothed to keep phi differentiable at zero shear."""
  tension = np.maximum(f_n, 0.0)
  in_plane = np.sqrt(f_t**2 + f_l_abs**2)
  theta = np.degrees(np.arctan2(f_l_abs, f_t))
  phi = np.degrees(np.arctan2(tension, in_plane))
  n_theta, n_phi = lut.shape
  ti = np.clip(theta / 180.0 * (n_theta - 1), 0.0, n_theta - 1)
  pi = np.clip(phi / 90.0 * (n_phi - 1), 0.0, n_phi - 1)
  t0 = np.minimum(np.floor(ti).astype(np.int64), n_theta - 2)
  p0 = np.minimum(np.floor(pi).astype(np.int64), n_phi - 2)
  tw = ti - t0
  pw = pi - p0
  return (
    lut[t0, p0] * (1.0 - tw) * (1.0 - pw)
    + lut[t0, p0 + 1] * (1.0 - tw) * pw
    + lut[t0 + 1, p0] * tw * (1.0 - pw)
    + lut[t0 + 1, p0 + 1] * tw * pw
  )


def _solve_frame(
  *,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  frame: int,
  segment: str,
  engaged_names: list[str],
  compression_contacts: list[tuple[str, int]],
  ground_contacts: list[tuple[str, int, np.ndarray]],
  ground_friction: float,
  toe_wall_friction: float,
  pad_site_ids: dict[str, int],
  ee_body_ids: dict[str, int],
  joint_dof_addr: np.ndarray,
  torque_limits: np.ndarray,
  lut: np.ndarray,
  f0: float,
  r_floor: float,
  max_compression_force: float,
  max_iterations: int,
  wall_outward: np.ndarray = _WALL_OUTWARD_W,
) -> _FrameResult:
  mass = float(model.body_mass.sum())
  com = np.einsum("i,ij->j", model.body_mass, data.xipos) / mass
  target = np.concatenate([mass * model.opt.gravity, np.zeros(3)])
  n_anchor = len(engaged_names)
  anchor_force_dim = 3 * n_anchor
  n_comp = len(compression_contacts)
  ground_offset = anchor_force_dim + 3 * n_comp
  load_dim = ground_offset + 3 * len(ground_contacts)
  equilibrium = np.zeros((6, load_dim), dtype=np.float64)
  torque_map = np.zeros(
    (len(joint_dof_addr), load_dim), dtype=np.float64
  )
  bases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  jacp = np.zeros((3, model.nv), dtype=np.float64)
  jacr = np.zeros_like(jacp)

  for i, ee in enumerate(engaged_names):
    site_id = pad_site_ids[ee]
    pos = data.site_xpos[site_id]
    sl = slice(3 * i, 3 * i + 3)
    equilibrium[0:3, sl] = np.eye(3)
    equilibrium[3:6, sl] = _skew(pos - com)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    # F is pad-on-wall load, hence wall-on-robot is -F.  Static actuator
    # torque is bias - passive - J.T(-F) = bias - passive + J.T F.
    torque_map[:, sl] = jacp[:, joint_dof_addr].T
    bases.append(engage_basis(data, ee_body_ids[ee], wall_outward))

  # Released support contacts: the raking pad face and/or the planted foot
  # toe. Each is a unilateral Coulomb contact at its own point: compression
  # INTO the surface (opposite the outward normal) plus tangential friction
  # inside a ``--toe-wall-friction`` cone — the toe-against-wall press is a
  # DESIGNED support that forms force couples with the hand microspines
  # (the sim's sphere-wall pairs carry mu 0.8, condim 3; the audit runs a
  # margined 0.5 by default). No tensile reaction, no local moment.
  compression_direction = -wall_outward
  comp_t2 = np.array([0.0, 1.0, 0.0])
  comp_t1 = np.cross(compression_direction, comp_t2)
  comp_basis = (compression_direction, comp_t1, comp_t2)
  for i, (_, site_id) in enumerate(compression_contacts):
    pos = data.site_xpos[site_id]
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    for a, direction in enumerate(comp_basis):
      column = anchor_force_dim + 3 * i + a
      equilibrium[0:3, column] = direction
      equilibrium[3:6, column] = _skew(pos - com) @ direction
      torque_map[:, column] = jacp[:, joint_dof_addr].T @ direction

  # Floor supports: a 3D ROBOT-ON-GROUND force P at each scheduled sole
  # point, same sign convention as the anchors (P enters the balance
  # directly; ground-on-robot is -P). Unilaterality is P_z <= 0 (the robot
  # can only push down) and the Coulomb cone is |P_xy| <= mu * (-P_z).
  for i, (_, body_id, point_w) in enumerate(ground_contacts):
    sl = slice(ground_offset + 3 * i, ground_offset + 3 * i + 3)
    equilibrium[0:3, sl] = np.eye(3)
    equilibrium[3:6, sl] = _skew(point_w - com)
    mujoco.mj_jac(model, data, jacp, jacr, point_w, body_id)
    torque_map[:, sl] = jacp[:, joint_dof_addr].T

  # The Wang surface is direction-dependent and mildly non-convex. Starting
  # from the previous frame can follow the wrong capacity lobe after a support
  # transition even though a low-utilization equilibrium exists. The
  # deterministic minimum-norm equilibrium is a reliable, frame-independent
  # seed and makes the audit invariant to traversal order.
  seed_cols = np.concatenate(
    [
      np.arange(anchor_force_dim),
      np.arange(ground_offset, load_dim),
    ]
  ).astype(int)
  seed0 = (
    np.linalg.lstsq(equilibrium[:, seed_cols], target, rcond=None)[0]
    if len(seed_cols)
    else np.zeros(0)
  )
  load0 = np.zeros(load_dim, dtype=np.float64)
  load0[seed_cols] = seed0
  # Nudge the seed inside the unilateral bound.
  for i in range(len(ground_contacts)):
    zi = ground_offset + 3 * i + 2
    load0[zi] = min(load0[zi], -1.0)
  if n_anchor:
    util0 = _anchor_utilization(
      load0[:anchor_force_dim], bases, lut, f0, r_floor
    )
    slack0 = max(float(util0.max()) + 0.02, 0.05)
  else:
    slack0 = 0.05
  z0 = np.concatenate([load0, [slack0]])

  tau0 = (
    data.qfrc_bias[joint_dof_addr]
    - data.qfrc_passive[joint_dof_addr]
  )

  def utilization(z: np.ndarray) -> np.ndarray:
    if not n_anchor:
      return np.zeros(1)
    return _anchor_utilization(
      z[:anchor_force_dim], bases, lut, f0, r_floor
    )

  def torque(z: np.ndarray) -> np.ndarray:
    return tau0 + torque_map @ z[:load_dim]

  def ground_cone(z: np.ndarray) -> np.ndarray:
    if not ground_contacts:
      return np.zeros(1)
    vals = np.zeros(len(ground_contacts))
    for i in range(len(ground_contacts)):
      p = z[ground_offset + 3 * i : ground_offset + 3 * i + 3]
      # Bias-corrected 10 mN smoothing: the tangential norm is kinked at
      # zero shear and SLSQP's finite differences stall there (same disease
      # as the Wang utilization at a pure-tension anchor); subtracting
      # sqrt(eps) keeps the smoothed cone exact at the origin so an unloaded
      # contact does not read as a 10 mN violation.
      vals[i] = ground_friction * (-p[2]) - float(
        np.sqrt(p[0] * p[0] + p[1] * p[1] + 1.0e-4) - 1.0e-2
      )
    return vals

  def toe_cone(z: np.ndarray) -> np.ndarray:
    if not compression_contacts:
      return np.zeros(1)
    vals = np.zeros(n_comp)
    for i in range(n_comp):
      c = z[anchor_force_dim + 3 * i : anchor_force_dim + 3 * i + 3]
      vals[i] = toe_wall_friction * c[0] - float(
        np.sqrt(c[1] * c[1] + c[2] * c[2] + 1.0e-4) - 1.0e-2
      )
    return vals

  constraints = (
    {
      "type": "eq",
      "fun": lambda z: equilibrium @ z[:load_dim] - target,
    },
    {
      "type": "ineq",
      "fun": lambda z: z[-1] - utilization(z),
    },
    {
      "type": "ineq",
      "fun": lambda z: torque_limits - torque(z),
    },
    {
      "type": "ineq",
      "fun": lambda z: torque_limits + torque(z),
    },
    {
      "type": "ineq",
      "fun": ground_cone,
    },
    {
      "type": "ineq",
      "fun": toe_cone,
    },
  )
  tangential_cap = toe_wall_friction * max_compression_force
  bounds = (
    [(-2000.0, 2000.0)] * anchor_force_dim
    + [
      (0.0, max_compression_force),
      (-tangential_cap, tangential_cap),
      (-tangential_cap, tangential_cap),
    ]
    * len(compression_contacts)
    + [(-2000.0, 2000.0), (-2000.0, 2000.0), (-2000.0, 0.0)]
    * len(ground_contacts)
    + [(0.0, 50.0)]
  )
  def _minimize(z_init: np.ndarray, iter_scale: int = 1):
    return minimize(
      lambda z: z[-1] + 1.0e-10 * np.dot(z[:load_dim], z[:load_dim]),
      z_init,
      method="SLSQP",
      bounds=bounds,
      constraints=constraints,
      options={
        "ftol": 1.0e-10,
        "maxiter": max_iterations * iter_scale,
        "disp": False,
      },
    )

  result = _minimize(z0)
  if not result.success or (
    n_anchor and float(utilization(result.x).max()) > 1.0 + 1.0e-5
  ):
    # Multi-start fallback: a hang-distribution seed (every anchor carrying
    # its share of the weight as rake-direction shear) approaches from the
    # high-capacity side of the Wang surface, away from the zero-shear
    # region the min-norm seed can start in.
    hang = np.zeros(load_dim)
    share = mass * float(np.linalg.norm(model.opt.gravity)) / max(n_anchor, 1)
    for i in range(n_anchor):
      t_hat = bases[i][0]
      hang[3 * i : 3 * i + 3] = share * t_hat
    for i in range(len(ground_contacts)):
      hang[ground_offset + 3 * i + 2] = -1.0
    alt = _minimize(np.concatenate([hang, [2.0]]), iter_scale=2)
    alt_eq = float(
      np.linalg.norm(equilibrium @ alt.x[:load_dim] - target, ord=np.inf)
    )
    if alt_eq <= 1.0e-6 and (
      (alt.success and not result.success)
      or float(utilization(alt.x).max()) < float(utilization(result.x).max())
    ):
      result = alt

  load = result.x[:load_dim]
  force = load[:anchor_force_dim]
  compression = (
    load[anchor_force_dim:ground_offset].reshape(-1, 3)[:, 0]
    if n_comp
    else load[anchor_force_dim:ground_offset]
  )
  ground = load[ground_offset:].reshape(-1, 3)
  util = utilization(result.x)
  tau = torque(result.x)
  tau_util = np.abs(tau) / torque_limits
  eq_residual = float(
    np.linalg.norm(equilibrium @ load - target, ord=np.inf)
  )
  anchor_index = int(np.argmax(util))
  torque_index = int(np.argmax(tau_util))
  if len(compression):
    compression_index = int(np.argmax(compression))
    compression_force = float(compression[compression_index])
    compression_name = compression_contacts[compression_index][0]
  else:
    compression_force = 0.0
    compression_name = "<none>"
  cone = ground_cone(result.x)
  inequality_residual = min(
    float(result.x[-1] - util.max()),
    float((torque_limits - np.abs(tau)).min()),
    float(cone.min()) if ground_contacts else np.inf,
    float(toe_cone(result.x).min()) if compression_contacts else np.inf,
  )
  # Inequality tolerance 1e-4 (N / N*m): a sub-millinewton cone or torque
  # exit violation is SLSQP numerical noise, not a physical infeasibility.
  accepted = (
    eq_residual <= 1.0e-6
    and inequality_residual >= -1.0e-4
    and float(util.max()) <= 1.0 + 1.0e-5
    and float(tau_util.max()) <= 1.0 + 1.0e-5
  )
  if not accepted:
    print(
      f"   frame {frame}: {result.message}; "
      f"minimum inequality residual {inequality_residual:.3e}"
    )
  return _FrameResult(
    frame=frame,
    segment=segment,
    accepted=accepted,
    solver_success=bool(result.success),
    eq_residual=eq_residual,
    anchor_util=float(util[anchor_index]) if n_anchor else 0.0,
    anchor_name=engaged_names[anchor_index] if n_anchor else "<none>",
    torque_util=float(tau_util[torque_index]),
    torque_name=PHYBOT_C2_POLICY_JOINT_NAMES[torque_index],
    compression_force=compression_force,
    compression_name=compression_name,
    forces={
      ee: force[3 * i : 3 * i + 3].copy()
      for i, ee in enumerate(engaged_names)
    },
    ground_normal_max=float((-ground[:, 2]).max()) if len(ground) else 0.0,
    ground_cone_min=float(cone.min()) if ground_contacts else np.inf,
  )


def timing_audit(clip: dict, args: argparse.Namespace) -> bool:
  """Schedule-timing feasibility: pure clip+config arithmetic, no sim.

  Two families of checks over the loopable cycle (cycle 0 minus end_hold,
  circular — the exact schedule ClimbMotionClip trains on):

  1. Every scheduled OFF window must fit the engage pipeline: release
     cooldown + press debounce + a maneuver floor for the actual travel.
  2. Fresh-anchor maturity: from each scheduled regrip to the NEXT release
     of any other limb there must be at least ``--ramp-time-s`` of dwell —
     a just-latched pad holds capacity_ramp_floor of its capacity and
     recruits to full only over ramp-time under load, so a shorter gap
     loads an immature anchor (the motionref16 failure: seat_hold 0.2 s
     under a 0.25 s ramp, right foot immature when the left hand swung).
     A WARN tier flags gaps shorter than ramp + the latch-lateness grace
     (``--grace-fraction`` x the closing window), where a legally-late
     latch still enters the next transfer partially recruited.
  """
  fps = float(clip["fps"])
  engaged_all = np.asarray(clip["ee_engaged"], dtype=bool)
  ee_names = tuple(str(name) for name in clip["ee_list"])
  seg_names = list(str(s) for s in clip["segment_names"])
  labels = np.asarray(clip["phase_segment"], dtype=np.int64)
  cycles = np.asarray(clip["cycle_index"], dtype=np.int64)
  end_id = seg_names.index("end_hold") if "end_hold" in seg_names else -1
  cycle_mask = (cycles == cycles.min()) & (labels != end_id)
  engaged = engaged_all[cycle_mask]
  cycle = engaged.shape[0]
  # Mount prologues do not loop: no wrap edges, and an OFF window that
  # touches frame 0 is a pad that simply has not engaged yet.
  acyclic = bool(np.asarray(clip.get("foot_on_ground", 0)).any())

  pipeline_floor = args.cooldown_s + args.debounce_s + args.maneuver_floor_s
  print(
    f"== schedule-timing audit: cycle {cycle} frames ({cycle / fps:.2f} s"
    + (", acyclic" if acyclic else "")
    + ")\n"
    f"   OFF-window floor {pipeline_floor:.2f} s (cooldown {args.cooldown_s}"
    f" + debounce {args.debounce_s} + maneuver {args.maneuver_floor_s}); "
    f"regrip->next-release gap floor {args.ramp_time_s:.2f} s (recruitment "
    f"ramp), WARN under ramp + {args.grace_fraction:.2f} x window"
  )
  ok = True
  falling = {}  # ee -> sorted frame indices of scheduled releases
  rising = {}
  for i, ee in enumerate(ee_names):
    col = engaged[:, i]
    nxt = np.append(col[1:], col[0] if not acyclic else col[-1])
    falling[i] = np.flatnonzero(col & ~nxt)  # last ON frame before OFF
    rising[i] = np.flatnonzero(~col & nxt)  # last OFF frame before ON
    # OFF windows (circular runs of ~col; linear when acyclic).
    for r in rising[i]:
      # Walk back from the rising edge to the window's start.
      length = 0
      f = r
      hit_start = False
      while not col[f]:
        length += 1
        if acyclic and f == 0:
          hit_start = True
          break
        f = (f - 1) % cycle
        if length > cycle:
          break
      if hit_start:
        continue  # first-ever engagement: the whole prologue is its window
      window_s = length / fps
      if window_s + 1.0e-9 < pipeline_floor:
        print(
          f"   FAIL {ee}: OFF window ending at frame {r} is "
          f"{window_s:.2f} s < {pipeline_floor:.2f} s pipeline floor"
        )
        ok = False
  # Regrip -> next other-limb release maturity gaps.
  worst_gap = None
  for i, ee in enumerate(ee_names):
    for r in rising[i]:
      latch = (r + 1) % cycle  # first scheduled ON frame
      gaps = []
      for j in range(len(ee_names)):
        if j == i or falling[j].size == 0:
          continue
        if acyclic:
          rel = falling[j] - latch
          rel = rel[rel >= 0]
          if rel.size == 0:
            continue
        else:
          rel = (falling[j] - latch) % cycle
        gaps.append(rel.min())
      if not gaps:
        continue
      gap_s = min(gaps) / fps
      # Latch-lateness allowance: the closing OFF window's grace.
      length = 0
      f = r
      while not engaged[f, i]:
        length += 1
        if acyclic and f == 0:
          break
        f = (f - 1) % cycle
        if length > cycle:
          break
      grace_s = args.grace_fraction * (length / fps)
      tag = "ok"
      if gap_s + 1.0e-9 < args.ramp_time_s:
        tag = "FAIL"
        ok = False
      elif gap_s + 1.0e-9 < args.ramp_time_s + grace_s:
        tag = "WARN"
      if worst_gap is None or gap_s < worst_gap[0]:
        worst_gap = (gap_s, ee)
      print(
        f"   {tag:4s} {ee:18s} regrip@f{latch:4d}: next other-limb release "
        f"in {gap_s:.2f} s (ramp {args.ramp_time_s:.2f}, late-latch grace "
        f"{grace_s:.2f})"
      )
  if worst_gap is not None:
    print(f"   worst maturity gap {worst_gap[0]:.2f} s ({worst_gap[1]})")
  print(
    "PASS: schedule timing feasible."
    if ok
    else "FAIL: schedule timing infeasible."
  )
  return ok


def audit(args: argparse.Namespace) -> bool:
  path = args.motion_file.resolve()
  clip = np.load(path, allow_pickle=True).item()
  if clip.get("generator") == "scripts/gen_cad_climb_reference.py":
    raise ValueError(
      "The new empirical CAD reference includes passive gimbal coordinates "
      "and empirical unit capacities. Use scripts/audit_cad_climb_reference.py "
      "instead of this legacy Wang audit."
    )
  ee_names = tuple(str(name) for name in clip["ee_list"])
  clip_joint_names = tuple(str(name) for name in clip["joints_list"])
  if clip_joint_names != PHYBOT_C2_POLICY_JOINT_NAMES:
    raise ValueError(
      "Motion joint order does not match PHYBOT_C2_POLICY_JOINT_NAMES."
    )

  timing_ok = True
  if not args.skip_timing:
    timing_ok = timing_audit(clip, args)

  model, pad_site_ids, ee_body_ids, toe_sites, ground_points = build_reference_model(
    ee_names
  )
  data = mujoco.MjData(model)
  free_joints = np.flatnonzero(
    model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
  )
  if len(free_joints) != 1:
    raise ValueError(f"Expected one free joint, found {len(free_joints)}.")
  free_qpos_addr = int(model.jnt_qposadr[free_joints[0]])
  joint_ids = np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in clip_joint_names
    ],
    dtype=np.int32,
  )
  joint_qpos_addr = model.jnt_qposadr[joint_ids].astype(np.int32)
  joint_dof_addr = model.jnt_dofadr[joint_ids].astype(np.int32)
  torque_limits = np.asarray(
    [
      PHYBOT_C2_DEPLOY_TORQUE_LIMITS[joint_group(name)]
      for name in clip_joint_names
    ],
    dtype=np.float64,
  )

  ls_cfg = MicrospineLimitSurfaceCfg()
  lut = build_capacity_lut(ls_cfg, torch.device("cpu")).numpy().astype(
    np.float64
  )
  segment_names = tuple(str(name) for name in clip["segment_names"])
  cycle_index = np.asarray(
    clip.get("cycle_index", np.zeros(len(clip["joint_positions"]))),
    dtype=np.int64,
  )
  cycle_frames = np.flatnonzero(cycle_index == cycle_index.min())
  if getattr(args, "frames", None):
    frames = np.unique(np.asarray(args.frames, dtype=np.int64))
    if not np.isin(frames, cycle_frames).all():
      raise ValueError("--frames must select frames from the first cycle")
  else:
    frames = cycle_frames[:: args.frame_stride]
    if frames[-1] != cycle_frames[-1]:
      frames = np.append(frames, cycle_frames[-1])

  print(
    f"== static moment-free audit: {path}\n"
    f"   {len(frames)}/{len(cycle_frames)} first-cycle frames, "
    f"F0={args.f0:.1f} N, local pad moments fixed to 0 N*m\n"
    "   scheduled-off contact: "
    + (
      "disabled"
      if args.compression_contact_tolerance_mm < 0.0
      else (
        "compression-only (pad face or planted toe) within "
        f"{args.compression_contact_tolerance_mm:.3f} mm of wall"
      )
    )
  )
  results: list[_FrameResult] = []
  engaged = np.asarray(clip["ee_engaged"], dtype=bool)
  phases = np.asarray(clip["phase_segment"], dtype=np.int64)
  wall_surface_x = float(clip["wall_surface_x"])
  # Audit the current asset's actual face C, including when the clip's old
  # metadata described a contact-sphere center behind that face.
  pad_touch_x = wall_surface_x
  wall = WallGeometry(float(clip.get("wall_angle", 90.0)), wall_surface_x)
  pad_touch_offset = 0.0
  if not wall.identity:
    print(
      f"   native wall angle {wall.angle_deg:.1f} deg: canonical frames "
      "rotated back about the base line, statics audited in the tilted "
      "native world (real gravity, horizontal floor)"
    )
  foot_on_ground = np.asarray(
    clip.get("foot_on_ground", np.zeros((len(engaged), 2))), dtype=bool
  )
  foot_ees = [ee for ee in ee_names if ee in ground_points]
  if foot_on_ground.any():
    print(
      f"   floor supports enabled (mount clip): heel+toe friction cones, "
      f"mu={args.ground_friction:.2f}"
    )
  for count, frame in enumerate(frames, start=1):
    _set_motion_frame(
      model,
      data,
      clip,
      int(frame),
      joint_qpos_addr,
      free_qpos_addr,
      wall,
    )
    engaged_names = [
      ee for i, ee in enumerate(ee_names) if engaged[frame, i]
    ]
    tolerance_m = args.compression_contact_tolerance_mm * 1.0e-3
    compression_contacts: list[tuple[str, int]] = []
    if tolerance_m >= 0.0:
      for i, ee in enumerate(ee_names):
        # Pad face on the wall: the pre-engage rake press (released pads
        # only — an engaged pad's face load is the anchor force itself).
        if wall.identity:
          pad_on_wall = (
            data.site_xpos[pad_site_ids[ee], 0] >= pad_touch_x - tolerance_m
          )
        else:
          pad_on_wall = (
            wall.distance(data.site_xpos[pad_site_ids[ee]])
            <= pad_touch_offset + tolerance_m
          )
        if not engaged[frame, i] and pad_on_wall:
          compression_contacts.append((ee, pad_site_ids[ee]))
        # Planted bracing toe (feet only) — REGARDLESS of the pad's engaged
        # state: a toe cap touching the wall is a unilateral compression
        # path whether or not the spine pad above it is latched (the mount
        # prologue braces the toes of engaged feet through its hand steps).
        if ee in toe_sites:
          toe_site_id, toe_radius = toe_sites[ee]
          if wall.identity:
            toe_on_wall = (
              data.site_xpos[toe_site_id, 0]
              >= wall_surface_x - toe_radius - tolerance_m
            )
          else:
            toe_on_wall = (
              wall.distance(data.site_xpos[toe_site_id])
              <= toe_radius + tolerance_m
            )
          if toe_on_wall:
            compression_contacts.append((f"{ee}@toe", toe_site_id))
    ground_contacts: list[tuple[str, int, np.ndarray]] = []
    for gi, ee in enumerate(foot_ees):
      if not foot_on_ground[frame, gi]:
        continue
      bid, locals_ = ground_points[ee]
      xmat = data.xmat[bid].reshape(3, 3)
      for tag, local in zip(("heel", "toe"), locals_):
        point_w = data.xpos[bid] + xmat @ local
        # Per-point height gate: a toe-pivot frame keeps the foot scheduled
        # as a floor support while the heel has already lifted — only sole
        # points actually at the floor may carry force.
        if point_w[2] > 0.008:
          continue
        ground_contacts.append((f"{ee}@{tag}", bid, point_w))
    if len(engaged_names) < 3 and not ground_contacts:
      result = _FrameResult(
        frame=int(frame),
        segment=segment_names[phases[frame]],
        accepted=False,
        solver_success=False,
        eq_residual=np.inf,
        anchor_util=np.inf,
        anchor_name="<support-count>",
        torque_util=np.inf,
        torque_name="<support-count>",
        compression_force=np.inf,
        compression_name="<support-count>",
        forces={},
      )
    else:
      solve_kwargs = dict(
        model=model,
        data=data,
        frame=int(frame),
        segment=segment_names[phases[frame]],
        engaged_names=engaged_names,
        compression_contacts=compression_contacts,
        ground_contacts=ground_contacts,
        ground_friction=args.ground_friction,
        toe_wall_friction=args.toe_wall_friction,
        pad_site_ids=pad_site_ids,
        ee_body_ids=ee_body_ids,
        joint_dof_addr=joint_dof_addr,
        torque_limits=torque_limits,
        lut=lut,
        f0=args.f0,
        r_floor=ls_cfg.wang_r_floor,
        max_compression_force=args.max_compression_force,
        max_iterations=args.max_iterations,
        wall_outward=wall.outward,
      )
      result = _solve_frame(**solve_kwargs)
      retry_iterations = getattr(args, "retry_iterations", 3000)
      if (
        not result.accepted and not result.solver_success
        and retry_iterations > args.max_iterations
      ):
        # A failed nonlinear solve is not a proof of infeasibility. Give
        # the same frame more iterations without changing any constraint
        # or acceptance tolerance (CAD A80 frame 582 needs this fallback).
        print(f"   retry frame {frame} with {retry_iterations} iterations")
        retry = _solve_frame(**{**solve_kwargs, "max_iterations": retry_iterations})
        if retry.accepted:
          result = retry
    results.append(result)
    if args.progress_every and (
      count % args.progress_every == 0 or count == len(frames)
    ):
      print(f"   solved {count:4d}/{len(frames)} frames")

  failures = [result for result in results if not result.accepted]
  worst_anchor = max(results, key=lambda item: item.anchor_util)
  worst_torque = max(results, key=lambda item: item.torque_util)
  worst_eq = max(results, key=lambda item: item.eq_residual)
  worst_compression = max(
    results, key=lambda item: item.compression_force
  )
  mass = float(model.body_mass.sum())
  print(
    f"   robot mass {mass:.3f} kg, scheduled support count "
    f"{engaged[frames].sum(axis=1).min()}.."
    f"{engaged[frames].sum(axis=1).max()}"
  )
  print(
    "   worst anchor utilization "
    f"{worst_anchor.anchor_util:.4f} "
    f"(margin {(1.0 - worst_anchor.anchor_util) * 100.0:.1f}%, "
    f"frame {worst_anchor.frame} {worst_anchor.segment}, "
    f"{worst_anchor.anchor_name})"
  )
  print(
    "   worst actuator utilization "
    f"{worst_torque.torque_util:.4f} "
    f"(margin {(1.0 - worst_torque.torque_util) * 100.0:.1f}%, "
    f"frame {worst_torque.frame} {worst_torque.segment}, "
    f"{worst_torque.torque_name})"
  )
  print(
    f"   equilibrium residual max {worst_eq.eq_residual:.3e} N/N*m"
  )
  if worst_compression.compression_force > 0.0:
    print(
      "   worst released-pad compression "
      f"{worst_compression.compression_force:.2f} N "
      f"(frame {worst_compression.frame} {worst_compression.segment}, "
      f"{worst_compression.compression_name})"
    )
  worst_ground = max(results, key=lambda item: item.ground_normal_max)
  if worst_ground.ground_normal_max > 0.0:
    cone_min = min(
      item.ground_cone_min for item in results if np.isfinite(item.ground_cone_min)
    )
    print(
      "   worst floor sole-point normal "
      f"{worst_ground.ground_normal_max:.2f} N "
      f"(frame {worst_ground.frame} {worst_ground.segment}); "
      f"friction-cone slack min {cone_min:.2f} N"
    )
  print("   per-segment maxima:")
  for segment in segment_names:
    selected = [item for item in results if item.segment == segment]
    if not selected:
      continue
    print(
      f"     {segment:20s} "
      f"anchor {max(item.anchor_util for item in selected):.4f}, "
      f"actuator {max(item.torque_util for item in selected):.4f}"
    )

  if failures:
    print(f"FAIL: {len(failures)}/{len(results)} audited frames are infeasible.")
    for result in failures[:20]:
      worst_force = result.forces.get(result.anchor_name)
      decomp = ""
      if worst_force is not None:
        _set_motion_frame(
          model, data, clip, result.frame, joint_qpos_addr, free_qpos_addr
        )
        t_hat, l_hat, n_hat = engage_basis(
          data, ee_body_ids[result.anchor_name]
        )
        decomp = (
          f", {result.anchor_name}: f_t={worst_force @ t_hat:+.0f} "
          f"f_l={worst_force @ l_hat:+.0f} f_n={worst_force @ n_hat:+.0f} N"
        )
      print(
        f"   frame {result.frame:4d} {result.segment:20s}: "
        f"anchor={result.anchor_util:.4f}, "
        f"actuator={result.torque_util:.4f}, "
        f"eq={result.eq_residual:.2e}, "
        f"solver_success={result.solver_success}{decomp}"
      )
    return False
  solver_warnings = sum(not result.solver_success for result in results)
  print(
    f"PASS: all {len(results)} audited frames have a force-only equilibrium "
    "inside both limits."
  )
  if solver_warnings:
    print(
      f"   note: {solver_warnings} SLSQP exits reported a warning but their "
      "returned points satisfy every audited constraint."
    )
  if not timing_ok:
    print("OVERALL FAIL: static audit passed but the schedule timing did not.")
  return timing_ok


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_file",
    type=Path,
    nargs="?",
    default=_DEFAULT_CLIP,
  )
  parser.add_argument(
    "--f0",
    type=float,
    default=400.0,
    help="pure-rake capacity per pad (N); default is task DR minimum",
  )
  parser.add_argument(
    "--frame-stride",
    type=int,
    default=1,
    help="audit every Nth first-cycle frame (last frame is always included)",
  )
  parser.add_argument(
    "--max-iterations",
    type=int,
    default=300,
    help="maximum SLSQP iterations per frame",
  )
  parser.add_argument("--frames", type=int, nargs="+", help="audit only these frame indices (overrides stride)")
  parser.add_argument(
    "--retry-iterations", type=int, default=3000,
    help="iteration budget for an unaccepted, unsuccessful solve; 0 disables retry",
  )
  parser.add_argument(
    "--compression-contact-tolerance-mm",
    type=float,
    default=0.5,
    help="scheduled-off pads this close to the wall may push normally; "
    "negative disables released-pad contact (default: 0.5 mm)",
  )
  parser.add_argument(
    "--toe-wall-friction",
    type=float,
    default=0.5,
    help="Coulomb friction for released toe/pad wall presses (the sim's "
    "sphere-wall pairs carry 0.8 condim 3; 0.5 is the audit's margin — "
    "the toe press is a designed support forming couples with the "
    "microspine hands)",
  )
  parser.add_argument(
    "--max-compression-force",
    type=float,
    default=500.0,
    help="normal-force cap for each released-pad wall contact (N)",
  )
  parser.add_argument(
    "--ground-friction",
    type=float,
    default=0.6,
    help="Coulomb friction coefficient for scheduled floor supports "
    "(mount clips; the sim floor ships 1.0 — 0.6 is the audit's margin)",
  )
  parser.add_argument(
    "--progress-every",
    type=int,
    default=50,
    help="print progress every N solved frames; 0 disables progress output",
  )
  # Schedule-timing audit constants. Defaults are the climb task's operating
  # point: reengage_cooldown 10 steps / debounce 3 steps at 0.02 s task rate
  # (env_cfgs), capacity_ramp_time_s 0.25 and the mismatch grace_fraction
  # 0.3 (MicrospineLimitSurfaceCfg / reward config).
  parser.add_argument("--skip-timing", action="store_true")
  parser.add_argument("--cooldown-s", type=float, default=0.20)
  parser.add_argument("--debounce-s", type=float, default=0.06)
  parser.add_argument(
    "--maneuver-floor-s",
    type=float,
    default=0.25,
    help="minimum time budgeted for the actual pad travel inside a window",
  )
  parser.add_argument("--ramp-time-s", type=float, default=0.25)
  parser.add_argument("--grace-fraction", type=float, default=0.3)
  args = parser.parse_args()
  if args.f0 <= 0.0:
    parser.error("--f0 must be positive")
  if args.frame_stride <= 0:
    parser.error("--frame-stride must be positive")
  if args.max_compression_force <= 0.0:
    parser.error("--max-compression-force must be positive")
  raise SystemExit(0 if audit(args) else 1)


if __name__ == "__main__":
  main()
