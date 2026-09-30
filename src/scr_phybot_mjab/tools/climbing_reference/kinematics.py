"""Shared whole-body IK and task construction for offline climbing references."""

from __future__ import annotations

import dataclasses
import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_FOOT_SITES,
  PHYBOT_C2_FOOT_TOES,
  get_spec,
)
from scr_phybot_mjab.assets.wall.wall_constants import WALL_HALF_SIZE, WALL_OFFSET
from scr_phybot_mjab.microspine.contact import (
  ToeContact,
  add_toe_geoms,
  pad_contact_center,
  sphere_names,
  toe_site_name,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg

WALL_SURFACE_X = float(WALL_OFFSET[0] - WALL_HALF_SIZE[0])
PAD_TOUCH_X = WALL_SURFACE_X  # IK pad sites now represent the contact face C.

FOOT_SITE_NAMES = tuple(site for _, site in PHYBOT_C2_FOOT_SITES)


def smootherstep(x: float | np.ndarray) -> float | np.ndarray:
  x = np.clip(x, 0.0, 1.0)
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def orientation_error_world(rot_cur: np.ndarray, rot_tar: np.ndarray) -> np.ndarray:
  """World-frame small-angle error rotating ``rot_cur`` toward ``rot_tar``."""
  return 0.5 * (
    np.cross(rot_cur[:, 0], rot_tar[:, 0])
    + np.cross(rot_cur[:, 1], rot_tar[:, 1])
    + np.cross(rot_cur[:, 2], rot_tar[:, 2])
  )


def bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
  u = 1.0 - s
  return u * u * p0 + 2.0 * u * s * p1 + s * s * p2


def slerp_matrix(rot0: np.ndarray, rot1: np.ndarray, s: float) -> np.ndarray:
  """Geodesic interpolation between two rotation matrices."""
  q0, q1 = np.zeros(4), np.zeros(4)
  mujoco.mju_mat2Quat(q0, rot0.reshape(9))
  mujoco.mju_mat2Quat(q1, rot1.reshape(9))
  q0_inv, q_rel = np.zeros(4), np.zeros(4)
  mujoco.mju_negQuat(q0_inv, q0)
  mujoco.mju_mulQuat(q_rel, q1, q0_inv)
  vel = np.zeros(3)
  mujoco.mju_quat2Vel(vel, q_rel, 1.0)
  angle = float(np.linalg.norm(vel))
  q_s = np.array([1.0, 0.0, 0.0, 0.0])
  if angle > 1e-12:
    mujoco.mju_axisAngle2Quat(q_s, vel / angle, angle * float(s))
  q_out, mat = np.zeros(4), np.zeros(9)
  mujoco.mju_mulQuat(q_out, q_s, q0)
  mujoco.mju_quat2Mat(mat, q_out)
  return mat.reshape(3, 3)


def rotation_angle(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
  """Angle (rad) of the relative rotation between two rotation matrices."""
  cos = 0.5 * (np.trace(rot_a.T @ rot_b) - 1.0)
  return float(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclasses.dataclass
class IKTask:
  """One weighted IK target: a pad site or the floating base."""

  kind: str  # "site" | "base"
  sid: int
  pos: np.ndarray
  rot: np.ndarray | None
  w_pos: np.ndarray  # (3,) per-axis row weight
  w_rot: float
  track: bool = True  # counts toward the convergence residual
  hard: bool = False  # top-priority row block (null-space hierarchy)


@dataclasses.dataclass
class IKResidual:
  pos: float = 0.0
  rot: float = 0.0

  def merge(self, other: "IKResidual") -> None:
    self.pos = max(self.pos, other.pos)
    self.rot = max(self.rot, other.rot)


class ClimbIK:
  """Damped-least-squares whole-body IK on the C2 model with pad sites.

  ``limit_margin_frac`` / ``limit_weight`` add per-joint limit-avoidance
  springs to every solve: a joint inside the margin zone (that fraction of
  its full range next to either hard stop) is pulled back to the zone
  boundary with weight ``limit_weight``. Clearance from the stops is thereby
  part of the optimization — welds still dominate, but all null-space slack
  is spent away from the limits instead of on the posture reference alone.
  """

  def __init__(
    self,
    limit_margin_frac: float = 0.10,
    limit_weight: float = 3.0,
    initial_state: dict | None = None,
  ) -> None:
    self.limit_margin_frac = float(limit_margin_frac)
    self.limit_weight = float(limit_weight)
    # Per-joint limit-spring weight (default: ``limit_weight``). A joint whose
    # range is a hard mechanical stop the solver must not overshoot (the pad
    # gimbals) gets a weight that dominates the task rows.
    self.limit_weight_overrides: dict[str, float] = {}
    self.debug_hard = False  # print hard-level diagnostics for poorly converged solves
    cfg = phybot_c2_climb_wall_env_cfg()
    robot_cfg = cfg.scene.entities["robot"]
    self.init_base_pos = np.asarray(robot_cfg.init_state.pos, dtype=np.float64)
    self.init_joint_pos = dict(robot_cfg.init_state.joint_pos)
    if initial_state is not None:
      self.init_base_pos = np.asarray(initial_state["root_position"], dtype=np.float64)
      self.init_joint_pos = dict(initial_state["joint_positions"])
    microspine_cfg = cfg.commands["microspine_affordance"].microspine_cfg
    self.ee_names: tuple[str, ...] = tuple(microspine_cfg.ee_body_names)
    rows = microspine_cfg.spheres_per_ee_rows
    cols = microspine_cfg.spheres_per_ee_cols
    # A 30 mm crawl step keeps the low, wristless support arm away from its
    # shoulder-roll stop during the one-hand support beats.  The old 60 mm
    # hop was kinematically possible only by spending nearly all of that
    # joint's remaining range.
    self.default_step_height = 0.03

    spec = get_spec()
    self.pad_site_names = {}
    for ee in self.ee_names:
      center = pad_contact_center(spec, microspine_cfg, ee, robot_prefix="")
      site_name = f"{ee}_pad_center"
      spec.body(ee).add_site(name=site_name, pos=center, size=[0.005, 0.005, 0.005])
      self.pad_site_names[ee] = site_name

    # Foot toes: the bracing tips a released foot presses against the wall
    # instead of sliding its spine pad (same definition the climb env's
    # spec_fn instantiates for the physical contact pairs).
    self.toes = {
      ee: ToeContact(ee_body_name=ee, body_name=body, pos=pos, radius=radius)
      for ee, body, pos, radius in PHYBOT_C2_FOOT_TOES
    }
    add_toe_geoms(spec, tuple(self.toes.values()), robot_prefix="", with_sites=True)

    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model

    self.qpos_addresses = {}
    self.dof_addresses = {}
    self.joint_ranges = {}
    for j in range(m.njnt):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
      self.qpos_addresses[name] = int(m.jnt_qposadr[j])
      self.dof_addresses[name] = int(m.jnt_dofadr[j])
      self.joint_ranges[name] = np.array(m.jnt_range[j])

    free_jid = int(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0])
    self.root_bid = int(m.jnt_bodyid[free_jid])
    self.base_dofs = np.arange(6) + int(m.jnt_dofadr[free_jid])

    self.pad_sid = {ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for ee, s in self.pad_site_names.items()}
    self.toe_sid = {
      ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, toe_site_name(toe)) for ee, toe in self.toes.items()
    }
    self.ee_bid = {ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee) for ee in self.ee_names}
    self.foot_sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in FOOT_SITE_NAMES]
    self.sphere_bids = {
      ee: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in sphere_names(ee, rows, cols)]
      for ee in self.ee_names
    }

  # -- state helpers ---------------------------------------------------------

  def set_state1(self) -> None:
    d = self.data
    d.qpos[:] = 0.0
    d.qpos[0:3] = self.init_base_pos
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in self.init_joint_pos.items():
      d.qpos[self.qpos_addresses[name]] = angle
    mujoco.mj_forward(self.model, d)

  def pad_pose(self, ee: str) -> tuple[np.ndarray, np.ndarray]:
    sid = self.pad_sid[ee]
    return (
      self.data.site_xpos[sid].copy(),
      self.data.site_xmat[sid].reshape(3, 3).copy(),
    )

  def dof_selection(self, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([self.base_dofs, [self.dof_addresses[n] for n in joint_names]]).astype(int)

  # -- solver ----------------------------------------------------------------

  def solve_frame(
    self,
    tasks: list[IKTask],
    sel: np.ndarray,
    posture_ref: dict[str, float],
    posture_weight: float,
    iters: int = 100,
    tol_pos: float = 1.5e-4,
    tol_rot: float = 2.0e-3,
    tol_dq: float = 1.0e-4,
    lam2: float = 0.5,
    max_step: float = 0.12,
    posture_weight_overrides: dict[str, float] | None = None,
  ) -> IKResidual:
    m, d = self.model, self.data
    nv = m.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    sel_set = set(sel.tolist())
    sel_joints = [(name, self.dof_addresses[name]) for name in posture_ref if self.dof_addresses[name] in sel_set]
    sel_index = {dof: i for i, dof in enumerate(sel.tolist())}

    residual = IKResidual()
    for _ in range(iters):
      mujoco.mj_forward(m, d)
      rows_j, rows_e, rows_w = [], [], []
      residual = IKResidual()
      hard_j, hard_e = [], []
      for task in tasks:
        if task.kind == "site":
          pos = d.site_xpos[task.sid]
          rot = d.site_xmat[task.sid].reshape(3, 3)
          mujoco.mj_jacSite(m, d, jacp, jacr, task.sid)
        else:
          pos = d.xpos[self.root_bid]
          rot = d.xmat[self.root_bid].reshape(3, 3)
          mujoco.mj_jacBody(m, d, jacp, jacr, self.root_bid)
        err_p = task.pos - pos
        # Hard rows (welds under ``hard``): top priority, solved exactly
        # (damped only for conditioning); every soft row then moves in
        # their null space. A welded pad therefore never drifts -- the
        # infeasibility lands on the swing/base targets, where the physics
        # puts it.
        dst_j, dst_e = (hard_j, hard_e) if task.hard else (rows_j, rows_e)
        dst_j.append(jacp[:, sel].copy())
        dst_e.append(err_p)
        if not task.hard:
          rows_w.append(np.asarray(task.w_pos, dtype=np.float64))
        if task.track:
          residual.pos = max(residual.pos, float(np.linalg.norm(err_p)))
        if task.rot is not None:
          err_r = orientation_error_world(rot, task.rot)
          dst_j.append(jacr[:, sel].copy())
          dst_e.append(err_r)
          if not task.hard:
            rows_w.append(np.full(3, task.w_rot))
          if task.track:
            residual.rot = max(residual.rot, float(np.linalg.norm(err_r)))

      if rows_j:
        jac = np.vstack(rows_j)
        err = np.concatenate(rows_e)
        w2 = np.concatenate(rows_w) ** 2
        a = (jac * w2[:, None]).T @ jac + lam2 * np.eye(len(sel))
        b = (jac * w2[:, None]).T @ err
      else:
        a = lam2 * np.eye(len(sel))
        b = np.zeros(len(sel))
      overrides = posture_weight_overrides or {}
      for name, dof in sel_joints:
        i = sel_index[dof]
        w_lim2 = self.limit_weight_overrides.get(name, self.limit_weight) ** 2
        w_post2 = overrides.get(name, posture_weight) ** 2
        a[i, i] += w_post2
        b[i] += w_post2 * (posture_ref[name] - d.qpos[self.qpos_addresses[name]])
        # Limit-avoidance spring: active only inside the margin zone, pulls
        # back to the zone boundary. Zero force in the interior, so it never
        # fights the posture reference where clearance is healthy.
        lo, hi = self.joint_ranges[name]
        margin = self.limit_margin_frac * (hi - lo)
        q = d.qpos[self.qpos_addresses[name]]
        if q < lo + margin:
          a[i, i] += w_lim2
          b[i] += w_lim2 * ((lo + margin) - q)
        elif q > hi - margin:
          a[i, i] += w_lim2
          b[i] += w_lim2 * ((hi - margin) - q)

      if hard_j:
        # Exact (rank-truncated) projector: a damped pseudo-inverse leaks
        # the soft rows into the hard ones along near-singular directions
        # (measured: 40-54 mm weld residuals at lambda 1e-6).
        jac1 = np.vstack(hard_j)
        err1 = np.concatenate(hard_e)
        n_sel = len(sel)
        u1, s1, vt1 = np.linalg.svd(jac1, full_matrices=False)
        _hard_stats = (jac1.shape[0], float(s1[0]), float(s1[-1]), float(np.linalg.norm(err1)))
        rank = int(np.sum(s1 > 1.0e-9 * s1[0])) if s1.size else 0
        u1, s1, vt1 = u1[:, :rank], s1[:rank], vt1[:rank]
        # Damped step along the hard rows (conditioning only), exact null space.
        # lam 0.03: an undamped hard step along a near-singular direction
        # (straight arm) swallowed the whole max_step budget every iteration.
        dq1 = vt1.T @ ((s1 / (s1 * s1 + 1.0e-3)) * (u1.T @ err1))
        null1 = np.eye(n_sel) - vt1.T @ vt1
        a2 = null1 @ a @ null1 + 1.0e-9 * np.eye(n_sel)
        b2 = null1 @ (b - a @ dq1)
        dq = dq1 + null1 @ np.linalg.solve(a2, b2)
      else:
        dq = np.linalg.solve(a, b)
      # Converged only when the tracked (hard) errors are met AND the soft
      # tasks are stationary. Checking the hard errors alone would freeze the
      # base whenever the welds are already satisfied by the warm start.
      if residual.pos < tol_pos and residual.rot < tol_rot and float(np.abs(dq).max()) < tol_dq:
        break
      peak = float(np.abs(dq).max())
      if peak > max_step:
        dq *= max_step / peak
      qvel = np.zeros(nv)
      qvel[sel] = dq
      mujoco.mj_integratePos(m, d.qpos, qvel, 1.0)
      for name, (lo, hi) in self.joint_ranges.items():
        if self.dof_addresses[name] not in sel_set:
          continue
        adr = self.qpos_addresses[name]
        d.qpos[adr] = np.clip(d.qpos[adr], lo + 0.005, hi - 0.005)

    mujoco.mj_forward(m, d)
    if self.debug_hard and hard_j and (residual.pos > 2.0e-3 or residual.rot > 0.02):
      print(
        f"    [hard-ik] rows {_hard_stats[0]} smax {_hard_stats[1]:.3f} smin {_hard_stats[2]:.2e} "
        f"|err1| {_hard_stats[3] * 1000:.1f} mm -> residual pos {residual.pos * 1000:.1f} mm rot {np.degrees(residual.rot):.1f} deg "
        f"iters {_ + 1} peak_dq {float(np.abs(dq).max()):.3f}"
      )
    return residual


def weld_task(ik: ClimbIK, ee: str, anchor: tuple[np.ndarray, np.ndarray]) -> IKTask:
  return IKTask(
    kind="site",
    sid=ik.pad_sid[ee],
    pos=anchor[0],
    rot=anchor[1],
    # Support welds are hard kinematic facts; posture and swing shaping must
    # spend their error elsewhere.
    w_pos=np.full(3, 1000.0),
    w_rot=500.0,
  )


def toe_task(
  ik: ClimbIK,
  ee: str,
  pos: np.ndarray,
  w_pos: tuple[float, float, float] = (300.0, 30.0, 30.0),
) -> IKTask:
  """Position-only brace target for a foot's toe tip.

  The 2-DoF ankle cannot satisfy all three components exactly, so the wall
  press (x) dominates and y/z absorb the residual; ``track=False`` keeps the
  intentionally-unreachable components out of the convergence residual (toe
  quality is reported separately).
  """
  return IKTask(
    kind="site",
    sid=ik.toe_sid[ee],
    pos=pos,
    rot=None,
    w_pos=np.asarray(w_pos, dtype=np.float64),
    w_rot=0.0,
    track=False,
  )


def base_task(
  ik: ClimbIK,
  pos: np.ndarray,
  w_pos: tuple[float, float, float],
  rot: np.ndarray | None = None,
  w_rot: float = 2.0,
) -> IKTask:
  return IKTask(
    kind="base",
    sid=-1,
    pos=np.asarray(pos, dtype=np.float64),
    rot=np.eye(3) if rot is None else rot,
    w_pos=np.asarray(w_pos, dtype=np.float64),
    w_rot=w_rot,
    track=False,
  )
