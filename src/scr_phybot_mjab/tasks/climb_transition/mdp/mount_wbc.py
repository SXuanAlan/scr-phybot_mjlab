"""Kinematic whole-body controller for the scripted mount base.

Port of the climb scripted base's WBC (climb-osc-wbc-base branch,
``tasks/climb_wall/mdp/wbc.py``) with the mount's two additions:

  * the anchored set generalizes to welded pads ∪ STANCE FEET — a foot the
    schedule keeps on the floor gets the same 6 zero-velocity rows (at the
    ankle body) that a welded pad gets, so base motion emerges through
    floor contact exactly like it does through anchors;
  * a base-XY anchoring task: flat-pad reach on this wristless arm is a
    function of base standoff (stance v3 note), so the body must stay
    pinned to the clip's reference station instead of drifting on the CoM
    regularizer.

One weighted least-squares velocity solve per control step over
``v = (V_B ∈ R⁶, q̇_j ∈ R²¹)``; the base rows carry no actuator — the
solution's V_B is discarded and only q̇_j is commanded.

Jacobians come from a CPU shadow ``mujoco.MjData`` synced from the live env
state each step. The passive pad-gimbal hinges (2026-08-15 plant) are synced
into the shadow qpos for correct FK, but their velocity columns are simply
never selected (``sel`` keeps base + policy joints only), so the solver
treats pad orientation as the passive DOF it physically is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import pad_contact_center


def orientation_error_w(rot_cur: np.ndarray, rot_tar: np.ndarray) -> np.ndarray:
  """World-frame small-angle error rotating ``rot_cur`` toward ``rot_tar``."""
  return 0.5 * (
    np.cross(rot_cur[:, 0], rot_tar[:, 0])
    + np.cross(rot_cur[:, 1], rot_tar[:, 1])
    + np.cross(rot_cur[:, 2], rot_tar[:, 2])
  )


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
  return mat.reshape(3, 3)


def flat_on_wall_rotation(
  face_local_axis: np.ndarray,
  rake_local_axis: np.ndarray,
  wall_inward_w: np.ndarray,
  rake_down_w: np.ndarray,
) -> np.ndarray:
  """Pad world rotation seating the face into the wall, rake down-wall."""
  f_l = face_local_axis / np.linalg.norm(face_local_axis)
  r_l = rake_local_axis / np.linalg.norm(rake_local_axis)
  t_l = np.cross(f_l, r_l)
  f_w = wall_inward_w / np.linalg.norm(wall_inward_w)
  r_w = rake_down_w - (rake_down_w @ f_w) * f_w
  r_w = r_w / np.linalg.norm(r_w)
  t_w = np.cross(f_w, r_w)
  local = np.stack([f_l, r_l, t_l], axis=1)
  world = np.stack([f_w, r_w, t_w], axis=1)
  return world @ local.T


@dataclass
class MountWbcGains:
  """Task weights (squared, applied to residual rows) and servo gains."""

  w_latch: float = 1.0e4
  w_swing: float = 1.0e2
  w_com: float = 1.0e2
  w_base: float = 1.0e2
  """Kept at the parent's raised value: at 1e1 the solver rotates the base
  backwards to 'achieve' a press (measured on the climb P0)."""

  w_base_xy: float = 30.0
  """Base-station anchoring (horizontal position rows). Below w_com — the
  contact QP's CoM command wins when the load geometry demands a lean, the
  station keeps the residual drift bounded."""

  w_posture: float = 1.0
  reg: float = 1.0e-2

  # Servo gains sized for quasi-static execution on the underdamped leg-PD
  # chain (faster task motion rings a ~3 Hz body wobble that bounces the
  # pressing pad out of the engage debounce window).
  k_swing: float = 1.5
  k_swing_rot: float = 2.0
  k_com: float = 2.0
  """Raised from the parent's 1.0: the mount's single-stance weight
  transfer must finish inside the one-second pre-unload window."""

  k_base: float = 1.5
  k_base_xy: float = 1.0
  k_posture: float = 2.0

  clamp_swing: float = 0.12  # m/s
  clamp_swing_rot: float = 1.5  # rad/s
  clamp_com: float = 0.12  # m/s
  clamp_base: float = 1.0  # rad/s
  clamp_base_xy: float = 0.06  # m/s
  clamp_posture: float = 1.5  # rad/s


@dataclass
class SwingTarget:
  pos_w: np.ndarray
  rot_w: np.ndarray | None = None
  vel_ff_w: np.ndarray | None = None
  point: str = "pad"
  """"pad" = sphere-grid centroid + radius (the physical contact point, for
  seat/release micro-goals in wall coordinates); "origin" = the EE body
  frame origin (what the clip's ee_positions store). The two differ ~22 cm
  on the hand — mixing them was a silent target bias on the climb P0."""

  weight: float | None = None
  rot_weight: float | None = None


@dataclass
class MountWbcSolution:
  qdot_j: np.ndarray
  base_vel: np.ndarray
  task_residuals: dict[str, float] = field(default_factory=dict)


class MountKinematicWbc:
  """Weighted least-squares velocity WBC on a CPU shadow of the env model."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    microspine_cfg: MicrospineCfg,
    policy_joint_names: tuple[str, ...],
    foot_body_names: tuple[str, str],
    robot_prefix: str = "robot",
    gains: MountWbcGains | None = None,
  ) -> None:
    self.model = mj_model  # read-only
    self.data = mujoco.MjData(mj_model)
    self.gains = gains or MountWbcGains()
    self.ee_names = tuple(microspine_cfg.ee_body_names)
    self.policy_joint_names = tuple(policy_joint_names)

    m = mj_model
    self._qadr: dict[str, int] = {}
    self._dofadr: dict[str, int] = {}
    self._range: dict[str, np.ndarray] = {}
    for j in range(m.njnt):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
      self._qadr[name] = int(m.jnt_qposadr[j])
      self._dofadr[name] = int(m.jnt_dofadr[j])
      self._range[name] = np.array(m.jnt_range[j])

    free_jids = np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free_jids) != 1:
      raise ValueError(
        f"Expected exactly one free joint (the robot base); found "
        f"{len(free_jids)}."
      )
    free_jid = int(free_jids[0])
    self.root_bid = int(m.jnt_bodyid[free_jid])
    self._root_qadr = int(m.jnt_qposadr[free_jid])
    base_dofadr = int(m.jnt_dofadr[free_jid])
    self._base_dofs = np.arange(6) + base_dofadr

    def _joint_key(name: str) -> str:
      key = f"{robot_prefix}/{name}"
      if key in self._dofadr:
        return key
      if name in self._dofadr:
        return name
      raise KeyError(f"Joint '{name}' not found in model.")

    self._policy_keys = [_joint_key(n) for n in policy_joint_names]
    joint_dofs = np.array(
      [self._dofadr[k] for k in self._policy_keys], dtype=int
    )
    self.sel = np.concatenate([self._base_dofs, joint_dofs]).astype(int)
    self._nj = len(joint_dofs)

    def _body_id(name: str) -> int:
      for candidate in (f"{robot_prefix}/{name}", name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
          return bid
      raise KeyError(f"Body '{name}' not found in model.")

    self.ee_bids = [_body_id(name) for name in self.ee_names]
    self.foot_bids = [_body_id(name) for name in foot_body_names]
    self.foot_body_names = tuple(foot_body_names)

    # Contact-array center C, independent of proxy shape and gimbal pose.
    self.pad_local: list[np.ndarray] = []
    for ee, bid in zip(self.ee_names, self.ee_bids):
      self.pad_local.append(
        pad_contact_center(
          m, microspine_cfg, ee,
          m.body(bid).name.removesuffix(ee).rstrip("/"),
        )
      )

    # Heel/toe sole points in each foot body frame, from the union AABB of
    # the foot's collision geoms (the reference validator's ground model).
    corner_signs = np.array(
      [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
      dtype=np.float64,
    )
    self.foot_sole_local: list[list[np.ndarray]] = []
    for bid in self.foot_bids:
      lo = np.full(3, np.inf)
      hi = np.full(3, -np.inf)
      for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
          continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
          continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        corners = m.geom_pos[g] + (
          m.geom_aabb[g, :3] + m.geom_aabb[g, 3:] * corner_signs
        ) @ rot.reshape(3, 3).T
        lo = np.minimum(lo, corners.min(axis=0))
        hi = np.maximum(hi, corners.max(axis=0))
      sole_z = float(lo[2]) + 0.002
      self.foot_sole_local.append(
        [
          np.array([float(lo[0]) + 0.01, 0.0, sole_z]),  # heel
          np.array([float(hi[0]) - 0.01, 0.0, sole_z]),  # toe
        ]
      )

    self.robot_mass = float(m.body_subtreemass[self.root_bid])
    self._face_local = np.asarray(microspine_cfg.face_local_axis, float)
    self._rake_local = np.asarray(
      microspine_cfg.limit_surface.ee_rake_local_axis, float
    )

  # -- state sync ------------------------------------------------------------

  def sync(
    self,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: dict[str, float],
  ) -> None:
    """Write the live env state into the shadow and run FK."""
    d = self.data
    d.qpos[self._root_qadr : self._root_qadr + 3] = root_pos
    d.qpos[self._root_qadr + 3 : self._root_qadr + 7] = root_quat_wxyz
    for name, value in joint_pos.items():
      key = name if name in self._qadr else f"robot/{name}"
      if key in self._qadr:
        d.qpos[self._qadr[key]] = value
    mujoco.mj_forward(self.model, d)

  # -- kinematic readouts ----------------------------------------------------

  def pad_pose(self, ee_index: int) -> tuple[np.ndarray, np.ndarray]:
    bid = self.ee_bids[ee_index]
    rot = self.data.xmat[bid].reshape(3, 3)
    pos = self.data.xpos[bid] + rot @ self.pad_local[ee_index]
    return pos.copy(), rot.copy()

  def origin_pose(self, ee_index: int) -> tuple[np.ndarray, np.ndarray]:
    bid = self.ee_bids[ee_index]
    return (
      self.data.xpos[bid].copy(),
      self.data.xmat[bid].reshape(3, 3).copy(),
    )

  def toe_tip_z(self, foot_index: int) -> float:
    """World height of the bracing-toe sphere center minus its radius —
    the toe's actual clearance over the floor. Constants from
    PHYBOT_C2_FOOT_TOES: sphere at (0.150, 0, −0.049) r=0.012 on the
    ankle_roll body."""
    bid = self.foot_bids[foot_index]
    rot = self.data.xmat[bid].reshape(3, 3)
    pos = self.data.xpos[bid] + rot @ np.array([0.150, 0.0, -0.049])
    return float(pos[2]) - 0.012

  def foot_sole_points_w(self, foot_index: int) -> list[np.ndarray]:
    """World heel/toe sole points of a foot (ground-contact geometry)."""
    bid = self.foot_bids[foot_index]
    rot = self.data.xmat[bid].reshape(3, 3)
    pos = self.data.xpos[bid]
    return [pos + rot @ p for p in self.foot_sole_local[foot_index]]

  def pad_rot_rows(self, ee_index: int) -> np.ndarray:
    """(3, J) joint-space rows mapping q̇_j to the pad's angular velocity
    (world). Used by the weld-torque relax loop: the wristless arm renders
    any configuration change on a welded pad as pad torque, and these rows
    point back down the twist."""
    m, d = self.model, self.data
    bid = self.ee_bids[ee_index]
    pos, _ = self.pad_pose(ee_index)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, pos, bid)
    return jacr[:, self.sel][:, 6:].copy()

  def pad_axis_joint_row(self, ee_index: int, axis_w: np.ndarray) -> np.ndarray:
    """(J,) joint-space row mapping q̇_j to pad-center velocity along
    ``axis_w`` — the dedicated press channel the whole-body mixer cannot
    dilute."""
    m, d = self.model, self.data
    bid = self.ee_bids[ee_index]
    pos, _ = self.pad_pose(ee_index)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, pos, bid)
    row = np.asarray(axis_w, dtype=np.float64) @ jacp[:, self.sel]
    return row[6:].copy()

  def com(self) -> np.ndarray:
    return self.data.subtree_com[self.root_bid].copy()

  def reference_com(
    self,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    policy_joint_pos: np.ndarray,
  ) -> np.ndarray:
    """CoM of a reference pose (clip frame). Overwrites the shadow state —
    callers batch this at init and re-sync afterwards."""
    d = self.data
    d.qpos[self._root_qadr : self._root_qadr + 3] = root_pos
    d.qpos[self._root_qadr + 3 : self._root_qadr + 7] = root_quat_wxyz
    for key, val in zip(self._policy_keys, policy_joint_pos):
      d.qpos[self._qadr[key]] = val
    mujoco.mj_forward(self.model, d)
    return d.subtree_com[self.root_bid].copy()

  def base_pos(self) -> np.ndarray:
    return self.data.xpos[self.root_bid].copy()

  def base_rot(self) -> np.ndarray:
    return self.data.xmat[self.root_bid].reshape(3, 3).copy()

  def joint_pos(self) -> np.ndarray:
    return np.array(
      [self.data.qpos[self._qadr[k]] for k in self._policy_keys]
    )

  def joint_range(self) -> np.ndarray:
    return np.stack([self._range[k] for k in self._policy_keys])

  def policy_bias_forces(self) -> np.ndarray:
    """(J,) generalized bias force at the policy joints from the last
    ``sync`` (shadow qvel is never written, so this is the pure gravity
    load). Statics: the PD must produce exactly this torque, so
    ``q_ref + bias/kp`` is the sag-free position command."""
    return self.data.qfrc_bias[self.sel[6:]].copy()

  # -- solver ----------------------------------------------------------------

  def solve(
    self,
    latched: np.ndarray,
    stance_feet: tuple[bool, bool],
    swing_targets: dict[int, SwingTarget],
    com_target: np.ndarray | None,
    base_rot_target: np.ndarray | None,
    base_xy_target: np.ndarray | None,
    posture_ref: np.ndarray | None,
    dof_reg: np.ndarray | None = None,
  ) -> MountWbcSolution:
    """One velocity solve.

    ``latched`` masks pads held by the physics weld; ``stance_feet`` masks
    feet the schedule keeps planted (6 zero-velocity rows at the ankle
    body each). ``dof_reg``: optional (6+J,) extra Tikhonov weight per
    velocity DOF — the least-norm solution otherwise routes a hand press
    through knees carrying the body weight.
    """
    m, d, g = self.model, self.data, self.gains
    nv = m.nv
    n_sel = len(self.sel)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    rows_j: list[np.ndarray] = []
    rows_e: list[np.ndarray] = []
    rows_w: list[float] = []
    spans: dict[str, slice] = {}

    def _push(tag: str, jac: np.ndarray, err: np.ndarray, w2: float) -> None:
      if tag in spans:
        raise ValueError(f"Duplicate WBC task tag '{tag}'.")
      start = sum(r.shape[0] for r in rows_j)
      rows_j.append(jac)
      rows_e.append(err)
      rows_w.extend([w2] * jac.shape[0])
      spans[tag] = slice(start, start + jac.shape[0])

    def _clamped(vec: np.ndarray, limit: float) -> np.ndarray:
      norm = float(np.linalg.norm(vec))
      if norm > limit > 0.0:
        return vec * (limit / norm)
      return vec

    for h, bid in enumerate(self.ee_bids):
      name = self.ee_names[h]
      if latched[h]:
        pos, rot = self.pad_pose(h)
        mujoco.mj_jac(m, d, jacp, jacr, pos, bid)
        _push(f"latch:{name}", jacp[:, self.sel].copy(), np.zeros(3), g.w_latch)
        _push(
          f"latch_rot:{name}", jacr[:, self.sel].copy(), np.zeros(3), g.w_latch
        )
      elif h in swing_targets:
        target = swing_targets[h]
        if target.point == "origin":
          pos, rot = self.origin_pose(h)
        else:
          pos, rot = self.pad_pose(h)
        mujoco.mj_jac(m, d, jacp, jacr, pos, bid)
        w_row = target.weight if target.weight is not None else g.w_swing
        vel = g.k_swing * (target.pos_w - pos)
        vel = _clamped(vel, g.clamp_swing)
        if target.vel_ff_w is not None:
          vel = vel + target.vel_ff_w
        _push(f"swing:{name}", jacp[:, self.sel].copy(), vel, w_row)
        if target.rot_w is not None:
          w_rot = (
            target.rot_weight if target.rot_weight is not None else w_row
          )
          omega = g.k_swing_rot * orientation_error_w(rot, target.rot_w)
          omega = _clamped(omega, g.clamp_swing_rot)
          _push(f"swing_rot:{name}", jacr[:, self.sel].copy(), omega, w_rot)

    for f, bid in enumerate(self.foot_bids):
      if stance_feet[f]:
        mujoco.mj_jacBody(m, d, jacp, jacr, bid)
        name = self.foot_body_names[f]
        _push(
          f"stance:{name}", jacp[:, self.sel].copy(), np.zeros(3), g.w_latch
        )
        _push(
          f"stance_rot:{name}",
          jacr[:, self.sel].copy(),
          np.zeros(3),
          g.w_latch,
        )

    if com_target is not None:
      jac_com = np.zeros((3, nv))
      mujoco.mj_jacSubtreeCom(m, d, jac_com, self.root_bid)
      vel = _clamped(g.k_com * (com_target - self.com()), g.clamp_com)
      _push("com", jac_com[:, self.sel].copy(), vel, g.w_com)

    if base_rot_target is not None:
      mujoco.mj_jacBody(m, d, jacp, jacr, self.root_bid)
      omega = g.k_base * orientation_error_w(self.base_rot(), base_rot_target)
      omega = _clamped(omega, g.clamp_base)
      _push("base_ori", jacr[:, self.sel].copy(), omega, g.w_base)

    if base_xy_target is not None:
      mujoco.mj_jacBody(m, d, jacp, jacr, self.root_bid)
      err_xy = np.asarray(base_xy_target, dtype=np.float64) - self.base_pos()[:2]
      vel_xy = _clamped(g.k_base_xy * err_xy, g.clamp_base_xy)
      _push("base_xy", jacp[:2, self.sel].copy(), vel_xy, g.w_base_xy)

    if posture_ref is not None:
      jac_post = np.zeros((self._nj, n_sel))
      jac_post[:, 6:] = np.eye(self._nj)
      qerr = g.k_posture * (posture_ref - self.joint_pos())
      qerr = np.clip(qerr, -g.clamp_posture, g.clamp_posture)
      _push("posture", jac_post, qerr, g.w_posture)

    jac = np.vstack(rows_j)
    err = np.concatenate(rows_e)
    w2 = np.asarray(rows_w, dtype=np.float64)
    reg_diag = np.full(n_sel, g.reg)
    if dof_reg is not None:
      reg_diag = reg_diag + np.asarray(dof_reg, dtype=np.float64)
    a = (jac * w2[:, None]).T @ jac + np.diag(reg_diag)
    b = (jac * w2[:, None]).T @ err
    v = np.linalg.solve(a, b)

    achieved = jac @ v
    residuals = {
      tag: float(np.linalg.norm(achieved[sl] - err[sl]))
      for tag, sl in spans.items()
    }
    return MountWbcSolution(
      qdot_j=v[6:].copy(),
      base_vel=v[:6].copy(),
      task_residuals=residuals,
    )
