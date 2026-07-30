"""Generate a PHYBot C2 wall-climb regrip cycle clip via whole-body IK.

The reference is produced directly on the robot's own kinematics (no mocap /
retargeting), like ``gen_squat_clip.py`` / ``gen_stand_clip.py``. One cycle
uses a four-beat crawl that remains feasible with a finite pad-moment limit:

  state 1   climb-task reset stance ("climb zero"): all four microspine pads
            engaged on the wall, arms overhead, legs in the frog splay.
  1 -> 2    LEFT-FOOT REGRIP: only the left foot releases and travels one
            step upward on an off-wall arc while the TOE of the articulated
            foot (planted against the wall during the preceding dwell)
            braces the wall as an ordinary compression contact; both hands
            and the right foot remain anchored. The released spine pad
            never touches the wall mid-swing — dragging unloaded spine tips
            across asperities damages them — so the toe supplies the
            support the sliding pad used to.
  2 -> 3    RIGHT-FOOT REGRIP: mirror of 1 -> 2 (the intervening four-pad
            dwell retracts the left toe and plants the right one). The base
            finishes rising one step.
  3 -> 4    LEFT-HAND REGRIP: the left hand releases, swings along an off-wall
            arc to one step height above its old anchor, and engages.
  4 -> 5    RIGHT-HAND REGRIP: mirror of 3 -> 4. State 5 equals state 1
            translated up by exactly one step height (enforced by a closure
            blend over the tail of the last swing), so cycles tile vertically
            (``--cycles``) and the clip loops as a gait reference. All dwells
            sit on four engaged anchors; only the swings move a limb.

During every free swing exactly one pad is released (the hand swings begin
with a short all-four load-transfer dwell). The remaining three always
include at least one hand and one foot, preserving the meter-scale
upper-tension/lower-compression couple that balances gravity's wall-normal
moment. The braced toe (a pair-only sphere at the ankle_roll foot tip, see
``PHYBOT_C2_FOOT_TOES``) stays planted at one world point through its whole
swing — the ankle absorbs the 30 mm shin rise — so it adds compression
margin with zero sliding anywhere. The old two-foot hop removed the large
couple and demanded that two nearly level hand pads carry the full
49--60 N*m as local bending.

Every landing ends with a pre-engage RAKE: the pad first touches the wall
``--rake-dist`` (default 2 mm) above its scheduled anchor and then drags that
distance straight down the wall surface over the last ``--rake-s`` of the
swing. Microspines need a small downward displacement under compression to
load their tips into asperities, so the reference bakes that seating stroke in
before the engaged flag flips at the segment boundary.

Engaged pads are 6-DoF weld tasks at their anchors (the sim's engaged
microspine is a weld: ``anchor_lock_rotation=True``); the 4-DoF arms cannot
satisfy two hand welds from a prescribed base pose, so the base is a soft task
and tilts/sinks the way the hanging robot physically would. Swing pads track
position exactly and orientation with a weight that ramps up toward landing
(a wristless arm has no spare DoF mid-swing).

Output extends the AMP/motion ``.npy`` schema (``joints_list``,
``joint_positions``, ``root_position``, ``root_quaternion`` xyzw,
``foot_sites_list``, ``foot_positions``, ``fps`` — accepted by
``scripts/view_motion.py`` and ``scripts/validation/check_amp_dataset.py``)
with climb-specific keys:

    ee_list               list[str]         4 spine bodies, microspine order
                                            (left/right hand, left/right foot)
    ee_positions          (T, 4, 3) float32 spine body origins, world
    ee_quaternions        (T, 4, 4) float32 spine body orientation, xyzw
    pad_positions         (T, 4, 3) float32 pad centers (contact-sphere grid
                                            center), world
    ee_engaged            (T, 4)    uint8   1 while the pad is anchored
    ee_anchor_positions   (T, 4, 3) float32 pad-center anchor the weld holds
                                            (engaged) or the upcoming target
                                            anchor (swinging)
    toe_braced            (T, 2)    uint8   1 inside a foot swing's full-press
                                            toe-brace window (toe_list order)
    toe_list              list[str]         foot EEs owning a bracing toe
    phase_segment         (T,)      uint8   index into segment_names
    segment_names         list[str]
    cycle_index           (T,)      uint16
    step_height           float32           per-cycle rise of every anchor
    wall_surface_x        float32           wall plane
    pad_touch_x           float32           wall minus contact-sphere radius

Validate/preview:
    python scripts/view_motion.py \
      src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy
    python scripts/validation/check_amp_dataset.py --data-dir <output dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Initialize MJLab task entry points before importing PHYBot modules (see
# gen_squat_clip.py).
import mjlab  # noqa: F401
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_FOOT_SITES,
  PHYBOT_C2_FOOT_TOES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  get_spec,
)
from scr_phybot_mjab.assets.wall.wall_constants import (  # noqa: E402
  WALL_HALF_SIZE,
  WALL_OFFSET,
)
from scr_phybot_mjab.microspine.contact import (  # noqa: E402
  ToeContact,
  add_toe_geoms,
  sphere_names,
  toe_site_name,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (  # noqa: E402
  phybot_c2_climb_wall_env_cfg,
)

_WALL_SURFACE_X = float(WALL_OFFSET[0] - WALL_HALF_SIZE[0])
_SPHERE_RADIUS = 0.012
_PAD_TOUCH_X = _WALL_SURFACE_X - _SPHERE_RADIUS

_FOOT_SITE_NAMES = tuple(site for _, site in PHYBOT_C2_FOOT_SITES)

_SEGMENT_NAMES = (
  "hold_start",         # state 1
  "left_foot_swing",    # 1 -> 2
  "foot_transfer_hold", # state 2
  "right_foot_swing",   # 2 -> 3
  "seat_hold",          # state 3
  "left_hand_swing",    # 3 -> 4
  "regrip_hold",        # state 4
  "right_hand_swing",   # 4 -> 5
  "end_hold",           # state 5 == state 1 + step height
)


def _smootherstep(x: float | np.ndarray) -> float | np.ndarray:
  x = np.clip(x, 0.0, 1.0)
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _orientation_error_w(rot_cur: np.ndarray, rot_tar: np.ndarray) -> np.ndarray:
  """World-frame small-angle error rotating ``rot_cur`` toward ``rot_tar``."""
  return 0.5 * (
    np.cross(rot_cur[:, 0], rot_tar[:, 0])
    + np.cross(rot_cur[:, 1], rot_tar[:, 1])
    + np.cross(rot_cur[:, 2], rot_tar[:, 2])
  )


def _bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
  u = 1.0 - s
  return u * u * p0 + 2.0 * u * s * p1 + s * s * p2


def _slerp_mat(rot0: np.ndarray, rot1: np.ndarray, s: float) -> np.ndarray:
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


def _rot_angle(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
  """Angle (rad) of the relative rotation between two rotation matrices."""
  cos = 0.5 * (np.trace(rot_a.T @ rot_b) - 1.0)
  return float(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclasses.dataclass
class _Task:
  """One weighted IK target: a pad site or the floating base."""

  kind: str  # "site" | "base"
  sid: int
  pos: np.ndarray
  rot: np.ndarray | None
  w_pos: np.ndarray  # (3,) per-axis row weight
  w_rot: float
  track: bool = True  # counts toward the convergence residual


@dataclasses.dataclass
class _Residual:
  pos: float = 0.0
  rot: float = 0.0

  def merge(self, other: "_Residual") -> None:
    self.pos = max(self.pos, other.pos)
    self.rot = max(self.rot, other.rot)


class _ClimbIK:
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
  ) -> None:
    self.limit_margin_frac = float(limit_margin_frac)
    self.limit_weight = float(limit_weight)
    cfg = phybot_c2_climb_wall_env_cfg()
    robot_cfg = cfg.scene.entities["robot"]
    self.init_base_pos = np.asarray(robot_cfg.init_state.pos, dtype=np.float64)
    self.init_joint_pos = dict(robot_cfg.init_state.joint_pos)
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
      centers = np.array(
        [spec.body(name).pos for name in sphere_names(ee, rows, cols)]
      )
      site_name = f"{ee}_pad_center"
      spec.body(ee).add_site(
        name=site_name, pos=centers.mean(axis=0), size=[0.005, 0.005, 0.005]
      )
      self.pad_site_names[ee] = site_name

    # Foot toes: the bracing tips a released foot presses against the wall
    # instead of sliding its spine pad (same definition the climb env's
    # spec_fn instantiates for the physical contact pairs).
    self.toes = {
      ee: ToeContact(
        ee_body_name=ee, body_name=body, pos=pos, radius=radius
      )
      for ee, body, pos, radius in PHYBOT_C2_FOOT_TOES
    }
    add_toe_geoms(
      spec, tuple(self.toes.values()), robot_prefix="", with_sites=True
    )

    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model

    self._qadr = {}
    self._dofadr = {}
    self._range = {}
    for j in range(m.njnt):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
      self._qadr[name] = int(m.jnt_qposadr[j])
      self._dofadr[name] = int(m.jnt_dofadr[j])
      self._range[name] = np.array(m.jnt_range[j])

    free_jid = int(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0])
    self.root_bid = int(m.jnt_bodyid[free_jid])
    self._base_dofs = np.arange(6) + int(m.jnt_dofadr[free_jid])

    self.pad_sid = {
      ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
      for ee, s in self.pad_site_names.items()
    }
    self.toe_sid = {
      ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, toe_site_name(toe))
      for ee, toe in self.toes.items()
    }
    self.ee_bid = {
      ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee) for ee in self.ee_names
    }
    self.foot_sid = [
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in _FOOT_SITE_NAMES
    ]
    self.sphere_bids = {
      ee: [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in sphere_names(ee, rows, cols)
      ]
      for ee in self.ee_names
    }

  # -- state helpers ---------------------------------------------------------

  def set_state1(self) -> None:
    d = self.data
    d.qpos[:] = 0.0
    d.qpos[0:3] = self.init_base_pos
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in self.init_joint_pos.items():
      d.qpos[self._qadr[name]] = angle
    mujoco.mj_forward(self.model, d)

  def pad_pose(self, ee: str) -> tuple[np.ndarray, np.ndarray]:
    sid = self.pad_sid[ee]
    return (
      self.data.site_xpos[sid].copy(),
      self.data.site_xmat[sid].reshape(3, 3).copy(),
    )

  def dof_selection(self, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
      [self._base_dofs, [self._dofadr[n] for n in joint_names]]
    ).astype(int)

  # -- solver ----------------------------------------------------------------

  def solve_frame(
    self,
    tasks: list[_Task],
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
  ) -> _Residual:
    m, d = self.model, self.data
    nv = m.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    sel_set = set(sel.tolist())
    sel_joints = [
      (name, self._dofadr[name])
      for name in posture_ref
      if self._dofadr[name] in sel_set
    ]
    sel_index = {dof: i for i, dof in enumerate(sel.tolist())}

    residual = _Residual()
    for _ in range(iters):
      mujoco.mj_forward(m, d)
      rows_j, rows_e, rows_w = [], [], []
      residual = _Residual()
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
        rows_j.append(jacp[:, sel].copy())
        rows_e.append(err_p)
        rows_w.append(np.asarray(task.w_pos, dtype=np.float64))
        if task.track:
          residual.pos = max(residual.pos, float(np.linalg.norm(err_p)))
        if task.rot is not None:
          err_r = _orientation_error_w(rot, task.rot)
          rows_j.append(jacr[:, sel].copy())
          rows_e.append(err_r)
          rows_w.append(np.full(3, task.w_rot))
          if task.track:
            residual.rot = max(residual.rot, float(np.linalg.norm(err_r)))

      jac = np.vstack(rows_j)
      err = np.concatenate(rows_e)
      w2 = np.concatenate(rows_w) ** 2
      a = (jac * w2[:, None]).T @ jac + lam2 * np.eye(len(sel))
      b = (jac * w2[:, None]).T @ err
      w_lim2 = self.limit_weight**2
      overrides = posture_weight_overrides or {}
      for name, dof in sel_joints:
        i = sel_index[dof]
        w_post2 = overrides.get(name, posture_weight) ** 2
        a[i, i] += w_post2
        b[i] += w_post2 * (posture_ref[name] - d.qpos[self._qadr[name]])
        # Limit-avoidance spring: active only inside the margin zone, pulls
        # back to the zone boundary. Zero force in the interior, so it never
        # fights the posture reference where clearance is healthy.
        lo, hi = self._range[name]
        margin = self.limit_margin_frac * (hi - lo)
        q = d.qpos[self._qadr[name]]
        if q < lo + margin:
          a[i, i] += w_lim2
          b[i] += w_lim2 * ((lo + margin) - q)
        elif q > hi - margin:
          a[i, i] += w_lim2
          b[i] += w_lim2 * ((hi - margin) - q)

      dq = np.linalg.solve(a, b)
      # Converged only when the tracked (hard) errors are met AND the soft
      # tasks are stationary. Checking the hard errors alone would freeze the
      # base whenever the welds are already satisfied by the warm start.
      if (
        residual.pos < tol_pos
        and residual.rot < tol_rot
        and float(np.abs(dq).max()) < tol_dq
      ):
        break
      peak = float(np.abs(dq).max())
      if peak > max_step:
        dq *= max_step / peak
      qvel = np.zeros(nv)
      qvel[sel] = dq
      mujoco.mj_integratePos(m, d.qpos, qvel, 1.0)
      for name, (lo, hi) in self._range.items():
        adr = self._qadr[name]
        d.qpos[adr] = np.clip(d.qpos[adr], lo + 0.005, hi - 0.005)

    mujoco.mj_forward(m, d)
    return residual


def _weld_task(ik: _ClimbIK, ee: str, anchor: tuple[np.ndarray, np.ndarray]) -> _Task:
  return _Task(
    kind="site",
    sid=ik.pad_sid[ee],
    pos=anchor[0],
    rot=anchor[1],
    # Support welds are hard kinematic facts; posture and swing shaping must
    # spend their error elsewhere.
    w_pos=np.full(3, 1000.0),
    w_rot=500.0,
  )


def _toe_task(
  ik: _ClimbIK,
  ee: str,
  pos: np.ndarray,
  w_pos: tuple[float, float, float] = (300.0, 30.0, 30.0),
) -> _Task:
  """Position-only brace target for a foot's toe tip.

  The 2-DoF ankle cannot satisfy all three components exactly, so the wall
  press (x) dominates and y/z absorb the residual; ``track=False`` keeps the
  intentionally-unreachable components out of the convergence residual (toe
  quality is reported separately).
  """
  return _Task(
    kind="site",
    sid=ik.toe_sid[ee],
    pos=pos,
    rot=None,
    w_pos=np.asarray(w_pos, dtype=np.float64),
    w_rot=0.0,
    track=False,
  )


def _base_task(
  ik: _ClimbIK,
  pos: np.ndarray,
  w_pos: tuple[float, float, float],
  rot: np.ndarray | None = None,
  w_rot: float = 2.0,
) -> _Task:
  return _Task(
    kind="base",
    sid=-1,
    pos=np.asarray(pos, dtype=np.float64),
    rot=np.eye(3) if rot is None else rot,
    w_pos=np.asarray(w_pos, dtype=np.float64),
    w_rot=w_rot,
    track=False,
  )


def generate_clip(
  output: Path,
  step_height: float | None,
  cycles: int,
  fps: float,
  foot_swing_s: float,
  swing_s: float,
  hold_start_s: float,
  foot_transfer_hold_s: float,
  seat_hold_s: float,
  regrip_hold_s: float,
  end_hold_s: float,
  swing_bulge: float,
  swing_lateral: float,
  foot_bulge: float,
  body_press: float,
  lateral_weight_shift: float,
  hand_load_transfer_s: float,
  rake_s: float,
  rake_dist: float,
  limit_margin_frac: float = 0.10,
  limit_weight: float = 3.0,
) -> None:
  ik = _ClimbIK(limit_margin_frac=limit_margin_frac, limit_weight=limit_weight)
  height = ik.default_step_height if step_height is None else float(step_height)
  if cycles < 1:
    raise ValueError("cycles must be >= 1")

  n = {
    "hold_start": max(int(round(hold_start_s * fps)), 1),
    "left_foot_swing": max(int(round(foot_swing_s * fps)), 2),
    "foot_transfer_hold": max(int(round(foot_transfer_hold_s * fps)), 1),
    "right_foot_swing": max(int(round(foot_swing_s * fps)), 2),
    "seat_hold": max(int(round(seat_hold_s * fps)), 1),
    "left_hand_swing": max(int(round(swing_s * fps)), 2),
    "regrip_hold": max(int(round(regrip_hold_s * fps)), 1),
    "right_hand_swing": max(int(round(swing_s * fps)), 2),
    "end_hold": max(int(round(end_hold_s * fps)), 1),
  }
  n_rake = max(int(round(rake_s * fps)), 1)
  n_hand_load_transfer = max(int(round(hand_load_transfer_s * fps)), 0)
  for seg_name in (
    "left_foot_swing",
    "right_foot_swing",
    "left_hand_swing",
    "right_hand_swing",
  ):
    if n[seg_name] - n_rake < 2:
      raise ValueError(
        f"--rake-s leaves fewer than two arc frames in '{seg_name}'; "
        "shorten the rake or lengthen the segment."
      )
  if n_hand_load_transfer >= min(
    n["left_hand_swing"], n["right_hand_swing"]
  ):
    raise ValueError(
      "--hand-load-transfer-s must be shorter than each hand swing."
    )
  rake_vec = np.array([0.0, 0.0, float(rake_dist)])
  # Support handoff clearance: a dwelling toe hovers this far off the wall
  # (zero force, no fight with its own latched pad) and closes the gap only
  # while its pad is released; it lifts back out during the landing rake.
  toe_hover = 0.0015
  seg_id = {name: i for i, name in enumerate(_SEGMENT_NAMES)}

  def _arc_progress(t: int, n_seg: int) -> float:
    """Approach-arc progress in [0, 1]; saturated during the rake frames."""
    n_arc = n_seg - n_rake
    return float(_smootherstep(min(t / (n_arc - 1), 1.0)))

  def _swing_target(
    t: int, n_seg: int, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
  ) -> np.ndarray:
    """Swing-pad position: arc to ``rake_dist`` above the anchor, then the
    pre-engage rake — drag straight down the wall onto the anchor."""
    n_arc = n_seg - n_rake
    if t < n_arc:
      return _bezier(p0, p1, p2 + rake_vec, _arc_progress(t, n_seg))
    s_rake = float(_smootherstep((t - n_arc + 1) / n_rake))
    return p2 + (1.0 - s_rake) * rake_vec

  lh, rh, lf, rf = ik.ee_names  # microspine order: hands then feet
  hands, feet = (lh, rh), (lf, rf)

  ik.set_state1()
  state1_qpos = ik.data.qpos.copy()
  base_xy = ik.init_base_pos[:2]
  base_z1 = float(ik.init_base_pos[2])
  anchors = {ee: ik.pad_pose(ee) for ee in ik.ee_names}
  state1_pad_rot = {ee: anchors[ee][1].copy() for ee in ik.ee_names}
  posture_ref = {
    name: ik.init_joint_pos[name]
    for name in PHYBOT_C2_POLICY_JOINT_NAMES
  }

  # Per-frame records for one cycle.
  qpos_frames: list[np.ndarray] = []
  labels: list[int] = []
  engaged_frames: list[np.ndarray] = []
  anchor_frames: list[np.ndarray] = []
  toe_frames: list[np.ndarray] = []
  seg_residual: dict[str, _Residual] = {name: _Residual() for name in _SEGMENT_NAMES}

  # The weld anchor (or upcoming target anchor) each EE currently references.
  anchor_now = {ee: anchors[ee][0].copy() for ee in ik.ee_names}
  # Foot-toe wall-brace schedule: True only inside a swing's full-press
  # window (hover/plant/retract frames carry no load by design).
  toe_braced_now = {ee: False for ee in ik.toes}

  def push(segment: str, engaged: dict[str, bool]) -> None:
    qpos_frames.append(ik.data.qpos.copy())
    labels.append(seg_id[segment])
    engaged_frames.append(
      np.array([engaged[ee] for ee in ik.ee_names], dtype=np.uint8)
    )
    anchor_frames.append(np.stack([anchor_now[ee] for ee in ik.ee_names]))
    toe_frames.append(
      np.array([toe_braced_now[ee] for ee in ik.toes], dtype=np.uint8)
    )

  def hold(segment: str, engaged: dict[str, bool]) -> None:
    for _ in range(n[segment]):
      push(segment, engaged)

  all_on = {ee: True for ee in ik.ee_names}
  sel_all = ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES)

  # Ankle joints per bracing toe ("left_ankle_roll" -> left_ankle_*): the
  # only DOFs the toe work moves.
  ankle_joints = {
    ee: tuple(
      nm
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
      if nm.startswith(toe.body_name.removesuffix("_roll"))
    )
    for ee, toe in ik.toes.items()
  }
  stance_toe_clear = {
    ee: float(
      _WALL_SURFACE_X
      - ik.data.site_xpos[ik.toe_sid[ee]][0]
      - toe.radius
    )
    for ee, toe in ik.toes.items()
  }
  toe_report: dict[str, dict[str, float]] = {}

  def dwell_ik(
    segment: str,
    engaged: dict[str, bool],
    plant_ee: str | None = None,
    retract_ee: str | None = None,
  ) -> None:
    """Four-anchor dwell that moves only ankles: plant/retract a bracing toe.

    The spine pads mount on the shin and the toe on the articulated foot, so
    ankle motion is kinematically decoupled from every support anchor — the
    dwell replays the stance (welds + base/posture holds) while one toe
    ramps to the wall and/or the finished one folds back to stance angles.
    """
    for ee in ik.ee_names:
      anchors[ee] = ik.pad_pose(ee)
      anchor_now[ee] = anchors[ee][0].copy()
    base_hold = ik.data.qpos[0:3].copy()
    base_rot_hold = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_hold, ik.data.qpos[3:7])
    base_rot_hold = base_rot_hold.reshape(3, 3)
    posture_hold = {
      nm: float(ik.data.qpos[ik._qadr[nm]])
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    overrides: dict[str, float] = {}
    retract_from: dict[str, float] = {}
    if retract_ee is not None:
      # Fold the finished toe back to stance angles; the weak global posture
      # spring alone would leave it grazing the wall while the next beats
      # press the base in. The target BLENDS from the planted angles (an
      # instant stance-angle reference at weight 6 snapped the free ankle
      # 0.6 rad in one frame — a 32 rad/s spike in the clip).
      for nm in ankle_joints[retract_ee]:
        retract_from[nm] = posture_hold[nm]
        overrides[nm] = 6.0
    toe0 = goal = None
    if plant_ee is not None:
      toe0 = ik.data.site_xpos[ik.toe_sid[plant_ee]].copy()
      # The reachable plant point sits on the toe's ankle-centered reach
      # circle: HOVERING toe_hover shy of the wall, ~6 cm above the stance
      # toe height. The toe must not press while its own pad is still
      # latched — a PD-held planted toe reacts against the wall and loads
      # the fresh anchor toward reverse-shear/tension (the cheap release
      # directions); it closes the last 1.5 mm only once the pad lets go.
      # The frog splay tilts the foot's sagittal plane ~30 deg off world x,
      # so chasing the stance y here would demand a large ankle-roll sweep
      # and flipped the (pitch, roll) IK branch mid-plant (a 24-26 rad/s
      # roll spike in the clip). Let y ride (weight 3) and pin the roll to
      # its stance angle instead — the pitch does the reaching.
      goal = np.array(
        [
          _WALL_SURFACE_X - ik.toes[plant_ee].radius - toe_hover,
          toe0[1],
          toe0[2] + 0.06,
        ]
      )
      for nm in ankle_joints[plant_ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for t in range(n[segment]):
      s = float(_smootherstep(t / max(n[segment] - 1, 1)))
      tasks = [_weld_task(ik, ee, anchors[ee]) for ee in ik.ee_names]
      if plant_ee is not None:
        tasks.append(
          _toe_task(
            ik,
            plant_ee,
            (1.0 - s) * toe0 + s * goal,
            w_pos=(300.0, 3.0, 10.0),
          )
        )
      tasks.append(
        _base_task(
          ik, base_hold, (30.0, 30.0, 30.0), rot=base_rot_hold, w_rot=8.0
        )
      )
      for nm, start in retract_from.items():
        posture_hold[nm] = (1.0 - s) * start + s * posture_ref[nm]
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_hold,
        posture_weight=0.3,
        posture_weight_overrides=overrides,
      )
      seg_residual[segment].merge(res)
      push(segment, engaged)

  # -- state 1 hold: all four pads dwell while the left toe plants against
  # the wall, ready to brace the upcoming left-foot regrip.
  dwell_ik("hold_start", all_on, plant_ee=lf)

  # -- 1 -> 2 -> 3: sequential foot regrips ---------------------------------
  # Never release both feet together. The three-pad support set always
  # contains two hands and one foot, so the body moment closes through the
  # large hand/foot force couple instead of local pad bending.
  base_z2 = base_z1 + height

  step_vec = np.array([0.0, 0.0, height])
  foot_target = {
    ee: (anchors[ee][0] + step_vec, state1_pad_rot[ee]) for ee in feet
  }
  landing_report: dict[str, tuple[float, float]] = {}
  for foot_index, (segment, swing_ee) in enumerate(
    (("left_foot_swing", lf), ("right_foot_swing", rf))
  ):
    # Shift toward the opposite, still-anchored foot: left swing -> right
    # (negative y), right swing -> left (positive y). This removes the yaw
    # couple that otherwise concentrates the normal load on one hand.
    support_side = -1.0 if swing_ee == lf else 1.0
    engaged = {ee: ee != swing_ee for ee in ik.ee_names}
    for ee in ik.ee_names:
      if ee != swing_ee:
        anchors[ee] = ik.pad_pose(ee)
        anchor_now[ee] = anchors[ee][0].copy()
    p0, rot0 = ik.pad_pose(swing_ee)
    p2, rot2 = foot_target[swing_ee]
    anchor_now[swing_ee] = p2.copy()
    p1 = 0.5 * (p0 + p2) + np.array([-foot_bulge, 0.0, 0.0])
    # The toe was pre-positioned toe_hover shy of the wall during the
    # preceding dwell. The swing closes that gap over its first few frames
    # (the pad just released — no more co-press conflict), braces through
    # the travel, and lifts back to the hover during the landing rake while
    # the pad face takes over. Exact point-holding is geometrically
    # impossible for the fixed-radius toe once the pad bulges off the wall
    # (the ankle center retreats, so the toe glides a little ALONG the
    # surface); the wall-normal gap during the press window is the metric
    # that must stay ~0.
    toe_plant = ik.data.site_xpos[ik.toe_sid[swing_ee]].copy()
    toe_press_x = _WALL_SURFACE_X - ik.toes[swing_ee].radius
    n_press_in = max(int(round(0.06 * fps)), 2)
    press_end = n[segment] - n_rake
    toe_report[swing_ee] = {
      "plant_gap": float(
        _WALL_SURFACE_X - toe_plant[0] - ik.toes[swing_ee].radius
      ),
      "wall_gap": 0.0,
      "glide": 0.0,
    }
    base_start = ik.data.qpos[0:3].copy()
    # Press the body toward the wall during the transfer.  Besides keeping
    # the released pad close enough to re-seat, this shortens gravity's
    # wall-normal lever arm and gives the one-hand beats real tension margin.
    base_end = np.array(
      [
        base_xy[0] + body_press,
        base_xy[1] + support_side * lateral_weight_shift,
        base_z1 + height * (foot_index + 1) / len(feet),
      ]
    )
    posture_start = {
      nm: float(ik.data.qpos[ik._qadr[nm]])
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    for t in range(n[segment]):
      s = float(_smootherstep(t / (n[segment] - 1)))
      tasks = [
        _weld_task(ik, ee, anchors[ee])
        for ee in ik.ee_names
        if ee != swing_ee
      ]
      tasks.append(
        _Task(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(t, n[segment], p0, p1, p2),
          rot=_slerp_mat(rot0, rot2, s),
          w_pos=np.full(3, 1000.0),
          # Blend toward the canonical landing orientation while retaining
          # enough positional authority to keep the foot on its wall track.
          w_rot=100.0,
        )
      )
      # Press-in ramp -> full brace -> lift-out during the rake handoff.
      if t < n_press_in:
        press = (t + 1) / n_press_in
      elif t >= press_end:
        press = max(1.0 - (t - press_end + 1) / n_rake, 0.0)
      else:
        press = 1.0
      toe_braced_now[swing_ee] = n_press_in <= t < press_end
      toe_target = toe_plant.copy()
      toe_target[0] = (1.0 - press) * toe_plant[0] + press * toe_press_x
      tasks.append(_toe_task(ik, swing_ee, toe_target))
      tasks.append(
        _base_task(
          ik,
          (1.0 - s) * base_start + s * base_end,
          (100.0, 30.0, 50.0),
        )
      )
      posture_t = {
        nm: (1.0 - s) * posture_start[nm] + s * posture_ref[nm]
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES
      }
      res = ik.solve_frame(tasks, sel_all, posture_t, posture_weight=0.3)
      seg_residual[segment].merge(res)
      toe_now = ik.data.site_xpos[ik.toe_sid[swing_ee]]
      if n_press_in <= t < press_end:
        toe_report[swing_ee]["wall_gap"] = max(
          toe_report[swing_ee]["wall_gap"],
          float(
            _WALL_SURFACE_X - toe_now[0] - ik.toes[swing_ee].radius
          ),
        )
      toe_report[swing_ee]["glide"] = max(
        toe_report[swing_ee]["glide"],
        float(np.linalg.norm(toe_now[1:] - toe_plant[1:])),
      )
      push(segment, engaged)
    anchors[swing_ee] = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = anchors[swing_ee][0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(anchors[swing_ee][0] - p2)),
      _rot_angle(anchors[swing_ee][1], rot2),
    )
    if foot_index == 0:
      # One four-pad dwell folds the finished left toe back and plants the
      # right one (independent ankles, simultaneous).
      dwell_ik("foot_transfer_hold", all_on, plant_ee=rf, retract_ee=lf)

  dwell_ik("seat_hold", all_on, retract_ee=rf)

  # -- 3 -> 4 -> 5: single-hand regrips --------------------------------------
  arm_joints = {
    lh: tuple(n for n in PHYBOT_C2_POLICY_JOINT_NAMES if n.startswith("left_sh") or n == "left_elbow_pitch"),
    rh: tuple(n for n in PHYBOT_C2_POLICY_JOINT_NAMES if n.startswith("right_sh") or n == "right_elbow_pitch"),
  }
  # NOTE on shoulder roll: state 4 (one hand re-gripped high, the other
  # still anchored low, base upright) kinematically REQUIRES the low arm at
  # roll ~+0.29 and waist_yaw ~-0.14 on this wristless 4-DoF arm — verified
  # by static probes across base-orientation regimes and hand widths (the
  # hand's lateral position is not a free parameter; forcing it breaks the
  # stance). Fighting it with waist freezes or stronger springs only trades
  # 5-8 mm of landing accuracy for a few percent of margin. The task's 0.98
  # soft band leaves the reference 0.058 rad inside the target clamp there.
  for segment, swing_ee, lateral_sign in (
    ("left_hand_swing", lh, 1.0),
    ("right_hand_swing", rh, -1.0),
  ):
    engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
    # Re-seat every engaged weld at its current pad pose (the sim's
    # reanchor-after-grace does the same). Sub-mm weld residuals inherited
    # from the previous segment would otherwise re-balance under this
    # segment's changed spring set in the very first solve.
    for ee in ik.ee_names:
      if ee != swing_ee:
        anchors[ee] = ik.pad_pose(ee)
        anchor_now[ee] = anchors[ee][0].copy()
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = anchors[swing_ee][0] + np.array([0.0, 0.0, height])
    # Keep the hand latched briefly while the body/other three limbs take up
    # the new load.  The smootherstep target is essentially stationary during
    # these first frames; after the transfer dwell the outgoing hand releases
    # and its recorded anchor changes from the old grip to the upcoming one.
    anchor_now[swing_ee] = p0.copy()
    p1 = 0.5 * (p0 + p2) + np.array(
      [-swing_bulge, lateral_sign * swing_lateral, 0.0]
    )
    # Releasing the swing hand's weld changes the soft-task stiffness balance;
    # with CONSTANT posture / base-rotation references the very first solve
    # would jump straight to the new equilibrium (measured: 0.018 rad hip +
    # 2.9 mm root in one frame after an otherwise static seat hold). Anchor
    # every soft reference at the segment-start state and blend it to the
    # canonical one so the re-equilibration is spread over the whole swing.
    base_swing_start = ik.data.qpos[0:3].copy()
    base_rot_start = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_start, ik.data.qpos[3:7])
    base_rot_start = base_rot_start.reshape(3, 3)
    posture_start = {
      nm: float(ik.data.qpos[ik._qadr[nm]])
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    for t in range(n[segment]):
      engaged = all_on if t < n_hand_load_transfer else engaged_swing
      if t == n_hand_load_transfer:
        anchor_now[swing_ee] = p2.copy()
      s = float(_smootherstep(t / (n[segment] - 1)))
      s_arc = _arc_progress(t, n[segment])
      tasks = [
        _weld_task(ik, ee, anchors[ee])
        for ee in ik.ee_names
        if ee != swing_ee
      ]
      # The released hand starts strongly constrained and fades to swing
      # weights over the first 30% of the arc. An abrupt stiffness drop
      # released the limit-spring stress stored in the anchored arm
      # (shoulder roll is held deep in its margin zone by the low anchor) as
      # a one-frame whole-body pop at the segment boundary.
      w_fade = min(s_arc / 0.3, 1.0)
      tasks.append(
        _Task(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(t, n[segment], p0, p1, p2),
          rot=rot0,  # anchor orientation held; weight ramps toward landing
          w_pos=np.full(
            3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)
          ),
          # A near-free mid-swing orientation (old floor 2.0) let the
          # wristless arm resolve its null space by whipping shoulder_yaw
          # and tilting the pad corners through the wall (10.9 deg rot
          # residual / 11.6 mm sphere penetration on the v2 stance). Keep
          # the pad essentially flat the whole way, like the
          # orientation-constrained foot swings.
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
      # Keep the wall press through the first hand transfer; unwind it only
      # while the second hand returns the system to the vertically tiled
      # cycle-closure pose.
      base_target_x = (
        base_xy[0] + body_press
        if segment == "left_hand_swing"
        else base_xy[0]
      )
      support_side = -1.0 if swing_ee == lh else 1.0
      base_target = (1.0 - s) * base_swing_start + s * np.array(
        [
          base_target_x,
          base_xy[1] + support_side * lateral_weight_shift,
          base_z2,
        ]
      )
      tasks.append(
        _base_task(
          ik,
          base_target,
          (
            (100.0, 30.0, 50.0)
            if segment == "right_hand_swing"
            else (100.0, 30.0, 30.0)
          ),
          # State 3 is already square, so
          # this is a hold, not a mid-swing righting maneuver.
          rot=_slerp_mat(base_rot_start, np.eye(3), s),
          w_rot=50.0 if segment == "right_hand_swing" else 8.0,
        )
      )
      posture_t = {
        nm: (1.0 - s) * posture_start[nm] + s * posture_ref[nm]
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES
      }
      # The OTHER hand is still anchored one step BELOW its stance-relative
      # height, so the canonical stance angles are unreachable for that arm
      # while its weld holds. Blending it toward them anyway made the DLS
      # dump the contradiction into shoulder roll (1.7% margin peak) and
      # drag the anchored pad (3.2 mm weld drift). Its reference stays at
      # the segment-start angles until its own swing re-grips at stance
      # height.
      if segment == "left_hand_swing":
        for nm in arm_joints[rh]:
          posture_t[nm] = posture_start[nm]
      # Hand swings use a stiffer posture anchor than the foot transfers:
      # 4-DoF arm's position-only null space flips between elbow-in/out
      # branches frame to frame otherwise (8+ rad/s shoulder_yaw spikes).
      # The first few frames are throttled (2 iters, 3 mrad/iter): releasing
      # the swing hand moves the soft-task equilibrium ~0.02 rad (limit-
      # spring stress stored in the anchored arm lets go), and an
      # unthrottled first solve dumps that into a one-frame whole-body pop.
      # The commanded targets barely move here (smootherstep start), so the
      # throttle costs no tracking.
      throttled = t < 8
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_t,
        posture_weight=10.0 if segment == "right_hand_swing" else 0.6,
        iters=2 if throttled else 100,
        max_step=0.003 if throttled else 0.12,
      )
      seg_residual[segment].merge(res)
      push(segment, engaged)
    anchors[swing_ee] = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = anchors[swing_ee][0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(anchors[swing_ee][0] - p2)),
      _rot_angle(anchors[swing_ee][1], rot0),
    )
    if segment == "left_hand_swing":
      hold("regrip_hold", all_on)

  # -- closure: make state 5 exactly state 1 shifted one step height ---------
  state5_qpos = state1_qpos.copy()
  state5_qpos[2] += height
  delta = state5_qpos - qpos_frames[-1]
  closure_frames = max(int(round(0.4 * n["right_hand_swing"])), 2)
  for i in range(closure_frames):
    idx = len(qpos_frames) - closure_frames + i
    blend = float(_smootherstep((i + 1) / closure_frames))
    qpos_frames[idx] = qpos_frames[idx] + blend * delta
    qpos_frames[idx][3:7] /= np.linalg.norm(qpos_frames[idx][3:7])
  closure_pos = float(np.linalg.norm(delta[0:3]))
  closure_joint = float(np.abs(delta[7:]).max())

  # After the closure the final pose is exactly state 1 shifted, so the
  # end-hold anchors are exactly the state-1 anchors shifted.
  ik.data.qpos[:] = state5_qpos
  mujoco.mj_forward(ik.model, ik.data)
  for ee in ik.ee_names:
    anchor_now[ee] = ik.pad_pose(ee)[0]
  hold("end_hold", all_on)

  # -- tile cycles and extract outputs ---------------------------------------
  per_cycle = len(qpos_frames) - n["end_hold"]
  frames_qpos: list[np.ndarray] = []
  frames_label: list[int] = []
  frames_engaged: list[np.ndarray] = []
  frames_anchor: list[np.ndarray] = []
  frames_toe: list[np.ndarray] = []
  frames_cycle: list[int] = []
  z_shift = np.zeros(len(state1_qpos))
  for k in range(cycles):
    z_shift[2] = k * height
    rng = range(per_cycle) if k < cycles - 1 else range(len(qpos_frames))
    for i in rng:
      q = qpos_frames[i] + z_shift
      frames_qpos.append(q)
      frames_label.append(labels[i])
      frames_engaged.append(engaged_frames[i])
      frames_anchor.append(anchor_frames[i] + np.array([0.0, 0.0, k * height]))
      frames_toe.append(toe_frames[i])
      frames_cycle.append(k)

  frames = len(frames_qpos)
  joint_positions = np.zeros((frames, len(PHYBOT_C2_POLICY_JOINT_NAMES)), np.float32)
  root_position = np.zeros((frames, 3), np.float32)
  root_quaternion = np.zeros((frames, 4), np.float32)
  ee_positions = np.zeros((frames, 4, 3), np.float32)
  ee_quaternions = np.zeros((frames, 4, 4), np.float32)
  pad_positions = np.zeros((frames, 4, 3), np.float32)
  foot_positions = np.zeros((frames, 2, 3), np.float32)
  engaged_arr = np.stack(frames_engaged)
  anchor_arr = np.stack(frames_anchor).astype(np.float32)
  toe_arr = np.stack(frames_toe)
  label_arr = np.asarray(frames_label, dtype=np.uint8)
  cycle_arr = np.asarray(frames_cycle, dtype=np.uint16)

  d = ik.data
  max_weld_drift = 0.0
  max_wall_pen = -np.inf
  for t, q in enumerate(frames_qpos):
    d.qpos[:] = q
    mujoco.mj_forward(ik.model, d)
    joint_positions[t] = [
      d.qpos[ik._qadr[nm]] for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    ]
    root_position[t] = q[0:3]
    root_quaternion[t] = (q[4], q[5], q[6], q[3])  # wxyz -> xyzw
    for i, ee in enumerate(ik.ee_names):
      bid = ik.ee_bid[ee]
      ee_positions[t, i] = d.xpos[bid]
      xq = d.xquat[bid]
      ee_quaternions[t, i] = (xq[1], xq[2], xq[3], xq[0])
      pad_positions[t, i] = d.site_xpos[ik.pad_sid[ee]]
      if engaged_arr[t, i]:
        max_weld_drift = max(
          max_weld_drift,
          float(np.linalg.norm(pad_positions[t, i] - anchor_arr[t, i])),
        )
      else:
        for bid_s in ik.sphere_bids[ee]:
          max_wall_pen = max(
            max_wall_pen, float(d.xpos[bid_s][0] - _PAD_TOUCH_X)
          )
    for i, sid in enumerate(ik.foot_sid):
      foot_positions[t, i] = d.site_xpos[sid]

  data_out = {
    "joints_list": list(PHYBOT_C2_POLICY_JOINT_NAMES),
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "foot_sites_list": list(_FOOT_SITE_NAMES),
    "foot_positions": foot_positions,
    "fps": np.float32(fps),
    "ee_list": list(ik.ee_names),
    "ee_positions": ee_positions,
    "ee_quaternions": ee_quaternions,
    "pad_positions": pad_positions,
    "ee_engaged": engaged_arr,
    "ee_anchor_positions": anchor_arr,
    "toe_braced": toe_arr,
    "toe_list": list(ik.toes.keys()),
    "phase_segment": label_arr,
    "segment_names": list(_SEGMENT_NAMES),
    "cycle_index": cycle_arr,
    "step_height": np.float32(height),
    "wall_surface_x": np.float32(_WALL_SURFACE_X),
    "pad_touch_x": np.float32(_PAD_TOUCH_X),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data_out)

  # -- report -----------------------------------------------------------------
  print(
    f"wrote {output} ({frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps, "
    f"{cycles} cycle(s), step_height={height:.3f} m)"
  )
  for name in (
    "left_foot_swing",
    "right_foot_swing",
    "left_hand_swing",
    "right_hand_swing",
  ):
    r = seg_residual[name]
    print(
      f"  {name:18s} IK residual: pos {r.pos * 1000.0:6.2f} mm, "
      f"rot {np.degrees(r.rot):5.2f} deg"
    )
  # Peak speeds are an actuation sanity check; every swing retains three
  # engaged pads.
  joint_speed = np.abs(np.diff(joint_positions, axis=0)) * fps
  peak_joint = int(np.unravel_index(np.argmax(joint_speed), joint_speed.shape)[1])
  base_speed = float(np.abs(np.diff(root_position, axis=0)).max() * fps)
  print(
    f"  peak joint speed {joint_speed.max():.2f} rad/s "
    f"({PHYBOT_C2_POLICY_JOINT_NAMES[peak_joint]}), "
    f"peak base speed {base_speed:.3f} m/s, minimum engaged pads 3"
  )
  for ee, (pos_err, rot_err) in landing_report.items():
    print(
      f"  {ee:18s} landing vs scheduled anchor: "
      f"{pos_err * 1000.0:6.2f} mm, {np.degrees(rot_err):5.2f} deg"
    )
  # Toe-brace quality: hover gap is the swing-start toe-to-wall standoff
  # (should be ~= the 1.5 mm handoff hover), wall gap the worst
  # toe-surface-to-wall distance inside the press window (must stay ~0 —
  # the brace never lifts while it is the support), glide the worst
  # in-plane travel off the plant point (a small hard-tip glide is expected
  # when the pad bulges off the wall; spine tips never touch), stance
  # clearance the state-1 margin between the folded toe and the wall.
  for ee, rep in toe_report.items():
    print(
      f"  toe {ee:14s} hover gap {rep['plant_gap'] * 1000.0:5.2f} mm, "
      f"press wall gap max {rep['wall_gap'] * 1000.0:5.2f} mm, "
      f"glide {rep['glide'] * 1000.0:5.2f} mm, "
      f"stance clear {stance_toe_clear[ee] * 1000.0:6.1f} mm"
    )
  # Verify the pre-engage rake actually happened: downward on-wall stroke of
  # each landing pad over the final n_rake frames of its swing segment.
  for seg_name, ee_indices in (
    ("left_foot_swing", (2,)),
    ("right_foot_swing", (3,)),
    ("left_hand_swing", (0,)),
    ("right_hand_swing", (1,)),
  ):
    frames_in_seg = np.where(label_arr == seg_id[seg_name])[0]
    last = frames_in_seg[frames_in_seg < per_cycle][-1]
    for i in ee_indices:
      drop = float(pad_positions[last - n_rake, i, 2] - pad_positions[last, i, 2])
      off_wall = float(
        np.abs(pad_positions[last - n_rake : last + 1, i, 0] - anchor_arr[last, i, 0]).max()
      )
      print(
        f"  rake {ik.ee_names[i]:18s} ({seg_name}): down "
        f"{drop * 1000.0:5.2f} mm, off-anchor-plane max {off_wall * 1000.0:5.2f} mm"
      )
  print(
    f"  engaged-pad weld drift max      {max_weld_drift * 1000.0:6.2f} mm\n"
    f"  swing sphere wall penetration   {max_wall_pen * 1000.0:+6.2f} mm "
    f"(<= 0 means clear of the wall)\n"
    f"  cycle closure correction        {closure_pos * 1000.0:6.2f} mm base, "
    f"{closure_joint:.4f} rad joints"
  )
  # Per-joint worst clearance to the hard stops over the whole clip. The
  # limit-avoidance springs should keep every joint out of the margin zone
  # wherever the welds leave slack; anything below 5% of range is flagged
  # (the task's 0.98 soft band reserves only 1% — the clip must stay well
  # inside it so tracking never fights the target clamp).
  worst_name, worst_margin = "", np.inf
  flagged = []
  for i, nm in enumerate(PHYBOT_C2_POLICY_JOINT_NAMES):
    lo, hi = ik._range[nm]
    jmin = float(joint_positions[:, i].min())
    jmax = float(joint_positions[:, i].max())
    margin = min(jmin - lo, hi - jmax) / (hi - lo)
    if margin < worst_margin:
      worst_name, worst_margin = nm, margin
    if margin < 0.05:
      flagged.append(
        f"{nm} [{jmin:+.3f}, {jmax:+.3f}] range [{lo:+.3f}, {hi:+.3f}] "
        f"margin {margin * 100.0:.1f}%"
      )
  print(
    f"  joint-limit clearance min       {worst_margin * 100.0:6.2f}% of range "
    f"({worst_name})"
  )
  for line in flagged:
    print("  LOW LIMIT MARGIN: " + line)
  if max_weld_drift > 5.0e-3 or max_wall_pen > 2.0e-3:
    print("  WARNING: IK quality below tolerance; inspect the render.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=(
      _REPO
      / "src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy"
    ),
  )
  parser.add_argument(
    "--step-height",
    type=float,
    default=None,
    help="per-cycle anchor rise (m); default: the climb task's step_height",
  )
  parser.add_argument("--cycles", type=int, default=1, help="number of stacked cycles")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument(
    "--foot-swing-s",
    type=float,
    default=0.9,
    help="duration of each single-foot regrip; the other three pads stay engaged",
  )
  parser.add_argument(
    "--swing-s", type=float, default=1.2, help="single hand regrip duration"
  )
  parser.add_argument("--hold-start-s", type=float, default=0.5)
  parser.add_argument(
    "--foot-transfer-hold-s",
    type=float,
    default=0.4,
    help="four-pad dwell between the foot regrips; retracts the left toe "
    "and plants the right one, so keep the ankle sweep under ~4 rad/s",
  )
  parser.add_argument("--seat-hold-s", type=float, default=0.3)
  parser.add_argument("--regrip-hold-s", type=float, default=0.3)
  parser.add_argument("--end-hold-s", type=float, default=0.5)
  parser.add_argument(
    "--swing-bulge",
    type=float,
    default=0.04,
    help="hand swing off-wall Bezier control offset (m); peak clearance is half",
  )
  parser.add_argument(
    "--swing-lateral",
    type=float,
    default=0.0,
    help="hand swing outward Bezier control offset (m); peak offset is half "
    "(default 0: lateral excursion buys nothing and abducts the shoulder)",
  )
  parser.add_argument(
    "--foot-bulge",
    type=float,
    default=0.03,
    help="foot swing off-wall Bezier control offset (m; peak clearance is "
    "half): the released spine pad must clear the wall — the planted toe, "
    "not a sliding pad, braces the wall during foot regrips",
  )
  parser.add_argument(
    "--body-press",
    type=float,
    default=0.03,
    help="base shift toward the wall during the three-pad transfer beats (m)",
  )
  parser.add_argument(
    "--lateral-weight-shift",
    type=float,
    default=0.025,
    help="base shift toward the opposite-side support before each regrip (m)",
  )
  parser.add_argument(
    "--hand-load-transfer-s",
    type=float,
    default=0.1,
    help="all-four-anchor preload dwell at the start of each hand transfer (s)",
  )
  parser.add_argument(
    "--rake-s",
    type=float,
    default=0.1,
    help="pre-engage rake duration at the end of every swing (s)",
  )
  parser.add_argument(
    "--rake-dist",
    type=float,
    default=0.002,
    help="downward on-wall seating stroke before engage (m)",
  )
  parser.add_argument(
    "--limit-margin-frac",
    type=float,
    default=0.10,
    help="joint-limit avoidance zone as a fraction of each joint's range; "
    "inside it a spring pulls the joint back toward the interior",
  )
  parser.add_argument(
    "--limit-weight",
    type=float,
    default=3.0,
    help="IK weight of the limit-avoidance spring (posture weight is 0.3)",
  )
  args = parser.parse_args()

  generate_clip(
    output=args.output,
    step_height=args.step_height,
    cycles=args.cycles,
    fps=args.fps,
    foot_swing_s=args.foot_swing_s,
    swing_s=args.swing_s,
    hold_start_s=args.hold_start_s,
    foot_transfer_hold_s=args.foot_transfer_hold_s,
    seat_hold_s=args.seat_hold_s,
    regrip_hold_s=args.regrip_hold_s,
    end_hold_s=args.end_hold_s,
    swing_bulge=args.swing_bulge,
    swing_lateral=args.swing_lateral,
    foot_bulge=args.foot_bulge,
    body_press=args.body_press,
    lateral_weight_shift=args.lateral_weight_shift,
    hand_load_transfer_s=args.hand_load_transfer_s,
    rake_s=args.rake_s,
    rake_dist=args.rake_dist,
    limit_margin_frac=args.limit_margin_frac,
    limit_weight=args.limit_weight,
  )


if __name__ == "__main__":
  main()
