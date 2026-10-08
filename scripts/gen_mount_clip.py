"""Generate the PHYBot C2 walk-to-climb mount clip on a tilted wall via whole-body IK.

The mount bridges the walk task's standing keyframe (feet flat on the floor,
arms down) to the climb cycle's first frame (all four microspine pads
engaged on the wall), so the transition policy hands off to the climb
policy at an exact reference frame. The clip is acyclic; its final segment
is named ``end_hold`` so ``ClimbMotionClip.from_dict`` accepts it unchanged
(everything before ``end_hold`` reads as one cycle, and the tail supplies
the finite-difference velocity frame). ``step_height`` is 0.

The committed clip, assets/motions/climb_mount_prologue_a80.npy, comes from
the command in its recipe file. In it the left hand takes a low rung and
the right hand a middle rung while both feet stand on the floor; the right
foot and then the left foot (a toe pivot) mount the wall on the hand welds;
the right hand regrips once and the left hand twice up to the final rungs;
an all-four dwell at the climb cycle's first frame ends the clip. The
comments in ``generate_mount_clip`` describe each segment.

Output schema = the cycle clip (tools.climbing_reference.cycle) plus
``foot_on_ground`` (T, 2) uint8: 1 while that foot (left, right) is a flat
floor support (feet in ``toe_list``/``foot_sites_list`` order).

Validate:
    python scripts/audit_mount_reference.py <clip.npy>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mjlab  # noqa: F401  (registers mjlab entry points)
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "scripts") not in sys.path:
  sys.path.insert(0, str(_REPO / "scripts"))

from scr_phybot_mjab.assets.motions import CAD_CLIMB_MOTION  # noqa: E402
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)
from scr_phybot_mjab.tools.climbing_reference import kinematics as gcc  # noqa: E402

SEGMENT_NAMES = (
  "ground_stand",
  "hand_raise",
  "hand_place",
  "hand_set_left",
  "hand_place_right",
  "hand_set_hold",
  "left_foot_mount",
  "foot_switch_hold",
  "right_foot_mount",
  "brace_hold",
  "hand_up_left",
  "hand_up_hold",
  "hand_up_right",
  "handoff_hold",
  "end_hold",
  "hand_raise_right",
  "hand_set_right",
  "hand_up_left_mid",
  "hand_up_left_hold",
)

_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")


class _WallFrame:
  """Tilted-wall geometry: Rot_y(90 - angle) about the base line (x=1.5, z=0).

  Matches the runtime convention exactly (curriculums.wall_pose_for_angle):
  along-wall up u = (cos a, 0, sin a), outward normal n = (-sin a, 0, cos a),
  quat (wxyz) = (cos(t/2), 0, sin(t/2), 0) with t = radians(90 - a). The
  generator SOLVES in the native world (real gravity, horizontal floor,
  tilted wall) and DE-ROTATES all world-frame outputs to the canonical
  vertical-wall frame at write time, so the runtime's ``_to_world_*``
  machinery re-poses them at the sampled angle.
  """

  def __init__(self, angle_deg: float) -> None:
    self.angle_deg = float(angle_deg)
    self.pivot = np.array([gcc.WALL_SURFACE_X, 0.0, 0.0])
    theta = np.radians(90.0 - self.angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    self.R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    self.outward = self.R @ np.array([-1.0, 0.0, 0.0])
    self.up = self.R @ np.array([0.0, 0.0, 1.0])
    half = 0.5 * theta
    self.quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    self.quat_inv = self.quat * np.array([1.0, -1.0, -1.0, -1.0])

  def rotate_point(self, p: np.ndarray) -> np.ndarray:
    return self.pivot + self.R @ (np.asarray(p, dtype=np.float64) - self.pivot)

  def rotate_mat(self, rot: np.ndarray) -> np.ndarray:
    return self.R @ rot

  def rotate_root_qpos(self, qpos: np.ndarray) -> np.ndarray:
    """Rotate a free-root qpos (pos + wxyz quat) about the base line."""
    out = qpos.copy()
    out[0:3] = self.rotate_point(qpos[0:3])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, self.quat, qpos[3:7])
    out[3:7] = rotated
    return out

  def along_wall(self, p: np.ndarray) -> float:
    """Along-wall coordinate (the canonical-frame z) of a native point."""
    return float(self.up @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def wall_distance(self, p: np.ndarray) -> float:
    """Signed clearance to the wall surface (positive on the robot side)."""
    return float(self.outward @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def derotate_points(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 3) array."""
    shape = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float64)
    flat = (flat - self.pivot) @ self.R + self.pivot
    return flat.reshape(shape).astype(arr.dtype)

  def derotate_quats_xyzw(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 4) xyzw quaternion array."""
    shape = arr.shape
    flat = arr.reshape(-1, 4).astype(np.float64)
    out = np.empty_like(flat)
    q = np.zeros(4)
    for i in range(flat.shape[0]):
      wxyz = np.array([flat[i, 3], flat[i, 0], flat[i, 1], flat[i, 2]])
      mujoco.mju_mulQuat(q, self.quat_inv, wxyz)
      out[i] = (q[1], q[2], q[3], q[0])
    return out.reshape(shape).astype(arr.dtype)


def _leg_joints(side: str) -> tuple[str, ...]:
  return tuple(
    f"{side}_{j}"
    for j in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  )


def _arm_joints(side: str) -> tuple[str, ...]:
  return tuple(f"{side}_{j}" for j in _ARM_JOINTS)


def _key_blend(keys: tuple[tuple[float, float], ...], s: float) -> float:
  """Piecewise-smootherstep through (s_key, value) waypoints."""
  if s <= keys[0][0]:
    return keys[0][1]
  for (s0, v0), (s1, v1) in zip(keys[:-1], keys[1:]):
    if s <= s1:
      u = gcc.smootherstep((s - s0) / max(s1 - s0, 1.0e-9))
      return float((1.0 - u) * v0 + u * v1)
  return keys[-1][1]


def generate_mount_clip(
  output: Path,
  fps: float,
  wall_angle_deg: float,
  standoff: float,
  hand_z: float,
  hand_z_low: float,
  stand_s: float,
  raise_s: float,
  place_s: float,
  set_left_s: float,
  place_right_s: float,
  set_hold_s: float,
  foot_swing_s: float,
  foot_swing_right_s: float,
  switch_hold_s: float,
  brace_hold_s: float,
  hand_up_s: float,
  up_hold_s: float,
  handoff_hold_s: float,
  end_hold_s: float,
  place_bulge: float,
  foot_bulge: float,
  foot_lift: float,
  swing_bulge: float,
  place_lean: float,
  swing_press: float,
  raise_roll: float,
  raise_roll_right: float,
  step_hips_in: float,
  pivot_out: float,
  raise_pitch_end: float,
  touch_below: float,
  touch_frac: float,
  place_bow: float,
  reach_arm_pin: float,
  step_base_w: tuple[float, float, float],
  step_press: float,
  step_hips_lateral: float,
  switch_lateral: float,
  hand_z_mid: float,
  mid_rung_y: float,
  foot_rake_s: float,
  foot_rake_dist: float,
  low_rung_y: float,
  smooth_sigma: float,
  lateral_weight_shift: float,
  hand_load_transfer_s: float,
  rake_s: float,
  rake_right_s: float,
  rake_dist: float,
  limit_margin_frac: float,
  limit_weight: float,
  stand_y: float = 0.0,
  step_rise: float = 0.01,
  pivot_rise: float = 0.0,
  set_hold_press: float = 0.0,
  set_hold_rise: float = 0.0,
  pivot_smear_gap: float = 0.008,
  pivot_toe_release: float = 0.49,
  pivot_ctl_up: float = 0.05,
  pivot_ctl_out: float = 0.030,
  pivot_fold_u0: float = 0.45,
  raise_pitch_comp_sign: float = 1.0,
  stance_width: float = 0.16,
  place_right_pitch: float = 0.0,
  place_right_base_dz: float = 0.0,
  place_right_base_dx: float = 0.0,
  place_right_left_arm_w: float = 4.0,
  hand_up_left_mid: float = 1.57,
  up_left_mid_hold_s: float = 0.4,
  hand_up_left_mid_y: float = 0.0,
  raise_pitch_end_right: float = -2.4,
  handoff_motion: Path = CAD_CLIMB_MOTION,
  raise_yaw: float = 0.0,
) -> None:
  if not 0.0 < wall_angle_deg <= 90.0:
    raise ValueError("--wall-angle-deg must be in (0, 90].")
  wall = _WallFrame(wall_angle_deg)
  ik = gcc.ClimbIK(limit_margin_frac=limit_margin_frac, limit_weight=limit_weight)
  m, d = ik.model, ik.data
  lh, rh, lf, rf = ik.ee_names
  hands, feet = (lh, rh), (lf, rf)
  seg_id = {name: i for i, name in enumerate(SEGMENT_NAMES)}

  n = {
    "ground_stand": max(int(round(stand_s * fps)), 1),
    "hand_raise": max(int(round(raise_s * fps)), 2),
    "hand_place": max(int(round(place_s * fps)), 2),
    "hand_set_left": max(int(round((set_left_s) * fps)), 1),
    "hand_place_right": max(int(round(place_right_s * fps)), 2),
    "hand_set_hold": max(int(round(set_hold_s * fps)), 1),
    "left_foot_mount": max(int(round(foot_swing_s * fps)), 2),
    "foot_switch_hold": max(int(round(switch_hold_s * fps)), 1),
    "right_foot_mount": max(int(round(foot_swing_right_s * fps)), 2),
    "brace_hold": max(int(round(brace_hold_s * fps)), 2),
    "hand_up_left": max(int(round(hand_up_s * fps)), 2),
    "hand_up_hold": max(int(round(up_hold_s * fps)), 1),
    "hand_up_right": max(int(round(hand_up_s * fps)), 2),
    "hand_up_left_mid": max(int(round((hand_up_s) * fps)), 2),
    "hand_up_left_hold": max(int(round(up_left_mid_hold_s * fps)), 1),
    "handoff_hold": max(int(round(handoff_hold_s * fps)), 2),
    "end_hold": max(int(round(end_hold_s * fps)), 1),
  }
  # v5 hands-first order (--rh-first): the right hand's own raise / set dwell
  # reuse the left-hand durations; its place arc reuses --place-right-s.
  n["hand_raise_right"] = n["hand_raise"]
  n["hand_set_right"] = n["hand_set_left"]
  n_rake = max(int(round(rake_s * fps)), 1)
  # The right-foot swing gets a LONG rake: its floor support ends at the
  # toe-pad chain's reach (u~0.6) and the left-side-only low compression
  # audits infeasible until the pad face touches — the extended face-on-wall
  # window closes that gap. The early latch this induces (~0.24 s before the
  # scheduled engage) sits inside the mismatch bill's grace (0.3 x the
  # 1.5 s OFF window).
  n_rake_right = max(int(round(rake_right_s * fps)), 1)
  seg_rake = {seg: n_rake for seg in SEGMENT_NAMES}
  seg_rake["right_foot_mount"] = n_rake_right
  # v4: the FOOT rakes may keep their own (shorter) stroke while the hand
  # rakes get the classroom's longer, deeper press-and-drag demonstration —
  # a 0.5 s / 4 mm left-foot rake dipped the lifting ankle 4.7 mm into the
  # floor at the pivot's lift-off and grazed the base_link on the wall.
  n_rake_foot = max(int(round(foot_rake_s * fps)), 1)
  seg_rake["left_foot_mount"] = n_rake_foot
  n_lt = max(int(round(hand_load_transfer_s * fps)), 0)
  for seg in ("hand_place", "hand_place_right", "left_foot_mount", "right_foot_mount", "hand_up_left", "hand_up_right"):
    if n[seg] - seg_rake[seg] < 2:
      raise ValueError(f"rake leaves fewer than two arc frames in {seg}")
  if n_lt >= min(n["hand_up_left"], n["hand_up_right"]):
    raise ValueError("--hand-load-transfer-s must be shorter than a hand-up swing")
  # The pre-engage seating stroke rakes ALONG the wall surface.
  rake_vec = float(rake_dist) * wall.up
  rake_vec_default = rake_vec
  foot_dist = float(foot_rake_dist)
  rake_vec_foot = foot_dist * wall.up

  def _swing_target(t, n_seg, p0, p1, p2, n_rake_seg=None, rake_vec_seg=None):
    nr = n_rake if n_rake_seg is None else n_rake_seg
    rake_vec = rake_vec_default if rake_vec_seg is None else rake_vec_seg
    n_arc = n_seg - nr
    if t < n_arc:
      s = float(gcc.smootherstep(min(t / (n_arc - 1), 1.0)))
      return gcc.bezier(p0, p1, p2 + rake_vec, s)
    s_rake = float(gcc.smootherstep((t - n_arc + 1) / nr))
    return p2 + (1.0 - s_rake) * rake_vec

  # -- state 1 snapshot -> final (shifted) stance -----------------------------
  # The cycle's state-1 stance is canonical (vertical wall); at a tilted
  # angle the whole stance, its anchors and the anchor ladder rotate about
  # the wall base line. ``hand_z``/``hand_z_low`` are ALONG-WALL heights
  # (== world z at 90).
  ik.set_state1()
  handoff = np.load(handoff_motion, allow_pickle=True).item()
  d.qpos[:] = m.qpos0
  d.qpos[:3] = handoff["root_position"][0]
  d.qpos[3:7] = np.asarray(handoff["root_quaternion"])[0, [3, 0, 1, 2]]
  for name, value in zip(handoff["joints_list"], handoff["joint_positions"][0]):
    d.qpos[ik.qpos_addresses[name]] = value
  for i, name in enumerate(handoff.get("passive_joint_names", ())):
    d.qpos[ik.qpos_addresses[name]] = handoff["passive_joint_positions"][0, i]
  mujoco.mj_forward(m, d)
  state1_qpos = d.qpos.copy()
  state1_joint = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  state1_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names}
  state1_qpos = wall.rotate_root_qpos(state1_qpos)
  state1_anchor = {ee: (wall.rotate_point(pos), wall.rotate_mat(rot)) for ee, (pos, rot) in state1_anchor.items()}
  state1_pad_rot = {ee: state1_anchor[ee][1].copy() for ee in ik.ee_names}
  state1_hand_z = wall.along_wall(state1_anchor[lh][0])
  dz = hand_z - state1_hand_z
  final_qpos = state1_qpos.copy()
  final_qpos[0:3] = final_qpos[0:3] + dz * wall.up
  final_anchor_pos = {ee: state1_anchor[ee][0] + dz * wall.up for ee in ik.ee_names}
  low_anchor_pos = {ee: final_anchor_pos[ee] + (hand_z_low - hand_z) * wall.up for ee in hands}
  if hand_z_low >= hand_z:
    raise ValueError("--hand-z-low must be below --hand-z")

  # -- standing keyframe -------------------------------------------------------
  # ``standoff`` is the horizontal distance to the wall BASE LINE at every
  # angle (not the perpendicular distance at torso height): the binding
  # constraint for the standing robot is toe clearance at the wall's base,
  # where the surface sits at x = 1.5 regardless of tilt. A perpendicular
  # standoff at base height would walk the base FORWARD on a leaning wall
  # and push the toes through the base line. Identical to the old semantics
  # at 90.
  base_x0 = gcc.WALL_SURFACE_X - standoff

  # v5 (2026-08-22, geometry B as a STANDING shift): ``stand_y`` parks the
  # standing footprint laterally (world +y = robot's left) relative to the
  # anchor lane, which stays at y = 0 with the whole climb-state-1 handoff.
  # stand_y = -half_stance puts the LEFT sole under the lane centre, so the
  # hand_set_left weight transfer (+transfer_lateral) carries the CoM onto
  # the stance foot AND onto every v3 downstream station at once — no
  # walk-back, no outboard rung, no shoulder-roll stop on the closing step.
  def set_standing() -> None:
    d.qpos[:] = 0.0
    d.qpos[0:3] = (base_x0, stand_y, 0.726)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
      if name in ik.qpos_addresses:
        d.qpos[ik.qpos_addresses[name]] = angle
    # Keep the CAD gimbal rest pose during IK as well as at the seam.
    # Locking it at zero would rotate the forearm into the wall while
    # matching a pad orientation generated at the -3 degree stop.
    for name in handoff.get("passive_joint_names", ()):
      adr = ik.qpos_addresses[name]
      d.qpos[adr] = state1_qpos[adr]
    mujoco.mj_forward(m, d)
    # v6 (user call 2026-08-22): a NARROWER standing stance. The walk keyframe
    # plants the feet at +-0.148 m; re-plant them at +-stance_width/2 by IK
    # (same x, flat, base pose and attitude pinned) so a lifted foot leaves
    # the CoM closer to the stance foot and the hands carry less tipping.
    sel_st = ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES)
    post_st = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    base_st = d.qpos[0:3].copy()
    tasks_st = [gcc.base_task(ik, base_st, (1000.0, 1000.0, 1000.0), rot=np.eye(3), w_rot=500.0)]
    for sid_st in ik.foot_sid:
      pos_st = d.site_xpos[sid_st].copy()
      rot_st = d.site_xmat[sid_st].reshape(3, 3).copy()
      pos_st[1] = float(np.sign(pos_st[1])) * stance_width / 2.0
      tasks_st.append(
        gcc.IKTask(kind="site", sid=sid_st, pos=pos_st, rot=rot_st, w_pos=np.full(3, 1000.0), w_rot=500.0)
      )
    for _ in range(4):
      ik.solve_frame(tasks_st, sel_st, post_st, posture_weight=0.2, iters=200, max_step=0.1)
    mujoco.mj_forward(m, d)

  set_standing()
  standing_qpos = d.qpos.copy()
  standing_joint = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  foot_body_bid = {ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, toe.body_name) for ee, toe in ik.toes.items()}
  foot_hold_pose = {}
  for i, ee in enumerate(feet):
    sid = ik.foot_sid[i]
    foot_hold_pose[ee] = (
      d.site_xpos[sid].copy(),
      d.site_xmat[sid].reshape(3, 3).copy(),
      sid,
    )
  # Ground level = where the standing soles actually rest: the min world
  # AABB bottom over the foot COLLISION geoms (the keyframe carries ~0.7 mm
  # of sole penetration by design; the toe spheres hover 10 mm higher and
  # must NOT define the floor).
  ground_z = np.inf
  _corner_signs = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
  )
  for ee in feet:
    bid = foot_body_bid[ee]
    for g in range(m.ngeom):
      if m.geom_bodyid[g] != bid:
        continue
      if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
        continue
      world = (
        d.geom_xpos[g] + (m.geom_aabb[g, :3] + m.geom_aabb[g, 3:] * _corner_signs) @ d.geom_xmat[g].reshape(3, 3).T
      )
      ground_z = min(ground_z, float(world[:, 2].min()))

  # -- bookkeeping -------------------------------------------------------------
  qpos_frames: list[np.ndarray] = []
  labels: list[int] = []
  engaged_frames: list[np.ndarray] = []
  anchor_frames: list[np.ndarray] = []
  ground_frames: list[np.ndarray] = []
  seg_residual = {name: gcc.IKResidual() for name in SEGMENT_NAMES}
  landing_report: dict[str, tuple[float, float]] = {}
  toe_gap_report: dict[str, float] = {}

  # Upcoming/held anchor per EE (cycle-clip semantics: the weld target while
  # engaged, the scheduled landing target while free).
  anchor_now = {ee: low_anchor_pos[ee].copy() for ee in hands}
  anchor_now.update({ee: final_anchor_pos[ee].copy() for ee in feet})
  on_ground = {lf: True, rf: True}

  def push(segment: str, engaged: dict[str, bool]) -> None:
    qpos_frames.append(d.qpos.copy())
    labels.append(seg_id[segment])
    engaged_frames.append(np.array([engaged[ee] for ee in ik.ee_names], dtype=np.uint8))
    anchor_frames.append(np.stack([anchor_now[ee] for ee in ik.ee_names]))
    ground_frames.append(np.array([on_ground[lf], on_ground[rf]], dtype=np.uint8))

  def foot_hold_task(ee: str) -> gcc.IKTask:
    pos, rot, sid = foot_hold_pose[ee]
    return gcc.IKTask(
      kind="site",
      sid=sid,
      pos=pos,
      rot=rot,
      w_pos=np.full(3, 1000.0),
      w_rot=500.0,
    )

  sel_all = ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES)
  none_on = {ee: False for ee in ik.ee_names}
  all_on = {ee: True for ee in ik.ee_names}

  ankle_joints = {
    ee: tuple(nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.startswith(toe.body_name.removesuffix("_roll")))
    for ee, toe in ik.toes.items()
  }

  base_track: dict[str, np.ndarray] = {}

  def _note_base(segment: str, target: np.ndarray) -> None:
    """v4 audit: base tasks are track=False (they never enter the IK
    residual), so record |base - target| per axis per segment here."""
    err = np.abs(d.qpos[0:3] - np.asarray(target, dtype=np.float64))
    base_track[segment] = np.maximum(base_track.get(segment, np.zeros(3)), err)

  def dwell(
    segment: str,
    engaged: dict[str, bool],
    weld_ees: tuple[str, ...],
    hold_feet: tuple[str, ...],
    base_shift: np.ndarray | None = None,
    plant_toes: tuple[str, ...] = (),
    toe_hold_targets: dict[str, np.ndarray] | None = None,
    hold_pads: tuple[str, ...] = (),
    base_w: tuple[float, float, float] = (60.0, 60.0, 60.0),
    floor_toe_targets: dict[str, np.ndarray] | None = None,
    extra_overrides: dict[str, float] | None = None,
    base_rot_key: np.ndarray | None = None,
    base_w_rot: float = 8.0,
  ) -> None:
    """Static dwell on the current pose; optionally drift the base target,
    plant foot toes onto the wall (cycle dwell_ik's ankle-only maneuver),
    hold previously planted toes in place, and/or hold un-welded pads at
    their entry pose (``hold_pads`` — the staggered latch's hovering hand).
    ``floor_toe_targets`` keeps a TIPTOED foot planted on the floor by its
    toe only (toe site pinned at the given point, loose ankle pitch) — the
    interleaved press-up's trailing foot; a ``hold_feet`` entry would slam
    the flat 1000/500 pin back onto a lifted heel in one solve (measured:
    a 26 rad/s seam snap that walks the welded support arm off its branch)."""
    for ee in weld_ees:
      cur = ik.pad_pose(ee)
      anchor_now[ee] = cur[0].copy()
    weld_anchor = {ee: ik.pad_pose(ee) for ee in weld_ees}
    pad_hold = {ee: ik.pad_pose(ee) for ee in hold_pads}
    base_hold = d.qpos[0:3].copy()
    base_rot = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot, d.qpos[3:7])
    base_rot = base_rot.reshape(3, 3)
    posture_hold = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    overrides: dict[str, float] = {}
    plant0, plant_goal = {}, {}
    for ee in toe_hold_targets or ():
      for nm in ankle_joints[ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for ee in hold_feet:
      # A held foot stays FLAT by definition: at the 1.50 low ladder the
      # solver paid the higher reach by pitching the stance foot off its
      # 1000/500 hold (both sole points height-gated out -> the audit's
      # <support-count> auto-fail).
      for nm in ankle_joints[ee]:
        overrides[nm] = 6.0
    for ee in floor_toe_targets or ():
      for nm in ankle_joints[ee]:
        overrides[nm] = 3.0 if nm.endswith("_roll") else 1.0
    overrides.update(extra_overrides or {})
    for ee in plant_toes:
      toe0 = d.site_xpos[ik.toe_sid[ee]].copy()
      # Wall-touch target at exact contact: the static audit credits a
      # released support within 0.5 mm of the wall, and the IK undershoots
      # by a few tenths — aiming at the surface itself lands inside the
      # window. The reference still commands no press (pressing a toe whose
      # own pad is latched loads that anchor toward the cheap release
      # directions — cycle generator's warning).
      plant0[ee] = toe0
      # Same target on the tilted plane: project the toe onto the wall,
      # 5 cm up-wall, one sphere radius short of the surface.
      plant_goal[ee] = (
        wall.pivot
        + (wall.along_wall(toe0) + 0.05) * wall.up
        + np.array([0.0, toe0[1], 0.0])
        + ik.toes[ee].radius * wall.outward
      )
      for nm in ankle_joints[ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for t in range(n[segment]):
      s = float(gcc.smootherstep(t / max(n[segment] - 1, 1)))
      tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in weld_ees]
      tasks.extend(foot_hold_task(ee) for ee in hold_feet)
      for ee in hold_pads:
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.pad_sid[ee],
            pos=pad_hold[ee][0],
            rot=pad_hold[ee][1],
            w_pos=np.full(3, 800.0),
            w_rot=200.0,
          )
        )
      for ee in plant_toes:
        # World-axis task weights cannot express a tilted anisotropy
        # (gcc.IKTask is per-world-axis): with a fixed up-wall goal the
        # ankle's along-wall shortfall (~5 cm at weight 10) leaks
        # sin(90-a) of itself into the surface normal and the toe lands
        # ~2 mm THROUGH the wall at 88. Re-project the goal each frame at
        # the toe's OWN along-wall height with a decaying up-pull: at
        # s -> 1 the target is the plane at the toe's height, so the
        # strong normal weight converges to a clean touch.
        toe_now = d.site_xpos[ik.toe_sid[ee]]
        goal_now = (
          wall.pivot
          + (wall.along_wall(toe_now) + 0.05 * (1.0 - s)) * wall.up
          + np.array([0.0, plant0[ee][1], 0.0])
          + ik.toes[ee].radius * wall.outward
        )
        toe_target = (1.0 - s) * plant0[ee] + s * goal_now
        tasks.append(
          gcc.toe_task(
            ik,
            ee,
            toe_target,
            w_pos=(600.0, 3.0, 10.0),
          )
        )
      if toe_hold_targets:
        for ee, pos in toe_hold_targets.items():
          tasks.append(gcc.toe_task(ik, ee, pos, w_pos=(600.0, 3.0, 10.0)))
      for ee, pos in (floor_toe_targets or {}).items():
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.toe_sid[ee],
            pos=pos,
            rot=None,
            w_pos=np.array([600.0, 50.0, 1000.0]),
            w_rot=0.0,
            track=False,
          )
        )
      for ee in plant_toes:
        toe_gap_report[ee] = wall.wall_distance(d.site_xpos[ik.toe_sid[ee]]) - ik.toes[ee].radius
      target = base_hold if base_shift is None else base_hold + s * base_shift
      # v5: optional base ATTITUDE key (world rotation applied to the entry
      # attitude, slerped over the dwell) — the hands-first set dwell leans
      # the trunk off a high welded hand so the arm is in tension, not pushed.
      rot_t = base_rot if base_rot_key is None else gcc.slerp_matrix(base_rot, base_rot_key @ base_rot, s)
      tasks.append(gcc.base_task(ik, target, base_w, rot=rot_t, w_rot=base_w_rot))
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_hold,
        posture_weight=0.3,
        posture_weight_overrides=overrides,
      )
      _note_base(segment, target)
      seg_residual[segment].merge(res)
      push(segment, engaged)

  toe_wall_hold: dict[str, np.ndarray] = {}

  # ===========================================================================
  # INTERLEAVED order (user call, 08-21 — the 80-deg redesign): normal-human
  # vocabulary only — front raises, a bracing hand before every step, one
  # hand step — and every deep move on the strongest support stance the
  # mount has built so far. Two 80-deg facts drive it:
  #   * the surface at height h recedes by h*sin(90-a) (~0.25 m at the
  #     rungs), so the elbow-vs-wall geometry that forces the lateral
  #     windmill at 90 CHANGES SIGN — a SAGITTAL raise with a folded elbow
  #     clears where the straight-arm sweep never could;
  #   * the two-hands-overhead standing press is the 80-deg balance
  #     frontier (5-7 red frames, 08-15 audit) — so only ONE arm is ever
  #     overhead before a weld exists.
  # Order: LEFT hand raises sagittally and takes the LOW rung standing on
  # two flat feet; LEFT foot mounts braced by that weld; the RIGHT hand
  # rises and lands DIRECTLY on its FINAL rung in one arc, braced by two
  # welds + the flat right foot (the mount's longest reach rides its
  # strongest pre-closure stance — the foot-first deep reach drifted its
  # welds precisely because it ran on the weakest); the RIGHT foot
  # toe-pivots on three welds; the single LEFT-hand walk-up closes on four
  # supports. Total releases: ONE (the left hand's step). Every block is a
  # copy-adapt of its donor (hands-first raise pattern, foot-first
  # arcs/mounts/closure swing); no new motion vocabulary.
  # ===========================================================================
  lh_on = {lh: True, rh: False, rf: False, lf: False}
  hands_lf_on = {lh: True, rh: True, rf: True, lf: False}
  # The RIGHT hand never visits the low rung: re-seed its anchor lane at
  # the FINAL rung (the shared init aims hands at the low ladder; the lane
  # feeds ee_anchor_positions and the runtime press/tow rewards).
  anchor_now[rh] = final_anchor_pos[rh].copy()
  mid_anchor_rh = None
  # v4: the right hand lands an INTERMEDIATE rung reachable flat-footed
  # (probe: the final 1.52 rung is 8-27 mm out of a flat-footed reach and
  # the tiptoe that bought it is passively untenable on 50/5 ankles);
  # it walks up to the final rung in hand_up_right after the pivot.
  if not (hand_z_low < hand_z_mid < hand_z):
    raise ValueError("--hand-z-mid must lie between --hand-z-low and --hand-z")
  up_vec = wall.up
  mid_anchor_rh = final_anchor_pos[rh] + (hand_z_mid - hand_z) * up_vec + np.array([0.0, mid_rung_y, 0.0])
  anchor_now[rh] = mid_anchor_rh.copy()
  # v5 hands-first order: the right hand's first rung is --hand-z-mid when
  # given (a HIGHER rung still reachable flat-footed: RF1's shoulder-height
  # right hand left the pivot statically infeasible, RF util 4.2), else the
  # mirror of the left hand's low rung; the closure is the two-step walk-up.
  if mid_anchor_rh is None:
    mid_anchor_rh = low_anchor_pos[rh] + np.array([0.0, -low_rung_y, 0.0])
  anchor_now[rh] = mid_anchor_rh.copy()
  # The LEFT hand's first rung may sit OUTBOARD of its final lane
  # (--low-rung-y, world +y = the robot's left): a low first rung on the
  # final lane needs the shoulder adducted past its -0.35 rad stop once
  # the body rises into the right-hand reach (measured: roll pinned at
  # -0.34, weld torn 14 mm). The lane is the hand's to choose — it
  # leaves this rung on the closing step.
  low_anchor_pos[lh] = low_anchor_pos[lh] + np.array([0.0, low_rung_y, 0.0])
  anchor_now[lh] = low_anchor_pos[lh].copy()

  def stiff_foot_hold(ee: str) -> gcc.IKTask:
    # A floor foot that is the ONLY floor support of a braced reach or
    # step: 2x the shared flat hold, aimed 1 mm LOW. The audit gates each
    # sole point at 8 mm and the standing keyframe's toe points already
    # sit at 6.6-7.5 mm, so the 1-2 mm the 1000/500 hold concedes to a
    # deep reach drops a support corner (measured: 66 N unsupported
    # standing, one heel point against a hand weld on the step; at
    # 3000/1500 the toe still grazed 8.1 mm while the hand weld paid
    # 3.6 mm of drift for the stiffness). The millimetre bias buys the
    # margin the weight cannot.
    pos, rot, sid = foot_hold_pose[ee]
    return gcc.IKTask(
      kind="site",
      sid=sid,
      pos=pos - np.array([0.0, 0.0, 0.002]),
      rot=rot,
      w_pos=np.full(3, 2000.0),
      w_rot=1000.0,
    )

  # ground_stand: the walk-delivery keyframe, verbatim.
  set_standing()
  for _ in range(n["ground_stand"]):
    push("ground_stand", none_on)

  # hand_raise: LEFT arm only, joint-space SAGITTAL front raise (no IK;
  # legs, base and the right arm frozen at the standing keyframe). The
  # elbow folds deep while the upper arm sweeps past horizontal — the hand
  # tucks past the face instead of pointing a straight arm at the receding
  # surface — and re-extends to the cocked half-bend once overhead.
  # Fold FIRST, raise SECOND: the elbow tucks deep before the pitch sweep
  # passes horizontal, so the forearm rides up past the face instead of
  # pointing at the surface (a straight-ish front raise still clipped the
  # wall by 18 mm at the 0.19 standoff).
  elbow_fold_l = max(float(ik.joint_ranges["left_elbow_pitch"][0]) + 0.10, -2.2)
  out_sign_l = 1.0 if ik.joint_ranges["left_shoulder_roll"][1] > 1.5 else -1.0
  s1 = state1_joint
  st = standing_joint
  raise_keys = {
    "left_shoulder_pitch": (
      (0.0, st["left_shoulder_pitch"]),
      (0.25, st["left_shoulder_pitch"]),
      (0.88, raise_pitch_end),
      (1.0, raise_pitch_end),
    ),
    "left_shoulder_roll": (
      (0.0, st["left_shoulder_roll"]),
      (0.35, out_sign_l * raise_roll),
      (0.80, out_sign_l * raise_roll),
      (1.0, s1["left_shoulder_roll"]),
    ),
    "left_shoulder_yaw": (
      (0.0, st["left_shoulder_yaw"]),
      (1.0, s1["left_shoulder_yaw"]),
    ),
    "left_elbow_pitch": (
      (0.0, st["left_elbow_pitch"]),
      (0.10, st["left_elbow_pitch"]),
      (0.42, elbow_fold_l),
      (0.80, elbow_fold_l),
      (1.0, -1.15),
    ),
  }
  raise_keys["left_shoulder_yaw"] = (
    (0.0, st["left_shoulder_yaw"]),
    (0.25, raise_yaw),
    (0.55, raise_yaw),
    (1.0, s1["left_shoulder_yaw"]),
  )
  for t in range(n["hand_raise"]):
    s = t / max(n["hand_raise"] - 1, 1)
    d.qpos[:] = standing_qpos
    for nm, keys in raise_keys.items():
      d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s)
    mujoco.mj_forward(m, d)
    push("hand_raise", none_on)

  # hand_place: the LEFT pad arcs to the LOW rung + rake, standing on two
  # flat feet — the mount's only unbraced first contact (staggered-latch
  # rule), taken with the right arm DOWN so the standing CoM stays inside
  # the toe line where the two-arm overhead press audited red at 80.
  place_ctl_off = place_bulge * wall.outward + 0.03 * wall.up
  lh_start = ik.pad_pose(lh)
  p0_l = lh_start[0]
  p2_l = low_anchor_pos[lh]
  base_hold_l = d.qpos[0:3].copy()
  base_pl_end = base_hold_l - place_lean * wall.outward
  posture_hold_l = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  # The torso BOWS into the reach (base pitched 0.30 rad nose-down by
  # landing): at 80 deg the low rung sits 0.44 m out horizontally from
  # the standing hips (0.22 at 90), and with the base held level the
  # only torso DoF left to buy that reach was waist yaw — the solver
  # twisted it 0.55 rad (and 1.0 rad by the foot switch), the very
  # "whole body writhing" this redesign exists to remove. A human puts
  # a hand on a leaning wall by leaning at it.
  lean_th = place_bow
  lean_R = np.array(
    [
      [np.cos(lean_th), 0.0, np.sin(lean_th)],
      [0.0, 1.0, 0.0],
      [-np.sin(lean_th), 0.0, np.cos(lean_th)],
    ]
  )
  # Touch-and-slide place (user call, 08-21 morning): the pad first lands
  # on the wall LOW (``touch_below`` under the rung) with the trunk still
  # upright — a hand on the wall at chest height needs no lean — then
  # SLIDES up the surface to the rung under light compression while the
  # trunk bows onto it, so the lean is never unsupported (the audit
  # credits a pressing released pad inside a Coulomb cone). The single
  # arc it replaces bowed 22 deg over 1.6 s with the hand still 4-16 cm
  # off the wall (standing CoM margin 157 -> 40 mm, nothing to catch it).
  n_pl = n["hand_place"]
  n_arc_total = n_pl - n_rake
  # --touch-below <= 0 disables the slide (user call 08-21: a microspine
  # pad must not slide on the wall): one arc straight to the rung + the
  # 2 mm seating rake, the trunk held upright — the rung itself is then
  # chosen low enough (--hand-z-low) that no lean is needed, and the
  # single hand step later carries the hand up to its final rung.
  slide_on = touch_below > 0.0
  if slide_on:
    n_touch = max(int(round(touch_frac * n_arc_total)), 2)
    n_slide = n_arc_total - n_touch
    if n_slide < 2:
      raise ValueError("--place-s leaves fewer than two slide frames")
    p_touch = p2_l - touch_below * wall.up
  else:
    n_touch, n_slide = n_arc_total, 0
    p_touch = p2_l + rake_vec
  p1_t = 0.5 * (p0_l + p_touch) + place_ctl_off
  for t in range(n_pl):
    s = float(gcc.smootherstep(t / (n_pl - 1)))
    if t < n_touch:
      s_arc = float(gcc.smootherstep(t / (n_touch - 1)))
      w_fade = min(s_arc / 0.3, 1.0)
      pad_pos = gcc.bezier(p0_l, p1_t, p_touch, s_arc)
      w_p = 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)
      w_r = 50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc)
      # Without a slide the lean clock rides the arc (it is a no-op at
      # --place-bow 0).
      rot_s, slide_s = s_arc, (0.0 if slide_on else s_arc)
    elif t < n_arc_total:
      slide_s = float(gcc.smootherstep((t - n_touch + 1) / n_slide))
      pad_pos = p_touch + slide_s * (p2_l + rake_vec - p_touch)
      w_p, w_r, rot_s = 2000.0, 300.0, 1.0
    else:
      slide_s = 1.0
      s_rake = float(gcc.smootherstep((t - n_arc_total + 1) / n_rake))
      pad_pos = p2_l + (1.0 - s_rake) * rake_vec
      w_p, w_r, rot_s = 2000.0, 500.0, 1.0
    tasks = [stiff_foot_hold(ee) for ee in feet]
    tasks.append(
      gcc.IKTask(
        kind="site",
        sid=ik.pad_sid[lh],
        pos=pad_pos,
        rot=gcc.slerp_matrix(lh_start[1], state1_pad_rot[lh], rot_s),
        w_pos=np.full(3, w_p),
        w_rot=w_r,
      )
    )
    # Base lean and trunk bow ride the SLIDE clock: nothing leans until
    # the hand is on the wall. The trunk is held UPRIGHT hard until the
    # touch (w_rot 60): at the donor's 10 the reaching arm dragged the
    # soft base into a 10 deg bow before the pad arrived, which is the
    # unsupported lean this segment exists to remove.
    tasks.append(
      gcc.base_task(
        ik,
        (1.0 - slide_s) * base_hold_l + slide_s * base_pl_end,
        (60.0, 60.0, 100.0),
        rot=gcc.slerp_matrix(np.eye(3), lean_R, slide_s),
        w_rot=10.0 + 50.0 * (1.0 - slide_s),
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in _arm_joints("left"):
        posture_t[nm] = (1.0 - s) * posture_hold_l[nm] + s * state1_joint[nm]
      elif nm == "waist_yaw":
        posture_t[nm] = (1.0 - s) * posture_hold_l[nm]
      else:
        posture_t[nm] = posture_hold_l[nm]
    ov_pl = {nm: 1.2 for nm in _arm_joints("left")}
    ov_pl["waist_yaw"] = 12.0
    for ee in feet:
      for nm in ankle_joints[ee]:
        ov_pl[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=ov_pl,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    seg_residual["hand_place"].merge(res)
    push("hand_place", none_on)
  landed_l = ik.pad_pose(lh)
  landing_report[f"{lh}@low"] = (
    float(np.linalg.norm(landed_l[0] - p2_l)),
    gcc.rotation_angle(landed_l[1], state1_pad_rot[lh]),
  )

  # hand_set_left: the LEFT hand engages under its raked preload; both
  # feet stay flat. The right arm rests at the side — no hovering pad to
  # hold, the dwell's posture anchor keeps it down.
  # v4 WEIGHT TRANSFER: the mount's strongest stance (LH weld + two flat
  # feet) carries the CoM over the standing LEFT foot and toward the wall
  # before the right foot lifts (v3 kept it between the feet: 14-16 cm
  # right of the left sole through the whole step -> a passive robot tips
  # onto the welded hand). Needs the first rung outboard (--low-rung-y):
  # on the final lane the left shoulder roll stop blocks the shift.
  transfer_shift = None
  transfer_w = (60.0, 60.0, 60.0)
  # v6: ``set_left_pitch`` leans the trunk away from the wall (world +y pitch key)
  # while only the left hand is welded, so the right hand lands into an
  # already-hanging posture (its set dwell is otherwise pushed up the wall
  # until the lean completes: -25 N at the dwell start, -9 N once leaned).
  rot_key_sl = None
  dwell(
    "hand_set_left",
    lh_on,
    weld_ees=(lh,),
    hold_feet=feet,
    base_shift=transfer_shift,
    base_w=transfer_w,
    extra_overrides={"waist_yaw": 6.0},
    base_rot_key=rot_key_sl,
    base_w_rot=8.0 if rot_key_sl is None else 60.0,
  )

  # =======================================================================
  # v5 HANDS-FIRST ORDER (user call 2026-08-22): the RIGHT hand takes its
  # LOW rung on two flat feet, braced by the welded left hand, BEFORE any
  # foot lifts — the left place mirrored: keyed sagittal raise with the
  # body, legs and the welded left arm frozen (no IK, so the weld cannot
  # move), then the braced arc + seating rake, then a set dwell. Every
  # later beat rides on two hand welds, so no single hand ever carries
  # the right-foot lift's tipping moment (passive gate: 270-380 lost the
  # lone left hand in tension/lateral in every stance tried).
  # =======================================================================
  lh_weld = ik.pad_pose(lh)
  anchor_now[lh] = lh_weld[0].copy()
  elbow_fold_r = max(float(ik.joint_ranges["right_elbow_pitch"][0]) + 0.10, -2.2)
  out_sign_r = 1.0 if ik.joint_ranges["right_shoulder_roll"][1] > 1.5 else -1.0
  q_frozen_r = d.qpos.copy()
  st_r = {nm: float(q_frozen_r[ik.qpos_addresses[nm]]) for nm in _arm_joints("right")}
  # The raise is keyed in WORLD elevation (the reach block's lesson): if the
  # trunk arrives bowed toward the wall (--set-left-pitch), a joint-space
  # key sweeps the hand into the wall (12 cm at +10.7 deg, measured). Shift
  # the shoulder-pitch keys by the trunk's forward pitch so the arm follows
  # the same world arc as on an upright trunk.
  _R_tr = np.zeros(9)
  mujoco.mju_quat2Mat(_R_tr, q_frozen_r[3:7])
  _R_tr = _R_tr.reshape(3, 3)
  trunk_fwd_pitch = float(
    np.arcsin(np.clip(_R_tr[0, 2], -1.0, 1.0))
  )  # world-x component of the trunk up axis (+ = toward the wall)
  trunk_fwd_pitch -= float(np.arcsin(np.clip(wall.R[0, 2], -1.0, 1.0))) if hasattr(wall, "R") else 0.0
  raise_pitch_end_r = (raise_pitch_end_right) + raise_pitch_comp_sign * trunk_fwd_pitch
  raise_keys_r = {
    "right_shoulder_pitch": (
      (0.0, st_r["right_shoulder_pitch"]),
      (0.25, st_r["right_shoulder_pitch"]),
      (0.88, raise_pitch_end_r),
      (1.0, raise_pitch_end_r),
    ),
    "right_shoulder_roll": (
      (0.0, st_r["right_shoulder_roll"]),
      # --rh-raise-lateral: the cross reach block's LATERAL raise (the elbow
      # sweeps through +y where there is no wall, then comes inboard
      # overhead) for a trunk that arrives bowed toward the slab.
      (0.35, out_sign_r * (raise_roll_right)),
      (0.80, out_sign_r * (raise_roll_right)),
      (1.0, s1["right_shoulder_roll"]),
    ),
    "right_shoulder_yaw": (
      (0.0, st_r["right_shoulder_yaw"]),
      (1.0, s1["right_shoulder_yaw"]),
    ),
    "right_elbow_pitch": (
      (0.0, st_r["right_elbow_pitch"]),
      (0.10, st_r["right_elbow_pitch"]),
      (0.42, elbow_fold_r),
      (0.80, elbow_fold_r),
      (1.0, -1.15),
    ),
  }
  # Turn the folding forearm sideways before lifting it. The CAD
  # assembly reaches through the wall during a straight front fold.
  raise_keys_r["right_shoulder_yaw"] = (
    (0.0, st_r["right_shoulder_yaw"]),
    (0.25, -raise_yaw),
    (0.55, -raise_yaw),
    (1.0, s1["right_shoulder_yaw"]),
  )
  for t in range(n["hand_raise_right"]):
    s_r = t / max(n["hand_raise_right"] - 1, 1)
    d.qpos[:] = q_frozen_r
    for nm, keys in raise_keys_r.items():
      d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s_r)
    mujoco.mj_forward(m, d)
    push("hand_raise_right", lh_on)
  rh_start = ik.pad_pose(rh)
  p0_rl = rh_start[0]
  p2_rl = mid_anchor_rh.copy()
  base_hold_rl = d.qpos[0:3].copy()
  base_rot_rl = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_rl, d.qpos[3:7])
  base_rot_rl = base_rot_rl.reshape(3, 3)
  posture_hold_rl = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  n_plr = n["hand_place_right"]
  n_arc_rl = n_plr - n_rake
  if n_arc_rl < 2:
    raise ValueError("--place-right-s leaves fewer than two arc frames")
  p_touch_rl = p2_rl + rake_vec
  p1_rl = 0.5 * (p0_rl + p_touch_rl) + place_ctl_off
  for t in range(n_plr):
    s = float(gcc.smootherstep(t / (n_plr - 1)))
    if t < n_arc_rl:
      s_arc = float(gcc.smootherstep(t / (n_arc_rl - 1)))
      w_fade = min(s_arc / 0.3, 1.0)
      pad_pos = gcc.bezier(p0_rl, p1_rl, p_touch_rl, s_arc)
      w_p = 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)
      w_r = 50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc)
      rot_s = s_arc
    else:
      s_rake = float(gcc.smootherstep((t - n_arc_rl + 1) / n_rake))
      pad_pos = p2_rl + (1.0 - s_rake) * rake_vec
      w_p, w_r, rot_s = 2000.0, 500.0, 1.0
    tasks = [stiff_foot_hold(ee) for ee in feet]
    tasks.append(gcc.weld_task(ik, lh, lh_weld))
    tasks.append(
      gcc.IKTask(
        kind="site",
        sid=ik.pad_sid[rh],
        pos=pad_pos,
        rot=gcc.slerp_matrix(rh_start[1], state1_pad_rot[rh], rot_s),
        w_pos=np.full(3, w_p),
        w_rot=w_r,
      )
    )
    # Trunk held upright and the base parked (the left place's rule: a
    # hand on the wall at chest height needs no lean).
    # v6: ``place_right_pitch`` bows the trunk toward the wall DURING the
    # arc (the raise stays upright so its joint-space keys cannot sweep the
    # wall); by the landing the high right hand is loaded in arm tension.
    rot_pl_t = base_rot_rl
    cp_, sp_ = np.cos(place_right_pitch), np.sin(place_right_pitch)
    rot_pl_t = gcc.slerp_matrix(
      base_rot_rl, np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]]) @ base_rot_rl, s
    )
    base_pl_t = base_hold_rl
    # v6: reach relief for a high right rung -- the base rises /
    # approaches the wall over the arc instead of the IK dragging
    # the welded left hand off its anchor.
    base_pl_t = base_hold_rl - s * place_right_base_dx * wall.outward + s * place_right_base_dz * wall.up
    tasks.append(gcc.base_task(ik, base_pl_t, (60.0, 60.0, 100.0), rot=rot_pl_t, w_rot=60.0))
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in _arm_joints("right"):
        posture_t[nm] = (1.0 - s) * posture_hold_rl[nm] + s * state1_joint[nm]
      elif nm == "waist_yaw":
        posture_t[nm] = (1.0 - s) * posture_hold_rl[nm]
      else:
        posture_t[nm] = posture_hold_rl[nm]
    ov_plr = {nm: 1.2 for nm in _arm_joints("right")}
    # The welded left arm stays stiff (the hand-step support-arm rule).
    ov_plr.update({nm: place_right_left_arm_w for nm in _arm_joints("left")})
    ov_plr["waist_yaw"] = 12.0
    for ee in feet:
      for nm in ankle_joints[ee]:
        ov_plr[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=ov_plr,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    seg_residual["hand_place_right"].merge(res)
    _note_base("hand_place_right", base_hold_rl)
    push("hand_place_right", lh_on)
  landed_rl = ik.pad_pose(rh)
  landing_report[f"{rh}@low"] = (
    float(np.linalg.norm(landed_rl[0] - p2_rl)),
    gcc.rotation_angle(landed_rl[1], state1_pad_rot[rh]),
  )
  hands_on = {lh: True, rh: True, rf: False, lf: False}
  # The seated HIGH right hand is pushed UP the wall while the body stands
  # tall on two feet (passive gate: f_t -21 N at 1.40, -69 N at 1.46 —
  # reverse shear past the 15-25 N cap); the lift's hips-in flips it into a
  # hanging hand (+74 N down-shear). ``set_right_press`` / ``set_right_drop``
  # bring that hips-in (and a small drop) forward into this dwell.
  set_shift = None
  set_w = (60.0, 60.0, 60.0)
  # v6: ``set_right_pitch`` leans the trunk (world pitch about +y; negative =
  # away from the wall) so the high right hand is loaded in ARM TENSION
  # (passive gate: standing tall the trunk bows onto the straight arm and
  # pushes the pad up the wall, -69 N at 1.46; hanging off it holds +74 N).
  rot_key_sr = None
  dwell(
    "hand_set_right",
    hands_on,
    weld_ees=(lh, rh),
    hold_feet=feet,
    base_shift=set_shift,
    base_w=set_w,
    extra_overrides={"waist_yaw": 6.0},
    base_rot_key=rot_key_sr,
    base_w_rot=8.0 if rot_key_sr is None else 60.0,
  )
  rh_weld = ik.pad_pose(rh)
  anchor_now[rh] = rh_weld[0].copy()

  # left_foot_mount: the foot-first left mount WITH a hand brace — the LH
  # weld turns the free-balance step into a held one. Same bezier (launch
  # tangent UP first: the ankle hangs 0.20 m under the shin pad and
  # scrapes the floor otherwise), same delayed plantarflex clock, same
  # throttled opening.
  swing_ee, other_ee = rf, lf
  lh_weld = ik.pad_pose(lh)
  anchor_now[lh] = lh_weld[0].copy()
  p0, rot0 = ik.pad_pose(swing_ee)
  p2 = final_anchor_pos[swing_ee]
  rot2 = state1_pad_rot[swing_ee]
  anchor_now[swing_ee] = p2.copy()
  mount_ctl_off = foot_bulge * wall.outward + np.array([0.0, 0.0, foot_lift])
  p1 = 0.65 * p0 + 0.35 * p2 + mount_ctl_off
  base_seg_start = d.qpos[0:3].copy()
  base_rot_lf = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_lf, d.qpos[3:7])
  base_rot_lf = base_rot_lf.reshape(3, 3)
  # A held STEP's weight shift: hips over the stance foot, tall — and IN
  # toward the slab (8 cm, the free step's 1 cm): the torso un-bows from
  # the hand-place lean to the wall-parallel attitude across the step
  # (a bowed trunk carries the shoulder 15 cm closer to the wall and the
  # next arm raise sweeps through it, 130 mm measured), and the welded
  # low hand sits at the arm's reach limit, so the shoulder can only
  # come back if the hips go in by about as much (0.4 m trunk x sin 18
  # deg). The flat right foot dorsiflexes under it, a few degrees.
  # v4 (2026-08-22): the lateral shift over the stance (left) foot is a
  # knob, and the base gets a mid-swing wall PRESS (``step_press``) like
  # the toe pivot — at (100, 40, 60) the base task lost to the weld, the
  # floor pin and the swing pad: the hips actually drifted 6 cm OUT and
  # toward the swing side, loading the welded hand in tension + lateral
  # (the transition80 passive-hold probe: every seed lost within 0.4 s).
  # v5 (2026-08-22): ``step_rise`` replaces the hard-coded +1 cm rise of the
  # base through the braced right-foot mount. Negative = the body HANGS on
  # the welded left hand while the right foot is off the floor, giving the
  # pad the down-shear its lateral/tension capacity scales with (passive
  # gate @300: 43 N lateral on 14-29 N down-shear tears it; the validator's
  # static optimum hangs).
  base_seg_end = base_seg_start - step_hips_in * wall.outward + step_rise * wall.up
  base_seg_end[1] = base_seg_start[1] + step_hips_lateral
  base_rot_lf_end = wall.R
  posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  swing_leg = _leg_joints("right")
  for t in range(n["right_foot_mount"]):
    u = t / (n["right_foot_mount"] - 1)
    s = float(gcc.smootherstep(u))
    s_rot = float(gcc.smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
    on_ground[swing_ee] = False
    tasks = [
      gcc.weld_task(ik, lh, lh_weld),
      stiff_foot_hold(other_ee),
    ]
    tasks.append(gcc.weld_task(ik, rh, rh_weld))
    tasks.append(
      gcc.IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=_swing_target(
          t,
          n["right_foot_mount"],
          p0,
          p1,
          p2,
          n_rake_seg=seg_rake["right_foot_mount"],
          rake_vec_seg=rake_vec_foot,
        ),
        rot=gcc.slerp_matrix(rot0, rot2, s_rot),
        w_pos=np.full(3, 1000.0),
        w_rot=100.0,
      )
    )
    press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
    base_target_step = (1.0 - s) * base_seg_start + s * base_seg_end
    base_target_step = base_target_step - step_press * press * wall.outward
    tasks.append(
      gcc.base_task(
        ik,
        base_target_step,
        # The free step's weights: at (40, 40, 40) the hips wandered into
        # the wall and the hip roll hit its stop (51 mm residual); the
        # 3.7 mm of hand-weld drift the stiffer pull costs is the price.
        # v4: ``--step-base-w`` (cross v3 recipe 100/40/60 lost the vote).
        tuple(float(w) for w in step_base_w),
        # The trunk UN-BOWS to the wall-parallel attitude across the
        # step (w_rot 40 vs the donor's 10: it has to beat the posture
        # term that likes the bowed hips). Kept bowed, the shoulder sits
        # ~14 cm off the surface at the next raise and NO arm path
        # clears the wall (sagittal and lateral both 120-190 mm through
        # it, measured); the hips-in travel above hands the welded low
        # hand the slack the retreating shoulder takes.
        rot=gcc.slerp_matrix(base_rot_lf, base_rot_lf_end, s),
        w_rot=40.0,
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in swing_leg:
        posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * state1_joint[nm]
      else:
        posture_t[nm] = posture_start[nm]
    overrides = {}
    for nm in swing_leg:
      if "ankle" in nm:
        overrides[nm] = 4.0
    # Waist pinned hard through the un-bow (16): at 8 the solver paid
    # the retreating shoulder with 0.22-0.39 rad of twist.
    overrides["waist_yaw"] = 16.0
    # The stance foot stays FLAT (same 8 mm sole-point gate as the hand
    # place): the braced step let the right toe ride up 3 mm and the
    # audit was left with one heel point against a hand weld.
    for nm in ankle_joints[other_ee]:
      overrides[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    seg_residual["right_foot_mount"].merge(res)
    _note_base("right_foot_mount", base_target_step)
    push("right_foot_mount", hands_on)
  landed = ik.pad_pose(swing_ee)
  anchor_now[swing_ee] = landed[0].copy()
  landing_report[swing_ee] = (
    float(np.linalg.norm(landed[0] - p2)),
    gcc.rotation_angle(landed[1], rot2),
  )

  # foot_switch_hold: the fresh foot weld settles; the LEFT toe plants
  # onto the wall (the low compression every later beat leans on) and the
  # weight walks back to center.
  dwell(
    "foot_switch_hold",
    hands_lf_on,
    weld_ees=(lh, rh, rf),
    hold_feet=(lf,),
    base_shift=np.array([0.0, switch_lateral, 0.0]),
    plant_toes=(rf,),
    extra_overrides={"waist_yaw": 6.0},
  )
  toe_wall_hold[rf] = d.site_xpos[ik.toe_sid[rf]].copy()

  def rf_toe_hold():
    return gcc.toe_task(ik, rf, toe_wall_hold[rf], w_pos=(600.0, 3.0, 10.0))

  # hand_set_hold: the RIGHT hand engages on its final rung; three welds.
  # The base STAYS at the reach station — no settle to a compressed hang.
  # The donors' crouch existed because their hands were still LOW here;
  # with the right hand already on its FINAL rung, sinking the body
  # stretches that arm past its handoff branch (measured: pitch -2.47 ->
  # -2.99, elbow -0.56 -> -0.23 across the settle) and the rise-back
  # tears the weld — the yo-yo IS the foot-first 23 mm drift residual.
  # v5 (2026-08-22): ``set_hold_press`` / ``set_hold_rise`` move the base IN
  # toward the wall / UP along it during hand_set_hold (three welds + the
  # flat left foot). In the hands-first order the pivot otherwise starts 7 cm
  # further out and 7 cm lower than v3's tiptoe-reach station, the swing leg
  # runs straight (knee +1 deg) and flips branches at 15 rad/s (RF2 f713).
  sh_shift = None
  sh_w = (60.0, 60.0, 60.0)
  up_vec_sh = wall.up
  sh_shift = -set_hold_press * wall.outward + set_hold_rise * up_vec_sh
  sh_w = (300.0, 300.0, 300.0)
  rot_key_sh = None
  # v4 flat-footed reach: the left foot never tiptoed, hold it flat.
  dwell(
    "hand_set_hold",
    hands_lf_on,
    weld_ees=(rh, lh, rf),
    hold_feet=(lf,),
    toe_hold_targets={rf: toe_wall_hold[rf]},
    base_shift=sh_shift,
    base_w=sh_w,
    extra_overrides={"waist_yaw": 6.0},
    base_rot_key=rot_key_sh,
    base_w_rot=8.0 if rot_key_sh is None else 60.0,
  )
  # right_foot_mount: the toe pivot on three welds, verbatim from the
  # foot-first copy (which is itself the hands-first machinery): heel-lift
  # knee-drive with the toe planted through most of the swing, the guided
  # fold across the bistable window, the long face-on-wall rake.
  seg_name, swing_ee, other_ee = "left_foot_mount", lf, rf
  wall_welds = [lh, rh, rf]
  weld_anchor = {ee: ik.pad_pose(ee) for ee in wall_welds}
  for ee in wall_welds:
    anchor_now[ee] = weld_anchor[ee][0].copy()
  p0, rot0 = ik.pad_pose(swing_ee)
  p2 = final_anchor_pos[swing_ee]
  rot2 = state1_pad_rot[swing_ee]
  anchor_now[swing_ee] = p2.copy()
  # Control point 30 mm off the surface (donor: 15): during the relaxed
  # fold window the pad task runs at 15% and the approaching pad sphere
  # overshot the touch plane by 2.7 mm on the 15 mm path.
  # v5 (2026-08-22): the pivot arc's control point is a knob (v3: 3 cm out,
  # 5 cm up). Lifting the swing foot earlier keeps the knee bent while the
  # fold raises the body to the state-1 height (hands-first order: no tiptoe
  # station, the leg otherwise passes full extension and flips at 15 rad/s).
  pivot_ctl_off = pivot_ctl_out * wall.outward + np.array([0.0, 0.0, pivot_ctl_up])
  p1 = 0.5 * (p0 + p2) + pivot_ctl_off
  # The planted-toe station of the press-up, not the entry pose: the toe
  # has been sitting there since the right-hand reach.
  toe_hold = d.site_xpos[ik.toe_sid[lf]].copy()
  # Smear terminus: same height and lateral spot, toe sphere 8 mm short
  # of the (tilted) wall surface.
  toe_r = ik.toes[swing_ee].radius
  toe_smear = toe_hold.copy()
  # v5 (2026-08-22): ``pivot_smear_gap`` = how far short of the wall base the toe
  # smear stops (v3: 8 mm). In the hands-first order the hips sit 5-10 cm
  # further out at the pivot, so the v3 smear drags the toe 16-20 cm ahead of
  # the hip, the swing leg runs straight and flips branches at 15 rad/s.
  toe_smear[0] = wall.pivot[0] + (wall.outward[2] * toe_hold[2] - (toe_r + pivot_smear_gap)) / (-wall.outward[0])
  swing_fold_from = {}
  base_seg_start = d.qpos[0:3].copy()
  # The pivot HOLDS the station the reach ended on (plus the lateral
  # bias over the left support side): the high support arm keeps its
  # handoff geometry all the way to the closure, which does the rising.
  # 12 mm off the reach station (6 mm grazed the base_link by 9 mm once
  # the tiptoe stood the body taller).
  # v5 (2026-08-22): ``pivot_rise`` lifts the base ALONG THE WALL through the
  # toe pivot (three welds; the welded right leg's extension loads its pad in
  # down-shear, the safe direction) so the hand steps start at height and do
  # not rise on extended legs (RF2 passive gate: the left foot pad was pulled
  # off in tension / reverse shear through the non-closing left-hand step's
  # 4.6 cm rise). 0 = v3/v4.
  up_vec_pv = wall.up
  base_seg_end = base_seg_start + pivot_out * wall.outward + pivot_rise * up_vec_pv
  base_seg_end[1] = -lateral_weight_shift
  posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  swing_leg = _leg_joints("left")
  fold_u0 = pivot_fold_u0  # guided fold + stiff leg posture start (donor value 0.45; v5 knob)
  for t in range(n[seg_name]):
    u = t / (n[seg_name] - 1)
    s = float(gcc.smootherstep(u))
    s_rot = float(gcc.smootherstep(min(u / 0.9, 1.0)))
    # v5 (2026-08-22): ``pivot_toe_release`` = u at which the toe pin starts
    # fading (v3: 0.49, gone by 0.66). The flat-footed hands-first pivot keeps
    # the toe pinned while the pad target rises, the hip is pushed up on a
    # straight leg and the fold flips at 15 rad/s; an earlier release lets
    # the foot leave the floor on a bent knee.
    rel_end = pivot_toe_release + 0.17
    on_ground[swing_ee] = u < rel_end - 0.04
    tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in wall_welds]
    fade = float(np.clip((rel_end - u) / 0.17, 0.0, 1.0))
    if fade > 0.01:
      # Toe SMEAR toward the wall base (u 0.05-0.45, x tracked at 150
      # instead of the donor's free glide): with the toe honest at its
      # standing spot the shin must tilt ~35 deg to carry the pad to the
      # wall and the knee runs straight (-0.05) before toe-off — the
      # bistable flip then costs 15-21 rad/s however the release is
      # timed. Dragging the toe 2 cm short of the base line shortens
      # the chain; the one draft where the toe had (wrongly) slid
      # forward folded at 4 rad/s.
      smear_s = float(gcc.smootherstep(np.clip((u - 0.05) / 0.40, 0.0, 1.0)))
      # z at the donor's 250: pinning it at 1000 neither kept the toe
      # down (knee drive still lifts it ~1 cm) nor helped the low hand's
      # pull-out, and cost the cross pivot an early audit frame.
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.toe_sid[swing_ee],
          pos=toe_hold + smear_s * (toe_smear - toe_hold),
          rot=None,
          w_pos=np.array([150.0 * fade, 5.0, 250.0 * fade]),
          w_rot=0.0,
          track=False,
        )
      )
    tasks.append(rf_toe_hold())
    relax = 1.0 - 0.85 * float(
      gcc.smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0)) - gcc.smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
    )
    tasks.append(
      gcc.IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=_swing_target(
          t,
          n[seg_name],
          p0,
          p1,
          p2,
          n_rake_seg=seg_rake[seg_name],
          rake_vec_seg=rake_vec_foot,
        ),
        rot=gcc.slerp_matrix(rot0, rot2, s_rot),
        w_pos=np.array([1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]),
        w_rot=100.0 * relax,
        track=relax > 0.9,
      )
    )
    base_target = (1.0 - s) * base_seg_start + s * base_seg_end
    press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
    base_target = base_target - swing_press * press * wall.outward + np.array([0.0, -0.03 * press, 0.0])
    tasks.append(
      gcc.base_task(
        ik,
        base_target,
        (100.0, 40.0, 60.0),
        rot=wall.R,
        w_rot=10.0,
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in swing_leg:
        posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * state1_joint[nm]
      else:
        posture_t[nm] = posture_start[nm]
    if u >= fold_u0:
      if not swing_fold_from:
        for nm in swing_leg:
          swing_fold_from[nm] = float(d.qpos[ik.qpos_addresses[nm]])
      w = float(gcc.smootherstep(np.clip((u - fold_u0) / 0.50, 0.0, 1.0)))
      for nm in swing_leg:
        posture_t[nm] = (1.0 - w) * swing_fold_from[nm] + w * state1_joint[nm]
    overrides = {}
    for nm in swing_leg:
      if "ankle_roll" in nm:
        overrides[nm] = 4.0
      elif u >= fold_u0:
        overrides[nm] = 4.0
      elif "ankle_pitch" in nm:
        overrides[nm] = 1.0
      elif "hip_yaw" in nm:
        overrides[nm] = 2.0
    # The trunk does not twist to feed the pivot (0.54 rad measured
    # mid-swing with the waist free, 0.32 at 4; 16 bought nothing).
    overrides["waist_yaw"] = 8.0
    throttled = t < 8
    in_fold = 0.52 <= u <= 0.85
    res = ik.solve_frame(
      tasks,
      sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.05,
      lam2=2.0 if in_fold else 0.5,
    )
    seg_residual[seg_name].merge(res)
    _note_base(seg_name, base_target)
    push(seg_name, hands_lf_on)
  landed = ik.pad_pose(swing_ee)
  anchor_now[swing_ee] = landed[0].copy()
  landing_report[swing_ee] = (
    float(np.linalg.norm(landed[0] - p2)),
    gcc.rotation_angle(landed[1], rot2),
  )

  # brace_hold: all four welded; the RIGHT toe joins the planted left one
  # at the wall touch.
  dwell(
    "brace_hold",
    all_on,
    weld_ees=ik.ee_names,
    hold_feet=(),
    plant_toes=(lf,),
    toe_hold_targets={rf: toe_wall_hold[rf]},
    extra_overrides={"waist_yaw": 6.0},
  )
  toe_wall_hold[lf] = d.site_xpos[ik.toe_sid[lf]].copy()

  # hand_up_left / hand_up_right: the closure swing machinery (toe holds
  # release first, fade^2 done by u=0.55; the guided ankle fold walks both
  # feet to the state-1 hang; the base squares up; the shared closure
  # blend lands the handoff frame exactly), written once and called with
  # the sides swapped. v4: with an intermediate right rung the mount has
  # TWO hand steps — a non-closing hand_up_left (toe holds kept, no fold,
  # base 1.2 cm under the final station) -> hand_up_hold -> the closing
  # hand_up_right. The hand_up rule from the 08-20 post-mortem stands for
  # both: the base is nailed (200/100/150, w_rot 60) and the support arm
  # held stiff (override 4) inside a hand step.
  def cross_hand_up(
    seg_name, swing_ee, support_hand, base_end, closing, target_override=None, report_key=None, pin_swing=True
  ):
    for ee in ik.ee_names:
      if ee != swing_ee:
        anchor_now[ee] = ik.pad_pose(ee)[0].copy()
    weld_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee}
    p0, rot0 = ik.pad_pose(swing_ee)
    # v6g: an optional mid-rung target (the left hand's two-step walk-up).
    p2 = final_anchor_pos[swing_ee] if target_override is None else np.asarray(target_override, dtype=np.float64)
    anchor_now[swing_ee] = p0.copy()
    swing_ctl_off = swing_bulge * wall.outward
    p1 = 0.5 * (p0 + p2) + swing_ctl_off
    engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
    base_seg_start = d.qpos[0:3].copy()
    base_rot_start = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_start, d.qpos[3:7])
    base_rot_start = base_rot_start.reshape(3, 3)
    base_seg_end = np.asarray(base_end, dtype=np.float64).copy()
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    support_arm = _arm_joints("right" if support_hand == rh else "left")
    swing_arm = _arm_joints("left" if swing_ee == lh else "right")
    ankle_pitches = tuple(nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.endswith("ankle_pitch"))
    fold_from = {}
    for t in range(n[seg_name]):
      engaged = all_on if t < n_lt else engaged_swing
      if t == n_lt:
        anchor_now[swing_ee] = p2.copy()
      u = t / (n[seg_name] - 1)
      s = float(gcc.smootherstep(u))
      n_arc = n[seg_name] - n_rake
      s_arc = float(gcc.smootherstep(min(t / (n_arc - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in ik.ee_names if ee != swing_ee]
      toe_w = float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2 if closing else 1.0
      if toe_w > 1.0e-3:
        for ee in feet:
          tasks.append(
            gcc.toe_task(
              ik,
              ee,
              toe_wall_hold[ee],
              w_pos=(600.0 * toe_w, 3.0, 10.0),
            )
          )
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(t, n[seg_name], p0, p1, p2),
          rot=rot0,
          w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
      base_target = (1.0 - s) * base_seg_start + s * base_seg_end
      tasks.append(
        gcc.base_task(
          ik,
          base_target,
          (200.0, 100.0, 150.0),
          rot=gcc.slerp_matrix(base_rot_start, wall.R, s),
          w_rot=60.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in swing_arm:
          # v6g mid step: no state-1 branch pull -- the pad task alone
          # steers the arm to the mid rung (the state-1 pose is the 1.52
          # landing's and dragged the mid landing 29 mm off).
          target = state1_joint[nm] if (closing or pin_swing) else posture_start[nm]
        elif nm in support_arm or nm == "waist_yaw":
          target = posture_start[nm]
        else:
          # Legs walk toward the state-1 hang in BOTH steps: the
          # non-closing right step carries the rise the v3 tiptoe used
          # to buy (feet welded, knees extend); the toes stay held.
          target = state1_joint[nm]
        posture_t[nm] = (1.0 - s) * posture_start[nm] + s * target
      # The 08-20 rule pins the support arm (override 4) against
      # hyper-extension onto the shoulder stop — that is the HIGH support
      # of the closing step. The non-closing right step hangs from the
      # LOW left hand while the body rises 6 cm: a pinned low arm freezes
      # the shoulder to the hand and the rise never happens (base z 60 mm
      # short, measured); rising only flexes that elbow, so it stays soft.
      overrides = {nm: (4.0 if closing else 1.0) for nm in support_arm}
      if not closing and pin_swing:
        # Branch pin (the reach's lesson): the stepping arm rides its
        # state-1 branch or the closure pays 0.8 rad of shoulder pitch.
        overrides.update({nm: reach_arm_pin for nm in swing_arm})
      if closing and u >= 0.42:
        for nm in ankle_pitches:
          if nm not in fold_from:
            fold_from[nm] = float(d.qpos[ik.qpos_addresses[nm]])
          w = float(gcc.smootherstep(np.clip((u - 0.42) / 0.50, 0.0, 1.0)))
          posture_t[nm] = (1.0 - w) * fold_from[nm] + w * state1_joint[nm]
          overrides[nm] = 3.0
      throttled = t < 8
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_t,
        posture_weight=0.6,
        posture_weight_overrides=overrides,
        iters=2 if throttled else 100,
        max_step=0.003 if throttled else 0.12,
      )
      seg_residual[seg_name].merge(res)
      _note_base(seg_name, base_target)
      push(seg_name, engaged)
    landed = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = landed[0].copy()
    landing_report[report_key or swing_ee] = (
      float(np.linalg.norm(landed[0] - p2)),
      gcc.rotation_angle(landed[1], state1_pad_rot[swing_ee]),
    )

  if mid_anchor_rh is None:
    cross_hand_up("hand_up_left", lh, rh, final_qpos[0:3].copy(), closing=True)
  else:
    # v4 order: the LEFT hand steps first (outboard low rung -> final rung,
    # the mid-rung right hand + both feet supporting; this step carries
    # the rise the v3 tiptoe used to buy — the right arm below the
    # shoulder only flexes); then the RIGHT hand's closing step with the
    # high left hand as support = cross v3's proven closing step mirrored.
    # (Stepping the right hand first failed: the outboard low left rung
    # puts that arm on its shoulder-roll/elbow stops as the body rises.)
    up_vec = wall.up
    out_vec = wall.outward
    base_lh = final_qpos[0:3] - 0.012 * up_vec + 0.002 * out_vec
    base_lh[1] = float(d.qpos[1])
    # v5: the HIGH right hand steps first (1.46 -> 1.52, a 6 cm step with the
    # low left hand + both feet supporting and no rise), then the left hand
    # closes from the high right support = cross v3's proven closing step.
    cross_hand_up("hand_up_right", rh, lh, base_lh, closing=False)
    dwell(
      "hand_up_hold",
      all_on,
      weld_ees=ik.ee_names,
      hold_feet=(),
      toe_hold_targets=toe_wall_hold,
    )
    # v6g: the 30 cm left walk-up splits at a mid rung -- each landing
    # keeps elbow flexion in reserve instead of arriving at the reach
    # frontier (closing-school plateau: supports held, the swing hand
    # missed the far landing on ~half its edges).
    up_dir = wall.up
    mid_lh = final_anchor_pos[lh] + (hand_up_left_mid - hand_z) * up_dir
    mid_lh = mid_lh + np.array([0.0, hand_up_left_mid_y, 0.0])
    # The weight-shift hold parks the base ~5 cm toward the support
    # side; at the mid height the left arm cannot cross back to the
    # lane against its shoulder-roll stop (measured: a flat +24 mm
    # lateral landing miss). The mid step walks the base halfway home.
    base_mid = np.asarray(base_lh, dtype=np.float64).copy()
    base_mid[1] = 0.5 * base_mid[1]
    cross_hand_up(
      "hand_up_left_mid",
      lh,
      rh,
      base_mid,
      closing=False,
      target_override=mid_lh,
      report_key=f"{lh}@mid_up",
      pin_swing=False,
    )
    dwell(
      "hand_up_left_hold",
      all_on,
      weld_ees=ik.ee_names,
      hold_feet=(),
      toe_hold_targets=toe_wall_hold,
    )
    cross_hand_up("hand_up_left", lh, rh, final_qpos[0:3].copy(), closing=True)

  closure_seg = "hand_up_right"
  delta = final_qpos - qpos_frames[-1]
  closure_frames = max(int(round(0.5 * n[closure_seg])), 2)
  for i in range(closure_frames):
    idx = len(qpos_frames) - closure_frames + i
    blend = float(gcc.smootherstep((i + 1) / closure_frames))
    qpos_frames[idx] = qpos_frames[idx] + blend * delta
    qpos_frames[idx][3:7] /= np.linalg.norm(qpos_frames[idx][3:7])
  closure_pos = float(np.linalg.norm(delta[0:3]))
  closure_joint = float(np.abs(delta[7:]).max())
  closure_joint_name = "<none>"
  if delta[7:].size:
    _wj = int(np.argmax(np.abs(delta[7:])))
    for _nm, _adr in ((nm, ik.qpos_addresses[nm]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES):
      if _adr == 7 + _wj:
        closure_joint_name = _nm
        break

  d.qpos[:] = final_qpos
  mujoco.mj_forward(m, d)
  for ee in ik.ee_names:
    anchor_now[ee] = ik.pad_pose(ee)[0].copy()
  # handoff_hold: the all-four stance held INSIDE the playable region (the
  # loader truncates the end_hold tail). A terminal clock clamped here sits
  # a full segment away from the last engage edge, so letting go bills at
  # full price — clamping on a last-frame flip would ride the edge grace
  # forever ("just latched, releases are free").
  for _ in range(n["handoff_hold"]):
    push("handoff_hold", all_on)
  for _ in range(n["end_hold"]):
    push("end_hold", all_on)

  if smooth_sigma > 0.0:
    # Zero-phase Gaussian smoothing of the SOLVED frames (root position,
    # root quaternion, joints) up to the handoff: the per-frame DLS leaves
    # one-frame steps at task hand-overs that read as jitter in the
    # render. The handoff/end frames are exact state-1 copies and pad the
    # right edge, so the closure still lands on them exactly; the audits
    # below run on the smoothed frames, so any drift the blur costs is
    # measured, not hidden.
    n_play = len(qpos_frames) - n["handoff_hold"] - n["end_hold"]
    radius = max(int(round(3.0 * smooth_sigma)), 1)
    kern = np.exp(-0.5 * (np.arange(-radius, radius + 1) / smooth_sigma) ** 2)
    kern /= kern.sum()
    arr = np.stack(qpos_frames)
    left = np.repeat(arr[:1], radius, axis=0)
    right = np.repeat(arr[n_play : n_play + 1], radius, axis=0)
    padded = np.concatenate([left, arr[:n_play], right], axis=0)
    sm = np.empty_like(arr[:n_play])
    for col in range(arr.shape[1]):
      sm[:, col] = np.convolve(padded[:, col], kern, mode="valid")
    sm[:, 3:7] /= np.linalg.norm(sm[:, 3:7], axis=1, keepdims=True)
    for i in range(n_play):
      qpos_frames[i] = sm[i]

  # -- extract outputs ---------------------------------------------------------
  frames = len(qpos_frames)
  joint_positions = np.zeros((frames, len(PHYBOT_C2_POLICY_JOINT_NAMES)), np.float32)
  root_position = np.zeros((frames, 3), np.float32)
  root_quaternion = np.zeros((frames, 4), np.float32)
  ee_positions = np.zeros((frames, 4, 3), np.float32)
  ee_quaternions = np.zeros((frames, 4, 4), np.float32)
  pad_positions = np.zeros((frames, 4, 3), np.float32)
  foot_positions = np.zeros((frames, 2, 3), np.float32)
  engaged_arr = np.stack(engaged_frames)
  anchor_arr = np.stack(anchor_frames).astype(np.float32)
  ground_arr = np.stack(ground_frames)
  label_arr = np.asarray(labels, dtype=np.uint8)

  # Clearance sweeps against the wall plane and the floor, over COLLISION
  # geoms only (visual meshes have contype=conaffinity=0 and never contact
  # anything in sim; their conservative AABBs read ~1 cm through the wall
  # even in the cycle's own state-1 stance). The FOOT (ankle_roll) geoms are
  # exempt from the wall sweep: whenever a toe touches the wall the foot
  # mesh AABB corner reads ~3 cm "through", while the true mesh tip stops
  # 2 mm short of the toe sphere by design (the sphere protrudes exactly so
  # wall contact lands on the pair'd sphere — phybot_c2_constants note).
  pad_or_toe_geoms = set()
  visual_geoms = set()
  # Shank (ankle_pitch) geoms are exempt with the feet: the spine pads
  # stand off the shin by their 12 mm sphere radius, so a welded pad puts
  # the shank's AABB corner within ~1 mm of the plane by construction.
  shank_bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_pitch") for side in ("left", "right")}
  all_foot_geoms = set()
  for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "_spine_c_" in name or "toe" in name:
      pad_or_toe_geoms.add(g)
    if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
      visual_geoms.add(g)
    if m.geom_bodyid[g] in foot_body_bid.values() or m.geom_bodyid[g] in shank_bids:
      all_foot_geoms.add(g)
  foot_geoms = {ee: {g for g in range(m.ngeom) if m.geom_bodyid[g] == foot_body_bid[ee]} for ee in feet}
  corner_signs = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
  )

  wall_clear_min = (np.inf, -1, "")
  ground_clear_min = (np.inf, -1, "")
  max_weld_drift = 0.0
  max_wall_pen = -np.inf
  max_wall_pen_at = (-1, "")
  wall_pen_hits: list = []
  com_margin_min = (np.inf, -1)
  mass = m.body_mass.sum()
  com_single: dict[str, list[float]] = {}  # v4: per segment [min lateral-toward-stance, max ahead-of-toe]

  for t, q in enumerate(qpos_frames):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    joint_positions[t] = [d.qpos[ik.qpos_addresses[nm]] for nm in PHYBOT_C2_POLICY_JOINT_NAMES]
    root_position[t] = q[0:3]
    root_quaternion[t] = (q[4], q[5], q[6], q[3])
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
          # Penetration past the pad-touch offset plane (one sphere
          # radius short of the tilted surface).
          pen = (gcc.WALL_SURFACE_X - gcc.PAD_TOUCH_X) - wall.wall_distance(d.xpos[bid_s])
          if pen > max_wall_pen:
            max_wall_pen_at = (t, ee)
          max_wall_pen = max(max_wall_pen, pen)
          if pen > 0.0:
            wall_pen_hits.append((float(pen), int(t), ee))
    for i, sid in enumerate(ik.foot_sid):
      foot_positions[t, i] = d.site_xpos[sid]

    seg_name_t = SEGMENT_NAMES[label_arr[t]]
    if seg_name_t in ("hand_up_left", "hand_up_hold", "hand_up_right"):
      for ee in feet:
        gap = wall.wall_distance(d.site_xpos[ik.toe_sid[ee]]) - ik.toes[ee].radius
        key = f"{ee}@{seg_name_t}"
        toe_gap_report[key] = max(toe_gap_report.get(key, -np.inf), gap)
    grounded_geoms = set()
    for i, ee in enumerate(feet):
      # A foot on the floor legitimately touches it; a foot whose pad is
      # ENGAGED on the wall stands 4 cm over the floor at the wall base —
      # its proximity is a stance fact, not a swing scrape.
      if ground_arr[t, i] or engaged_arr[t, 2 + i]:
        grounded_geoms |= foot_geoms[ee]
    for g in range(m.ngeom):
      if m.geom_bodyid[g] == 0 or g in visual_geoms:
        continue
      center = m.geom_aabb[g, :3]
      half = m.geom_aabb[g, 3:]
      world = d.geom_xpos[g] + (center + half * corner_signs) @ d.geom_xmat[g].reshape(3, 3).T
      if g not in pad_or_toe_geoms and g not in all_foot_geoms:
        clear = float(((world - wall.pivot) @ wall.outward).min())
        if clear < wall_clear_min[0]:
          wall_clear_min = (
            clear,
            t,
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}",
          )
      if g not in grounded_geoms:
        clear = float(world[:, 2].min()) - ground_z
        if clear < ground_clear_min[0]:
          ground_clear_min = (
            clear,
            t,
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}",
          )
    if int(ground_arr[t].sum()) == 1:
      # v4 audit: single-foot floor support — where is the CoM relative to
      # the standing sole (lateral, + = toward the stance side, i.e. over or
      # beyond the foot) and to the toe front (+ = tipping over the toe)?
      com_w = (m.body_mass[:, None] * d.xipos).sum(axis=0) / mass
      i_st = int(np.flatnonzero(ground_arr[t])[0])
      sole = d.site_xpos[ik.foot_sid[i_st]]
      dy = float(com_w[1] - sole[1]) * (1.0 if i_st == 0 else -1.0)
      tw = -np.array([wall.outward[0], wall.outward[1], 0.0])
      tw = tw / max(np.linalg.norm(tw), 1.0e-9)
      toe_front = float(d.site_xpos[ik.toe_sid[feet[i_st]]] @ tw + ik.toes[feet[i_st]].radius)
      dx = float(com_w @ tw) - toe_front
      rec = com_single.setdefault(SEGMENT_NAMES[label_arr[t]], [np.inf, -np.inf])
      rec[0] = min(rec[0], dy)
      rec[1] = max(rec[1], dx)
    if ground_arr[t].all() and not engaged_arr[t].any():
      com_x = float((m.body_mass[:, None] * d.xipos).sum(axis=0)[0] / mass)
      toe_x = max(float(d.site_xpos[ik.toe_sid[ee]][0] + ik.toes[ee].radius) for ee in feet)
      heel_x = min(float(d.xpos[foot_body_bid[ee]][0] - 0.06) for ee in feet)
      margin = min(toe_x - com_x, com_x - heel_x)
      if margin < com_margin_min[0]:
        com_margin_min = (margin, t)

  # Canonical-frame storage: de-rotate every world-frame output about the
  # base line so the runtime re-poses the clip at the sampled wall angle
  # through its existing wall-frame machinery. The audits above ran in the
  # NATIVE tilted world; only the stored arrays change frame.
  root_position = wall.derotate_points(root_position)
  root_quaternion = wall.derotate_quats_xyzw(root_quaternion)
  ee_positions = wall.derotate_points(ee_positions)
  ee_quaternions = wall.derotate_quats_xyzw(ee_quaternions)
  pad_positions = wall.derotate_points(pad_positions)
  foot_positions = wall.derotate_points(foot_positions)
  anchor_arr = wall.derotate_points(anchor_arr)

  data_out = {
    "joints_list": list(PHYBOT_C2_POLICY_JOINT_NAMES),
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "foot_sites_list": list(gcc.FOOT_SITE_NAMES),
    "foot_positions": foot_positions,
    "fps": np.float32(fps),
    "ee_list": list(ik.ee_names),
    "ee_positions": ee_positions,
    "ee_quaternions": ee_quaternions,
    "pad_positions": pad_positions,
    "ee_engaged": engaged_arr,
    "ee_anchor_positions": anchor_arr,
    "toe_braced": np.zeros((frames, 2), dtype=np.uint8),
    "toe_list": list(ik.toes.keys()),
    "foot_on_ground": ground_arr,
    "phase_segment": label_arr,
    "segment_names": list(SEGMENT_NAMES),
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.0),
    "wall_surface_x": np.float32(gcc.WALL_SURFACE_X),
    "pad_touch_x": np.float32(gcc.PAD_TOUCH_X),
    # Native wall angle (deg from horizontal, 90 = vertical). Frame data is
    # stored CANONICAL (de-rotated); the validator re-rotates to this angle
    # for its native-world statics, the runtime poses the wall wherever the
    # band sampler says and absorbs |sampled - native| <= 2 deg.
    "wall_angle": np.float32(wall.angle_deg),
  }
  # The current CAD stance uses passive gimbal coordinates. They are part of
  # reference geometry and RSI, never additional policy actuator targets.
  passive_names = tuple(name for name in ik.qpos_addresses if "_gimbal_" in name)
  data_out["passive_joint_names"] = list(passive_names)
  data_out["passive_joint_positions"] = np.stack(qpos_frames)[
    :, [ik.qpos_addresses[name] for name in passive_names]
  ].astype(np.float32)
  data_out["pad_reference_point"] = "C"
  data_out["generator"] = "scripts/gen_mount_clip.py"
  data_out["handoff_motion"] = handoff_motion.name
  data_out["handoff_motion_sha256"] = hashlib.sha256(handoff_motion.read_bytes()).hexdigest()
  asset_xml = _REPO / "src/scr_phybot_mjab/assets/phybot_c2/xml/phybot_c2.xml"
  data_out["asset_xml_sha256"] = hashlib.sha256(asset_xml.read_bytes()).hexdigest()
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data_out)

  # -- report ------------------------------------------------------------------
  print(
    f"wrote {output} ({frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps, "
    f"dz={dz:+.4f} m: hands {hand_z_low:.3f} -> {hand_z:.3f}, feet "
    f"{final_anchor_pos[lf][2]:.3f})"
  )
  for name in SEGMENT_NAMES:
    r = seg_residual[name]
    if r.pos > 0.0 or r.rot > 0.0:
      print(f"  {name:18s} IK residual: pos {r.pos * 1000.0:6.2f} mm, rot {np.degrees(r.rot):5.2f} deg")
  for key, (pos_err, rot_err) in landing_report.items():
    print(f"  {key:22s} landing vs scheduled anchor: {pos_err * 1000.0:6.2f} mm, {np.degrees(rot_err):5.2f} deg")
  for ee, gap in toe_gap_report.items():
    print(
      f"  toe {ee:18s} wall-touch gap {gap * 1000.0:+6.2f} mm "
      "(must be within +0.5 mm for the static audit to credit the brace)"
    )
  joint_speed = np.abs(np.diff(joint_positions, axis=0)) * fps
  peak_frame, peak_joint = np.unravel_index(np.argmax(joint_speed), joint_speed.shape)
  base_speed = float(np.abs(np.diff(root_position, axis=0)).max() * fps)
  print(
    f"  peak joint speed {joint_speed.max():.2f} rad/s "
    f"({PHYBOT_C2_POLICY_JOINT_NAMES[int(peak_joint)]} @f{int(peak_frame)} "
    f"{SEGMENT_NAMES[label_arr[int(peak_frame)]]}), peak base speed "
    f"{base_speed:.3f} m/s"
  )
  if joint_speed.max() > 8.0:
    print(
      "    note: the residual single-frame step sits at the right-swing"
      " leg's bistable fold (u~0.66). Guided-fold posture, press shaping"
      " and DLS damping brought it 56 -> ~15 rad/s; the remainder is a"
      " ~0.3 rad one-frame reference step on one joint, tracked through"
      " by the policy (the reference is never replayed open loop)."
    )
  # Rake verification for every landing pad.
  for seg_name, i in (
    ("hand_place", 0),
    ("hand_place_right", 1),
    ("left_foot_mount", 2),
    ("right_foot_mount", 3),
    ("hand_up_left", 0),
    ("hand_up_right", 1),
  ):
    frames_in_seg = np.flatnonzero(label_arr == seg_id[seg_name])
    last = int(frames_in_seg[-1])
    nr = seg_rake[seg_name]
    drop = float(pad_positions[last - nr, i, 2] - pad_positions[last, i, 2])
    print(f"  rake {ik.ee_names[i]:18s} ({seg_name}): down {drop * 1000.0:5.2f} mm")
  print(
    f"  engaged-pad weld drift max      {max_weld_drift * 1000.0:6.2f} mm\n"
    f"  swing sphere wall penetration   {max_wall_pen * 1000.0:+6.2f} mm @f{max_wall_pen_at[0]} {max_wall_pen_at[1]} "
    f"(<= 0 clear)\n"
    f"  closure correction              {closure_pos * 1000.0:6.2f} mm base, "
    f"{closure_joint:.4f} rad joints ({closure_joint_name})"
  )
  if wall_pen_hits:
    _top = sorted(set((round(p_, 4), t, ee) for p_, t, ee in wall_pen_hits), key=lambda x: -x[0])[:12]
    print("    penetration frames: " + ", ".join(f"f{t} {ee[:2]} {p_ * 1000:+.1f}" for p_, t, ee in _top))
  print(
    f"  wall clearance min (non-pad)    {wall_clear_min[0] * 1000.0:+7.2f} mm "
    f"@f{wall_clear_min[1]} {SEGMENT_NAMES[label_arr[wall_clear_min[1]]]} "
    f"({wall_clear_min[2]})"
  )
  print(
    f"  ground clearance min (airborne) {ground_clear_min[0] * 1000.0:+7.2f} mm "
    f"@f{ground_clear_min[1]} {SEGMENT_NAMES[label_arr[ground_clear_min[1]]]} "
    f"({ground_clear_min[2]})"
  )
  if com_margin_min[1] >= 0:
    print(
      f"  standing CoM support margin min {com_margin_min[0] * 1000.0:6.1f} mm "
      f"@f{com_margin_min[1]} (pre-engage frames)"
    )
  for seg_name_c, (dy_min, dx_max) in com_single.items():
    print(
      f"  single-foot CoM {seg_name_c:18s} lateral toward stance min "
      f"{dy_min * 1000.0:+6.1f} mm (>= -50 ok), ahead of toe max "
      f"{dx_max * 1000.0:+6.1f} mm (<= 0 ok)"
    )
  for seg_name_b, err in base_track.items():
    print(
      f"  base track {seg_name_b:18s} max |base - target| "
      f"x {err[0] * 1000.0:5.1f} y {err[1] * 1000.0:5.1f} z {err[2] * 1000.0:5.1f} mm"
    )
  flagged = []
  for i, nm in enumerate(PHYBOT_C2_POLICY_JOINT_NAMES):
    lo, hi = ik.joint_ranges[nm]
    col = joint_positions[:, i]
    lo_m = float(col.min()) - lo
    hi_m = hi - float(col.max())
    margin = min(lo_m, hi_m) / (hi - lo)
    if margin < 0.05:
      fr = int(np.argmin(col) if lo_m < hi_m else np.argmax(col))
      flagged.append((nm, margin, fr))
  for nm, margin, fr in flagged:
    print(f"  LIMIT {nm:22s} {margin * 100.0:5.2f}% of range @f{fr} {SEGMENT_NAMES[label_arr[fr]]}")
  worst_name, worst_margin, worst_frame = "", np.inf, -1
  for i, nm in enumerate(PHYBOT_C2_POLICY_JOINT_NAMES):
    lo, hi = ik.joint_ranges[nm]
    col = joint_positions[:, i]
    lo_m = float(col.min()) - lo
    hi_m = hi - float(col.max())
    margin = min(lo_m, hi_m) / (hi - lo)
    if margin < worst_margin:
      worst_name, worst_margin = nm, margin
      worst_frame = int(np.argmin(col) if lo_m < hi_m else np.argmax(col))
  print(
    f"  joint-limit clearance min       {worst_margin * 100.0:6.2f}% of range "
    f"({worst_name} @f{worst_frame} "
    f"{SEGMENT_NAMES[label_arr[worst_frame]]})"
  )
  if "shoulder_roll" in worst_name and "hand_up" in SEGMENT_NAMES[label_arr[worst_frame]]:
    print(
      "    note: one-hand-high / one-hand-low support kinematically forces "
      "the LOW arm's shoulder roll onto its stop on this wristless 4-DoF "
      "arm (cycle generator's state-4 note, amplified by the 0.09 m hand "
      "step). Base-pose relief was probed and does not exist; accepted as "
      "a ~0.3 s transient, same precedent as the cycle's 0.058 rad case."
    )
  # Handoff contract: the final frame must BE state 1 shifted by dz.
  handoff_joint = float(
    np.abs(joint_positions[-1] - np.array([state1_joint[nm] for nm in PHYBOT_C2_POLICY_JOINT_NAMES])).max()
  )
  print(f"  handoff joint delta vs state-1  {handoff_joint:.2e} rad (must be ~0)")
  if max_weld_drift > 5.0e-3 or max_wall_pen > 2.0e-3:
    print("  WARNING: IK quality below tolerance; inspect the render.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Clip to write (the committed one is assets/motions/climb_mount_prologue_a80.npy).",
  )
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument(
    "--handoff-motion",
    type=Path,
    default=CAD_CLIMB_MOTION,
    help="Cycle whose first complete pose (including passive joints) defines the handoff.",
  )
  parser.add_argument(
    "--wall-angle-deg",
    type=float,
    default=90.0,
    help="native wall angle (deg from horizontal, 90 = vertical). The clip "
    "is solved in the tilted native world and stored de-rotated to the "
    "canonical frame with a wall_angle metadata key; 90 reproduces the "
    "shipped clip bitwise",
  )
  parser.add_argument(
    "--standoff",
    type=float,
    default=0.23,
    help="standing horizontal distance to the wall BASE LINE (m) — at every "
    "angle the binding constraint is toe clearance at the base, where the "
    "surface sits at x=1.5 regardless of tilt; toes clear by "
    "standoff - 0.163 and the flat-foot hand reach ceiling shrinks ~1 cm "
    "per +1 cm of standoff",
  )
  parser.add_argument(
    "--hand-z",
    type=float,
    default=1.53,
    help="FINAL hand anchor ALONG-WALL height (m, == world z at 90) = "
    "state-1 height + dz. The final stance's lowest point (ankle collision) "
    "hangs 1.512 m below the hand anchors ALONG the wall; at 90, 1.53 "
    "leaves 1.8 cm of ground; every extra cm here costs a cm of hand-step "
    "reach margin",
  )
  parser.add_argument(
    "--hand-z-low",
    type=float,
    default=1.44,
    help="first-placement hand anchor ALONG-WALL height (m); must sit "
    "inside the standing reach ceiling, which SHRINKS as the wall leans "
    "away (~1.47 at 90/std standoff; the surface recedes by h*sin(90-a))",
  )
  parser.add_argument("--stand-s", type=float, default=1.0)
  parser.add_argument("--raise-s", type=float, default=2.4)
  parser.add_argument("--place-s", type=float, default=1.4)
  parser.add_argument(
    "--set-left-s",
    type=float,
    default=1.5,
    help="hand_set_left dwell (s): the left hand welded, the right "
    "hovering — long enough for the RSI window to sample the one-hand "
    "classroom",
  )
  parser.add_argument(
    "--place-right-s",
    type=float,
    default=1.0,
    help="hand_place_right duration (s): the braced right-hand close hover -> anchor + rake",
  )
  parser.add_argument("--set-hold-s", type=float, default=0.5)
  parser.add_argument("--foot-swing-s", type=float, default=1.6)
  parser.add_argument(
    "--foot-swing-right-s",
    type=float,
    default=2.4,
    help="right foot mount duration (s); longer than the left because the "
    "toe-pivot leg folds through its whole reconfiguration here — at 1.6 s "
    "the thigh sweeps ~300 deg/s through the fold window, visually out of "
    "step with the paced reference",
  )
  parser.add_argument("--switch-hold-s", type=float, default=0.4)
  parser.add_argument("--brace-hold-s", type=float, default=0.5)
  parser.add_argument("--hand-up-s", type=float, default=1.6)
  parser.add_argument("--up-hold-s", type=float, default=0.4)
  parser.add_argument(
    "--handoff-hold-s",
    type=float,
    default=0.4,
    help="all-four handoff dwell inside the playable region (s); a terminal "
    "clock clamps here, one full engage-run away from the last edge",
  )
  parser.add_argument("--end-hold-s", type=float, default=0.6)
  parser.add_argument("--place-bulge", type=float, default=0.04)
  parser.add_argument(
    "--raise-yaw",
    type=float,
    default=0.0,
    help="cross sequence: mirrored shoulder-yaw excursion during the arm fold (rad)",
  )
  parser.add_argument("--foot-bulge", type=float, default=0.04)
  parser.add_argument(
    "--foot-lift",
    type=float,
    default=0.10,
    help="extra +z on the foot-mount Bezier control point (ground clearance)",
  )
  parser.add_argument("--swing-bulge", type=float, default=0.04)
  parser.add_argument(
    "--place-lean",
    type=float,
    default=0.02,
    help="forward base drift allowed during hand placement (m)",
  )
  parser.add_argument(
    "--swing-press",
    type=float,
    default=0.05,
    help="base wall-press amplitude through the foot-mount swings (m); "
    "shrinks the anchored hands' pull-out in the statically weak "
    "mid-swing window (v58: the audit's 0.99-util frames live there)",
  )
  parser.add_argument(
    "--raise-roll",
    type=float,
    default=0.0,
    help="interleaved only: outboard shoulder-roll amplitude (rad) held "
    "through the sagittal raise's mid sweep. 0.0 = pure sagittal front "
    "raise; ~0.8 = diagonal (the elbow-clearance fallback); 1.6 "
    "reproduces the lateral windmill",
  )
  parser.add_argument(
    "--raise-roll-right",
    type=float,
    default=0.0,
    help="interleaved only: outboard shoulder-roll amplitude (rad) of the "
    "RIGHT arm's raise inside hand_place_right (the trunk is bowed toward "
    "the slab there; ~1.3 sweeps the elbow through +y clear of the wall)",
  )
  parser.add_argument(
    "--step-hips-in",
    type=float,
    default=0.06,
    help="interleaved only: how far the hips travel toward the wall base "
    "during the braced first foot mount (m); pays the welded low hand the "
    "slack the un-bowing trunk takes from it",
  )
  parser.add_argument(
    "--pivot-out",
    type=float,
    default=0.02,
    help="interleaved/cross: base station change across the toe-pivot "
    "mount, positive = away from the wall (m); trades the pivot's "
    "base_link wall clearance against the low hand's pull-out",
  )
  parser.add_argument(
    "--raise-pitch-end",
    type=float,
    default=-2.35,
    help="interleaved/cross: shoulder pitch the LEFT arm's standing raise "
    "ends at (rad; -2.35 = overhead cocked, ~-1.8 = hand in front of the "
    "face, where the touch-and-slide place starts)",
  )
  parser.add_argument(
    "--touch-below",
    type=float,
    default=0.10,
    help="interleaved/cross: how far BELOW the low rung (along-wall, m) the "
    "left pad first touches the wall before sliding up to the rung",
  )
  parser.add_argument(
    "--touch-frac",
    type=float,
    default=0.5,
    help="interleaved/cross: fraction of the hand_place arc time spent "
    "reaching the touch point (the rest slides up the wall)",
  )
  parser.add_argument(
    "--place-bow",
    type=float,
    default=0.25,
    help="interleaved/cross: trunk pitch target (rad, nose-down) the base "
    "bows to while the left hand lands its first rung; 0 = stay upright "
    "(pair with a --hand-z-low the upright arm reaches)",
  )
  parser.add_argument(
    "--reach-arm-pin",
    type=float,
    default=6.0,
    help="interleaved/cross: posture weight override on the RIGHT arm "
    "through its final-rung reach (branch pin against the second arm "
    "solution family)",
  )
  parser.add_argument(
    "--step-base-w",
    type=float,
    nargs=3,
    default=(100.0, 40.0, 60.0),
    metavar=("WX", "WY", "WZ"),
    help="cross: base position task weights through the braced right-foot "
    "mount (v3 100/40/60 lost the vote; v4 recipe 300/300/60)",
  )
  parser.add_argument(
    "--step-press",
    type=float,
    default=0.0,
    help="cross: mid-swing base wall-press amplitude through the braced "
    "right-foot mount (like --swing-press for the pivot); 0 = v3",
  )
  parser.add_argument(
    "--step-hips-lateral",
    type=float,
    default=0.05,
    help="cross: lateral hip shift over the standing LEFT foot across the right-foot mount (world +y)",
  )
  parser.add_argument(
    "--switch-lateral",
    type=float,
    default=-0.05,
    help="cross: lateral base shift during foot_switch_hold (v3 walked the "
    "weight back to center with -0.05; v4 recipe 0 keeps it over the left foot)",
  )
  parser.add_argument(
    "--hand-z-mid",
    type=float,
    default=1.53,
    help="along-wall height of the right hand's first rung, reachable flat-footed; "
    "the right hand walks up to --hand-z in hand_up_right after the feet mount",
  )
  parser.add_argument(
    "--mid-rung-y",
    type=float,
    default=0.0,
    help="cross v4: lateral offset of the intermediate right-hand rung",
  )
  parser.add_argument(
    "--foot-rake-s",
    type=float,
    default=0.3,
    help="rake duration of the left foot mount (s); the right foot uses --rake-right-s",
  )
  parser.add_argument(
    "--foot-rake-dist",
    type=float,
    default=0.002,
    help="rake stroke of both foot mounts (m)",
  )
  parser.add_argument(
    "--low-rung-y",
    type=float,
    default=0.0,
    help="interleaved/cross: lateral offset (m, world +y = robot's left) "
    "of the LEFT hand's first rung off its final lane",
  )
  parser.add_argument(
    "--smooth-sigma",
    type=float,
    default=0.0,
    help="zero-phase Gaussian smoothing of the solved frames (sigma in "
    "frames, 0 = off) applied before output extraction and the audits; "
    "the exact handoff/end frames are never touched. Takes the per-frame "
    "IK's one-frame steps out of the reference (visible as jitter)",
  )
  parser.add_argument(
    "--stand-y",
    type=float,
    default=0.0,
    help="v5: lateral station of the STANDING footprint (m, world +y = robot's "
    "left) relative to the anchor lane (y = 0). Negative parks the body right of "
    "the lane so the left sole sits under it; pair with --transfer-lateral = "
    "-stand-y and --step-hips-lateral/--switch-lateral/--reach-lateral/"
    "--pivot-lateral 0 so the body stays on the lane after the transfer. 0 = v3.",
  )
  parser.add_argument(
    "--step-rise",
    type=float,
    default=0.01,
    help="cross: along-wall base rise (m) through the braced right-foot mount; v3 = 0.01. "
    "Negative hangs the body on the welded left hand during the lift (v5 passive-gate fix).",
  )
  parser.add_argument(
    "--pivot-rise",
    type=float,
    default=0.0,
    help="cross v5: along-wall base rise (m) through the left-foot toe pivot (three welds); 0 = v3.",
  )
  parser.add_argument(
    "--set-hold-press",
    type=float,
    default=0.0,
    help="cross v5: base shift toward the wall (m) during hand_set_hold (pivot starts closer).",
  )
  parser.add_argument(
    "--set-hold-rise",
    type=float,
    default=0.0,
    help="cross v5: along-wall base rise (m) during hand_set_hold (pivot starts higher).",
  )
  parser.add_argument(
    "--pivot-smear-gap",
    type=float,
    default=0.008,
    help="cross v5: the toe smear stops this far short of the wall base (m); v3 = 0.008.",
  )
  parser.add_argument(
    "--pivot-toe-release",
    type=float,
    default=0.49,
    help="cross v5: pivot progress u at which the planted toe pin starts fading (gone 0.17 later); v3 = 0.49.",
  )
  parser.add_argument(
    "--pivot-ctl-up", type=float, default=0.05, help="cross v5: pivot arc control point lift (m); v3 = 0.05."
  )
  parser.add_argument(
    "--pivot-ctl-out",
    type=float,
    default=0.030,
    help="cross v5: pivot arc control point outward offset (m); v3 = 0.03.",
  )
  parser.add_argument(
    "--pivot-fold-u0",
    type=float,
    default=0.45,
    help="cross v5: pivot progress u at which the guided leg fold (posture -> state-1, stiff leg) starts; v3 = 0.45.",
  )
  parser.add_argument(
    "--raise-pitch-comp-sign",
    type=float,
    default=1.0,
    help="cross v6 --rh-first: sign/scale of the trunk-pitch compensation of the right-arm raise keys (1 = add the trunk forward pitch to the end key).",
  )
  parser.add_argument(
    "--stance-width",
    type=float,
    default=0.16,
    help="standing stance width (m): IK re-plants the feet at +-w/2 (the walk keyframe stands at 0.296)",
  )
  parser.add_argument(
    "--place-right-pitch",
    type=float,
    default=0.0,
    help="cross v6 --rh-first: trunk pitch key (rad, + = toward the wall) reached over the right-hand place arc.",
  )
  parser.add_argument(
    "--hand-up-left-mid",
    type=float,
    default=1.57,
    help="height of the middle rung the left hand's two-step walk-up passes (m); each landing keeps elbow flexion in reserve",
  )
  parser.add_argument(
    "--up-left-mid-hold-s", type=float, default=0.4, help="four-anchor re-seat dwell between the two left-hand steps."
  )
  parser.add_argument(
    "--hand-up-left-mid-y",
    type=float,
    default=0.0,
    help="lateral offset (m, world +y = outboard for the left hand) of the mid rung off the final lane: at 1.40 the lane sits past the shoulder-roll stop (+24 mm flat landing miss measured); the lane is the hand's to choose on the closing step.",
  )
  parser.add_argument(
    "--weld-w-scale",
    type=float,
    default=1.0,
    help="multiplier on every support-weld IK task weight (pos 1000 / rot 500 x scale): >1 trades base/press realization for less engaged-pad drift (the welded pads sink 1-3 mm into the wall under the hips-in press at 1.0).",
  )
  parser.add_argument(
    "--place-right-base-dz",
    type=float,
    default=0.0,
    help="cross v6 --rh-first: base rise (m) reached over the right-hand place arc (reach relief for high rungs: the parked base over-stretches the welded left arm, 13-21 mm weld drift at 1.44-1.46).",
  )
  parser.add_argument(
    "--place-right-base-dx",
    type=float,
    default=0.0,
    help="cross v6 --rh-first: base shift toward the wall (m) reached over the right-hand place arc.",
  )
  parser.add_argument(
    "--place-right-left-arm-w",
    type=float,
    default=4.0,
    help="cross v6 --rh-first: posture-override weight of the welded left arm during the right-hand place (4.0 = stiff support-arm rule).",
  )
  parser.add_argument(
    "--raise-pitch-end-right",
    type=float,
    default=-2.4,
    help="end shoulder-pitch key of the right-arm raise (rad)",
  )
  parser.add_argument("--lateral-weight-shift", type=float, default=0.03)
  parser.add_argument("--hand-load-transfer-s", type=float, default=0.1)
  parser.add_argument("--rake-s", type=float, default=0.1)
  parser.add_argument(
    "--rake-right-s",
    type=float,
    default=0.45,
    help="extended face-on-wall rake for the right foot mount (s); covers "
    "the statically weak window between toe-off and landing",
  )
  parser.add_argument("--rake-dist", type=float, default=0.002)
  parser.add_argument("--limit-margin-frac", type=float, default=0.10)
  parser.add_argument("--limit-weight", type=float, default=3.0)
  args = parser.parse_args()
  if args.weld_w_scale != 1.0:
    _weld_task_base = gcc.weld_task

    def _weld_task_scaled(ik, ee, anchor):  # noqa: ANN001
      task = _weld_task_base(ik, ee, anchor)
      task.w_pos = task.w_pos * args.weld_w_scale
      task.w_rot = task.w_rot * args.weld_w_scale
      return task

    gcc.weld_task = _weld_task_scaled

  generate_mount_clip(
    output=args.output,
    handoff_motion=args.handoff_motion,
    raise_yaw=args.raise_yaw,
    fps=args.fps,
    wall_angle_deg=args.wall_angle_deg,
    standoff=args.standoff,
    hand_z=args.hand_z,
    hand_z_low=args.hand_z_low,
    stand_s=args.stand_s,
    raise_s=args.raise_s,
    place_s=args.place_s,
    set_left_s=args.set_left_s,
    place_right_s=args.place_right_s,
    set_hold_s=args.set_hold_s,
    foot_swing_s=args.foot_swing_s,
    foot_swing_right_s=args.foot_swing_right_s,
    switch_hold_s=args.switch_hold_s,
    brace_hold_s=args.brace_hold_s,
    hand_up_s=args.hand_up_s,
    up_hold_s=args.up_hold_s,
    handoff_hold_s=args.handoff_hold_s,
    end_hold_s=args.end_hold_s,
    place_bulge=args.place_bulge,
    foot_bulge=args.foot_bulge,
    foot_lift=args.foot_lift,
    swing_bulge=args.swing_bulge,
    place_lean=args.place_lean,
    swing_press=args.swing_press,
    raise_roll=args.raise_roll,
    raise_roll_right=args.raise_roll_right,
    step_hips_in=args.step_hips_in,
    pivot_out=args.pivot_out,
    raise_pitch_end=args.raise_pitch_end,
    touch_below=args.touch_below,
    touch_frac=args.touch_frac,
    place_bow=args.place_bow,
    reach_arm_pin=args.reach_arm_pin,
    step_base_w=tuple(args.step_base_w),
    step_press=args.step_press,
    step_hips_lateral=args.step_hips_lateral,
    switch_lateral=args.switch_lateral,
    hand_z_mid=args.hand_z_mid,
    mid_rung_y=args.mid_rung_y,
    foot_rake_s=args.foot_rake_s,
    foot_rake_dist=args.foot_rake_dist,
    low_rung_y=args.low_rung_y,
    smooth_sigma=args.smooth_sigma,
    lateral_weight_shift=args.lateral_weight_shift,
    stand_y=args.stand_y,
    step_rise=args.step_rise,
    pivot_rise=args.pivot_rise,
    set_hold_press=args.set_hold_press,
    set_hold_rise=args.set_hold_rise,
    pivot_smear_gap=args.pivot_smear_gap,
    pivot_toe_release=args.pivot_toe_release,
    pivot_ctl_up=args.pivot_ctl_up,
    pivot_ctl_out=args.pivot_ctl_out,
    pivot_fold_u0=args.pivot_fold_u0,
    raise_pitch_comp_sign=args.raise_pitch_comp_sign,
    stance_width=args.stance_width,
    place_right_pitch=args.place_right_pitch,
    place_right_base_dz=args.place_right_base_dz,
    place_right_base_dx=args.place_right_base_dx,
    place_right_left_arm_w=args.place_right_left_arm_w,
    hand_up_left_mid=args.hand_up_left_mid,
    up_left_mid_hold_s=args.up_left_mid_hold_s,
    hand_up_left_mid_y=args.hand_up_left_mid_y,
    raise_pitch_end_right=args.raise_pitch_end_right,
    hand_load_transfer_s=args.hand_load_transfer_s,
    rake_s=args.rake_s,
    rake_right_s=args.rake_right_s,
    rake_dist=args.rake_dist,
    limit_margin_frac=args.limit_margin_frac,
    limit_weight=args.limit_weight,
  )


if __name__ == "__main__":
  main()
