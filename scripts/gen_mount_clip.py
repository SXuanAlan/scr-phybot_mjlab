"""Generate the PHYBot C2 walk->climb MOUNT prologue clip via whole-body IK.

The mount bridges the walk task's standing keyframe (feet flat on the floor,
arms down) to the climb cycle's state 1 ("climb zero": all four microspine
pads engaged on the wall) so a standalone mount policy can hand off to the
climb policy at an exact reference frame. The clip is ACYCLIC; its final
segment is named ``end_hold`` so ``ClimbMotionClip.from_dict`` accepts it
unchanged (everything before ``end_hold`` reads as one "cycle", the tail
supplies the finite-difference velocity frame). ``step_height`` is 0.

Choreography (probe-driven, 2026-08-05):

  ground_stand      walk-delivery stance at ``--standoff`` from the wall.
  hand_raise        joint-space LATERAL windmill raise to a cocked pose.
                    A sagittal raise is infeasible: at shoulder pitch
                    -0.4..-0.6 the elbow protrudes ~0.32 m and hits the wall
                    for any standoff under ~0.33 m, while hand placement
                    wants the base well inside that. Sideways (shoulder
                    roll) the elbow sweeps through +-y where there is no
                    wall.
  hand_place        both pads arc together toward the LOW hand anchors at
                    ``--hand-z-low``, but only the LEFT lands and rakes
                    (2 mm downward seating stroke); the RIGHT parks at a
                    hover point ``--hand-hover`` short of its anchor. Low
                    first because the state-1 anchor height is out of
                    standing reach: the flat-foot reach ceiling at the
                    placement standoff is ~1.47 m (probe), while ground
                    clearance of the final stance needs hands >= 1.53 m
                    (state-1's lowest point, the ankle collision volume,
                    hangs 1.512 m under the hand anchors; 2 cm floor
                    margin). A tiptoe-assisted single placement reaches
                    ~1.55 only by standing on two frontiers at once (arm
                    extension + toe balance) and was rejected per the v3
                    "reach frontier" precedent.
  hand_set_left     the LEFT hand engages (weld) under its raked preload
                    and holds; the right hovers. STAGGERED latch
                    (2026-08-15): the old both-hands-same-frame engage
                    demanded two simultaneous unbraced first contacts —
                    a condition the cycle never faces (it always regrips
                    one limb from a 3-anchor brace). One needle at a
                    time; the second press below is cycle-style braced.
  hand_place_right  the RIGHT pad closes hover -> anchor + rake, braced
                    by the welded left hand (and both flat feet).
  hand_set_hold     the RIGHT hand engages; both welded, feet flat.
  left_foot_mount   left shin pad ground -> its FINAL wall anchor; support
                    is two hand welds + the flat right foot.
  foot_switch_hold  four-contact dwell (2 welds + left pad + right foot);
                    also plants the LEFT toe to a wall touch — once the
                    right foot's floor support runs out mid-swing it is the
                    only right-swing low compression besides the left pad.
  right_foot_mount  TOE PIVOT, not an air swing: the right toe stays on the
                    floor (z-only hold, free to glide — an x pin locks the
                    chain against the wall-bound pad and the DLS snaps leg
                    branches) until the toe-pad chain geometrically runs
                    out of floor at pad height ~0.31 m (u~0.6); a long
                    face-on-wall rake (--rake-right-s) then bridges the
                    statically weak tail until landing. Ends in a
                    COMPRESSED all-four stance (hands still low). The
                    swing leg crosses a bistable fold at u~0.66 (heel-lift
                    pivot shape -> frog wall-plant shape): at full swing-
                    task weight the Bezier drags the leg through it as a
                    one-frame branch snap (56 rad/s unguided). The cure is
                    a HANDOVER, all on one window: the toe hold fades, the
                    mid-swing pad task relaxes to 15% (z keeps 50% — the
                    reconfiguring leg otherwise sags into the floor), and
                    a rate-limited guided fold walks the leg to the frog
                    shape before the task re-tightens for the rake. Peak
                    residual step ~5 rad/s, same scale as the arm raise.
  brace_hold        the RIGHT toe joins the left at the wall touch. The
                    touching toes extend the compression couple ~0.2 m
                    below the foot anchors; without them the one-support-
                    hand beats below audit at util 1.1-1.4 (F0=400). The
                    toes TOUCH (audit's 0.5 mm released-contact tolerance)
                    but the reference commands no press: pressing a toe
                    whose own pad is latched loads that anchor toward the
                    cheap release directions (cycle generator's warning).
  hand_up_left      cycle-style single-hand regrip, low anchor -> final
                    state-1 anchor (one +(hand_z - hand_z_low) step, with
                    the all-four load-transfer dwell and landing rake).
  hand_up_hold      four-anchor re-seat dwell; carries the lateral weight
                    bias from the right support side to the left.
  hand_up_right     mirror; toe holds release first (fade^2 by u=0.55),
                    then a rate-limited ankle blend folds the toes to the
                    state-1 hang, and the closure blend over the tail lands
                    the clip EXACTLY on state-1 shifted by
                    dz = hand_z - 1.7534.
  handoff_hold      0.4 s all-four dwell at the exact final stance, INSIDE
                    the playable region (the loader truncates the end_hold
                    tail): a terminal clock clamps here a full engage-run
                    away from the last edge, so releases bill full price.
  end_hold          the handoff dwell tail: the climb cycle's frame 0 pose.

Sequence variants (--sequence): the table above narrates ``hands-first``
(the shipped 90-deg choreography, byte-frozen). ``foot-first`` (2026-08-20)
resequences the same blocks foot-first. ``interleaved`` (2026-08-21) is the
80-deg redesign; note the "sagittal raise is infeasible" argument under
``hand_raise`` is 90-DEG GEOMETRY ONLY — as the wall leans away the surface
at height h recedes by h*sin(90-a), the elbow clears in FRONT, and the
lateral windmill has no reason to exist. See the branch banners for the
per-variant narratives.

Output schema == the cycle clip (tools.climbing_reference.cycle) plus:

    foot_on_ground   (T, 2) uint8   1 while that foot (left, right) is a
                                    flat floor support (feet order matches
                                    ``toe_list``/``foot_sites_list``)

Validate:
    python scripts/audit_mount_reference.py <clip.npy>
"""

from __future__ import annotations

import argparse
import hashlib
import os
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
  machinery re-poses them at the sampled angle. ``identity`` (a == 90) is an
  explicit fast path: every consumer keeps its original expression verbatim,
  so the 90-degree output stays bitwise identical to the shipped clip.
  """

  def __init__(self, angle_deg: float) -> None:
    self.angle_deg = float(angle_deg)
    self.identity = self.angle_deg == 90.0
    self.pivot = np.array([gcc.WALL_SURFACE_X, 0.0, 0.0])
    theta = np.radians(90.0 - self.angle_deg)
    if self.identity:
      self.R = np.eye(3)
      self.outward = np.array([-1.0, 0.0, 0.0])
      self.up = np.array([0.0, 0.0, 1.0])
      self.quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
      c, s = float(np.cos(theta)), float(np.sin(theta))
      self.R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
      self.outward = self.R @ np.array([-1.0, 0.0, 0.0])
      self.up = self.R @ np.array([0.0, 0.0, 1.0])
      half = 0.5 * theta
      self.quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    self.quat_inv = self.quat * np.array([1.0, -1.0, -1.0, -1.0])

  def rotate_point(self, p: np.ndarray) -> np.ndarray:
    if self.identity:
      return np.asarray(p, dtype=np.float64).copy()
    return self.pivot + self.R @ (np.asarray(p, dtype=np.float64) - self.pivot)

  def rotate_mat(self, rot: np.ndarray) -> np.ndarray:
    if self.identity:
      return np.asarray(rot, dtype=np.float64).copy()
    return self.R @ rot

  def rotate_root_qpos(self, qpos: np.ndarray) -> np.ndarray:
    """Rotate a free-root qpos (pos + wxyz quat) about the base line."""
    if self.identity:
      return qpos.copy()
    out = qpos.copy()
    out[0:3] = self.rotate_point(qpos[0:3])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, self.quat, qpos[3:7])
    out[3:7] = rotated
    return out

  def along_wall(self, p: np.ndarray) -> float:
    """Along-wall coordinate (the canonical-frame z) of a native point."""
    if self.identity:
      return float(p[2])
    return float(self.up @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def wall_distance(self, p: np.ndarray) -> float:
    """Signed clearance to the wall surface (positive on the robot side)."""
    if self.identity:
      return float(gcc.WALL_SURFACE_X - p[0])
    return float(self.outward @ (np.asarray(p, dtype=np.float64) - self.pivot))

  def derotate_points(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 3) array."""
    if self.identity:
      return arr
    shape = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float64)
    flat = (flat - self.pivot) @ self.R + self.pivot
    return flat.reshape(shape).astype(arr.dtype)

  def derotate_quats_xyzw(self, arr: np.ndarray) -> np.ndarray:
    """Native world -> canonical frame, any (..., 4) xyzw quaternion array."""
    if self.identity:
      return arr
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
  sequence: str,
  standoff: float,
  hand_z: float,
  hand_z_low: float,
  stand_s: float,
  raise_s: float,
  place_s: float,
  set_left_s: float,
  place_right_s: float,
  hand_hover: float,
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
  reach_base_z_w: float,
  step_base_w: tuple[float, float, float],
  step_press: float,
  step_hips_lateral: float,
  reach_rise: float | None,
  reach_rot_w: float,
  reach_rot_frac: float,
  reach_heel_lift: float,
  reach_heel_start: float,
  transfer_lateral: float,
  transfer_press: float,
  switch_lateral: float,
  reach_lateral: float | None,
  hand_z_mid: float | None,
  mid_rung_y: float,
  pivot_lateral: float | None,
  step_trunk_pitch: float,
  foot_rake_s: float | None,
  foot_rake_dist: float | None,
  low_rung_y: float,
  smooth_sigma: float,
  left_mount_frac: float,
  lateral_weight_shift: float,
  hand_load_transfer_s: float,
  rake_s: float,
  rake_right_s: float,
  rake_dist: float,
  limit_margin_frac: float,
  limit_weight: float,
  stand_y: float = 0.0,
  step_rise: float = 0.01,
  rh_first: bool = False,
  pivot_rise: float = 0.0,
  set_right_press: float = 0.0,
  set_right_drop: float = 0.0,
  close_rh_first: bool = False,
  set_hold_press: float = 0.0,
  set_hold_rise: float = 0.0,
  pivot_heel_lift: float | None = None,
  pivot_smear_gap: float = 0.008,
  pivot_toe_release: float = 0.49,
  pivot_ctl_up: float = 0.05,
  pivot_ctl_out: float = 0.030,
  pivot_fold_u0: float = 0.45,
  set_right_pitch: float = 0.0,
  set_right_s: float | None = None,
  set_left_pitch: float = 0.0,
  raise_pitch_comp_sign: float = 1.0,
  rh_raise_lateral: bool = False,
  set_left_s_override: float | None = None,
  stance_width: float | None = None,
  place_right_pitch: float = 0.0,
  place_right_base_dz: float = 0.0,
  place_right_base_dx: float = 0.0,
  place_right_left_arm_w: float = 4.0,
  weld_gimbal_deg: float = 0.0,
  weld_gimbal_w: float = 3.0,
  weld_hard: bool = False,
  hand_up_left_mid: float | None = None,
  hand_up_left_mid_s: float | None = None,
  up_left_mid_hold_s: float = 0.4,
  hand_up_left_mid_y: float = 0.0,
  set_right_lateral: float = 0.0,
  set_hold_pitch: float = 0.0,
  raise_pitch_end_right: float | None = None,
  handoff_motion: Path | None = None,
  raise_yaw: float = 0.0,
) -> None:
  if not 0.0 < wall_angle_deg <= 90.0:
    raise ValueError("--wall-angle-deg must be in (0, 90].")
  wall = _WallFrame(wall_angle_deg)
  ik = gcc.ClimbIK(limit_margin_frac=limit_margin_frac, limit_weight=limit_weight)
  m, d = ik.model, ik.data
  lh, rh, lf, rf = ik.ee_names
  gimbal_joint_names: dict[str, list[str]] = {}
  if weld_hard:
    _weld_task_soft = gcc.weld_task

    def _weld_task_hard(ik_, ee, anchor):  # noqa: ANN001
      task = _weld_task_soft(ik_, ee, anchor)
      task.hard = True
      return task

    gcc.weld_task = _weld_task_hard
    ik.debug_hard = bool(os.environ.get("IK_DEBUG"))
  if weld_gimbal_deg > 0.0:
    # v6: the plant's passive pad gimbals as IK DoF for WELDED pads only.
    # A wristless 4-DoF arm under a 6-DoF weld leaves the torso ~2 DoF; the
    # legacy solve (gimbals locked) met any further rotation demand by
    # translating the pad off its anchor -- 13-21 mm at right rungs 1.44-1.46
    # (left shoulder roll on its stop). Physically that pad cannot move; the
    # wrist ball joint can turn. Range clamp +-D deg (below the 3 deg stop so
    # the policy keeps travel for its own tracking error); swing pads stay
    # pinned at zero (their gimbal is passive in the sim, the IK must not use
    # it to fake a landing orientation).
    rad = np.radians(weld_gimbal_deg)
    for ee in ik.ee_names:
      names = [f"{ee}_gimbal_{ax}" for ax in ("x", "y", "z")]
      missing = [nm for nm in names if nm not in ik.qpos_addresses]
      if missing:
        raise ValueError(f"--weld-gimbal-deg needs the pad gimbal joints; missing {missing}")
      gimbal_joint_names[ee] = names
      for nm in names:
        ik.joint_ranges[nm] = np.array([-rad, rad])
        # The clamp is a hard stop the solver must see: a weak spring lets
        # every iteration ask the hinge for the whole rotation, the clamp
        # truncates it and the arm never takes the correction (13-33 deg
        # weld rot residuals measured at weight 3).
        ik.limit_weight_overrides[nm] = 1.0e4
    gimbal_dofs = np.array(
      [ik.dof_addresses[nm] for names in gimbal_joint_names.values() for nm in names],
      dtype=int,
    )
    _welded_now: set[str] = set()
    _weld_task_prev = gcc.weld_task

    def _weld_task_recording(ik_, ee, anchor):  # noqa: ANN001
      _welded_now.add(ee)
      return _weld_task_prev(ik_, ee, anchor)

    gcc.weld_task = _weld_task_recording
    _solve_frame_prev = ik.solve_frame

    def _solve_frame_gimbal(tasks, sel, posture_ref, *args, **kwargs):  # noqa: ANN001
      welded = set(_welded_now)
      _welded_now.clear()
      sel2 = np.concatenate([np.asarray(sel, dtype=int), gimbal_dofs]).astype(int)
      posture2 = dict(posture_ref)
      overrides = dict(kwargs.get("posture_weight_overrides") or {})
      for ee, names in gimbal_joint_names.items():
        for nm in names:
          posture2[nm] = 0.0
          overrides[nm] = weld_gimbal_w if ee in welded else 1000.0
      kwargs["posture_weight_overrides"] = overrides
      return _solve_frame_prev(tasks, sel2, posture2, *args, **kwargs)

    ik.solve_frame = _solve_frame_gimbal
  hands, feet = (lh, rh), (lf, rf)
  seg_id = {name: i for i, name in enumerate(SEGMENT_NAMES)}

  n = {
    "ground_stand": max(int(round(stand_s * fps)), 1),
    "hand_raise": max(int(round(raise_s * fps)), 2),
    "hand_place": max(int(round(place_s * fps)), 2),
    "hand_set_left": max(int(round((set_left_s if set_left_s_override is None else set_left_s_override) * fps)), 1),
    "hand_place_right": max(int(round(place_right_s * fps)), 2),
    "hand_set_hold": max(int(round(set_hold_s * fps)), 1),
    "left_foot_mount": max(int(round(foot_swing_s * fps)), 2),
    "foot_switch_hold": max(int(round(switch_hold_s * fps)), 1),
    "right_foot_mount": max(int(round(foot_swing_right_s * fps)), 2),
    "brace_hold": max(int(round(brace_hold_s * fps)), 2),
    "hand_up_left": max(int(round(hand_up_s * fps)), 2),
    "hand_up_hold": max(int(round(up_hold_s * fps)), 1),
    "hand_up_right": max(int(round(hand_up_s * fps)), 2),
    "hand_up_left_mid": max(int(round((hand_up_left_mid_s if hand_up_left_mid_s is not None else hand_up_s) * fps)), 2),
    "hand_up_left_hold": max(int(round(up_left_mid_hold_s * fps)), 1),
    "handoff_hold": max(int(round(handoff_hold_s * fps)), 2),
    "end_hold": max(int(round(end_hold_s * fps)), 1),
  }
  # v5 hands-first order (--rh-first): the right hand's own raise / set dwell
  # reuse the left-hand durations; its place arc reuses --place-right-s.
  n["hand_raise_right"] = n["hand_raise"]
  n["hand_set_right"] = n["hand_set_left"] if set_right_s is None else max(int(round(set_right_s * fps)), 1)
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
  n_rake_foot = n_rake if foot_rake_s is None else max(int(round(foot_rake_s * fps)), 1)
  seg_rake["left_foot_mount"] = n_rake_foot
  n_lt = max(int(round(hand_load_transfer_s * fps)), 0)
  for seg in ("hand_place", "hand_place_right", "left_foot_mount", "right_foot_mount", "hand_up_left", "hand_up_right"):
    if n[seg] - seg_rake[seg] < 2:
      raise ValueError(f"rake leaves fewer than two arc frames in {seg}")
  if n_lt >= min(n["hand_up_left"], n["hand_up_right"]):
    raise ValueError("--hand-load-transfer-s must be shorter than a hand-up swing")
  if wall.identity:
    rake_vec = np.array([0.0, 0.0, float(rake_dist)])
  else:
    # The pre-engage seating stroke rakes ALONG the wall surface.
    rake_vec = float(rake_dist) * wall.up
  rake_vec_default = rake_vec
  foot_dist = float(rake_dist if foot_rake_dist is None else foot_rake_dist)
  rake_vec_foot = np.array([0.0, 0.0, foot_dist]) if wall.identity else foot_dist * wall.up

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
  if handoff_motion is not None:
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
  if not wall.identity:
    state1_qpos = wall.rotate_root_qpos(state1_qpos)
    state1_anchor = {ee: (wall.rotate_point(pos), wall.rotate_mat(rot)) for ee, (pos, rot) in state1_anchor.items()}
  state1_pad_rot = {ee: state1_anchor[ee][1].copy() for ee in ik.ee_names}
  state1_hand_z = wall.along_wall(state1_anchor[lh][0])
  dz = hand_z - state1_hand_z
  final_qpos = state1_qpos.copy()
  if wall.identity:
    final_qpos[2] += dz
    final_anchor_pos = {ee: state1_anchor[ee][0] + np.array([0.0, 0.0, dz]) for ee in ik.ee_names}
    low_anchor_pos = {ee: final_anchor_pos[ee] + np.array([0.0, 0.0, hand_z_low - hand_z]) for ee in hands}
  else:
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
    if handoff_motion is not None and weld_gimbal_deg == 0.0:
      # Keep the CAD gimbal rest pose during IK as well as at the seam.
      # Locking it at zero would rotate the forearm into the wall while
      # matching a pad orientation generated at the -3 degree stop.
      for name in handoff.get("passive_joint_names", ()):
        adr = ik.qpos_addresses[name]
        d.qpos[adr] = state1_qpos[adr]
    mujoco.mj_forward(m, d)
    if stance_width is not None:
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
      hard=weld_hard,
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
      if wall.identity:
        plant_goal[ee] = np.array(
          [
            gcc.WALL_SURFACE_X - ik.toes[ee].radius,
            toe0[1],
            toe0[2] + 0.05,
          ]
        )
      else:
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
        if wall.identity:
          toe_target = (1.0 - s) * plant0[ee] + s * plant_goal[ee]
        else:
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
        if wall.identity:
          toe_gap_report[ee] = float(gcc.WALL_SURFACE_X - d.site_xpos[ik.toe_sid[ee]][0] - ik.toes[ee].radius)
        else:
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

  if sequence == "foot-first":
    # ===========================================================================
    # FOOT-FIRST order (user call, 08-20 night): the SAME choreography
    # vocabulary as hands-first — windmill raise, bezier arc + pre-engage
    # rake placements, dwell engages, toe-pivot right mount, cycle-style
    # hand step — RESEQUENCED per the user's spec: the LEFT foot takes its
    # final wall anchor first (at 80 deg the low anchor is a knee-high
    # step away from a receding surface), the RIGHT hand latches the LOW
    # rung braced by that weld, the LEFT hand goes DIRECTLY to its FINAL
    # rung, the RIGHT foot mounts as the usual toe pivot on three welds,
    # and a single right-hand walk-up closes onto the handoff stance.
    # Every block is a copy of its hands-first counterpart with only the
    # support set and targets adapted; no new motion vocabulary.
    # ===========================================================================
    lf_first_on = {lh: False, rh: False, lf: True, rf: False}
    rhlf_on = {lh: False, rh: True, lf: True, rf: False}
    three_on = {lh: True, rh: True, lf: True, rf: False}

    # ground_stand: the walk-delivery keyframe, verbatim.
    set_standing()
    for _ in range(n["ground_stand"]):
      push("ground_stand", none_on)

    # left_foot_mount: the hands-first LEFT mount block run FIRST — no hand
    # welds exist yet, so the only support is the flat right foot plus the
    # weight shift onto it. Same bezier (launch tangent UP first: the ankle
    # hangs 0.20 m under the shin pad and scrapes the floor otherwise),
    # same delayed plantarflex clock, same throttled opening.
    swing_ee, other_ee = lf, rf
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    rot2 = state1_pad_rot[swing_ee]
    anchor_now[swing_ee] = p2.copy()
    if wall.identity:
      mount_ctl_off = np.array([-foot_bulge, 0.0, foot_lift])
    else:
      mount_ctl_off = foot_bulge * wall.outward + np.array([0.0, 0.0, foot_lift])
    p1 = 0.65 * p0 + 0.35 * p2 + mount_ctl_off
    base_seg_start = d.qpos[0:3].copy()
    # A STEP's weight shift, not the hands-first lean: the hips move over
    # the stance foot and stay tall (the compressed-hang target does not
    # exist before the hands are up).
    if wall.identity:
      base_seg_end = base_seg_start + np.array([0.01, -0.05, 0.01])
    else:
      base_seg_end = base_seg_start - 0.01 * wall.outward + 0.01 * wall.up
      base_seg_end[1] = base_seg_start[1] - 0.05
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    swing_leg = _leg_joints("left")
    for t in range(n["left_foot_mount"]):
      u = t / (n["left_foot_mount"] - 1)
      s = float(gcc.smootherstep(u))
      # Delayed plantarflex (hands-first left-mount note): the pad lifts on
      # the Bezier first; the orientation follows once airborne.
      s_rot = float(gcc.smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
      on_ground[swing_ee] = False
      tasks = [foot_hold_task(other_ee)]
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(
            t,
            n["left_foot_mount"],
            p0,
            p1,
            p2,
            n_rake_seg=seg_rake["left_foot_mount"],
          ),
          rot=gcc.slerp_matrix(rot0, rot2, s_rot),
          w_pos=np.full(3, 1000.0),
          w_rot=100.0,
        )
      )
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_seg_start + s * base_seg_end,
          (100.0, 40.0, 60.0),
          rot=np.eye(3),
          w_rot=10.0,
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
      seg_residual["left_foot_mount"].merge(res)
      push("left_foot_mount", none_on)
    landed = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = landed[0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(landed[0] - p2)),
      gcc.rotation_angle(landed[1], rot2),
    )

    # foot_switch_hold: the fresh weld's settle dwell; the LEFT toe plants
    # onto the wall (hands-first dwell verbatim) — the low compression
    # point every later beat leans on — and the weight walks back to
    # center.
    dwell(
      "foot_switch_hold",
      lf_first_on,
      weld_ees=(lf,),
      hold_feet=(rf,),
      base_shift=np.array([0.0, 0.05, 0.0]),
      plant_toes=(lf,),
    )
    toe_wall_hold[lf] = d.site_xpos[ik.toe_sid[lf]].copy()

    def lf_toe_hold():
      return gcc.toe_task(ik, lf, toe_wall_hold[lf], w_pos=(600.0, 3.0, 10.0))

    # hand_raise: the hands-first windmill, keyed over the CURRENT pose
    # (legs and base frozen each frame so the fresh foot weld cannot
    # drift) — the same lateral roll-out choreography, both arms.
    raise_base = d.qpos.copy()
    cur_joint = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    raise_keys = {}
    for side, ee in (("left", lh), ("right", rh)):
      roll_range = ik.joint_ranges[f"{side}_shoulder_roll"]
      out_sign = 1.0 if roll_range[1] > 1.5 else -1.0
      s1 = state1_joint
      st = cur_joint
      raise_keys[f"{side}_shoulder_roll"] = (
        (0.0, st[f"{side}_shoulder_roll"]),
        (0.30, out_sign * 1.6),
        (0.70, out_sign * 1.6),
        (1.0, s1[f"{side}_shoulder_roll"]),
      )
      raise_keys[f"{side}_shoulder_pitch"] = (
        (0.0, st[f"{side}_shoulder_pitch"]),
        (0.35, st[f"{side}_shoulder_pitch"]),
        (0.85, -2.35),
        (1.0, -2.35),
      )
      raise_keys[f"{side}_shoulder_yaw"] = (
        (0.0, st[f"{side}_shoulder_yaw"]),
        (1.0, s1[f"{side}_shoulder_yaw"]),
      )
      raise_keys[f"{side}_elbow_pitch"] = (
        (0.0, st[f"{side}_elbow_pitch"]),
        (0.55, st[f"{side}_elbow_pitch"]),
        (0.90, -1.15),
        (1.0, -1.15),
      )
    for t in range(n["hand_raise"]):
      s = t / max(n["hand_raise"] - 1, 1)
      d.qpos[:] = raise_base
      for nm, keys in raise_keys.items():
        d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s)
      mujoco.mj_forward(m, d)
      push("hand_raise", lf_first_on)

    # The lean-in target: the hands-first compressed all-four stance. With
    # the LEFT foot already welded the body can walk toward it SAFELY
    # while the hands place — each reach is then short instead of the
    # standing frontier. Staged like the hands-first base schedule: 45%
    # by the right-hand landing, 70% through its set dwell, 95% by the
    # left-hand landing, 100% through hand_set_left.
    if wall.identity:
      base_compressed = np.array([final_qpos[0] - 0.012, 0.0, final_qpos[2] - 0.030])
    else:
      base_compressed = final_qpos[0:3] + 0.012 * wall.outward - 0.030 * wall.up
      base_compressed[1] = 0.0

    # hand_place_right: the RIGHT pad arcs from the cocked pose to the LOW
    # anchor + rake (hands-first hand_place machinery, single arm), braced
    # by the LF weld, the flat right foot and the planted left toe. The
    # LEFT arm holds its cocked pose (its own reach comes next).
    lf_weld = ik.pad_pose(lf)
    anchor_now[lf] = lf_weld[0].copy()
    rh_start = ik.pad_pose(rh)
    base_hold_r = d.qpos[0:3].copy()
    base_rot_r = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_r, d.qpos[3:7])
    base_rot_r = base_rot_r.reshape(3, 3)
    posture_hold_r = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    if wall.identity:
      place_ctl_off = np.array([-place_bulge, 0.0, 0.03])
    else:
      place_ctl_off = place_bulge * wall.outward + 0.03 * wall.up
    p0_r = rh_start[0]
    p2_r = low_anchor_pos[rh]
    p1_r = 0.5 * (p0_r + p2_r) + place_ctl_off
    base_pr_end = base_hold_r + 0.45 * (base_compressed - base_hold_r)
    for t in range(n["hand_place_right"]):
      s = float(gcc.smootherstep(t / (n["hand_place_right"] - 1)))
      n_arc_r = n["hand_place_right"] - n_rake
      s_arc = float(gcc.smootherstep(min(t / (n_arc_r - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [gcc.weld_task(ik, lf, lf_weld), foot_hold_task(rf), lf_toe_hold()]
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[rh],
          pos=_swing_target(t, n["hand_place_right"], p0_r, p1_r, p2_r),
          rot=gcc.slerp_matrix(rh_start[1], state1_pad_rot[rh], s),
          w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_hold_r + s * base_pr_end,
          (60.0, 60.0, 100.0),
          rot=gcc.slerp_matrix(base_rot_r, np.eye(3) if wall.identity else wall.R, 0.45 * s),
          w_rot=8.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in _arm_joints("right"):
          posture_t[nm] = (1.0 - s) * posture_hold_r[nm] + s * state1_joint[nm]
        elif nm == "waist_yaw":
          posture_t[nm] = (1.0 - s) * posture_hold_r[nm]
        else:
          posture_t[nm] = posture_hold_r[nm]
      ov_pr = {nm: 1.2 for nm in _arm_joints("right")}
      ov_pr["waist_yaw"] = 4.0
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_t,
        posture_weight=0.4,
        posture_weight_overrides=ov_pr,
      )
      seg_residual["hand_place_right"].merge(res)
      push("hand_place_right", lf_first_on)
    landed_r = ik.pad_pose(rh)
    landing_report[f"{rh}@low"] = (
      float(np.linalg.norm(landed_r[0] - low_anchor_pos[rh])),
      gcc.rotation_angle(landed_r[1], state1_pad_rot[rh]),
    )

    # hand_set_hold: the RIGHT hand engages under its raked preload; the
    # lean-in carries its middle stretch here, on two welds (v6 lesson:
    # the deep lean and the next reach must not share one segment).
    dwell(
      "hand_set_hold",
      rhlf_on,
      weld_ees=(rh, lf),
      hold_feet=(rf,),
      toe_hold_targets={lf: toe_wall_hold[lf]},
      base_shift=0.25 * (base_compressed - d.qpos[0:3]) * np.array([1.0, 0.0, 1.0]),
    )

    # hand_place: the LEFT pad arcs from its cocked pose DIRECTLY to its
    # FINAL anchor + rake — the best-supported moment of the mount takes
    # the highest reach (LF weld + RH weld + flat right foot + left toe).
    # The control point detours laterally around the head (the anchor sits
    # up-left of the face while the body leans on the slab).
    lf_weld = ik.pad_pose(lf)
    rh_weld = ik.pad_pose(rh)
    anchor_now[lf] = lf_weld[0].copy()
    anchor_now[rh] = rh_weld[0].copy()
    lh_start = ik.pad_pose(lh)
    base_hold_l = d.qpos[0:3].copy()
    base_rot_l = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_l, d.qpos[3:7])
    base_rot_l = base_rot_l.reshape(3, 3)
    posture_hold_l = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    p0_l = lh_start[0]
    p2_l = final_anchor_pos[lh]
    p1_l = 0.5 * (p0_l + p2_l) + place_ctl_off
    # The base RISES with the reach (the hands-first hand-step rule: rise
    # most of the way on the left step): landing the FINAL rung from the
    # compressed-hang height put the shoulder too low and the 4-DoF arm
    # welded onto its second branch (pitch -3.34/yaw -0.40/waist -0.41),
    # which the closure then had to snap 0.98 rad. Same end station as
    # the hands-first hand_up_left; hand_set_left settles back down to
    # the compressed hang for the right swing afterwards.
    if wall.identity:
      base_pl_end = np.array([final_qpos[0] - 0.002, -lateral_weight_shift, final_qpos[2] - 0.012])
    else:
      base_pl_end = final_qpos[0:3] + 0.002 * wall.outward - 0.012 * wall.up
      base_pl_end[1] = -lateral_weight_shift
    for t in range(n["hand_place"]):
      s = float(gcc.smootherstep(t / (n["hand_place"] - 1)))
      n_arc_l = n["hand_place"] - n_rake
      s_arc = float(gcc.smootherstep(min(t / (n_arc_l - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [
        gcc.weld_task(ik, lf, lf_weld),
        gcc.weld_task(ik, rh, rh_weld),
        foot_hold_task(rf),
        lf_toe_hold(),
      ]
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[lh],
          pos=_swing_target(t, n["hand_place"], p0_l, p1_l, p2_l),
          # CONSTANT state-1 palm target, no slerp: the slerp's transition
          # orientations at the leaned-in geometry are only reachable by
          # twisting the waist (-0.74 measured), which walks the whole arm
          # onto a solution family that does not exist at the handoff pose
          # (left shoulder pitch past its stop, the 0.975 rad closure
          # snap). Aligned EARLY, while the pad is still low and the
          # w_rot ramp is soft, the palm sets in free space and the arm
          # rides the state-1 branch the whole way up.
          rot=state1_pad_rot[lh],
          w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_hold_l + s * base_pl_end,
          (60.0, 60.0, 100.0),
          rot=gcc.slerp_matrix(base_rot_l, np.eye(3) if wall.identity else wall.R, 0.8 * s),
          w_rot=8.0,
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
      throttled = t < 8
      # Branch pin (cycle hand-swing lesson): the 4-DoF arm admits a second
      # (pitch -3.34, yaw -0.40, waist -0.41) solution for the SAME final
      # pad pose, and the DLS lands there from the cocked start — the
      # support-arm hold rule then carries it to the closure, which snaps
      # 0.98 rad. A stiff posture anchor on the reaching arm + the waist
      # keeps the landing on the state-1 branch the handoff requires.
      ov_pl = {nm: 6.0 for nm in _arm_joints("left")}
      ov_pl["waist_yaw"] = 6.0
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
      push("hand_place", rhlf_on)
    landed_l = ik.pad_pose(lh)
    landing_report[f"{lh}@final"] = (
      float(np.linalg.norm(landed_l[0] - p2_l)),
      gcc.rotation_angle(landed_l[1], state1_pad_rot[lh]),
    )

    # hand_set_left: the LEFT hand engages; three welds; the base finishes
    # its descent to the compressed hang so the right swing starts from
    # the same crouched geometry the hands-first executor was tuned on.
    dwell(
      "hand_set_left",
      three_on,
      weld_ees=(lh, rh, lf),
      hold_feet=(rf,),
      toe_hold_targets={lf: toe_wall_hold[lf]},
      base_shift=base_compressed - d.qpos[0:3],
    )

    # right_foot_mount: the hands-first TOE PIVOT, verbatim — heel-lift
    # knee-drive with the toe planted through most of the swing, the
    # guided fold across the bistable window, the long face-on-wall rake.
    # Support here is THREE welds (right hand low, LEFT HAND HIGH, left
    # foot) plus the left toe: a longer moment baseline than the
    # hands-first version ever had at this beat.
    seg_name, swing_ee, other_ee = "right_foot_mount", rf, lf
    wall_welds = [lh, rh, lf]
    weld_anchor = {ee: ik.pad_pose(ee) for ee in wall_welds}
    for ee in wall_welds:
      anchor_now[ee] = weld_anchor[ee][0].copy()
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    rot2 = state1_pad_rot[swing_ee]
    anchor_now[swing_ee] = p2.copy()
    if wall.identity:
      pivot_ctl_off = np.array([-0.015, 0.0, 0.05])
    else:
      pivot_ctl_off = 0.015 * wall.outward + np.array([0.0, 0.0, 0.05])
    p1 = 0.5 * (p0 + p2) + pivot_ctl_off
    toe_hold = d.site_xpos[ik.toe_sid[swing_ee]].copy()
    swing_fold_from = {}
    engaged_rswing = {lh: True, rh: True, lf: True, rf: False}
    base_seg_start = d.qpos[0:3].copy()
    base_seg_end = base_compressed + np.array([0.0, lateral_weight_shift, 0.0])
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    swing_leg = _leg_joints("right")
    for t in range(n[seg_name]):
      u = t / (n[seg_name] - 1)
      s = float(gcc.smootherstep(u))
      s_rot = float(gcc.smootherstep(min(u / 0.9, 1.0)))
      on_ground[swing_ee] = u < 0.60
      tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in wall_welds]
      fade = float(np.clip((0.62 - u) / 0.17, 0.0, 1.0))
      if fade > 0.01:
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.toe_sid[swing_ee],
            pos=toe_hold,
            rot=None,
            w_pos=np.array([2.0, 5.0, 250.0 * fade]),
            w_rot=0.0,
            track=False,
          )
        )
      tasks.append(lf_toe_hold())
      relax = 1.0 - 0.85 * float(
        gcc.smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0)) - gcc.smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
      )
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(t, n[seg_name], p0, p1, p2, n_rake_seg=seg_rake[seg_name]),
          rot=gcc.slerp_matrix(rot0, rot2, s_rot),
          w_pos=np.array([1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]),
          w_rot=100.0 * relax,
          track=relax > 0.9,
        )
      )
      base_target = (1.0 - s) * base_seg_start + s * base_seg_end
      press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
      if wall.identity:
        base_target = base_target + np.array([swing_press * press, 0.03 * press, 0.0])
      else:
        base_target = base_target - swing_press * press * wall.outward + np.array([0.0, 0.03 * press, 0.0])
      tasks.append(
        gcc.base_task(
          ik,
          base_target,
          (100.0, 40.0, 60.0),
          rot=np.eye(3) if wall.identity else wall.R,
          w_rot=10.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in swing_leg:
          posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * state1_joint[nm]
        else:
          posture_t[nm] = posture_start[nm]
      if u >= 0.45:
        if not swing_fold_from:
          for nm in swing_leg:
            swing_fold_from[nm] = float(d.qpos[ik.qpos_addresses[nm]])
        w = float(gcc.smootherstep(np.clip((u - 0.45) / 0.50, 0.0, 1.0)))
        for nm in swing_leg:
          posture_t[nm] = (1.0 - w) * swing_fold_from[nm] + w * state1_joint[nm]
      overrides = {}
      for nm in swing_leg:
        if "ankle_roll" in nm:
          overrides[nm] = 4.0
        elif u >= 0.45:
          overrides[nm] = 4.0
        elif "ankle_pitch" in nm:
          overrides[nm] = 1.0
        elif "hip_yaw" in nm:
          overrides[nm] = 2.0
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
      push(seg_name, engaged_rswing)
    landed = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = landed[0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(landed[0] - p2)),
      gcc.rotation_angle(landed[1], rot2),
    )

    # brace_hold: all four welded; the RIGHT toe joins the planted left
    # one at the wall touch (hands-first dwell verbatim).
    dwell(
      "brace_hold",
      all_on,
      weld_ees=ik.ee_names,
      hold_feet=(),
      plant_toes=(rf,),
      toe_hold_targets={lf: toe_wall_hold[lf]},
    )
    toe_wall_hold[rf] = d.site_xpos[ik.toe_sid[rf]].copy()

    # hand_up_right: the mount's ONLY hand walk-up (the left hand already
    # sits at its final rung) — the hands-first closure swing verbatim:
    # toe holds release first (fade^2, done by u=0.55), the guided ankle
    # fold walks both feet to the state-1 hang, the base squares up, and
    # the shared closure blend lands the handoff frame exactly.
    seg_name, swing_ee, support_hand = "hand_up_right", rh, lh
    for ee in ik.ee_names:
      if ee != swing_ee:
        anchor_now[ee] = ik.pad_pose(ee)[0].copy()
    weld_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee}
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    anchor_now[swing_ee] = p0.copy()
    if wall.identity:
      swing_ctl_off = np.array([-swing_bulge, 0.0, 0.0])
    else:
      swing_ctl_off = swing_bulge * wall.outward
    p1 = 0.5 * (p0 + p2) + swing_ctl_off
    engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
    base_seg_start = d.qpos[0:3].copy()
    base_rot_start = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_start, d.qpos[3:7])
    base_rot_start = base_rot_start.reshape(3, 3)
    base_seg_end = final_qpos[0:3].copy()
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    support_arm = _arm_joints("left")
    swing_arm = _arm_joints("right")
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
      toe_w = float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2
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
          (60.0, 30.0, 50.0),
          rot=gcc.slerp_matrix(base_rot_start, np.eye(3) if wall.identity else wall.R, s),
          w_rot=30.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in swing_arm:
          target = state1_joint[nm]
        elif nm in support_arm or nm == "waist_yaw":
          target = posture_start[nm]
        else:
          target = state1_joint[nm]
        posture_t[nm] = (1.0 - s) * posture_start[nm] + s * target
      overrides = {}
      if u >= 0.42:
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
      push(seg_name, engaged)
    landed = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = landed[0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(landed[0] - p2)),
      gcc.rotation_angle(landed[1], state1_pad_rot[swing_ee]),
    )

  elif sequence == "interleaved":
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
    lh_on = {lh: True, rh: False, lf: False, rf: False}
    lhlf_on = {lh: True, rh: False, lf: True, rf: False}
    hands_lf_on = {lh: True, rh: True, lf: True, rf: False}
    # The RIGHT hand never visits the low rung: re-seed its anchor lane at
    # the FINAL rung (the shared init aims hands at the low ladder; the lane
    # feeds ee_anchor_positions and the runtime press/tow rewards).
    anchor_now[rh] = final_anchor_pos[rh].copy()
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
        hard=weld_hard,
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
    if wall.identity:
      place_ctl_off = np.array([-place_bulge, 0.0, 0.03])
    else:
      place_ctl_off = place_bulge * wall.outward + 0.03 * wall.up
    lh_start = ik.pad_pose(lh)
    p0_l = lh_start[0]
    p2_l = low_anchor_pos[lh]
    p1_l = 0.5 * (p0_l + p2_l) + place_ctl_off
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
    dwell(
      "hand_set_left",
      lh_on,
      weld_ees=(lh,),
      hold_feet=feet,
      extra_overrides={"waist_yaw": 6.0},
    )

    # left_foot_mount: the foot-first left mount WITH a hand brace — the LH
    # weld turns the free-balance step into a held one. Same bezier (launch
    # tangent UP first: the ankle hangs 0.20 m under the shin pad and
    # scrapes the floor otherwise), same delayed plantarflex clock, same
    # throttled opening.
    swing_ee, other_ee = lf, rf
    lh_weld = ik.pad_pose(lh)
    anchor_now[lh] = lh_weld[0].copy()
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    rot2 = state1_pad_rot[swing_ee]
    anchor_now[swing_ee] = p2.copy()
    if wall.identity:
      mount_ctl_off = np.array([-foot_bulge, 0.0, foot_lift])
    else:
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
    if wall.identity:
      base_seg_end = base_seg_start + np.array([step_hips_in, -0.05, 0.01])
    else:
      base_seg_end = base_seg_start - step_hips_in * wall.outward + 0.01 * wall.up
      base_seg_end[1] = base_seg_start[1] - 0.05
    base_rot_lf_end = np.eye(3) if wall.identity else wall.R
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    swing_leg = _leg_joints("left")
    for t in range(n["left_foot_mount"]):
      u = t / (n["left_foot_mount"] - 1)
      s = float(gcc.smootherstep(u))
      s_rot = float(gcc.smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
      on_ground[swing_ee] = False
      tasks = [
        gcc.weld_task(ik, lh, lh_weld),
        stiff_foot_hold(other_ee),
      ]
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(
            t,
            n["left_foot_mount"],
            p0,
            p1,
            p2,
            n_rake_seg=seg_rake["left_foot_mount"],
          ),
          rot=gcc.slerp_matrix(rot0, rot2, s_rot),
          w_pos=np.full(3, 1000.0),
          w_rot=100.0,
        )
      )
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_seg_start + s * base_seg_end,
          # The free step's weights: at (40, 40, 40) the hips wandered into
          # the wall and the hip roll hit its stop (51 mm residual); the
          # 3.7 mm of hand-weld drift the stiffer pull costs is the price.
          (100.0, 40.0, 60.0),
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
      seg_residual["left_foot_mount"].merge(res)
      push("left_foot_mount", lh_on)
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
      lhlf_on,
      weld_ees=(lh, lf),
      hold_feet=(rf,),
      base_shift=np.array([0.0, 0.05, 0.0]),
      plant_toes=(lf,),
      extra_overrides={"waist_yaw": 6.0},
    )
    toe_wall_hold[lf] = d.site_xpos[ik.toe_sid[lf]].copy()

    def lf_toe_hold():
      return gcc.toe_task(ik, lf, toe_wall_hold[lf], w_pos=(600.0, 3.0, 10.0))

    # hand_place_right: the RIGHT arm rises AND lands in ONE sagittal arc,
    # DIRECTLY on its FINAL rung — no low hang, no later step. The raise is
    # folded into this segment on purpose: a second disjoint hand_raise
    # label window would stretch every label-resolved RSI window across the
    # foot mounts (resolve_segment_window spans first..last hit). Support:
    # LH weld + LF weld + the flat right foot + the planted left toe — the
    # strongest stance before the closure carries the longest reach. The
    # base RISES toward the final station with the reach (foot-first
    # lesson: landing the final rung from the compressed hang put the
    # shoulder too low and welded the arm onto its second branch).
    lh_weld = ik.pad_pose(lh)
    lf_weld = ik.pad_pose(lf)
    anchor_now[lh] = lh_weld[0].copy()
    anchor_now[lf] = lf_weld[0].copy()
    base_hold_r = d.qpos[0:3].copy()
    base_rot_r = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_r, d.qpos[3:7])
    base_rot_r = base_rot_r.reshape(3, 3)
    p2_r = final_anchor_pos[rh]
    if wall.identity:
      base_pr_end = np.array([final_qpos[0] - 0.002, lateral_weight_shift, final_qpos[2] - 0.012])
    else:
      base_pr_end = final_qpos[0:3] + 0.002 * wall.outward - 0.012 * wall.up
      base_pr_end[1] = lateral_weight_shift
    # Phase A — keyed sagittal raise with the body FROZEN (foot-first's
    # braced-raise pattern: base, legs and the left arm pinned frame to
    # frame, so the welds cannot move and no IK is needed). Reaching for
    # the anchor straight off the hip capsized on first try: forcing the
    # state-1 palm while the pad is still low twists the arm onto the
    # reverse-wrap branch (654 mm / 174 deg landing, shoulder at its stop)
    # — the exact "early capture" pathology the donors stage around with a
    # cocked pose. So: raise first, reach second, both inside this label.
    elbow_fold_r = max(float(ik.joint_ranges["right_elbow_pitch"][0]) + 0.10, -2.2)
    out_sign_r = 1.0 if ik.joint_ranges["right_shoulder_roll"][1] > 1.5 else -1.0
    n_pr = n["hand_place_right"]
    n_raise_r = max(int(round(0.48 * n_pr)), 2)
    if n_pr - n_raise_r - n_rake < 2:
      raise ValueError("--place-right-s leaves fewer than two arc frames")
    raise_base_r = d.qpos.copy()
    cur_r = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in _arm_joints("right")}
    # The raise is keyed in WORLD elevation, not joint space: the trunk
    # arrives bowed toward the slab (the hand place leaned ~18 deg into
    # its reach and the welded low hand holds that bow), and a joint-space
    # -2.35 on a bowed trunk points the forearm at the wall (elbow 134 mm
    # through it, measured). Shift the shoulder-pitch keys by the native
    # trunk pitch so the arm sweeps the same world arc as on an upright
    # trunk.
    _qw, _qx, _qy, _qz = (float(v) for v in d.qpos[3:7])
    trunk_pitch = float(np.arcsin(np.clip(2.0 * (_qw * _qy - _qz * _qx), -1.0, 1.0)))
    # The wall-parallel attitude (native pitch 90 - a) is the reference
    # the raise was validated on; only the bow PAST it shifts the keys.
    trunk_bow = max(trunk_pitch - np.radians(90.0 - wall.angle_deg), 0.0)
    cocked_pitch_r = -2.35 + trunk_bow
    rh_keys = {
      "right_shoulder_pitch": (
        (0.0, cur_r["right_shoulder_pitch"]),
        (0.25, cur_r["right_shoulder_pitch"]),
        (0.88, cocked_pitch_r),
        (1.0, cocked_pitch_r),
      ),
      # LATERAL for this arm (--raise-roll-right, 1.3 rad at 80): the
      # trunk is still bowed toward the slab and the shoulder sits ~15 cm
      # off the surface, where any sagittal sweep puts the forearm through
      # the wall (130-200 mm measured); the elbow sweeps through +y where
      # there is no wall, then the arm comes inboard overhead.
      "right_shoulder_roll": (
        (0.0, cur_r["right_shoulder_roll"]),
        (0.35, out_sign_r * raise_roll_right),
        (0.80, out_sign_r * raise_roll_right),
        (1.0, s1["right_shoulder_roll"]),
      ),
      "right_shoulder_yaw": (
        (0.0, cur_r["right_shoulder_yaw"]),
        (1.0, s1["right_shoulder_yaw"]),
      ),
      "right_elbow_pitch": (
        (0.0, cur_r["right_elbow_pitch"]),
        (0.10, cur_r["right_elbow_pitch"]),
        (0.42, elbow_fold_r),
        (0.80, elbow_fold_r),
        (1.0, -1.15),
      ),
    }
    for t in range(n_raise_r):
      s = t / max(n_raise_r - 1, 1)
      d.qpos[:] = raise_base_r
      for nm, keys in rh_keys.items():
        d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s)
      mujoco.mj_forward(m, d)
      push("hand_place_right", lhlf_on)
    # Phase B — the braced arc from the cocked staging to the FINAL rung:
    # donor machinery verbatim (CONSTANT state-1 palm — from the cocked
    # pose the palm sets in free space and the arm rides the state-1
    # branch the whole way; stiff posture pin against the second-branch
    # family).
    rh_cocked = ik.pad_pose(rh)
    p0_r = rh_cocked[0]
    p1_r = 0.5 * (p0_r + p2_r) + place_ctl_off
    n_arc_seg = n_pr - n_raise_r
    posture_hold_b = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    rf_toe0 = d.site_xpos[ik.toe_sid[rf]].copy()
    rf_ankles = ankle_joints[rf]
    rf_ankle_pitch = next(nm for nm in rf_ankles if nm.endswith("_pitch"))
    for t in range(n_arc_seg):
      s = float(gcc.smootherstep(t / (n_arc_seg - 1)))
      n_arc_r = n_arc_seg - n_rake
      s_arc = float(gcc.smootherstep(min(t / (n_arc_r - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [
        gcc.weld_task(ik, lh, lh_weld),
        gcc.weld_task(ik, lf, lf_weld),
        lf_toe_hold(),
      ]
      # The trailing foot TIPTOES through the reach — the human press-up.
      # The flat 1000/500 pin lets go at once (s 0-0.12) and a KEYED heel
      # lift takes over (ankle plantarflexion ramped over s 0.05-0.45, the
      # toe pinned to its floor spot): releasing a fading hold instead of
      # leading the heel stored the reach in the arm (shoulder -2.9 rad
      # against the flat foot) and the body popped 50 mm in one frame the
      # instant the hold weight ran out (9-15 rad/s ankle/shoulder steps,
      # at every fade window tried).
      heel_s = float(gcc.smootherstep(np.clip((s - 0.05) / 0.40, 0.0, 1.0)))
      # The flat hold is never RELEASED — its target ROTATES about the
      # planted toe with the heel key (R_y(+theta): heel up), and only
      # its weight eases (1000/500 -> 300/150) once the heel is up so the
      # reach can fine-tune. Fading the weight to zero at any window was a
      # threshold: the foot popped the instant the hold lost to the reach
      # (15-22 rad/s knee steps).
      pos_h, rot_h, sid_h = foot_hold_pose[rf]
      heel_th = 0.38 * heel_s
      heel_R = np.array(
        [
          [np.cos(heel_th), 0.0, np.sin(heel_th)],
          [0.0, 1.0, 0.0],
          [-np.sin(heel_th), 0.0, np.cos(heel_th)],
        ]
      )
      # The hold pivots about the toe pin's CURRENT target (it settles 4 mm
      # as the heel comes up) and eases to 15%: pivoted about the standing
      # toe spot at 30% it out-voted the toe pin and the sole toe point
      # rode up to 9 mm — past the audit's 8 mm gate (8 frames with no
      # floor support counted).
      # ...and at 15% it still lifted the toe site 4.6 mm off its pin
      # (sole toe point 8.8 mm): the hold LEADS the heel and hands over
      # completely (weight -> 0 with the key), the toe pin settles 6 mm
      # (the sphere still clears the floor by 4 mm).
      hold_w = 1.0 - heel_s
      toe_pivot = rf_toe0 - np.array([0.0, 0.0, 0.006 * heel_s])
      if hold_w > 0.01:
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=sid_h,
            pos=toe_pivot + heel_R @ (pos_h - rf_toe0),
            rot=heel_R @ rot_h,
            w_pos=np.full(3, 1000.0 * hold_w),
            w_rot=500.0 * hold_w,
          )
        )
      # The toe stays PLANTED (x/y pinned like a real toe under friction —
      # the pivot's free glide let it slide 7 cm into the wall base here)
      # and settles 4 mm as the heel comes up: the audit's sole toe point
      # rides the 8 mm gate at the standing toe height (7.5 mm), so the
      # planted toe must buy its margin by sitting lower once the heel no
      # longer defines the sole plane.
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.toe_sid[rf],
          pos=toe_pivot,
          rot=None,
          w_pos=np.array([600.0, 50.0, 2000.0]),
          w_rot=0.0,
          track=False,
        )
      )
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[rh],
          pos=_swing_target(t, n_arc_seg, p0_r, p1_r, p2_r),
          # Palm converges EARLY but GRADUALLY: from the cocked overhead
          # staging there is no reverse-wrap branch to fall into, and the
          # ramp keeps the throttle release from dumping the full 500-
          # weight orientation gap into the body in one step (v2: that
          # yank ran left_hip_roll onto its stop and dragged the welds
          # 157 mm).
          # Palm completes by s=0.35 — BEFORE the lean-in deepens. The
          # donor's warning is load-bearing: transition orientations in the
          # leaned geometry recruit the waist and detonate the arm (0.55
          # measured: waist snap 2.0 rad, welded left hand dragged through
          # the wall).
          rot=gcc.slerp_matrix(rh_cocked[1], state1_pad_rot[rh], min(s / 0.35, 1.0)),
          w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
      # The base height is a SUGGESTION here (z weight 40, not the donors'
      # 100): the station sits at the edge of what a tiptoed leg reaches
      # (knee ran to its -0.045 stop), and a 100-weight height pull either
      # held the body down until the heel popped (s clock) or snapped the
      # knee 0.3 rad the instant the flat hold let go (heel clock). The
      # keyed heel and the reach negotiate the height; the later segments
      # HOLD wherever the reach ended (no yo-yo), and the closure step
      # rises to the handoff station as a cycle hand step does.
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_hold_r + s * base_pr_end,
          (60.0, 60.0, reach_base_z_w),
          rot=gcc.slerp_matrix(
            base_rot_r,
            np.eye(3) if wall.identity else wall.R,
            0.8 * s,
          ),
          w_rot=8.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in _arm_joints("right"):
          posture_t[nm] = (1.0 - s) * posture_hold_b[nm] + s * state1_joint[nm]
        elif nm == "waist_yaw":
          posture_t[nm] = (1.0 - s) * posture_hold_b[nm]
        else:
          posture_t[nm] = posture_hold_b[nm]
      # Keyed heel lift: +0.45 rad plantarflexion (the solved tiptoe sat at
      # +0.49 over the standing -0.295) led by a stiff posture anchor while
      # the ramp runs, then handed to the reach.
      posture_t[rf_ankle_pitch] = posture_hold_b[rf_ankle_pitch] + 0.38 * heel_s
      # Branch pin (foot-first final-rung lesson): stiff posture anchor on
      # the reaching arm + the waist keeps the landing on the state-1
      # branch the handoff requires.
      ov_pr = {nm: reach_arm_pin for nm in _arm_joints("right")}
      ov_pr["waist_yaw"] = 6.0
      for nm in rf_ankles:
        # The roll stays braced; the pitch follows the heel key.
        ov_pr[nm] = 4.0 if nm.endswith("_roll") else (6.0 if heel_s < 1.0 else 1.0)
      # The tiptoed leg stays in its sagittal plane: left free, the heel
      # swung 20 cm outboard (a duck-footed tiptoe) when the height got
      # tight.
      ov_pr["right_hip_yaw"] = 3.0
      ov_pr["right_hip_roll"] = 3.0
      throttled = t < 8
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_t,
        posture_weight=0.4,
        posture_weight_overrides=ov_pr,
        iters=2 if throttled else 100,
        max_step=0.003 if throttled else 0.12,
      )
      seg_residual["hand_place_right"].merge(res)
      push("hand_place_right", lhlf_on)
    landed_r = ik.pad_pose(rh)
    landing_report[f"{rh}@final"] = (
      float(np.linalg.norm(landed_r[0] - p2_r)),
      gcc.rotation_angle(landed_r[1], state1_pad_rot[rh]),
    )
    rf_floor_toe = rf_toe0 - np.array([0.0, 0.0, 0.006])

    # hand_set_hold: the RIGHT hand engages on its final rung; three welds.
    # The base STAYS at the reach station — no settle to a compressed hang.
    # The donors' crouch existed because their hands were still LOW here;
    # with the right hand already on its FINAL rung, sinking the body
    # stretches that arm past its handoff branch (measured: pitch -2.47 ->
    # -2.99, elbow -0.56 -> -0.23 across the settle) and the rise-back
    # tears the weld — the yo-yo IS the foot-first 23 mm drift residual.
    dwell(
      "hand_set_hold",
      hands_lf_on,
      weld_ees=(rh, lh, lf),
      hold_feet=(),
      floor_toe_targets={rf: rf_floor_toe},
      toe_hold_targets={lf: toe_wall_hold[lf]},
      extra_overrides={"waist_yaw": 6.0},
    )

    # right_foot_mount: the toe pivot on three welds, verbatim from the
    # foot-first copy (which is itself the hands-first machinery): heel-lift
    # knee-drive with the toe planted through most of the swing, the guided
    # fold across the bistable window, the long face-on-wall rake.
    seg_name, swing_ee, other_ee = "right_foot_mount", rf, lf
    wall_welds = [lh, rh, lf]
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
    if wall.identity:
      pivot_ctl_off = np.array([-0.030, 0.0, 0.05])
    else:
      pivot_ctl_off = 0.030 * wall.outward + np.array([0.0, 0.0, 0.05])
    p1 = 0.5 * (p0 + p2) + pivot_ctl_off
    # The planted-toe station of the press-up, not the entry pose: the toe
    # has been sitting there since the right-hand reach.
    toe_hold = rf_floor_toe.copy()
    # Smear terminus: same height and lateral spot, toe sphere 8 mm short
    # of the (tilted) wall surface.
    toe_r = ik.toes[swing_ee].radius
    toe_smear = toe_hold.copy()
    if wall.identity:
      toe_smear[0] = gcc.WALL_SURFACE_X - toe_r - 0.008
    else:
      toe_smear[0] = wall.pivot[0] + (wall.outward[2] * toe_hold[2] - (toe_r + 0.008)) / (-wall.outward[0])
    swing_fold_from = {}
    base_seg_start = d.qpos[0:3].copy()
    # The pivot HOLDS the station the reach ended on (plus the lateral
    # bias over the left support side): the high support arm keeps its
    # handoff geometry all the way to the closure, which does the rising.
    # 12 mm off the reach station (6 mm grazed the base_link by 9 mm once
    # the tiptoe stood the body taller).
    base_seg_end = base_seg_start + pivot_out * wall.outward
    base_seg_end[1] = lateral_weight_shift
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    swing_leg = _leg_joints("right")
    fold_u0 = 0.45  # guided fold + stiff leg posture start (donor value)
    for t in range(n[seg_name]):
      u = t / (n[seg_name] - 1)
      s = float(gcc.smootherstep(u))
      s_rot = float(gcc.smootherstep(min(u / 0.9, 1.0)))
      on_ground[swing_ee] = u < 0.62
      tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in wall_welds]
      fade = float(np.clip((0.66 - u) / 0.17, 0.0, 1.0))
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
      tasks.append(lf_toe_hold())
      relax = 1.0 - 0.85 * float(
        gcc.smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0)) - gcc.smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
      )
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(t, n[seg_name], p0, p1, p2, n_rake_seg=seg_rake[seg_name]),
          rot=gcc.slerp_matrix(rot0, rot2, s_rot),
          w_pos=np.array([1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]),
          w_rot=100.0 * relax,
          track=relax > 0.9,
        )
      )
      base_target = (1.0 - s) * base_seg_start + s * base_seg_end
      press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
      if wall.identity:
        base_target = base_target + np.array([swing_press * press, 0.03 * press, 0.0])
      else:
        base_target = base_target - swing_press * press * wall.outward + np.array([0.0, 0.03 * press, 0.0])
      tasks.append(
        gcc.base_task(
          ik,
          base_target,
          (100.0, 40.0, 60.0),
          rot=np.eye(3) if wall.identity else wall.R,
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
      plant_toes=(rf,),
      toe_hold_targets={lf: toe_wall_hold[lf]},
      extra_overrides={"waist_yaw": 6.0},
    )
    toe_wall_hold[rf] = d.site_xpos[ik.toe_sid[rf]].copy()

    # hand_up_left: the mount's ONLY hand step (the right hand already sits
    # at its final rung) — the closure swing verbatim, sides swapped: toe
    # holds release first (fade^2, done by u=0.55), the guided ankle fold
    # walks both feet to the state-1 hang, the base squares up, and the
    # shared closure blend lands the handoff frame exactly. The deepest
    # move rides the strongest support of the whole mount: RH final weld +
    # both foot welds.
    seg_name, swing_ee, support_hand = "hand_up_left", lh, rh
    for ee in ik.ee_names:
      if ee != swing_ee:
        anchor_now[ee] = ik.pad_pose(ee)[0].copy()
    weld_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee}
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    anchor_now[swing_ee] = p0.copy()
    if wall.identity:
      swing_ctl_off = np.array([-swing_bulge, 0.0, 0.0])
    else:
      swing_ctl_off = swing_bulge * wall.outward
    p1 = 0.5 * (p0 + p2) + swing_ctl_off
    engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
    base_seg_start = d.qpos[0:3].copy()
    base_rot_start = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_start, d.qpos[3:7])
    base_rot_start = base_rot_start.reshape(3, 3)
    base_seg_end = final_qpos[0:3].copy()
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    support_arm = _arm_joints("right")
    swing_arm = _arm_joints("left")
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
      toe_w = float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2
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
      # Base PINNED through the step (200/100/150 vs the donor's 60/30/50)
      # and the support arm held stiff (override 4): from a low first rung
      # the 20 cm left-hand step bulges out through the wall plane and the
      # soft base followed it 2 cm away from the wall, hyper-extending the
      # welded right arm onto its shoulder stop (pitch -3.14) and tearing
      # that weld 33 mm. The hand_up rule from the 08-20 post-mortem: the
      # base must be nailed inside a hand step.
      tasks.append(
        gcc.base_task(
          ik,
          base_target,
          (200.0, 100.0, 150.0),
          rot=gcc.slerp_matrix(base_rot_start, np.eye(3) if wall.identity else wall.R, s),
          w_rot=60.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in swing_arm:
          target = state1_joint[nm]
        elif nm in support_arm or nm == "waist_yaw":
          target = posture_start[nm]
        else:
          target = state1_joint[nm]
        posture_t[nm] = (1.0 - s) * posture_start[nm] + s * target
      overrides = {nm: 4.0 for nm in support_arm}
      if u >= 0.42:
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
      push(seg_name, engaged)
    landed = ik.pad_pose(swing_ee)
    anchor_now[swing_ee] = landed[0].copy()
    landing_report[swing_ee] = (
      float(np.linalg.norm(landed[0] - p2)),
      gcc.rotation_angle(landed[1], state1_pad_rot[swing_ee]),
    )
  elif sequence == "cross":
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
    lhlf_on = {lh: True, rh: False, rf: True, lf: False}
    hands_lf_on = {lh: True, rh: True, rf: True, lf: False}
    # The RIGHT hand never visits the low rung: re-seed its anchor lane at
    # the FINAL rung (the shared init aims hands at the low ladder; the lane
    # feeds ee_anchor_positions and the runtime press/tow rewards).
    anchor_now[rh] = final_anchor_pos[rh].copy()
    mid_anchor_rh = None
    if hand_z_mid is not None:
      # v4: the right hand lands an INTERMEDIATE rung reachable flat-footed
      # (probe: the final 1.52 rung is 8-27 mm out of a flat-footed reach and
      # the tiptoe that bought it is passively untenable on 50/5 ankles);
      # it walks up to the final rung in hand_up_right after the pivot.
      if not (hand_z_low < hand_z_mid < hand_z):
        raise ValueError("--hand-z-mid must lie between --hand-z-low and --hand-z")
      up_vec = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
      mid_anchor_rh = final_anchor_pos[rh] + (hand_z_mid - hand_z) * up_vec + np.array([0.0, mid_rung_y, 0.0])
      anchor_now[rh] = mid_anchor_rh.copy()
    if rh_first:
      # v5 hands-first order: the right hand's first rung is --hand-z-mid when
      # given (a HIGHER rung still reachable flat-footed: RF1's shoulder-height
      # right hand left the pivot statically infeasible, RF util 4.2), else the
      # mirror of the left hand's low rung; the closure is the two-step walk-up.
      if mid_anchor_rh is None:
        mid_anchor_rh = low_anchor_pos[rh] + np.array([0.0, -low_rung_y, 0.0])
      anchor_now[rh] = mid_anchor_rh.copy()
      reach_heel_lift = 0.0  # no tiptoe reach in this order
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
        hard=weld_hard,
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
    if raise_yaw != 0.0:
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
    if wall.identity:
      place_ctl_off = np.array([-place_bulge, 0.0, 0.03])
    else:
      place_ctl_off = place_bulge * wall.outward + 0.03 * wall.up
    lh_start = ik.pad_pose(lh)
    p0_l = lh_start[0]
    p2_l = low_anchor_pos[lh]
    p1_l = 0.5 * (p0_l + p2_l) + place_ctl_off
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
    if transfer_lateral != 0.0 or transfer_press != 0.0:
      transfer_shift = np.array([0.0, transfer_lateral, 0.0])
      if wall.identity:
        transfer_shift = transfer_shift + np.array([transfer_press, 0.0, 0.0])
      else:
        transfer_shift = transfer_shift - transfer_press * wall.outward
      transfer_w = (300.0, 300.0, 100.0)
    # v6: ``set_left_pitch`` leans the trunk away from the wall (world +y pitch key)
    # while only the left hand is welded, so the right hand lands into an
    # already-hanging posture (its set dwell is otherwise pushed up the wall
    # until the lean completes: -25 N at the dwell start, -9 N once leaned).
    rot_key_sl = None
    if set_left_pitch != 0.0:
      cp_, sp_ = np.cos(set_left_pitch), np.sin(set_left_pitch)
      rot_key_sl = np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]])
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

    if rh_first:
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
      if not wall.identity:
        trunk_fwd_pitch -= float(np.arcsin(np.clip(wall.R[0, 2], -1.0, 1.0))) if hasattr(wall, "R") else 0.0
      raise_pitch_end_r = (
        raise_pitch_end if raise_pitch_end_right is None else raise_pitch_end_right
      ) + raise_pitch_comp_sign * trunk_fwd_pitch
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
          (0.35, out_sign_r * (raise_roll_right if rh_raise_lateral else raise_roll)),
          (0.80, out_sign_r * (raise_roll_right if rh_raise_lateral else raise_roll)),
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
      if raise_yaw != 0.0:
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
        if place_right_pitch != 0.0:
          cp_, sp_ = np.cos(place_right_pitch), np.sin(place_right_pitch)
          rot_pl_t = gcc.slerp_matrix(
            base_rot_rl, np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]]) @ base_rot_rl, s
          )
        base_pl_t = base_hold_rl
        if place_right_base_dz != 0.0 or place_right_base_dx != 0.0:
          # v6: reach relief for a high right rung -- the base rises /
          # approaches the wall over the arc instead of the IK dragging
          # the welded left hand off its anchor.
          if wall.identity:
            base_pl_t = base_hold_rl + s * np.array([place_right_base_dx, 0.0, place_right_base_dz])
          else:
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
      if set_right_press != 0.0 or set_right_drop != 0.0 or set_right_lateral != 0.0:
        up_vec_sr = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
        set_shift = -set_right_press * wall.outward - set_right_drop * up_vec_sr
        if wall.identity:
          set_shift = np.array([set_right_press, 0.0, -set_right_drop])
        set_shift = set_shift + np.array([0.0, set_right_lateral, 0.0])
        set_w = (300.0, 300.0, 100.0)
      # v6: ``set_right_pitch`` leans the trunk (world pitch about +y; negative =
      # away from the wall) so the high right hand is loaded in ARM TENSION
      # (passive gate: standing tall the trunk bows onto the straight arm and
      # pushes the pad up the wall, -69 N at 1.46; hanging off it holds +74 N).
      rot_key_sr = None
      if set_right_pitch != 0.0:
        cp_, sp_ = np.cos(set_right_pitch), np.sin(set_right_pitch)
        rot_key_sr = np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]])
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
    if wall.identity:
      mount_ctl_off = np.array([-foot_bulge, 0.0, foot_lift])
    else:
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
    if wall.identity:
      base_seg_end = base_seg_start + np.array([step_hips_in, step_hips_lateral, step_rise])
    else:
      base_seg_end = base_seg_start - step_hips_in * wall.outward + step_rise * wall.up
      base_seg_end[1] = base_seg_start[1] + step_hips_lateral
    base_rot_lf_end = np.eye(3) if wall.identity else wall.R
    if step_trunk_pitch != 0.0:
      cp, sp = np.cos(step_trunk_pitch), np.sin(step_trunk_pitch)
      base_rot_lf_end = base_rot_lf_end @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
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
      if rh_first:
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
      if wall.identity:
        base_target_step = base_target_step + np.array([step_press * press, 0.0, 0.0])
      else:
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
      push("right_foot_mount", hands_on if rh_first else lh_on)
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
      hands_lf_on if rh_first else lhlf_on,
      weld_ees=(lh, rh, rf) if rh_first else (lh, rf),
      hold_feet=(lf,),
      base_shift=np.array([0.0, switch_lateral, 0.0]),
      plant_toes=(rf,),
      extra_overrides={"waist_yaw": 6.0},
    )
    toe_wall_hold[rf] = d.site_xpos[ik.toe_sid[rf]].copy()

    def rf_toe_hold():
      return gcc.toe_task(ik, rf, toe_wall_hold[rf], w_pos=(600.0, 3.0, 10.0))

    if not rh_first:
      # hand_place_right: the RIGHT arm rises AND lands in ONE sagittal arc,
      # DIRECTLY on its FINAL rung — no low hang, no later step. The raise is
      # folded into this segment on purpose: a second disjoint hand_raise
      # label window would stretch every label-resolved RSI window across the
      # foot mounts (resolve_segment_window spans first..last hit). Support:
      # LH weld + LF weld + the flat right foot + the planted left toe — the
      # strongest stance before the closure carries the longest reach. The
      # base RISES toward the final station with the reach (foot-first
      # lesson: landing the final rung from the compressed hang put the
      # shoulder too low and welded the arm onto its second branch).
      lh_weld = ik.pad_pose(lh)
      rf_weld = ik.pad_pose(rf)
      anchor_now[lh] = lh_weld[0].copy()
      anchor_now[rf] = rf_weld[0].copy()
      base_hold_r = d.qpos[0:3].copy()
      base_rot_r = np.zeros(9)
      mujoco.mju_quat2Mat(base_rot_r, d.qpos[3:7])
      base_rot_r = base_rot_r.reshape(3, 3)
      p2_r = final_anchor_pos[rh] if mid_anchor_rh is None else mid_anchor_rh
      reach_y = -lateral_weight_shift if reach_lateral is None else reach_lateral
      if wall.identity:
        base_pr_end = np.array([final_qpos[0] - 0.002, reach_y, final_qpos[2] - 0.012])
      else:
        base_pr_end = final_qpos[0:3] + 0.002 * wall.outward - 0.012 * wall.up
        base_pr_end[1] = reach_y
      # v4 (2026-08-22): cap the along-wall RISE of the reach station. The
      # v3 station rose 7.6 cm on tiptoe with the right foot pad already
      # welded: the extending right leg pushed its toe pad UP the wall
      # (reverse shear, 128/128 in the passive-hold probe) and the left hand
      # over its ultimate shear. Reach from the hold height instead; the
      # pivot / closure take the body up later, as a cycle hand step does.
      if reach_rise is not None:
        up_vec = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
        rise = float(wall.along_wall(base_pr_end) - wall.along_wall(base_hold_r))
        if rise > reach_rise:
          base_pr_end = base_pr_end - (rise - reach_rise) * up_vec
      # Phase A — keyed sagittal raise with the body FROZEN (foot-first's
      # braced-raise pattern: base, legs and the left arm pinned frame to
      # frame, so the welds cannot move and no IK is needed). Reaching for
      # the anchor straight off the hip capsized on first try: forcing the
      # state-1 palm while the pad is still low twists the arm onto the
      # reverse-wrap branch (654 mm / 174 deg landing, shoulder at its stop)
      # — the exact "early capture" pathology the donors stage around with a
      # cocked pose. So: raise first, reach second, both inside this label.
      elbow_fold_r = max(float(ik.joint_ranges["right_elbow_pitch"][0]) + 0.10, -2.2)
      out_sign_r = 1.0 if ik.joint_ranges["right_shoulder_roll"][1] > 1.5 else -1.0
      n_pr = n["hand_place_right"]
      n_raise_r = max(int(round(0.48 * n_pr)), 2)
      if n_pr - n_raise_r - n_rake < 2:
        raise ValueError("--place-right-s leaves fewer than two arc frames")
      raise_base_r = d.qpos.copy()
      cur_r = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in _arm_joints("right")}
      # The raise is keyed in WORLD elevation, not joint space: the trunk
      # arrives bowed toward the slab (the hand place leaned ~18 deg into
      # its reach and the welded low hand holds that bow), and a joint-space
      # -2.35 on a bowed trunk points the forearm at the wall (elbow 134 mm
      # through it, measured). Shift the shoulder-pitch keys by the native
      # trunk pitch so the arm sweeps the same world arc as on an upright
      # trunk.
      _qw, _qx, _qy, _qz = (float(v) for v in d.qpos[3:7])
      trunk_pitch = float(np.arcsin(np.clip(2.0 * (_qw * _qy - _qz * _qx), -1.0, 1.0)))
      # The wall-parallel attitude (native pitch 90 - a) is the reference
      # the raise was validated on; only the bow PAST it shifts the keys.
      trunk_bow = max(trunk_pitch - np.radians(90.0 - wall.angle_deg), 0.0)
      cocked_pitch_r = -2.35 + trunk_bow
      rh_keys = {
        "right_shoulder_pitch": (
          (0.0, cur_r["right_shoulder_pitch"]),
          (0.25, cur_r["right_shoulder_pitch"]),
          (0.88, cocked_pitch_r),
          (1.0, cocked_pitch_r),
        ),
        # LATERAL for this arm (--raise-roll-right, 1.3 rad at 80): the
        # trunk is still bowed toward the slab and the shoulder sits ~15 cm
        # off the surface, where any sagittal sweep puts the forearm through
        # the wall (130-200 mm measured); the elbow sweeps through +y where
        # there is no wall, then the arm comes inboard overhead.
        "right_shoulder_roll": (
          (0.0, cur_r["right_shoulder_roll"]),
          (0.35, out_sign_r * raise_roll_right),
          (0.80, out_sign_r * raise_roll_right),
          (1.0, s1["right_shoulder_roll"]),
        ),
        "right_shoulder_yaw": (
          (0.0, cur_r["right_shoulder_yaw"]),
          (1.0, s1["right_shoulder_yaw"]),
        ),
        "right_elbow_pitch": (
          (0.0, cur_r["right_elbow_pitch"]),
          (0.10, cur_r["right_elbow_pitch"]),
          (0.42, elbow_fold_r),
          (0.80, elbow_fold_r),
          (1.0, -1.15),
        ),
      }
      for t in range(n_raise_r):
        s = t / max(n_raise_r - 1, 1)
        d.qpos[:] = raise_base_r
        for nm, keys in rh_keys.items():
          d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s)
        mujoco.mj_forward(m, d)
        push("hand_place_right", lhlf_on)
      # Phase B — the braced arc from the cocked staging to the FINAL rung:
      # donor machinery verbatim (CONSTANT state-1 palm — from the cocked
      # pose the palm sets in free space and the arm rides the state-1
      # branch the whole way; stiff posture pin against the second-branch
      # family).
      rh_cocked = ik.pad_pose(rh)
      p0_r = rh_cocked[0]
      p1_r = 0.5 * (p0_r + p2_r) + place_ctl_off
      n_arc_seg = n_pr - n_raise_r
      posture_hold_b = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
      lf_toe0 = d.site_xpos[ik.toe_sid[lf]].copy()
      lf_ankles = ankle_joints[lf]
      lf_ankle_pitch = next(nm for nm in lf_ankles if nm.endswith("_pitch"))
      for t in range(n_arc_seg):
        s = float(gcc.smootherstep(t / (n_arc_seg - 1)))
        n_arc_r = n_arc_seg - n_rake
        s_arc = float(gcc.smootherstep(min(t / (n_arc_r - 1), 1.0)))
        w_fade = min(s_arc / 0.3, 1.0)
        tasks = [
          gcc.weld_task(ik, lh, lh_weld),
          gcc.weld_task(ik, rf, rf_weld),
          rf_toe_hold(),
        ]
        # The trailing foot TIPTOES through the reach — the human press-up.
        # The flat 1000/500 pin lets go at once (s 0-0.12) and a KEYED heel
        # lift takes over (ankle plantarflexion ramped over s 0.05-0.45, the
        # toe pinned to its floor spot): releasing a fading hold instead of
        # leading the heel stored the reach in the arm (shoulder -2.9 rad
        # against the flat foot) and the body popped 50 mm in one frame the
        # instant the hold weight ran out (9-15 rad/s ankle/shoulder steps,
        # at every fade window tried).
        heel_s = float(gcc.smootherstep(np.clip((s - reach_heel_start) / 0.40, 0.0, 1.0)))
        if reach_heel_lift <= 0.0:
          heel_s = 0.0  # v4: no tiptoe — the flat hold stays at full weight
        # The flat hold is never RELEASED — its target ROTATES about the
        # planted toe with the heel key (R_y(+theta): heel up), and only
        # its weight eases (1000/500 -> 300/150) once the heel is up so the
        # reach can fine-tune. Fading the weight to zero at any window was a
        # threshold: the foot popped the instant the hold lost to the reach
        # (15-22 rad/s knee steps).
        pos_h, rot_h, sid_h = foot_hold_pose[lf]
        heel_th = reach_heel_lift * heel_s
        heel_R = np.array(
          [
            [np.cos(heel_th), 0.0, np.sin(heel_th)],
            [0.0, 1.0, 0.0],
            [-np.sin(heel_th), 0.0, np.cos(heel_th)],
          ]
        )
        # The hold pivots about the toe pin's CURRENT target (it settles 4 mm
        # as the heel comes up) and eases to 15%: pivoted about the standing
        # toe spot at 30% it out-voted the toe pin and the sole toe point
        # rode up to 9 mm — past the audit's 8 mm gate (8 frames with no
        # floor support counted).
        # ...and at 15% it still lifted the toe site 4.6 mm off its pin
        # (sole toe point 8.8 mm): the hold LEADS the heel and hands over
        # completely (weight -> 0 with the key), the toe pin settles 6 mm
        # (the sphere still clears the floor by 4 mm).
        hold_w = 1.0 - heel_s
        toe_pivot = lf_toe0 - np.array([0.0, 0.0, 0.006 * heel_s])
        if hold_w > 0.01:
          tasks.append(
            gcc.IKTask(
              kind="site",
              sid=sid_h,
              pos=toe_pivot + heel_R @ (pos_h - lf_toe0),
              rot=heel_R @ rot_h,
              w_pos=np.full(3, 1000.0 * hold_w),
              w_rot=500.0 * hold_w,
            )
          )
        # The toe stays PLANTED (x/y pinned like a real toe under friction —
        # the pivot's free glide let it slide 7 cm into the wall base here)
        # and settles 4 mm as the heel comes up: the audit's sole toe point
        # rides the 8 mm gate at the standing toe height (7.5 mm), so the
        # planted toe must buy its margin by sitting lower once the heel no
        # longer defines the sole plane.
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.toe_sid[lf],
            pos=toe_pivot,
            rot=None,
            w_pos=np.array([600.0, 50.0, 2000.0]),
            w_rot=0.0,
            track=False,
          )
        )
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.pad_sid[rh],
            pos=_swing_target(t, n_arc_seg, p0_r, p1_r, p2_r),
            # Palm converges EARLY but GRADUALLY: from the cocked overhead
            # staging there is no reverse-wrap branch to fall into, and the
            # ramp keeps the throttle release from dumping the full 500-
            # weight orientation gap into the body in one step (v2: that
            # yank ran left_hip_roll onto its stop and dragged the welds
            # 157 mm).
            # Palm completes by s=0.35 — BEFORE the lean-in deepens. The
            # donor's warning is load-bearing: transition orientations in the
            # leaned geometry recruit the waist and detonate the arm (0.55
            # measured: waist snap 2.0 rad, welded left hand dragged through
            # the wall).
            rot=gcc.slerp_matrix(rh_cocked[1], state1_pad_rot[rh], min(s / 0.35, 1.0)),
            w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
            w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
          )
        )
        # The base height is a SUGGESTION here (z weight 40, not the donors'
        # 100): the station sits at the edge of what a tiptoed leg reaches
        # (knee ran to its -0.045 stop), and a 100-weight height pull either
        # held the body down until the heel popped (s clock) or snapped the
        # knee 0.3 rad the instant the flat hold let go (heel clock). The
        # keyed heel and the reach negotiate the height; the later segments
        # HOLD wherever the reach ended (no yo-yo), and the closure step
        # rises to the handoff station as a cycle hand step does.
        tasks.append(
          gcc.base_task(
            ik,
            (1.0 - s) * base_hold_r + s * base_pr_end,
            (60.0, 60.0, reach_base_z_w),
            # v4: the trunk must actually reach wall-parallel (v3: w_rot 8 lost
            # ~5 deg to the welds + reach + tiptoe = the lean-back).
            rot=gcc.slerp_matrix(
              base_rot_r,
              np.eye(3) if wall.identity else wall.R,
              reach_rot_frac * s,
            ),
            w_rot=reach_rot_w,
          )
        )
        posture_t = {}
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
          if nm in _arm_joints("right"):
            posture_t[nm] = (1.0 - s) * posture_hold_b[nm] + s * state1_joint[nm]
          elif nm == "waist_yaw":
            posture_t[nm] = (1.0 - s) * posture_hold_b[nm]
          else:
            posture_t[nm] = posture_hold_b[nm]
        # Keyed heel lift: +0.45 rad plantarflexion (the solved tiptoe sat at
        # +0.49 over the standing -0.295) led by a stiff posture anchor while
        # the ramp runs, then handed to the reach.
        posture_t[lf_ankle_pitch] = posture_hold_b[lf_ankle_pitch] + reach_heel_lift * heel_s
        # Branch pin (foot-first final-rung lesson): stiff posture anchor on
        # the reaching arm + the waist keeps the landing on the state-1
        # branch the handoff requires.
        ov_pr = {nm: reach_arm_pin for nm in _arm_joints("right")}
        ov_pr["waist_yaw"] = 6.0
        for nm in lf_ankles:
          # The roll stays braced; the pitch follows the heel key.
          ov_pr[nm] = 4.0 if nm.endswith("_roll") else (6.0 if heel_s < 1.0 else 1.0)
        # The tiptoed leg stays in its sagittal plane: left free, the heel
        # swung 20 cm outboard (a duck-footed tiptoe) when the height got
        # tight.
        ov_pr["left_hip_yaw"] = 3.0
        ov_pr["left_hip_roll"] = 3.0
        throttled = t < 8
        res = ik.solve_frame(
          tasks,
          sel_all,
          posture_t,
          posture_weight=0.4,
          posture_weight_overrides=ov_pr,
          iters=2 if throttled else 100,
          max_step=0.003 if throttled else 0.12,
        )
        seg_residual["hand_place_right"].merge(res)
        _note_base("hand_place_right", (1.0 - s) * base_hold_r + s * base_pr_end)
        push("hand_place_right", lhlf_on)
      landed_r = ik.pad_pose(rh)
      landing_report[f"{rh}@mid" if mid_anchor_rh is not None else f"{rh}@final"] = (
        float(np.linalg.norm(landed_r[0] - p2_r)),
        gcc.rotation_angle(landed_r[1], state1_pad_rot[rh]),
      )
    else:
      lf_toe0 = d.site_xpos[ik.toe_sid[lf]].copy()
    lf_floor_toe = lf_toe0 - np.array([0.0, 0.0, 0.006])

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
    if set_hold_press != 0.0 or set_hold_rise != 0.0:
      up_vec_sh = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
      sh_shift = -set_hold_press * wall.outward + set_hold_rise * up_vec_sh
      if wall.identity:
        sh_shift = np.array([set_hold_press, 0.0, set_hold_rise])
      sh_w = (300.0, 300.0, 300.0)
    rot_key_sh = None
    if set_hold_pitch != 0.0:
      cp_, sp_ = np.cos(set_hold_pitch), np.sin(set_hold_pitch)
      rot_key_sh = np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]])
    if reach_heel_lift <= 0.0:
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
    else:
      dwell(
        "hand_set_hold",
        hands_lf_on,
        weld_ees=(rh, lh, rf),
        hold_feet=(),
        floor_toe_targets={lf: lf_floor_toe},
        toe_hold_targets={rf: toe_wall_hold[rf]},
        extra_overrides={"waist_yaw": 6.0},
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
    if wall.identity:
      pivot_ctl_off = np.array([-pivot_ctl_out, 0.0, pivot_ctl_up])
    else:
      pivot_ctl_off = pivot_ctl_out * wall.outward + np.array([0.0, 0.0, pivot_ctl_up])
    p1 = 0.5 * (p0 + p2) + pivot_ctl_off
    # The planted-toe station of the press-up, not the entry pose: the toe
    # has been sitting there since the right-hand reach.
    toe_hold = d.site_xpos[ik.toe_sid[lf]].copy() if reach_heel_lift <= 0.0 else lf_floor_toe.copy()
    # Smear terminus: same height and lateral spot, toe sphere 8 mm short
    # of the (tilted) wall surface.
    toe_r = ik.toes[swing_ee].radius
    toe_smear = toe_hold.copy()
    # v5 (2026-08-22): ``pivot_smear_gap`` = how far short of the wall base the toe
    # smear stops (v3: 8 mm). In the hands-first order the hips sit 5-10 cm
    # further out at the pivot, so the v3 smear drags the toe 16-20 cm ahead of
    # the hip, the swing leg runs straight and flips branches at 15 rad/s.
    if wall.identity:
      toe_smear[0] = gcc.WALL_SURFACE_X - toe_r - pivot_smear_gap
    else:
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
    up_vec_pv = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
    base_seg_end = base_seg_start + pivot_out * wall.outward + pivot_rise * up_vec_pv
    base_seg_end[1] = -lateral_weight_shift if pivot_lateral is None else pivot_lateral
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
        if pivot_heel_lift is not None:
          # v5 HEEL-UP start: the reach block's keyed heel (R_y(+theta), heel up,
          # pivoting about the planted toe, the hold handing over to the key as
          # the heel comes up) replayed over the first 30 % of the pivot, so the
          # flat-footed hands-first order enters the fold from v3's tiptoe stance
          # instead of rolling the foot onto its toe with a straight leg.
          heel_s_pv = float(gcc.smootherstep(np.clip(u / 0.30, 0.0, 1.0)))
          hold_w_pv = 1.0 - heel_s_pv
          if hold_w_pv > 0.01:
            th_pv = (
              float(pivot_heel_lift - posture_start[next(nm for nm in swing_leg if nm.endswith("ankle_pitch"))])
              * heel_s_pv
            )
            heel_R_pv = np.array(
              [[np.cos(th_pv), 0.0, np.sin(th_pv)], [0.0, 1.0, 0.0], [-np.sin(th_pv), 0.0, np.cos(th_pv)]]
            )
            pos_hp, rot_hp, sid_hp = foot_hold_pose[swing_ee]
            toe_pivot_pv = toe_hold - np.array([0.0, 0.0, 0.006 * heel_s_pv])
            tasks.append(
              gcc.IKTask(
                kind="site",
                sid=sid_hp,
                pos=toe_pivot_pv + heel_R_pv @ (pos_hp - toe_hold),
                rot=heel_R_pv @ rot_hp,
                w_pos=np.full(3, 1000.0 * hold_w_pv),
                w_rot=500.0 * hold_w_pv,
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
      if wall.identity:
        base_target = base_target + np.array([swing_press * press, -0.03 * press, 0.0])
      else:
        base_target = base_target - swing_press * press * wall.outward + np.array([0.0, -0.03 * press, 0.0])
      tasks.append(
        gcc.base_task(
          ik,
          base_target,
          (100.0, 40.0, 60.0),
          rot=np.eye(3) if wall.identity else wall.R,
          w_rot=10.0,
        )
      )
      posture_t = {}
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
        if nm in swing_leg:
          posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * state1_joint[nm]
        else:
          posture_t[nm] = posture_start[nm]
      if pivot_heel_lift is not None and u < fold_u0:
        # v5 (2026-08-22): HEEL-UP start. The hands-first order enters the pivot
        # on a flat left foot (ankle -18 deg) 7 cm lower than v3's tiptoe-reach
        # station; the toe smear then has to roll the foot onto its toe while
        # the leg runs straight, and the hip flips 70 deg in 4 frames (15 rad/s,
        # a visible jump). Key the swing ankle to v3's tiptoe plantarflexion
        # over the first 30 % of the pivot so the fold starts from a bent leg.
        hl = float(gcc.smootherstep(np.clip(u / 0.30, 0.0, 1.0)))
        nm_ap = next(nm for nm in swing_leg if nm.endswith("ankle_pitch"))
        posture_t[nm_ap] = posture_start[nm_ap] + hl * (pivot_heel_lift - posture_start[nm_ap])
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
          overrides[nm] = 1.0 if pivot_heel_lift is None else 4.0
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
      if wall.identity:
        swing_ctl_off = np.array([-swing_bulge, 0.0, 0.0])
      else:
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
            rot=gcc.slerp_matrix(base_rot_start, np.eye(3) if wall.identity else wall.R, s),
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
      up_vec = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
      out_vec = np.array([-1.0, 0.0, 0.0]) if wall.identity else wall.outward
      base_lh = final_qpos[0:3] - 0.012 * up_vec + 0.002 * out_vec
      base_lh[1] = float(d.qpos[1])
      if close_rh_first:
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
        if hand_up_left_mid is not None:
          # v6g: the 30 cm left walk-up splits at a mid rung -- each landing
          # keeps elbow flexion in reserve instead of arriving at the reach
          # frontier (closing-school plateau: supports held, the swing hand
          # missed the far landing on ~half its edges).
          up_dir = np.array([0.0, 0.0, 1.0]) if wall.identity else wall.up
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
      else:
        cross_hand_up("hand_up_left", lh, rh, base_lh, closing=False)
        dwell(
          "hand_up_hold",
          all_on,
          weld_ees=ik.ee_names,
          hold_feet=(),
          toe_hold_targets=toe_wall_hold,
        )
        cross_hand_up("hand_up_right", rh, lh, final_qpos[0:3].copy(), closing=True)

  else:
    # ===========================================================================
    # ground_stand: the walk-delivery keyframe, verbatim.
    set_standing()
    for _ in range(n["ground_stand"]):
      push("ground_stand", none_on)

    # hand_raise: joint-space lateral windmill (no IK; legs and base frozen).
    # Waypoint at s=0.45: arm out sideways and half-raised, elbow folding; the
    # elbow tracks through +-y, never toward the wall.
    # Strict sequencing keeps the elbow out of the wall: the roll carries the
    # arm fully sideways FIRST and holds there while the pitch sweeps overhead
    # (the elbow then tracks through +-y, radius cos(roll) toward the wall),
    # and only returns inboard once the arm is high.
    raise_keys: dict[str, tuple[tuple[float, float], ...]] = {}
    for side, ee in (("left", lh), ("right", rh)):
      roll_range = ik.joint_ranges[f"{side}_shoulder_roll"]
      out_sign = 1.0 if roll_range[1] > 1.5 else -1.0
      s1 = state1_joint
      st = standing_joint
      # roll 1.6 is enough: the elbow's forward protrusion scales ~cos(roll)
      # and vanishes there; the fold waits until the pitch sweep is done so
      # the forearm never points at the wall.
      raise_keys[f"{side}_shoulder_roll"] = (
        (0.0, st[f"{side}_shoulder_roll"]),
        (0.30, out_sign * 1.6),
        (0.70, out_sign * 1.6),
        (1.0, s1[f"{side}_shoulder_roll"]),
      )
      raise_keys[f"{side}_shoulder_pitch"] = (
        (0.0, st[f"{side}_shoulder_pitch"]),
        (0.35, st[f"{side}_shoulder_pitch"]),
        (0.85, -2.35),
        (1.0, -2.35),
      )
      raise_keys[f"{side}_shoulder_yaw"] = (
        (0.0, st[f"{side}_shoulder_yaw"]),
        (1.0, s1[f"{side}_shoulder_yaw"]),
      )
      raise_keys[f"{side}_elbow_pitch"] = (
        (0.0, st[f"{side}_elbow_pitch"]),
        (0.55, st[f"{side}_elbow_pitch"]),
        (0.90, -1.15),
        (1.0, -1.15),
      )
    for t in range(n["hand_raise"]):
      s = t / max(n["hand_raise"] - 1, 1)
      d.qpos[:] = standing_qpos
      for nm, keys in raise_keys.items():
        d.qpos[ik.qpos_addresses[nm]] = _key_blend(keys, s)
      mujoco.mj_forward(m, d)
      push("hand_raise", none_on)

    # hand_place: both pads arc together, but only the LEFT lands and rakes —
    # the RIGHT parks at a hover point ``hand_hover`` short of its anchor.
    # Staggered latch (2026-08-15): the both-hands-same-frame engage demanded
    # two simultaneous unbraced first contacts; one needle at a time, and the
    # second press (hand_place_right) is braced by the first weld.
    place_start = {ee: ik.pad_pose(ee) for ee in hands}
    if wall.identity:
      hover_vec = np.array([-float(hand_hover), 0.0, 0.0])
    else:
      hover_vec = float(hand_hover) * wall.outward
    place_end = {lh: low_anchor_pos[lh], rh: low_anchor_pos[rh] + hover_vec}
    base_start = d.qpos[0:3].copy()
    base_place_end = base_start + np.array([place_lean, 0.0, 0.0])
    posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    arm_target = {nm: state1_joint[nm] for side in ("left", "right") for nm in _arm_joints(side)}
    if wall.identity:
      place_ctl_off = np.array([-place_bulge, 0.0, 0.03])
    else:
      # Bulge away from the tilted surface, lift along the wall.
      place_ctl_off = place_bulge * wall.outward + 0.03 * wall.up
    p_ctl = {}
    for ee in hands:
      p0 = place_start[ee][0]
      p2 = place_end[ee]
      p_ctl[ee] = (p0, 0.5 * (p0 + p2) + place_ctl_off, p2)
    for t in range(n["hand_place"]):
      s = float(gcc.smootherstep(t / (n["hand_place"] - 1)))
      n_arc = n["hand_place"] - n_rake
      s_arc = float(gcc.smootherstep(min(t / (n_arc - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [foot_hold_task(ee) for ee in feet]
      for ee in hands:
        p0, p1, p2 = p_ctl[ee]
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.pad_sid[ee],
            pos=_swing_target(t, n["hand_place"], p0, p1, p2),
            rot=gcc.slerp_matrix(place_start[ee][1], state1_pad_rot[ee], s),
            w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
            w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
          )
        )
      tasks.append(
        gcc.base_task(
          ik,
          (1.0 - s) * base_start + s * base_place_end,
          (60.0, 60.0, 100.0),
          rot=np.eye(3),
          w_rot=10.0,
        )
      )
      posture_t = {
        nm: (1.0 - s) * posture_start[nm] + s * arm_target.get(nm, standing_joint[nm])
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES
      }
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_t,
        posture_weight=0.6,
        posture_weight_overrides={nm: 1.2 for side in ("left", "right") for nm in _arm_joints(side)},
      )
      seg_residual["hand_place"].merge(res)
      push("hand_place", none_on)
    landed = ik.pad_pose(lh)
    landing_report[f"{lh}@low"] = (
      float(np.linalg.norm(landed[0] - low_anchor_pos[lh])),
      gcc.rotation_angle(landed[1], state1_pad_rot[lh]),
    )

    # hand_set_left: the LEFT hand engages and holds; the right hovers.
    set_left_on = {lh: True, rh: False, lf: False, rf: False}
    dwell(
      "hand_set_left",
      set_left_on,
      weld_ees=(lh,),
      hold_feet=feet,
      hold_pads=(rh,),
    )

    # hand_place_right: the RIGHT pad closes hover -> anchor + rake, braced
    # by the welded left hand and both flat feet — the cycle's regrip
    # condition, imported: the second first-contact is never unbraced.
    rh_start = ik.pad_pose(rh)
    lh_brace = ik.pad_pose(lh)
    anchor_now[lh] = lh_brace[0].copy()
    base_hold_r = d.qpos[0:3].copy()
    base_rot_r = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot_r, d.qpos[3:7])
    base_rot_r = base_rot_r.reshape(3, 3)
    posture_hold_r = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
    p0_r = rh_start[0]
    p2_r = low_anchor_pos[rh]
    p1_r = 0.5 * (p0_r + p2_r) + 0.3 * place_ctl_off
    for t in range(n["hand_place_right"]):
      s = float(gcc.smootherstep(t / (n["hand_place_right"] - 1)))
      tasks = [gcc.weld_task(ik, lh, lh_brace)]
      tasks.extend(foot_hold_task(ee) for ee in feet)
      tasks.append(
        gcc.IKTask(
          kind="site",
          sid=ik.pad_sid[rh],
          pos=_swing_target(t, n["hand_place_right"], p0_r, p1_r, p2_r),
          rot=gcc.slerp_matrix(rh_start[1], state1_pad_rot[rh], s),
          w_pos=np.full(3, 1500.0),
          w_rot=300.0,
        )
      )
      tasks.append(gcc.base_task(ik, base_hold_r, (60.0, 60.0, 100.0), rot=base_rot_r, w_rot=10.0))
      res = ik.solve_frame(tasks, sel_all, posture_hold_r, posture_weight=0.4)
      seg_residual["hand_place_right"].merge(res)
      push("hand_place_right", set_left_on)
    landed_r = ik.pad_pose(rh)
    landing_report[f"{rh}@low"] = (
      float(np.linalg.norm(landed_r[0] - low_anchor_pos[rh])),
      gcc.rotation_angle(landed_r[1], state1_pad_rot[rh]),
    )

    # hand_set_hold: the RIGHT hand engages; both welded, feet flat. The
    # left weld re-captures at its held pose (house style, same as the foot
    # mounts' per-segment re-pin).
    hands_on = {lh: True, rh: True, lf: False, rf: False}
    dwell("hand_set_hold", hands_on, weld_ees=hands, hold_feet=feet)

    # foot mounts ---------------------------------------------------------------
    # The swinging shin pad travels from its standing pose to its FINAL wall
    # anchor. Base targets walk the standing pose toward the compressed
    # all-four stance (hands still at the low anchors).
    base_after_place = d.qpos[0:3].copy()
    if wall.identity:
      base_compressed = np.array(
        [
          final_qpos[0] - 0.012,
          0.0,
          final_qpos[2] - 0.030,
        ]
      )
    else:
      # 12 mm off the surface, 30 mm down-wall from the final stance base.
      base_compressed = final_qpos[0:3] + 0.012 * wall.outward - 0.030 * wall.up
      base_compressed[1] = 0.0
    hand_weld = {ee: ik.pad_pose(ee) for ee in hands}
    toe_wall_hold.clear()

    for seg_name, swing_ee, other_ee, shift_sign, seg_frac in (
      ("left_foot_mount", lf, rf, -1.0, left_mount_frac),
      ("right_foot_mount", rf, lf, 1.0, 1.0),
    ):
      for ee in hands:
        hand_weld[ee] = ik.pad_pose(ee)
        anchor_now[ee] = hand_weld[ee][0].copy()
      wall_welds = list(hands) + ([lf] if swing_ee == rf else [])
      weld_anchor = {ee: ik.pad_pose(ee) for ee in wall_welds}
      p0, rot0 = ik.pad_pose(swing_ee)
      p2 = final_anchor_pos[swing_ee]
      rot2 = state1_pad_rot[swing_ee]
      anchor_now[swing_ee] = p2.copy()
      # LEFT mount (flat right foot still grounded): the launch tangent points
      # UP first — the ankle hangs 0.20 m under the shin pad and scrapes the
      # floor if the pad drives forward before it has lifted.
      # RIGHT mount is a TOE PIVOT instead: with the left foot already a
      # one-sided low wall compression, losing the right floor support hands
      # the yaw moment to the left hand as pure pull-out (static audit: util
      # 1.2-1.4 at F0=400). Keeping the right TOE planted through most of the
      # swing (heel-lift knee-drive, like stepping onto a high hold) preserves
      # a right-side low support until the pad face reaches the wall.
      toe_pivot = swing_ee == rf
      if toe_pivot:
        if wall.identity:
          pivot_ctl_off = np.array([-0.015, 0.0, 0.05])
        else:
          # Small off-surface bulge; the lift is GROUND clearance (world z).
          pivot_ctl_off = 0.015 * wall.outward + np.array([0.0, 0.0, 0.05])
        p1 = 0.5 * (p0 + p2) + pivot_ctl_off
        toe_hold = d.site_xpos[ik.toe_sid[swing_ee]].copy()
        toe_off_u = 0.62  # where the toe-pad chain runs out of floor
        swing_fold_from: dict[str, float] = {}
      else:
        if wall.identity:
          mount_ctl_off = np.array([-foot_bulge, 0.0, foot_lift])
        else:
          mount_ctl_off = foot_bulge * wall.outward + np.array([0.0, 0.0, foot_lift])
        p1 = 0.65 * p0 + 0.35 * p2 + mount_ctl_off
      engaged = {
        lh: True,
        rh: True,
        lf: swing_ee == rf,
        rf: False,
      }
      base_seg_start = d.qpos[0:3].copy()
      base_seg_end = (
        base_after_place
        + seg_frac * (base_compressed - base_after_place)
        + np.array([0.0, shift_sign * lateral_weight_shift, 0.0])
      )
      posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
      swing_leg = _leg_joints("left" if swing_ee == lf else "right")
      for t in range(n[seg_name]):
        u = t / (n[seg_name] - 1)
        s = float(gcc.smootherstep(u))
        if toe_pivot:
          # Toe stays planted: the heel-lift plantarflex starts immediately
          # (it pivots the ankle volume UP off the floor); the floor support
          # ends where the toe-pad chain runs out (see the hold fade below).
          s_rot = float(gcc.smootherstep(min(u / 0.9, 1.0)))
          on_ground[swing_ee] = u < 0.60
        else:
          # Delay the plantarflex: rotating the foot down while it is still
          # low scrapes the ankle collision volume on the floor. The pad
          # lifts on the Bezier first; the orientation follows once airborne.
          s_rot = float(gcc.smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
          on_ground[swing_ee] = False
        tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in wall_welds]
        if on_ground[other_ee]:
          tasks.append(foot_hold_task(other_ee))
        if toe_pivot:
          # Z-ONLY hold: the toe stays ON the floor (normal support survives
          # while the chain reaches) but is free to glide forward. Pinning x
          # builds stretch stress against the wall-bound pad — the chain
          # locks near u~0.65 and the DLS snaps to another leg branch
          # (measured 23 rad/s hip + 4 cm base pop). A hard toe cap may
          # glide; spine tips never touch the floor. The hold fades over the
          # window where the toe-pad chain geometrically runs out of floor
          # (pad above ~0.31 m); holding longer just stores snap stress.
          fade = float(np.clip((0.62 - u) / 0.17, 0.0, 1.0))
          if fade > 0.01:
            tasks.append(
              gcc.IKTask(
                kind="site",
                sid=ik.toe_sid[swing_ee],
                pos=toe_hold,
                rot=None,
                w_pos=np.array([2.0, 5.0, 250.0 * fade]),
                w_rot=0.0,
                track=False,
              )
            )
          if lf in toe_wall_hold:
            # The left toe keeps its wall touch through the whole right
            # swing; it is the right-swing tail's low-couple support.
            tasks.append(gcc.toe_task(ik, lf, toe_wall_hold[lf], w_pos=(600.0, 3.0, 10.0)))
        # Fold-window relaxation (right swing): between toe-off and the rake
        # the leg must cross a bistable fold ("heel-lift pivot" shape ->
        # "frog wall-plant" shape). Mid-swing the free pad's Bezier tracking
        # is cosmetic, and at full weight it is exactly what DRAGS the leg
        # through the fold in one frame (a 0.3 rad hip / 3 cm base teleport
        # survived every damping knob). Dropping the swing-task weight there
        # hands the path to the rate-limited guided fold; the weight returns
        # before the rake so the landing is unaffected.
        if toe_pivot:
          # The relaxation window opens WITH the toe fade (0.45) so the leg
          # never sits between two strong masters, and closes before the
          # rake (~0.81). The z component keeps half its weight throughout:
          # fully released it let the reconfiguring leg sag the ankle volume
          # into the floor.
          relax = 1.0 - 0.85 * float(
            gcc.smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0))
            - gcc.smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
          )
        else:
          relax = 1.0
        tasks.append(
          gcc.IKTask(
            kind="site",
            sid=ik.pad_sid[swing_ee],
            pos=_swing_target(t, n[seg_name], p0, p1, p2, n_rake_seg=seg_rake[seg_name]),
            rot=gcc.slerp_matrix(rot0, rot2, s_rot),
            w_pos=np.array([1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]),
            w_rot=100.0 * relax,
            # A relaxed mid-swing task is guidance, not a kinematic fact —
            # its commanded deviation must not read as solver failure.
            track=relax > 0.9,
          )
        )
        base_target = (1.0 - s) * base_seg_start + s * base_seg_end
        if toe_pivot:
          # Wall press through the transfer (cycle's body_press logic) plus a
          # weight bias toward the left support column: both shrink the
          # anchored left hand's pull-out through the mid-swing window where
          # the floor toe has run out and the pad face has not yet touched
          # (the audit's binding frames sit at u~0.4-0.55). The profile peaks
          # there and DECAYS before u~0.66, where the swing leg crosses its
          # bistable fold — holding a deep press through the fold stores the
          # stress the branch snap releases (20 rad/s hip, 1.8 m/s base).
          press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
          if wall.identity:
            base_target = base_target + np.array([swing_press * press, 0.03 * press, 0.0])
          else:
            # Press INTO the surface (against outward) + the lateral bias.
            base_target = base_target - swing_press * press * wall.outward + np.array([0.0, 0.03 * press, 0.0])
        tasks.append(
          gcc.base_task(
            ik,
            base_target,
            (100.0, 40.0, 60.0),
            # The compressed all-four stance the mounts walk toward carries the
            # (rotated) state-1 base orientation; upright at 90.
            rot=np.eye(3) if wall.identity else wall.R,
            w_rot=10.0,
          )
        )
        posture_t = {}
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
          if nm in swing_leg:
            # The pad task pins the SHIN; the articulated foot below the
            # ankle is otherwise free to flop into the floor. Blend the leg
            # on the delayed clock and hold its ankle with a firm spring.
            target = state1_joint[nm]
            posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * target
          else:
            posture_t[nm] = posture_start[nm]
        if toe_pivot and u >= 0.45:
          # Guided fold: from where the floor support runs out to the wall
          # plant the swing leg crosses a bistable fold (the DLS measured a
          # 20 rad/s hip + 1.8 m/s base branch snap at u~0.66 whenever this
          # ran unguided). Rate-limited targets from the leg's ACTUAL angles
          # at u=0.45 to the state-1 angles pick one branch and walk it; the
          # window matches the toe fade and the swing-task relaxation above.
          if not swing_fold_from:
            for nm in swing_leg:
              swing_fold_from[nm] = float(d.qpos[ik.qpos_addresses[nm]])
          w = float(gcc.smootherstep(np.clip((u - 0.45) / 0.50, 0.0, 1.0)))
          for nm in swing_leg:
            posture_t[nm] = (1.0 - w) * swing_fold_from[nm] + w * state1_joint[nm]
        overrides = {}
        for nm in swing_leg:
          if "ankle_roll" in nm:
            overrides[nm] = 4.0
          elif toe_pivot and u >= 0.45:
            # Guided-fold authority (see the posture block above).
            overrides[nm] = 4.0
          elif "ankle_pitch" in nm:
            # Toe pivot: the pitch is owned by the toe-hold + pad geometry
            # early on (a stiff spring fights the pivot and snaps).
            overrides[nm] = 1.0 if toe_pivot else 4.0
          elif "hip_yaw" in nm and toe_pivot:
            # The x-free toe hold leaves a null-space corridor the DLS
            # wanders through as hip-yaw jitter; a moderate guide damps it.
            overrides[nm] = 2.0
        # Throttled opening frames: swapping the flat-foot hold for the
        # toe-only hold (and releasing the other foot's floor pin) moves the
        # soft-task equilibrium; an unthrottled first solve dumps that into a
        # one-frame whole-body pop (cycle hand-swing lesson). The toe-pivot
        # swing additionally caps the per-iteration step for its whole
        # duration: near the toe's stretch limit the DLS can flip leg
        # branches, and a capped step turns that into a spread-out glide.
        throttled = t < 8
        # Extra DLS damping through the pivot leg's bistable fold window:
        # hysteresis against basin hopping (the guided fold and press shaping
        # cut the snap from 23 to ~15 rad/s; the damping absorbs the rest).
        in_fold = toe_pivot and 0.52 <= u <= 0.85
        res = ik.solve_frame(
          tasks,
          sel_all,
          posture_t,
          posture_weight=0.4,
          posture_weight_overrides=overrides,
          iters=2 if throttled else 100,
          max_step=0.003 if throttled else (0.05 if toe_pivot else 0.12),
          lam2=2.0 if in_fold else 0.5,
        )
        seg_residual[seg_name].merge(res)
        push(seg_name, engaged)
      landed = ik.pad_pose(swing_ee)
      anchor_now[swing_ee] = landed[0].copy()
      landing_report[swing_ee] = (
        float(np.linalg.norm(landed[0] - p2)),
        gcc.rotation_angle(landed[1], rot2),
      )
      if seg_name == "left_foot_mount":
        # The dwell also plants the LEFT toe onto the wall: once the right
        # foot's floor support runs out mid-swing (the toe-pad chain cannot
        # reach the floor past pad height ~0.31 m), the left toe extends the
        # low compression couple ~0.2 m below the foot anchor — without it
        # the swing tail audits at util 1.2 (F0=400).
        dwell(
          "foot_switch_hold",
          {lh: True, rh: True, lf: True, rf: False},
          weld_ees=(lh, rh, lf),
          hold_feet=(rf,),
          base_shift=np.array([0.0, -shift_sign * lateral_weight_shift, 0.0]),
          plant_toes=(lf,),
        )
        toe_wall_hold[lf] = d.site_xpos[ik.toe_sid[lf]].copy()

    # brace_hold: with all four pads on the wall, the RIGHT toe joins the
    # (already planted) left one at the wall touch. During the hand steps the
    # single support hand carries the whole pitch-moment pull-out; the
    # touching toes extend the compression couple ~0.2 m below the foot
    # anchors, which is the difference between util ~1.10 and a pass at
    # F0=400 (the same job the cycle's toe braces do for its foot swings).
    dwell(
      "brace_hold",
      all_on,
      weld_ees=ik.ee_names,
      hold_feet=(),
      plant_toes=(rf,),
      toe_hold_targets={lf: toe_wall_hold[lf]},
    )
    toe_wall_hold[rf] = d.site_xpos[ik.toe_sid[rf]].copy()

    # hand walk-up --------------------------------------------------------------
    # Two cycle-style single-hand regrips lift the hands from the low anchors
    # to the state-1 anchors; everything else hangs on three wall welds.
    for seg_name, swing_ee, support_hand in (
      ("hand_up_left", lh, rh),
      ("hand_up_right", rh, lh),
    ):
      for ee in ik.ee_names:
        if ee != swing_ee:
          anchor_now[ee] = ik.pad_pose(ee)[0].copy()
      weld_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee}
      p0, rot0 = ik.pad_pose(swing_ee)
      p2 = final_anchor_pos[swing_ee]
      anchor_now[swing_ee] = p0.copy()
      if wall.identity:
        swing_ctl_off = np.array([-swing_bulge, 0.0, 0.0])
      else:
        swing_ctl_off = swing_bulge * wall.outward
      p1 = 0.5 * (p0 + p2) + swing_ctl_off
      engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
      base_seg_start = d.qpos[0:3].copy()
      base_rot_start = np.zeros(9)
      mujoco.mju_quat2Mat(base_rot_start, d.qpos[3:7])
      base_rot_start = base_rot_start.reshape(3, 3)
      # The base rises WITH each hand step: staying low after a hand lands
      # high stretches that arm onto the reach frontier; rising early folds
      # the still-low support arm past its minimum. Left step: rise most of
      # the way (the low right support caps the shoulder ~0.27 m under its
      # anchor); right step: finish exactly at the state-1 base. The lateral
      # shift toward the SUPPORT hand relieves the anchored low arm's
      # shoulder-roll demand (cycle-generator lesson: the low support arm
      # otherwise rides its roll stop).
      support_side = -1.0 if support_hand == rh else 1.0
      if seg_name == "hand_up_left":
        if wall.identity:
          base_seg_end = np.array(
            [
              final_qpos[0] - 0.002,
              support_side * lateral_weight_shift,
              final_qpos[2] - 0.012,
            ]
          )
        else:
          base_seg_end = final_qpos[0:3] + 0.002 * wall.outward - 0.012 * wall.up
          base_seg_end[1] = support_side * lateral_weight_shift
      else:
        base_seg_end = final_qpos[0:3].copy()
      posture_start = {nm: float(d.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
      support_arm = _arm_joints("left" if support_hand == lh else "right")
      swing_arm = _arm_joints("left" if swing_ee == lh else "right")
      ankle_pitches = tuple(nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.endswith("ankle_pitch"))
      fold_from: dict[str, float] = {}
      for t in range(n[seg_name]):
        # Engage flips at the segment boundary (cycle convention); the
        # all-four handoff stance lives in the handoff_hold segment that
        # follows the second hand step.
        engaged = all_on if t < n_lt else engaged_swing
        if t == n_lt:
          anchor_now[swing_ee] = p2.copy()
        u = t / (n[seg_name] - 1)
        s = float(gcc.smootherstep(u))
        n_arc = n[seg_name] - n_rake
        s_arc = float(gcc.smootherstep(min(t / (n_arc - 1), 1.0)))
        w_fade = min(s_arc / 0.3, 1.0)
        tasks = [gcc.weld_task(ik, ee, weld_anchor[ee]) for ee in ik.ee_names if ee != swing_ee]
        # Both toes stay at their wall touch through the first step; on the
        # closure swing the hold RELEASES first (fade^2, done by u=0.55) and
        # the fold to the state-1 hang runs afterwards under its own
        # rate-limited posture blend below — spring release and fold must not
        # overlap or the stored toe-task stress snaps the ankle (24 rad/s
        # measured at the old hard fade-end).
        toe_w = 1.0 if seg_name == "hand_up_left" else float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2
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
        if seg_name == "hand_up_left":
          # Wall press through the first hand transfer (cycle's body_press):
          # the anchored low hand's pull-out shrinks with the CoM lever.
          if wall.identity:
            base_target = base_target + np.array([0.02 * np.sin(np.pi * s), 0.0, 0.0])
          else:
            base_target = base_target - (0.02 * np.sin(np.pi * s)) * wall.outward
        tasks.append(
          gcc.base_task(
            ik,
            base_target,
            (60.0, 30.0, 50.0),
            rot=gcc.slerp_matrix(base_rot_start, np.eye(3) if wall.identity else wall.R, s),
            # A 4-DoF arm under a 6-DoF weld is overdetermined: given the base
            # pose the support arm's roll is FORCED, and near the stop only a
            # small base tilt can relieve it. Keep the base rotation nearly
            # free while the right support hangs low; square up on the closure
            # swing.
            w_rot=8.0 if seg_name == "hand_up_left" else 30.0,
          )
        )
        posture_t = {}
        for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
          if nm in swing_arm:
            target = state1_joint[nm]
          elif nm in support_arm or nm == "waist_yaw":
            # The support hand is still welded BELOW its stance height; its
            # canonical angles (and the waist yaw that squares the torso) are
            # unreachable while that weld holds (cycle's anchored-arm note).
            target = posture_start[nm]
          else:
            target = state1_joint[nm]
          posture_t[nm] = (1.0 - s) * posture_start[nm] + s * target
        overrides: dict[str, float] = {}
        if seg_name == "hand_up_right" and u >= 0.42:
          # Controlled toe fold, engaged BEFORE the wall-touch hold finishes
          # releasing (release ends u=0.55): both ankles blend from their
          # actual braced angles to the state-1 hang under a firm spring — a
          # gap between spring release and guided fold lets the ankle rush
          # the s-blend target (15 rad/s measured).
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
        push(seg_name, engaged)
      landed = ik.pad_pose(swing_ee)
      anchor_now[swing_ee] = landed[0].copy()
      landing_report[swing_ee] = (
        float(np.linalg.norm(landed[0] - p2)),
        gcc.rotation_angle(landed[1], state1_pad_rot[swing_ee]),
      )
      if seg_name == "hand_up_left":
        # Four-anchor dwell is the safe window to carry the weight bias from
        # the right-hand support side over to the left before the right swing.
        dwell(
          "hand_up_hold",
          all_on,
          weld_ees=ik.ee_names,
          hold_feet=(),
          base_shift=np.array([0.0, 2.0 * lateral_weight_shift, 0.0]),
          toe_hold_targets=toe_wall_hold,
        )

  # closure: land exactly on state 1 shifted by dz -----------------------------
  # The blend window keys off the LAST hand step of the sequence: interleaved
  # ends on hand_up_left (it has no hand_up_right frames); both shipped
  # branches keep their original key (same numeric value — both hand-ups
  # share --hand-up-s — so the two frozen branches stay bitwise identical).
  if sequence == "cross" and hand_z_mid is not None:
    closure_seg = "hand_up_right"  # v4: the right hand's step closes
  else:
    closure_seg = "hand_up_left" if sequence in ("interleaved", "cross") else "hand_up_right"
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
          if wall.identity:
            pen = float(d.xpos[bid_s][0] - gcc.PAD_TOUCH_X)
          else:
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
        if wall.identity:
          gap = float(gcc.WALL_SURFACE_X - d.site_xpos[ik.toe_sid[ee]][0] - ik.toes[ee].radius)
        else:
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
        if wall.identity:
          clear = gcc.WALL_SURFACE_X - float(world[:, 0].max())
        else:
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
      if wall.identity:
        tw = np.array([1.0, 0.0, 0.0])
      else:
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
  if not wall.identity:
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
  if handoff_motion is not None:
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
    if frames_in_seg.size == 0:
      continue  # sequence variants skip some steps (foot-first: no left hand-up)
    last = int(frames_in_seg[-1])
    nr = seg_rake[seg_name]
    drop = float(pad_positions[last - nr, i, 2] - pad_positions[last, i, 2])
    print(f"  rake {ik.ee_names[i]:18s} ({seg_name}): down {drop * 1000.0:5.2f} mm")
  if gimbal_joint_names:
    arr_g = np.stack(qpos_frames)
    for ee, names in gimbal_joint_names.items():
      ang = np.degrees(np.abs(arr_g[:, [ik.qpos_addresses[nm] for nm in names]]))
      fr, ax = np.unravel_index(int(np.argmax(ang)), ang.shape)
      print(
        f"  weld gimbal use {ee:18s} max {ang.max():5.2f} deg @f{int(fr)} axis {'xyz'[int(ax)]} "
        f"(clamp {weld_gimbal_deg:.1f}, stop 3.0)"
      )
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
    "--sequence",
    choices=("hands-first", "foot-first", "interleaved", "cross"),
    default="hands-first",
    help="mount order: hands-first = the shipped standing-lean "
    "choreography (byte-frozen); foot-first = the SAME blocks resequenced "
    "(user call 08-20): LEFT foot to its final anchor first, right hand "
    "to the LOW rung, left hand DIRECTLY to its final rung, toe-pivot "
    "right mount on three welds, single right-hand walk-up; interleaved = "
    "the 80-deg redesign (user call 08-21): sagittal one-arm raises "
    "(the receding surface clears the elbow in front), LEFT hand to the "
    "LOW rung standing, LEFT foot mounts braced by that weld, RIGHT hand "
    "DIRECTLY to its final rung on two welds + the flat right foot, "
    "toe-pivot right mount on three welds, single LEFT-hand walk-up; cross = "
    "interleaved with the feet mirrored (user call 08-21 night): LH low rung, "
    "RIGHT foot first braced by LH + flat LF, RH to its final rung on the "
    "diagonal LH+RF welds + tiptoed LF, LEFT foot toe-pivot, LEFT-hand walk-up",
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
  parser.add_argument(
    "--hand-hover",
    type=float,
    default=0.015,
    help="right-hand hover distance off its anchor (m) while the left welds first",
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
    "--reach-base-z-w",
    type=float,
    default=40.0,
    help="interleaved/cross: base height task weight through the right-hand "
    "reach (the station is a suggestion at 40; 100 = the donors' pull)",
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
    "--reach-rise",
    type=float,
    default=None,
    help="cross: cap on the along-wall RISE of the base through the "
    "right-hand reach (None = v3 station, final - 1.2 cm; 0 = reach from "
    "the hold height)",
  )
  parser.add_argument(
    "--reach-rot-w",
    type=float,
    default=8.0,
    help="cross: base orientation task weight through the right-hand reach",
  )
  parser.add_argument(
    "--reach-rot-frac",
    type=float,
    default=0.8,
    help="cross: fraction of the trunk slerp toward wall-parallel reached by the end of the right-hand reach",
  )
  parser.add_argument(
    "--reach-heel-lift",
    type=float,
    default=0.38,
    help="cross: keyed heel-lift amplitude (rad) of the floor foot through "
    "the reach; <= 0 keeps the foot flat (no tiptoe)",
  )
  parser.add_argument(
    "--reach-heel-start",
    type=float,
    default=0.05,
    help="cross: reach-progress s at which the heel key starts ramping",
  )
  parser.add_argument(
    "--transfer-lateral",
    type=float,
    default=0.0,
    help="cross v4: lateral (world +y, over the standing LEFT foot) base shift "
    "during the hand_set_left dwell — the weight transfer before the braced "
    "right-foot mount; 0 = v3",
  )
  parser.add_argument(
    "--transfer-press",
    type=float,
    default=0.0,
    help="cross v4: base shift toward the wall during the same transfer (m)",
  )
  parser.add_argument(
    "--switch-lateral",
    type=float,
    default=-0.05,
    help="cross: lateral base shift during foot_switch_hold (v3 walked the "
    "weight back to center with -0.05; v4 recipe 0 keeps it over the left foot)",
  )
  parser.add_argument(
    "--reach-lateral",
    type=float,
    default=None,
    help="cross: lateral base station through the right-hand reach (None = "
    "-lateral_weight_shift as in v3; v4 recipe +0.13 = over the left foot)",
  )
  parser.add_argument(
    "--hand-z-mid",
    type=float,
    default=None,
    help="cross v4: along-wall height of an INTERMEDIATE right-hand rung "
    "reachable flat-footed; the right hand then walks up to --hand-z in a "
    "hand_up_right step after the pivot (None = land the final rung directly)",
  )
  parser.add_argument(
    "--mid-rung-y",
    type=float,
    default=0.0,
    help="cross v4: lateral offset of the intermediate right-hand rung",
  )
  parser.add_argument(
    "--pivot-lateral",
    type=float,
    default=None,
    help="cross: lateral base target at the end of the toe pivot (None = -lateral_weight_shift as in v3)",
  )
  parser.add_argument(
    "--step-trunk-pitch",
    type=float,
    default=0.0,
    help="cross v4: extra trunk pitch toward the wall (rad) at the end of the "
    "braced right-foot mount, on top of wall-parallel",
  )
  parser.add_argument(
    "--foot-rake-s",
    type=float,
    default=None,
    help="cross v4: rake duration of the LEFT foot mount (None = --rake-s); the right foot keeps --rake-right-s",
  )
  parser.add_argument(
    "--foot-rake-dist",
    type=float,
    default=None,
    help="cross v4: rake stroke of both foot mounts (None = --rake-dist)",
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
    "--left-mount-frac",
    type=float,
    default=0.55,
    help="fraction of the compressed all-four base reached by the END of "
    "left_foot_mount (right mount always finishes at 1.0); higher = the "
    "body arrives closer to the wall before the weak right swing",
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
    "--rh-first",
    action="store_true",
    help="cross v5 (user call 2026-08-22): HANDS-FIRST order LH -> RH -> RF -> LF. The "
    "right hand takes its own LOW rung on two flat feet braced by the welded left hand "
    "(mirror of the left place), both feet then mount on two hand welds, and the closure "
    "is the two-step hand walk-up (left then right). No tiptoe reach.",
  )
  parser.add_argument(
    "--pivot-rise",
    type=float,
    default=0.0,
    help="cross v5: along-wall base rise (m) through the left-foot toe pivot (three welds); 0 = v3.",
  )
  parser.add_argument(
    "--set-right-press",
    type=float,
    default=0.0,
    help="cross v5 --rh-first: base shift toward the wall (m) during hand_set_right (hips-in before the lift).",
  )
  parser.add_argument(
    "--set-right-drop",
    type=float,
    default=0.0,
    help="cross v5 --rh-first: base drop along the wall (m) during hand_set_right.",
  )
  parser.add_argument(
    "--close-rh-first",
    action="store_true",
    help="cross v5 two-step closure: the right hand steps first (non-closing), the left hand closes.",
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
    "--pivot-heel-lift",
    type=float,
    default=None,
    help="cross v5: absolute swing-ankle plantarflexion target (rad) keyed over the first 30%% of the toe pivot (v3 tiptoe = 0.38); None = v3 behaviour.",
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
    "--set-right-pitch",
    type=float,
    default=0.0,
    help="cross v6 --rh-first: trunk pitch key (rad, world +y; negative = lean away from the wall) reached by the end of hand_set_right.",
  )
  parser.add_argument(
    "--set-right-s",
    type=float,
    default=None,
    help="cross v6 --rh-first: hand_set_right duration (s); None = --set-left-s.",
  )
  parser.add_argument(
    "--set-left-pitch",
    type=float,
    default=0.0,
    help="cross v6: trunk pitch key (rad; positive = lean away from the wall) reached by the end of hand_set_left.",
  )
  parser.add_argument(
    "--raise-pitch-comp-sign",
    type=float,
    default=1.0,
    help="cross v6 --rh-first: sign/scale of the trunk-pitch compensation of the right-arm raise keys (1 = add the trunk forward pitch to the end key).",
  )
  parser.add_argument(
    "--rh-raise-lateral",
    action="store_true",
    help="cross v6 --rh-first: raise the right arm laterally (--raise-roll-right) instead of sagittally.",
  )
  parser.add_argument("--set-left-s2", type=float, default=None, help="cross v6: hand_set_left duration override (s).")
  parser.add_argument(
    "--stance-width",
    type=float,
    default=None,
    help="v6: standing stance width (m, feet re-planted at +-w/2 by IK); None = the walk keyframe (0.296).",
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
    default=None,
    help="v6g closing rework: split the left hand's 30 cm walk-up (1.22 -> hand-z) into TWO steps through a mid rung at this height (m); each landing keeps elbow flexion in reserve. None = the shipped single step.",
  )
  parser.add_argument(
    "--hand-up-left-mid-s", type=float, default=None, help="duration of the mid step (default: --hand-up-s)."
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
    "--weld-hard",
    action="store_true",
    help="welds and planted floor feet become the IK's top-priority level (exact, null-space hierarchy): an engaged pad never drifts off its anchor -- the physics of a latched microspine -- and any reach shortfall lands on the swing target / base instead (landing error, closure).",
  )
  parser.add_argument(
    "--weld-gimbal-deg",
    type=float,
    default=0.0,
    help="let the IK use the WELDED pads' passive gimbal hinges (plant joints *_gimbal_{x,y,z}, real stop 3 deg) within +-D deg, so a closed-chain rotation demand is met by the wrist ball joint instead of dragging the pad off its anchor (weld drift). 0 = gimbals locked at zero (legacy). Swing pads stay pinned at zero.",
  )
  parser.add_argument(
    "--weld-gimbal-w",
    type=float,
    default=3.0,
    help="posture spring (to zero) on a freed gimbal hinge; higher = the arm joints are preferred over the gimbal.",
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
    "--set-right-lateral",
    type=float,
    default=0.0,
    help="cross v6 --rh-first: lateral base shift (m, +y = left) during hand_set_right (both hands welded).",
  )
  parser.add_argument(
    "--set-hold-pitch",
    type=float,
    default=0.0,
    help="cross v6: trunk pitch key (rad, + = toward the wall) over hand_set_hold (un-bow before the pivot).",
  )
  parser.add_argument(
    "--raise-pitch-end-right",
    type=float,
    default=None,
    help="cross v6 --rh-first: end shoulder-pitch key of the right-arm raise (rad); None = --raise-pitch-end.",
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
    sequence=args.sequence,
    standoff=args.standoff,
    hand_z=args.hand_z,
    hand_z_low=args.hand_z_low,
    stand_s=args.stand_s,
    raise_s=args.raise_s,
    place_s=args.place_s,
    set_left_s=args.set_left_s,
    place_right_s=args.place_right_s,
    hand_hover=args.hand_hover,
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
    reach_base_z_w=args.reach_base_z_w,
    step_base_w=tuple(args.step_base_w),
    step_press=args.step_press,
    step_hips_lateral=args.step_hips_lateral,
    reach_rise=args.reach_rise,
    reach_rot_w=args.reach_rot_w,
    reach_rot_frac=args.reach_rot_frac,
    reach_heel_lift=args.reach_heel_lift,
    reach_heel_start=args.reach_heel_start,
    transfer_lateral=args.transfer_lateral,
    transfer_press=args.transfer_press,
    switch_lateral=args.switch_lateral,
    reach_lateral=args.reach_lateral,
    hand_z_mid=args.hand_z_mid,
    mid_rung_y=args.mid_rung_y,
    pivot_lateral=args.pivot_lateral,
    step_trunk_pitch=args.step_trunk_pitch,
    foot_rake_s=args.foot_rake_s,
    foot_rake_dist=args.foot_rake_dist,
    low_rung_y=args.low_rung_y,
    smooth_sigma=args.smooth_sigma,
    left_mount_frac=args.left_mount_frac,
    lateral_weight_shift=args.lateral_weight_shift,
    stand_y=args.stand_y,
    step_rise=args.step_rise,
    rh_first=args.rh_first,
    pivot_rise=args.pivot_rise,
    set_right_press=args.set_right_press,
    set_right_drop=args.set_right_drop,
    close_rh_first=args.close_rh_first,
    set_hold_press=args.set_hold_press,
    set_hold_rise=args.set_hold_rise,
    pivot_heel_lift=args.pivot_heel_lift,
    pivot_smear_gap=args.pivot_smear_gap,
    pivot_toe_release=args.pivot_toe_release,
    pivot_ctl_up=args.pivot_ctl_up,
    pivot_ctl_out=args.pivot_ctl_out,
    pivot_fold_u0=args.pivot_fold_u0,
    set_right_pitch=args.set_right_pitch,
    set_right_s=args.set_right_s,
    set_left_pitch=args.set_left_pitch,
    raise_pitch_comp_sign=args.raise_pitch_comp_sign,
    rh_raise_lateral=args.rh_raise_lateral,
    set_left_s_override=args.set_left_s2,
    stance_width=args.stance_width,
    place_right_pitch=args.place_right_pitch,
    place_right_base_dz=args.place_right_base_dz,
    place_right_base_dx=args.place_right_base_dx,
    place_right_left_arm_w=args.place_right_left_arm_w,
    weld_gimbal_deg=args.weld_gimbal_deg,
    weld_gimbal_w=args.weld_gimbal_w,
    weld_hard=args.weld_hard,
    hand_up_left_mid=args.hand_up_left_mid,
    hand_up_left_mid_s=args.hand_up_left_mid_s,
    up_left_mid_hold_s=args.up_left_mid_hold_s,
    hand_up_left_mid_y=args.hand_up_left_mid_y,
    set_right_lateral=args.set_right_lateral,
    set_hold_pitch=args.set_hold_pitch,
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
