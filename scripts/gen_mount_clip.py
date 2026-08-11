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
  hand_place        both pads together: Bezier arc onto the LOW hand
                    anchors at ``--hand-z-low`` + the standard pre-engage
                    rake (2 mm downward seating stroke). Low first because
                    the state-1 anchor height is out of standing reach:
                    the flat-foot reach ceiling at the placement standoff
                    is ~1.47 m (probe), while ground clearance of the final
                    stance needs hands >= 1.53 m (state-1's lowest point,
                    the ankle collision volume, hangs 1.512 m under the
                    hand anchors; 2 cm floor margin). A tiptoe-assisted
                    single placement reaches ~1.55 only by standing on two
                    frontiers at once (arm extension + toe balance) and was
                    rejected per the v3 "reach frontier" precedent.
  hand_set_hold     hands engage (weld) under the raked preload, feet flat.
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

Output schema == gen_climb_clip.py plus:

    foot_on_ground   (T, 2) uint8   1 while that foot (left, right) is a
                                    flat floor support (feet order matches
                                    ``toe_list``/``foot_sites_list``)

Validate:
    python scripts/validate_climb_reference.py \
      src/scr_phybot_mjab/assets/motions/climb_mount_prologue.npy
    (ground-contact aware static audit; the acyclic timing audit engages
    automatically via the foot_on_ground key)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mjlab  # noqa: F401  (entry points; see gen_climb_clip.py)
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "scripts") not in sys.path:
  sys.path.insert(0, str(_REPO / "scripts"))

import gen_climb_clip as gcc  # noqa: E402
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)

_SEGMENT_NAMES = (
  "ground_stand",
  "hand_raise",
  "hand_place",
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
)

_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")


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
      u = gcc._smootherstep((s - s0) / max(s1 - s0, 1.0e-9))
      return float((1.0 - u) * v0 + u * v1)
  return keys[-1][1]


def generate_mount_clip(
  output: Path,
  fps: float,
  standoff: float,
  hand_z: float,
  hand_z_low: float,
  stand_s: float,
  raise_s: float,
  place_s: float,
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
  lateral_weight_shift: float,
  hand_load_transfer_s: float,
  rake_s: float,
  rake_right_s: float,
  rake_dist: float,
  limit_margin_frac: float,
  limit_weight: float,
) -> None:
  ik = gcc._ClimbIK(
    limit_margin_frac=limit_margin_frac, limit_weight=limit_weight
  )
  m, d = ik.model, ik.data
  lh, rh, lf, rf = ik.ee_names
  hands, feet = (lh, rh), (lf, rf)
  seg_id = {name: i for i, name in enumerate(_SEGMENT_NAMES)}

  n = {
    "ground_stand": max(int(round(stand_s * fps)), 1),
    "hand_raise": max(int(round(raise_s * fps)), 2),
    "hand_place": max(int(round(place_s * fps)), 2),
    "hand_set_hold": max(int(round(set_hold_s * fps)), 1),
    "left_foot_mount": max(int(round(foot_swing_s * fps)), 2),
    "foot_switch_hold": max(int(round(switch_hold_s * fps)), 1),
    "right_foot_mount": max(int(round(foot_swing_right_s * fps)), 2),
    "brace_hold": max(int(round(brace_hold_s * fps)), 2),
    "hand_up_left": max(int(round(hand_up_s * fps)), 2),
    "hand_up_hold": max(int(round(up_hold_s * fps)), 1),
    "hand_up_right": max(int(round(hand_up_s * fps)), 2),
    "handoff_hold": max(int(round(handoff_hold_s * fps)), 2),
    "end_hold": max(int(round(end_hold_s * fps)), 1),
  }
  n_rake = max(int(round(rake_s * fps)), 1)
  # The right-foot swing gets a LONG rake: its floor support ends at the
  # toe-pad chain's reach (u~0.6) and the left-side-only low compression
  # audits infeasible until the pad face touches — the extended face-on-wall
  # window closes that gap. The early latch this induces (~0.24 s before the
  # scheduled engage) sits inside the mismatch bill's grace (0.3 x the
  # 1.5 s OFF window).
  n_rake_right = max(int(round(rake_right_s * fps)), 1)
  seg_rake = {seg: n_rake for seg in _SEGMENT_NAMES}
  seg_rake["right_foot_mount"] = n_rake_right
  n_lt = max(int(round(hand_load_transfer_s * fps)), 0)
  for seg in ("hand_place", "left_foot_mount", "right_foot_mount",
              "hand_up_left", "hand_up_right"):
    if n[seg] - seg_rake[seg] < 2:
      raise ValueError(f"rake leaves fewer than two arc frames in {seg}")
  if n_lt >= min(n["hand_up_left"], n["hand_up_right"]):
    raise ValueError("--hand-load-transfer-s must be shorter than a hand-up swing")
  rake_vec = np.array([0.0, 0.0, float(rake_dist)])

  def _swing_target(t, n_seg, p0, p1, p2, n_rake_seg=None):
    nr = n_rake if n_rake_seg is None else n_rake_seg
    n_arc = n_seg - nr
    if t < n_arc:
      s = float(gcc._smootherstep(min(t / (n_arc - 1), 1.0)))
      return gcc._bezier(p0, p1, p2 + rake_vec, s)
    s_rake = float(gcc._smootherstep((t - n_arc + 1) / nr))
    return p2 + (1.0 - s_rake) * rake_vec

  # -- state 1 snapshot -> final (shifted) stance -----------------------------
  ik.set_state1()
  state1_qpos = d.qpos.copy()
  state1_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names}
  state1_pad_rot = {ee: state1_anchor[ee][1].copy() for ee in ik.ee_names}
  state1_joint = {
    nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
  }
  state1_hand_z = float(state1_anchor[lh][0][2])
  dz = hand_z - state1_hand_z
  final_qpos = state1_qpos.copy()
  final_qpos[2] += dz
  final_anchor_pos = {
    ee: state1_anchor[ee][0] + np.array([0.0, 0.0, dz]) for ee in ik.ee_names
  }
  low_anchor_pos = {
    ee: final_anchor_pos[ee] + np.array([0.0, 0.0, hand_z_low - hand_z])
    for ee in hands
  }
  if hand_z_low >= hand_z:
    raise ValueError("--hand-z-low must be below --hand-z")

  # -- standing keyframe -------------------------------------------------------
  base_x0 = gcc._WALL_SURFACE_X - standoff

  def set_standing() -> None:
    d.qpos[:] = 0.0
    d.qpos[0:3] = (base_x0, 0.0, 0.726)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
      if name in ik._qadr:
        d.qpos[ik._qadr[name]] = angle
    mujoco.mj_forward(m, d)

  set_standing()
  standing_qpos = d.qpos.copy()
  standing_joint = {
    nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
  }
  foot_body_bid = {
    ee: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, toe.body_name)
    for ee, toe in ik.toes.items()
  }
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
      world = d.geom_xpos[g] + (
        m.geom_aabb[g, :3] + m.geom_aabb[g, 3:] * _corner_signs
      ) @ d.geom_xmat[g].reshape(3, 3).T
      ground_z = min(ground_z, float(world[:, 2].min()))

  # -- bookkeeping -------------------------------------------------------------
  qpos_frames: list[np.ndarray] = []
  labels: list[int] = []
  engaged_frames: list[np.ndarray] = []
  anchor_frames: list[np.ndarray] = []
  ground_frames: list[np.ndarray] = []
  seg_residual = {name: gcc._Residual() for name in _SEGMENT_NAMES}
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
    engaged_frames.append(
      np.array([engaged[ee] for ee in ik.ee_names], dtype=np.uint8)
    )
    anchor_frames.append(np.stack([anchor_now[ee] for ee in ik.ee_names]))
    ground_frames.append(
      np.array([on_ground[lf], on_ground[rf]], dtype=np.uint8)
    )

  def foot_hold_task(ee: str) -> gcc._Task:
    pos, rot, sid = foot_hold_pose[ee]
    return gcc._Task(
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
    ee: tuple(
      nm
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
      if nm.startswith(toe.body_name.removesuffix("_roll"))
    )
    for ee, toe in ik.toes.items()
  }

  def dwell(
    segment: str,
    engaged: dict[str, bool],
    weld_ees: tuple[str, ...],
    hold_feet: tuple[str, ...],
    base_shift: np.ndarray | None = None,
    plant_toes: tuple[str, ...] = (),
    toe_hold_targets: dict[str, np.ndarray] | None = None,
  ) -> None:
    """Static dwell on the current pose; optionally drift the base target,
    plant foot toes onto the wall (cycle dwell_ik's ankle-only maneuver),
    and/or hold previously planted toes in place."""
    for ee in weld_ees:
      cur = ik.pad_pose(ee)
      anchor_now[ee] = cur[0].copy()
    weld_anchor = {ee: ik.pad_pose(ee) for ee in weld_ees}
    base_hold = d.qpos[0:3].copy()
    base_rot = np.zeros(9)
    mujoco.mju_quat2Mat(base_rot, d.qpos[3:7])
    base_rot = base_rot.reshape(3, 3)
    posture_hold = {
      nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    overrides: dict[str, float] = {}
    plant0, plant_goal = {}, {}
    for ee in plant_toes:
      toe0 = d.site_xpos[ik.toe_sid[ee]].copy()
      # Wall-touch target at exact contact: the static audit credits a
      # released support within 0.5 mm of the wall, and the IK undershoots
      # by a few tenths — aiming at the surface itself lands inside the
      # window. The reference still commands no press (pressing a toe whose
      # own pad is latched loads that anchor toward the cheap release
      # directions — cycle generator's warning).
      plant0[ee] = toe0
      plant_goal[ee] = np.array(
        [
          gcc._WALL_SURFACE_X - ik.toes[ee].radius,
          toe0[1],
          toe0[2] + 0.05,
        ]
      )
      for nm in ankle_joints[ee]:
        if nm.endswith("_roll"):
          overrides[nm] = 3.0
    for t in range(n[segment]):
      s = float(gcc._smootherstep(t / max(n[segment] - 1, 1)))
      tasks = [gcc._weld_task(ik, ee, weld_anchor[ee]) for ee in weld_ees]
      tasks.extend(foot_hold_task(ee) for ee in hold_feet)
      for ee in plant_toes:
        tasks.append(
          gcc._toe_task(
            ik,
            ee,
            (1.0 - s) * plant0[ee] + s * plant_goal[ee],
            w_pos=(600.0, 3.0, 10.0),
          )
        )
      if toe_hold_targets:
        for ee, pos in toe_hold_targets.items():
          tasks.append(gcc._toe_task(ik, ee, pos, w_pos=(600.0, 3.0, 10.0)))
      for ee in plant_toes:
        toe_gap_report[ee] = float(
          gcc._WALL_SURFACE_X
          - d.site_xpos[ik.toe_sid[ee]][0]
          - ik.toes[ee].radius
        )
      target = base_hold if base_shift is None else base_hold + s * base_shift
      tasks.append(
        gcc._base_task(ik, target, (60.0, 60.0, 60.0), rot=base_rot, w_rot=8.0)
      )
      res = ik.solve_frame(
        tasks,
        sel_all,
        posture_hold,
        posture_weight=0.3,
        posture_weight_overrides=overrides,
      )
      seg_residual[segment].merge(res)
      push(segment, engaged)

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
    roll_range = ik._range[f"{side}_shoulder_roll"]
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
      d.qpos[ik._qadr[nm]] = _key_blend(keys, s)
    mujoco.mj_forward(m, d)
    push("hand_raise", none_on)

  # hand_place: both pads arc onto the LOW anchors and rake down 2 mm.
  place_start = {ee: ik.pad_pose(ee) for ee in hands}
  base_start = d.qpos[0:3].copy()
  base_place_end = base_start + np.array([place_lean, 0.0, 0.0])
  posture_start = {
    nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
  }
  arm_target = {
    nm: state1_joint[nm] for side in ("left", "right") for nm in _arm_joints(side)
  }
  p_ctl = {}
  for ee in hands:
    p0 = place_start[ee][0]
    p2 = low_anchor_pos[ee]
    p_ctl[ee] = (p0, 0.5 * (p0 + p2) + np.array([-place_bulge, 0.0, 0.03]), p2)
  for t in range(n["hand_place"]):
    s = float(gcc._smootherstep(t / (n["hand_place"] - 1)))
    n_arc = n["hand_place"] - n_rake
    s_arc = float(gcc._smootherstep(min(t / (n_arc - 1), 1.0)))
    w_fade = min(s_arc / 0.3, 1.0)
    tasks = [foot_hold_task(ee) for ee in feet]
    for ee in hands:
      p0, p1, p2 = p_ctl[ee]
      tasks.append(
        gcc._Task(
          kind="site",
          sid=ik.pad_sid[ee],
          pos=_swing_target(t, n["hand_place"], p0, p1, p2),
          rot=gcc._slerp_mat(place_start[ee][1], state1_pad_rot[ee], s),
          w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
          w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
        )
      )
    tasks.append(
      gcc._base_task(
        ik,
        (1.0 - s) * base_start + s * base_place_end,
        (60.0, 60.0, 100.0),
        rot=np.eye(3),
        w_rot=10.0,
      )
    )
    posture_t = {
      nm: (1.0 - s) * posture_start[nm]
      + s * arm_target.get(nm, standing_joint[nm])
      for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    res = ik.solve_frame(
      tasks,
      sel_all,
      posture_t,
      posture_weight=0.6,
      posture_weight_overrides={
        nm: 1.2 for side in ("left", "right") for nm in _arm_joints(side)
      },
    )
    seg_residual["hand_place"].merge(res)
    push("hand_place", none_on)
  for ee in hands:
    landed = ik.pad_pose(ee)
    landing_report[f"{ee}@low"] = (
      float(np.linalg.norm(landed[0] - low_anchor_pos[ee])),
      gcc._rot_angle(landed[1], state1_pad_rot[ee]),
    )

  # hand_set_hold: hands engage under the raked preload.
  hands_on = {lh: True, rh: True, lf: False, rf: False}
  dwell("hand_set_hold", hands_on, weld_ees=hands, hold_feet=feet)

  # foot mounts ---------------------------------------------------------------
  # The swinging shin pad travels from its standing pose to its FINAL wall
  # anchor. Base targets walk the standing pose toward the compressed
  # all-four stance (hands still at the low anchors).
  base_after_place = d.qpos[0:3].copy()
  base_compressed = np.array(
    [
      final_qpos[0] - 0.012,
      0.0,
      final_qpos[2] - 0.030,
    ]
  )
  hand_weld = {ee: ik.pad_pose(ee) for ee in hands}
  toe_wall_hold: dict[str, np.ndarray] = {}

  for seg_name, swing_ee, other_ee, shift_sign, seg_frac in (
    ("left_foot_mount", lf, rf, -1.0, 0.55),
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
      p1 = 0.5 * (p0 + p2) + np.array([-0.015, 0.0, 0.05])
      toe_hold = d.site_xpos[ik.toe_sid[swing_ee]].copy()
      toe_off_u = 0.62  # where the toe-pad chain runs out of floor
      swing_fold_from: dict[str, float] = {}
    else:
      p1 = 0.65 * p0 + 0.35 * p2 + np.array([-foot_bulge, 0.0, foot_lift])
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
    posture_start = {
      nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    swing_leg = _leg_joints("left" if swing_ee == lf else "right")
    for t in range(n[seg_name]):
      u = t / (n[seg_name] - 1)
      s = float(gcc._smootherstep(u))
      if toe_pivot:
        # Toe stays planted: the heel-lift plantarflex starts immediately
        # (it pivots the ankle volume UP off the floor); the floor support
        # ends where the toe-pad chain runs out (see the hold fade below).
        s_rot = float(gcc._smootherstep(min(u / 0.9, 1.0)))
        on_ground[swing_ee] = u < 0.60
      else:
        # Delay the plantarflex: rotating the foot down while it is still
        # low scrapes the ankle collision volume on the floor. The pad
        # lifts on the Bezier first; the orientation follows once airborne.
        s_rot = float(gcc._smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
        on_ground[swing_ee] = False
      tasks = [gcc._weld_task(ik, ee, weld_anchor[ee]) for ee in wall_welds]
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
            gcc._Task(
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
          tasks.append(
            gcc._toe_task(
              ik, lf, toe_wall_hold[lf], w_pos=(600.0, 3.0, 10.0)
            )
          )
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
          gcc._smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0))
          - gcc._smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
        )
      else:
        relax = 1.0
      tasks.append(
        gcc._Task(
          kind="site",
          sid=ik.pad_sid[swing_ee],
          pos=_swing_target(
            t, n[seg_name], p0, p1, p2, n_rake_seg=seg_rake[seg_name]
          ),
          rot=gcc._slerp_mat(rot0, rot2, s_rot),
          w_pos=np.array(
            [1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]
          ),
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
        base_target = base_target + np.array(
          [0.05 * press, 0.03 * press, 0.0]
        )
      tasks.append(
        gcc._base_task(
          ik,
          base_target,
          (100.0, 40.0, 60.0),
          rot=np.eye(3),
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
            swing_fold_from[nm] = float(d.qpos[ik._qadr[nm]])
        w = float(gcc._smootherstep(np.clip((u - 0.45) / 0.50, 0.0, 1.0)))
        for nm in swing_leg:
          posture_t[nm] = (
            (1.0 - w) * swing_fold_from[nm] + w * state1_joint[nm]
          )
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
      gcc._rot_angle(landed[1], rot2),
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
    weld_anchor = {
      ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee
    }
    p0, rot0 = ik.pad_pose(swing_ee)
    p2 = final_anchor_pos[swing_ee]
    anchor_now[swing_ee] = p0.copy()
    p1 = 0.5 * (p0 + p2) + np.array([-swing_bulge, 0.0, 0.0])
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
    base_seg_end = (
      np.array(
        [
          final_qpos[0] - 0.002,
          support_side * lateral_weight_shift,
          final_qpos[2] - 0.012,
        ]
      )
      if seg_name == "hand_up_left"
      else final_qpos[0:3].copy()
    )
    posture_start = {
      nm: float(d.qpos[ik._qadr[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    }
    support_arm = _arm_joints("left" if support_hand == lh else "right")
    swing_arm = _arm_joints("left" if swing_ee == lh else "right")
    ankle_pitches = tuple(
      nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.endswith("ankle_pitch")
    )
    fold_from: dict[str, float] = {}
    for t in range(n[seg_name]):
      # Engage flips at the segment boundary (cycle convention); the
      # all-four handoff stance lives in the handoff_hold segment that
      # follows the second hand step.
      engaged = all_on if t < n_lt else engaged_swing
      if t == n_lt:
        anchor_now[swing_ee] = p2.copy()
      u = t / (n[seg_name] - 1)
      s = float(gcc._smootherstep(u))
      n_arc = n[seg_name] - n_rake
      s_arc = float(gcc._smootherstep(min(t / (n_arc - 1), 1.0)))
      w_fade = min(s_arc / 0.3, 1.0)
      tasks = [
        gcc._weld_task(ik, ee, weld_anchor[ee])
        for ee in ik.ee_names
        if ee != swing_ee
      ]
      # Both toes stay at their wall touch through the first step; on the
      # closure swing the hold RELEASES first (fade^2, done by u=0.55) and
      # the fold to the state-1 hang runs afterwards under its own
      # rate-limited posture blend below — spring release and fold must not
      # overlap or the stored toe-task stress snaps the ankle (24 rad/s
      # measured at the old hard fade-end).
      toe_w = (
        1.0
        if seg_name == "hand_up_left"
        else float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2
      )
      if toe_w > 1.0e-3:
        for ee in feet:
          tasks.append(
            gcc._toe_task(
              ik,
              ee,
              toe_wall_hold[ee],
              w_pos=(600.0 * toe_w, 3.0, 10.0),
            )
          )
      tasks.append(
        gcc._Task(
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
        base_target = base_target + np.array(
          [0.02 * np.sin(np.pi * s), 0.0, 0.0]
        )
      tasks.append(
        gcc._base_task(
          ik,
          base_target,
          (60.0, 30.0, 50.0),
          rot=gcc._slerp_mat(base_rot_start, np.eye(3), s),
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
            fold_from[nm] = float(d.qpos[ik._qadr[nm]])
          w = float(gcc._smootherstep(np.clip((u - 0.42) / 0.50, 0.0, 1.0)))
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
      gcc._rot_angle(landed[1], state1_pad_rot[swing_ee]),
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
  delta = final_qpos - qpos_frames[-1]
  closure_frames = max(int(round(0.5 * n["hand_up_right"])), 2)
  for i in range(closure_frames):
    idx = len(qpos_frames) - closure_frames + i
    blend = float(gcc._smootherstep((i + 1) / closure_frames))
    qpos_frames[idx] = qpos_frames[idx] + blend * delta
    qpos_frames[idx][3:7] /= np.linalg.norm(qpos_frames[idx][3:7])
  closure_pos = float(np.linalg.norm(delta[0:3]))
  closure_joint = float(np.abs(delta[7:]).max())

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
  shank_bids = {
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_pitch")
    for side in ("left", "right")
  }
  all_foot_geoms = set()
  for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "_spine_c_" in name or "toe" in name:
      pad_or_toe_geoms.add(g)
    if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
      visual_geoms.add(g)
    if (
      m.geom_bodyid[g] in foot_body_bid.values()
      or m.geom_bodyid[g] in shank_bids
    ):
      all_foot_geoms.add(g)
  foot_geoms = {
    ee: {
      g
      for g in range(m.ngeom)
      if m.geom_bodyid[g] == foot_body_bid[ee]
    }
    for ee in feet
  }
  corner_signs = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
  )

  wall_clear_min = (np.inf, -1, "")
  ground_clear_min = (np.inf, -1, "")
  max_weld_drift = 0.0
  max_wall_pen = -np.inf
  com_margin_min = (np.inf, -1)
  mass = m.body_mass.sum()

  for t, q in enumerate(qpos_frames):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    joint_positions[t] = [
      d.qpos[ik._qadr[nm]] for nm in PHYBOT_C2_POLICY_JOINT_NAMES
    ]
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
          max_wall_pen = max(
            max_wall_pen, float(d.xpos[bid_s][0] - gcc._PAD_TOUCH_X)
          )
    for i, sid in enumerate(ik.foot_sid):
      foot_positions[t, i] = d.site_xpos[sid]

    seg_name_t = _SEGMENT_NAMES[label_arr[t]]
    if seg_name_t in ("hand_up_left", "hand_up_hold", "hand_up_right"):
      for ee in feet:
        gap = float(
          gcc._WALL_SURFACE_X
          - d.site_xpos[ik.toe_sid[ee]][0]
          - ik.toes[ee].radius
        )
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
      world = d.geom_xpos[g] + (center + half * corner_signs) @ d.geom_xmat[
        g
      ].reshape(3, 3).T
      if g not in pad_or_toe_geoms and g not in all_foot_geoms:
        clear = gcc._WALL_SURFACE_X - float(world[:, 0].max())
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
    if ground_arr[t].all() and not engaged_arr[t].any():
      com_x = float((m.body_mass[:, None] * d.xipos).sum(axis=0)[0] / mass)
      toe_x = max(
        float(d.site_xpos[ik.toe_sid[ee]][0] + ik.toes[ee].radius)
        for ee in feet
      )
      heel_x = min(
        float(d.xpos[foot_body_bid[ee]][0] - 0.06) for ee in feet
      )
      margin = min(toe_x - com_x, com_x - heel_x)
      if margin < com_margin_min[0]:
        com_margin_min = (margin, t)

  data_out = {
    "joints_list": list(PHYBOT_C2_POLICY_JOINT_NAMES),
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "foot_sites_list": list(gcc._FOOT_SITE_NAMES),
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
    "segment_names": list(_SEGMENT_NAMES),
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.0),
    "wall_surface_x": np.float32(gcc._WALL_SURFACE_X),
    "pad_touch_x": np.float32(gcc._PAD_TOUCH_X),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data_out)

  # -- report ------------------------------------------------------------------
  print(
    f"wrote {output} ({frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps, "
    f"dz={dz:+.4f} m: hands {hand_z_low:.3f} -> {hand_z:.3f}, feet "
    f"{final_anchor_pos[lf][2]:.3f})"
  )
  for name in _SEGMENT_NAMES:
    r = seg_residual[name]
    if r.pos > 0.0 or r.rot > 0.0:
      print(
        f"  {name:18s} IK residual: pos {r.pos * 1000.0:6.2f} mm, "
        f"rot {np.degrees(r.rot):5.2f} deg"
      )
  for key, (pos_err, rot_err) in landing_report.items():
    print(
      f"  {key:22s} landing vs scheduled anchor: "
      f"{pos_err * 1000.0:6.2f} mm, {np.degrees(rot_err):5.2f} deg"
    )
  for ee, gap in toe_gap_report.items():
    print(
      f"  toe {ee:18s} wall-touch gap {gap * 1000.0:+6.2f} mm "
      "(must be within +0.5 mm for the static audit to credit the brace)"
    )
  joint_speed = np.abs(np.diff(joint_positions, axis=0)) * fps
  peak_frame, peak_joint = np.unravel_index(
    np.argmax(joint_speed), joint_speed.shape
  )
  base_speed = float(np.abs(np.diff(root_position, axis=0)).max() * fps)
  print(
    f"  peak joint speed {joint_speed.max():.2f} rad/s "
    f"({PHYBOT_C2_POLICY_JOINT_NAMES[int(peak_joint)]} @f{int(peak_frame)} "
    f"{_SEGMENT_NAMES[label_arr[int(peak_frame)]]}), peak base speed "
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
    ("hand_place", 1),
    ("left_foot_mount", 2),
    ("right_foot_mount", 3),
    ("hand_up_left", 0),
    ("hand_up_right", 1),
  ):
    frames_in_seg = np.flatnonzero(label_arr == seg_id[seg_name])
    last = int(frames_in_seg[-1])
    nr = seg_rake[seg_name]
    drop = float(pad_positions[last - nr, i, 2] - pad_positions[last, i, 2])
    print(
      f"  rake {ik.ee_names[i]:18s} ({seg_name}): down {drop * 1000.0:5.2f} mm"
    )
  print(
    f"  engaged-pad weld drift max      {max_weld_drift * 1000.0:6.2f} mm\n"
    f"  swing sphere wall penetration   {max_wall_pen * 1000.0:+6.2f} mm "
    f"(<= 0 clear)\n"
    f"  closure correction              {closure_pos * 1000.0:6.2f} mm base, "
    f"{closure_joint:.4f} rad joints"
  )
  print(
    f"  wall clearance min (non-pad)    {wall_clear_min[0] * 1000.0:+7.2f} mm "
    f"@f{wall_clear_min[1]} {_SEGMENT_NAMES[label_arr[wall_clear_min[1]]]} "
    f"({wall_clear_min[2]})"
  )
  print(
    f"  ground clearance min (airborne) {ground_clear_min[0] * 1000.0:+7.2f} mm "
    f"@f{ground_clear_min[1]} {_SEGMENT_NAMES[label_arr[ground_clear_min[1]]]} "
    f"({ground_clear_min[2]})"
  )
  if com_margin_min[1] >= 0:
    print(
      f"  standing CoM support margin min {com_margin_min[0] * 1000.0:6.1f} mm "
      f"@f{com_margin_min[1]} (pre-engage frames; statics audited properly by "
      "validate_climb_reference.py)"
    )
  worst_name, worst_margin, worst_frame = "", np.inf, -1
  for i, nm in enumerate(PHYBOT_C2_POLICY_JOINT_NAMES):
    lo, hi = ik._range[nm]
    col = joint_positions[:, i]
    lo_m = float(col.min()) - lo
    hi_m = hi - float(col.max())
    margin = min(lo_m, hi_m) / (hi - lo)
    if margin < worst_margin:
      worst_name, worst_margin = nm, margin
      worst_frame = int(
        np.argmin(col) if lo_m < hi_m else np.argmax(col)
      )
  print(
    f"  joint-limit clearance min       {worst_margin * 100.0:6.2f}% of range "
    f"({worst_name} @f{worst_frame} "
    f"{_SEGMENT_NAMES[label_arr[worst_frame]]})"
  )
  if "shoulder_roll" in worst_name and "hand_up" in _SEGMENT_NAMES[
    label_arr[worst_frame]
  ]:
    print(
      "    note: one-hand-high / one-hand-low support kinematically forces "
      "the LOW arm's shoulder roll onto its stop on this wristless 4-DoF "
      "arm (cycle generator's state-4 note, amplified by the 0.09 m hand "
      "step). Base-pose relief was probed and does not exist; accepted as "
      "a ~0.3 s transient, same precedent as the cycle's 0.058 rad case."
    )
  # Handoff contract: the final frame must BE state 1 shifted by dz.
  handoff_joint = float(
    np.abs(
      joint_positions[-1]
      - np.array([state1_joint[nm] for nm in PHYBOT_C2_POLICY_JOINT_NAMES])
    ).max()
  )
  print(f"  handoff joint delta vs state-1  {handoff_joint:.2e} rad (must be ~0)")
  if max_weld_drift > 5.0e-3 or max_wall_pen > 2.0e-3:
    print("  WARNING: IK quality below tolerance; inspect the render.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=_REPO / "src/scr_phybot_mjab/assets/motions/climb_mount_prologue.npy",
  )
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument(
    "--standoff",
    type=float,
    default=0.23,
    help="standing base-to-wall distance (m); toes clear the wall by "
    "standoff - 0.163 and the flat-foot hand reach ceiling shrinks ~1 cm "
    "per +1 cm of standoff",
  )
  parser.add_argument(
    "--hand-z",
    type=float,
    default=1.53,
    help="FINAL hand anchor height (m) = state-1 height + dz. The final "
    "stance's lowest point (ankle collision) hangs 1.512 m below the hand "
    "anchors: 1.53 leaves 1.8 cm of ground; every extra cm here costs a cm "
    "of hand-step reach margin",
  )
  parser.add_argument(
    "--hand-z-low",
    type=float,
    default=1.44,
    help="first-placement hand anchor height (m); must sit inside the "
    "flat-foot reach ceiling (~1.47 at the default standoff)",
  )
  parser.add_argument("--stand-s", type=float, default=1.0)
  parser.add_argument("--raise-s", type=float, default=2.4)
  parser.add_argument("--place-s", type=float, default=1.4)
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

  generate_mount_clip(
    output=args.output,
    fps=args.fps,
    standoff=args.standoff,
    hand_z=args.hand_z,
    hand_z_low=args.hand_z_low,
    stand_s=args.stand_s,
    raise_s=args.raise_s,
    place_s=args.place_s,
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
    lateral_weight_shift=args.lateral_weight_shift,
    hand_load_transfer_s=args.hand_load_transfer_s,
    rake_s=args.rake_s,
    rake_right_s=args.rake_right_s,
    rake_dist=args.rake_dist,
    limit_margin_frac=args.limit_margin_frac,
    limit_weight=args.limit_weight,
  )


if __name__ == "__main__":
  main()
