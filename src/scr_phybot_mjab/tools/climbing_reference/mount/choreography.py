"""The a80 mount choreography: one function per clip segment, in clip order."""

from __future__ import annotations

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tools.climbing_reference import kinematics as kin

from .builder import MountBuilder, arm_joints, key_blend, leg_joints


def perform(b: MountBuilder) -> None:
  """Every segment, in clip order."""
  ground_stand(b)
  hand_raise(b)
  hand_place(b)
  hand_set_left(b)
  hand_raise_right(b)
  hand_place_right(b)
  hand_set_right(b)
  right_foot_mount(b)
  foot_switch_hold(b)
  hand_set_hold(b)
  left_foot_mount(b)
  brace_hold(b)
  hand_up_steps(b)
  handoff(b)


def ground_stand(b: MountBuilder) -> None:
  """The walk-delivery keyframe, held."""
  b.set_standing()
  for _ in range(b.seg_frames["ground_stand"]):
    b.push("ground_stand", b.engaged())


def hand_raise(b: MountBuilder) -> None:
  """The left arm raises from the standing keyframe in joint space, the body frozen."""
  ik, data, model, recipe = b.ik, b.data, b.model, b.recipe

  # Joint space, no IK: the legs, base and right arm stay at the standing
  # keyframe. The elbow folds deep before the upper arm sweeps past
  # horizontal, so the forearm rides up past the face instead of pointing
  # at the receding surface (a straighter front raise clips the wall by
  # 18 mm at the 0.19 m standoff); overhead it re-extends to the cocked
  # half-bend.
  elbow_fold = max(float(ik.joint_ranges["left_elbow_pitch"][0]) + 0.10, -2.2)
  out_sign = 1.0 if ik.joint_ranges["left_shoulder_roll"][1] > 1.5 else -1.0
  s1 = b.state1_joint
  st = b.standing_joint
  raise_keys = {
    "left_shoulder_pitch": (
      (0.0, st["left_shoulder_pitch"]),
      (0.25, st["left_shoulder_pitch"]),
      (0.88, recipe.raise_pitch_end),
      (1.0, recipe.raise_pitch_end),
    ),
    "left_shoulder_roll": (
      (0.0, st["left_shoulder_roll"]),
      (0.35, out_sign * recipe.raise_roll),
      (0.80, out_sign * recipe.raise_roll),
      (1.0, s1["left_shoulder_roll"]),
    ),
    # The forearm turns sideways while it folds, then back to the handoff yaw.
    "left_shoulder_yaw": (
      (0.0, st["left_shoulder_yaw"]),
      (0.25, recipe.raise_yaw),
      (0.55, recipe.raise_yaw),
      (1.0, s1["left_shoulder_yaw"]),
    ),
    "left_elbow_pitch": (
      (0.0, st["left_elbow_pitch"]),
      (0.10, st["left_elbow_pitch"]),
      (0.42, elbow_fold),
      (0.80, elbow_fold),
      (1.0, -1.15),
    ),
  }
  for t in range(b.seg_frames["hand_raise"]):
    s = t / max(b.seg_frames["hand_raise"] - 1, 1)
    data.qpos[:] = b.standing_qpos
    for nm, keys in raise_keys.items():
      data.qpos[ik.qpos_addresses[nm]] = key_blend(keys, s)
    mujoco.mj_forward(model, data)
    b.push("hand_raise", b.engaged())


def hand_place(b: MountBuilder) -> None:
  """The left pad arcs to its low rung and rakes, standing on both flat feet."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)

  # The mount's only unbraced first contact, made with the right arm down:
  # with both arms overhead the standing CoM leaves the toe line at 80 deg.
  place_ctl_off = recipe.place_bulge * wall.outward + 0.03 * wall.up
  lh_start = ik.pad_pose(lh)
  p0 = lh_start[0]
  p2 = b.low_anchor_pos[lh]
  base_hold = data.qpos[0:3].copy()
  base_end = base_hold - recipe.place_lean * wall.outward
  posture_hold = {nm: float(data.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  n_frames = b.seg_frames["hand_place"]
  n_arc = n_frames - b.n_rake
  p_touch = p2 + b.rake_vec
  p1 = 0.5 * (p0 + p_touch) + place_ctl_off
  for t in range(n_frames):
    s = float(kin.smootherstep(t / (n_frames - 1)))
    if t < n_arc:
      s_arc = float(kin.smootherstep(t / (n_arc - 1)))
      w_fade = min(s_arc / 0.3, 1.0)
      pad_pos = kin.bezier(p0, p1, p_touch, s_arc)
      w_p = 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)
      w_r = 50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc)
      rot_s = lean_s = s_arc
    else:
      lean_s = 1.0
      s_rake = float(kin.smootherstep((t - n_arc + 1) / b.n_rake))
      pad_pos = p2 + (1.0 - s_rake) * b.rake_vec
      w_p, w_r, rot_s = 2000.0, 500.0, 1.0
    tasks = [b.stiff_foot_hold(ee) for ee in feet]
    tasks.append(
      kin.IKTask(
        kind="site",
        sid=ik.pad_sid[lh],
        pos=pad_pos,
        rot=kin.slerp_matrix(lh_start[1], b.state1_pad_rot[lh], rot_s),
        w_pos=np.full(3, w_p),
        w_rot=w_r,
      )
    )
    # The base lean rides the arc clock, so nothing leans before the hand is
    # on the wall, and the trunk is held upright hard until the touch: at
    # w_rot 10 the reaching arm drags the soft base into a 10 deg bow
    # before the pad arrives.
    tasks.append(
      kin.base_task(
        ik,
        (1.0 - lean_s) * base_hold + lean_s * base_end,
        (60.0, 60.0, 100.0),
        rot=np.eye(3),
        w_rot=10.0 + 50.0 * (1.0 - lean_s),
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in arm_joints("left"):
        posture_t[nm] = (1.0 - s) * posture_hold[nm] + s * b.state1_joint[nm]
      elif nm == "waist_yaw":
        posture_t[nm] = (1.0 - s) * posture_hold[nm]
      else:
        posture_t[nm] = posture_hold[nm]
    overrides = {nm: 1.2 for nm in arm_joints("left")}
    overrides["waist_yaw"] = 12.0
    for ee in feet:
      for nm in b.ankle_joints[ee]:
        overrides[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      b.sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    b.seg_residual["hand_place"].merge(res)
    b.push("hand_place", b.engaged())
  landed = ik.pad_pose(lh)
  b.landing_report[f"{lh}@low"] = (
    float(np.linalg.norm(landed[0] - p2)),
    kin.rotation_angle(landed[1], b.state1_pad_rot[lh]),
  )


def hand_set_left(b: MountBuilder) -> None:
  """The left hand welds; both feet stay flat."""
  ik = b.ik
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)

  # The right arm rests at the side; the dwell's posture anchor keeps it down.
  b.dwell(
    "hand_set_left",
    b.engaged(lh),
    weld_ees=(lh,),
    hold_feet=feet,
    extra_overrides={"waist_yaw": 6.0},
  )


def hand_raise_right(b: MountBuilder) -> None:
  """The right arm raises in joint space, the body and the welded left arm frozen."""
  ik, data, model, wall, recipe = b.ik, b.data, b.model, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names

  # The right hand takes its first rung before any foot lifts, so every
  # later move rides on two hand welds: a lone left hand fails in tension
  # or laterally under the right foot's lift in every stance tried.
  b.anchor_now[lh] = ik.pad_pose(lh)[0].copy()
  elbow_fold = max(float(ik.joint_ranges["right_elbow_pitch"][0]) + 0.10, -2.2)
  out_sign = 1.0 if ik.joint_ranges["right_shoulder_roll"][1] > 1.5 else -1.0
  q_frozen = data.qpos.copy()
  st = {nm: float(q_frozen[ik.qpos_addresses[nm]]) for nm in arm_joints("right")}
  # The raise is keyed in world elevation: the shoulder-pitch keys shift by
  # the trunk's forward pitch, so the arm follows the same world arc as on
  # an upright trunk (on a trunk bowed 10.7 deg toward the wall, a
  # joint-space key sweeps the hand 12 cm into it).
  trunk_rot = np.zeros(9)
  mujoco.mju_quat2Mat(trunk_rot, q_frozen[3:7])
  trunk_rot = trunk_rot.reshape(3, 3)
  trunk_fwd_pitch = float(
    np.arcsin(np.clip(trunk_rot[0, 2], -1.0, 1.0))
  )  # world-x component of the trunk up axis (+ = toward the wall)
  trunk_fwd_pitch -= float(np.arcsin(np.clip(wall.R[0, 2], -1.0, 1.0)))
  raise_pitch_end = (recipe.raise_pitch_end_right) + recipe.raise_pitch_comp_sign * trunk_fwd_pitch
  raise_keys = {
    "right_shoulder_pitch": (
      (0.0, st["right_shoulder_pitch"]),
      (0.25, st["right_shoulder_pitch"]),
      (0.88, raise_pitch_end),
      (1.0, raise_pitch_end),
    ),
    "right_shoulder_roll": (
      (0.0, st["right_shoulder_roll"]),
      # A lateral raise: the elbow sweeps through +y, where there is no
      # wall, and comes inboard overhead.
      (0.35, out_sign * (recipe.raise_roll_right)),
      (0.80, out_sign * (recipe.raise_roll_right)),
      (1.0, b.state1_joint["right_shoulder_roll"]),
    ),
    # Turn the folding forearm sideways before lifting it: a straight front
    # fold reaches through the wall.
    "right_shoulder_yaw": (
      (0.0, st["right_shoulder_yaw"]),
      (0.25, -recipe.raise_yaw),
      (0.55, -recipe.raise_yaw),
      (1.0, b.state1_joint["right_shoulder_yaw"]),
    ),
    "right_elbow_pitch": (
      (0.0, st["right_elbow_pitch"]),
      (0.10, st["right_elbow_pitch"]),
      (0.42, elbow_fold),
      (0.80, elbow_fold),
      (1.0, -1.15),
    ),
  }
  for t in range(b.seg_frames["hand_raise_right"]):
    s = t / max(b.seg_frames["hand_raise_right"] - 1, 1)
    data.qpos[:] = q_frozen
    for nm, keys in raise_keys.items():
      data.qpos[ik.qpos_addresses[nm]] = key_blend(keys, s)
    mujoco.mj_forward(model, data)
    b.push("hand_raise_right", b.engaged(lh))


def hand_place_right(b: MountBuilder) -> None:
  """The right pad arcs to its first rung and rakes, braced by the left weld and both feet."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)
  lh_weld = ik.pad_pose(lh)
  rh_start = ik.pad_pose(rh)
  p0 = rh_start[0]
  p2 = b.mid_anchor_rh.copy()
  base_hold = data.qpos[0:3].copy()
  base_rot = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot, data.qpos[3:7])
  base_rot = base_rot.reshape(3, 3)
  posture_hold = {nm: float(data.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  n_frames = b.seg_frames["hand_place_right"]
  n_arc = n_frames - b.n_rake
  if n_arc < 2:
    raise ValueError("--place-right-s leaves fewer than two arc frames")
  p_touch = p2 + b.rake_vec
  place_ctl_off = recipe.place_bulge * wall.outward + 0.03 * wall.up
  p1 = 0.5 * (p0 + p_touch) + place_ctl_off
  for t in range(n_frames):
    s = float(kin.smootherstep(t / (n_frames - 1)))
    if t < n_arc:
      s_arc = float(kin.smootherstep(t / (n_arc - 1)))
      w_fade = min(s_arc / 0.3, 1.0)
      pad_pos = kin.bezier(p0, p1, p_touch, s_arc)
      w_p = 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)
      w_r = 50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc)
      rot_s = s_arc
    else:
      s_rake = float(kin.smootherstep((t - n_arc + 1) / b.n_rake))
      pad_pos = p2 + (1.0 - s_rake) * b.rake_vec
      w_p, w_r, rot_s = 2000.0, 500.0, 1.0
    tasks = [b.stiff_foot_hold(ee) for ee in feet]
    tasks.append(b.weld_task(lh, lh_weld))
    tasks.append(
      kin.IKTask(
        kind="site",
        sid=ik.pad_sid[rh],
        pos=pad_pos,
        rot=kin.slerp_matrix(rh_start[1], b.state1_pad_rot[rh], rot_s),
        w_pos=np.full(3, w_p),
        w_rot=w_r,
      )
    )
    # The trunk bows toward the wall during the arc, not the raise (whose
    # joint-space keys would sweep the wall), so the landing loads the high
    # right hand in arm tension.
    cp, sp = np.cos(recipe.place_right_pitch), np.sin(recipe.place_right_pitch)
    rot_t = kin.slerp_matrix(base_rot, np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]) @ base_rot, s)
    # Reach relief for the high rung: the base rises over the arc instead
    # of the IK dragging the welded left hand off its anchor.
    base_t = base_hold - s * recipe.place_right_base_dx * wall.outward + s * recipe.place_right_base_dz * wall.up
    tasks.append(kin.base_task(ik, base_t, (60.0, 60.0, 100.0), rot=rot_t, w_rot=60.0))
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in arm_joints("right"):
        posture_t[nm] = (1.0 - s) * posture_hold[nm] + s * b.state1_joint[nm]
      elif nm == "waist_yaw":
        posture_t[nm] = (1.0 - s) * posture_hold[nm]
      else:
        posture_t[nm] = posture_hold[nm]
    overrides = {nm: 1.2 for nm in arm_joints("right")}
    # The welded left arm stays stiff, like every support arm in a hand step.
    overrides.update({nm: recipe.place_right_left_arm_w for nm in arm_joints("left")})
    overrides["waist_yaw"] = 12.0
    for ee in feet:
      for nm in b.ankle_joints[ee]:
        overrides[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      b.sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    b.seg_residual["hand_place_right"].merge(res)
    b.note_base("hand_place_right", base_hold)
    b.push("hand_place_right", b.engaged(lh))
  landed = ik.pad_pose(rh)
  b.landing_report[f"{rh}@low"] = (
    float(np.linalg.norm(landed[0] - p2)),
    kin.rotation_angle(landed[1], b.state1_pad_rot[rh]),
  )


def hand_set_right(b: MountBuilder) -> None:
  """The right hand welds; both feet stay flat."""
  ik = b.ik
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)
  b.dwell(
    "hand_set_right",
    b.engaged(lh, rh),
    weld_ees=(lh, rh),
    hold_feet=feet,
    extra_overrides={"waist_yaw": 6.0},
  )
  b.anchor_now[rh] = ik.pad_pose(rh)[0].copy()


def right_foot_mount(b: MountBuilder) -> None:
  """The right shin pad steps from the floor to its wall anchor on the two hand welds."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  rh_weld = ik.pad_pose(rh)

  # The hand welds turn the free-balance step into a held one. The arc
  # launches upward first: the ankle hangs 0.20 m under the shin pad and
  # scrapes the floor otherwise.
  swing_ee, other_ee = rf, lf
  lh_weld = ik.pad_pose(lh)
  b.anchor_now[lh] = lh_weld[0].copy()
  p0, rot0 = ik.pad_pose(swing_ee)
  p2 = b.final_anchor_pos[swing_ee]
  rot2 = b.state1_pad_rot[swing_ee]
  b.anchor_now[swing_ee] = p2.copy()
  mount_ctl_off = recipe.foot_bulge * wall.outward + np.array([0.0, 0.0, recipe.foot_lift])
  p1 = 0.65 * p0 + 0.35 * p2 + mount_ctl_off
  base_seg_start = data.qpos[0:3].copy()
  base_rot_start = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_start, data.qpos[3:7])
  base_rot_start = base_rot_start.reshape(3, 3)
  # The step's weight shift: hips over the stance foot, tall, and in
  # toward the wall, which gives the welded low hand slack at its reach
  # limit. The base presses toward the wall mid-swing by step_press and
  # rises by step_rise.
  base_seg_end = base_seg_start - recipe.step_hips_in * wall.outward + recipe.step_rise * wall.up
  base_seg_end[1] = base_seg_start[1] + recipe.step_hips_lateral
  base_rot_end = wall.R
  posture_start = {nm: float(data.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  swing_leg = leg_joints("right")
  for t in range(b.seg_frames["right_foot_mount"]):
    u = t / (b.seg_frames["right_foot_mount"] - 1)
    s = float(kin.smootherstep(u))
    s_rot = float(kin.smootherstep(np.clip((u - 0.15) / 0.80, 0.0, 1.0)))
    b.on_ground[swing_ee] = False
    tasks = [
      b.weld_task(lh, lh_weld),
      b.stiff_foot_hold(other_ee),
    ]
    tasks.append(b.weld_task(rh, rh_weld))
    tasks.append(
      kin.IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=b.swing_target(
          t,
          b.seg_frames["right_foot_mount"],
          p0,
          p1,
          p2,
          n_rake_seg=b.seg_rake["right_foot_mount"],
          rake_vec_seg=b.rake_vec_foot,
        ),
        rot=kin.slerp_matrix(rot0, rot2, s_rot),
        w_pos=np.full(3, 1000.0),
        w_rot=100.0,
      )
    )
    press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
    base_target = (1.0 - s) * base_seg_start + s * base_seg_end
    base_target = base_target - recipe.step_press * press * wall.outward
    tasks.append(
      kin.base_task(
        ik,
        base_target,
        # At (40, 40, 40) the hips wander into the wall and the hip roll
        # hits its stop (51 mm residual).
        tuple(float(w) for w in recipe.step_base_w),
        # The trunk returns to the wall-parallel attitude across the step;
        # w_rot 40 beats the posture term, which favors the bowed hips.
        rot=kin.slerp_matrix(base_rot_start, base_rot_end, s),
        w_rot=40.0,
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in swing_leg:
        posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * b.state1_joint[nm]
      else:
        posture_t[nm] = posture_start[nm]
    overrides = {}
    for nm in swing_leg:
      if "ankle" in nm:
        overrides[nm] = 4.0
    # The waist is pinned hard (16) through the attitude change: at 8 the
    # solver pays for it with 0.22-0.39 rad of twist.
    overrides["waist_yaw"] = 16.0
    # The stance foot stays flat (the audit gates each sole point at 8 mm):
    # otherwise its toe rides up 3 mm and leaves one heel point bearing
    # against a hand weld.
    for nm in b.ankle_joints[other_ee]:
      overrides[nm] = 12.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      b.sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    b.seg_residual["right_foot_mount"].merge(res)
    b.note_base("right_foot_mount", base_target)
    b.push("right_foot_mount", b.engaged(lh, rh))
  landed = ik.pad_pose(swing_ee)
  b.anchor_now[swing_ee] = landed[0].copy()
  b.landing_report[swing_ee] = (
    float(np.linalg.norm(landed[0] - p2)),
    kin.rotation_angle(landed[1], rot2),
  )


def foot_switch_hold(b: MountBuilder) -> None:
  """Three welds settle and the right toe plants on the wall."""
  ik, data, recipe = b.ik, b.data, b.recipe
  lh, rh, lf, rf = ik.ee_names

  # The right toe plants on the wall (the low compression later moves lean
  # on) and the weight shifts back toward the center.
  b.dwell(
    "foot_switch_hold",
    b.engaged(lh, rh, rf),
    weld_ees=(lh, rh, rf),
    hold_feet=(lf,),
    base_shift=np.array([0.0, recipe.switch_lateral, 0.0]),
    plant_toes=(rf,),
    extra_overrides={"waist_yaw": 6.0},
  )
  b.toe_wall_hold[rf] = data.site_xpos[ik.toe_sid[rf]].copy()


def hand_set_hold(b: MountBuilder) -> None:
  """Three-weld dwell; the base presses toward the wall so the pivot starts closer."""
  ik, wall, recipe = b.ik, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names

  # The base does not settle into a crouch: with a hand on a high rung,
  # sinking stretches that arm past its handoff branch and rising back
  # tears the weld. It presses toward the wall instead; from a station
  # further out the pivot's swing leg runs straight and flips branches at
  # 15 rad/s.
  base_shift_in = -recipe.set_hold_press * wall.outward + recipe.set_hold_rise * wall.up
  b.dwell(
    "hand_set_hold",
    b.engaged(lh, rh, rf),
    weld_ees=(rh, lh, rf),
    hold_feet=(lf,),
    toe_hold_targets={rf: b.toe_wall_hold[rf]},
    base_shift=base_shift_in,
    base_w=(300.0, 300.0, 300.0),
    extra_overrides={"waist_yaw": 6.0},
  )


def left_foot_mount(b: MountBuilder) -> None:
  """The left foot toe-pivots onto the wall on three welds."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  # A heel-lift knee drive with the toe planted through most of the swing,
  # a guided fold across the leg's bistable window, then the face-on-wall
  # rake.
  seg_name, swing_ee = "left_foot_mount", lf
  wall_welds = [lh, rh, rf]
  weld_anchor = {ee: ik.pad_pose(ee) for ee in wall_welds}
  for ee in wall_welds:
    b.anchor_now[ee] = weld_anchor[ee][0].copy()
  p0, rot0 = ik.pad_pose(swing_ee)
  p2 = b.final_anchor_pos[swing_ee]
  rot2 = b.state1_pad_rot[swing_ee]
  b.anchor_now[swing_ee] = p2.copy()
  # The arc's control point sits pivot_ctl_out off the surface and
  # pivot_ctl_up high. During the relaxed fold window the pad task runs at
  # 15 %, and a 15 mm offset lets the approaching pad sphere overshoot the
  # touch plane by 2.7 mm. Lifting the swing foot early keeps the knee bent
  # while the fold raises the body; a straight leg flips at 15 rad/s.
  pivot_ctl_off = recipe.pivot_ctl_out * wall.outward + np.array([0.0, 0.0, recipe.pivot_ctl_up])
  p1 = 0.5 * (p0 + p2) + pivot_ctl_off
  # The left toe's standing station.
  toe_hold = data.site_xpos[ik.toe_sid[lf]].copy()
  # The smear's end: same height and lateral spot, the toe sphere
  # pivot_smear_gap short of the tilted wall surface.
  toe_r = ik.toes[swing_ee].radius
  toe_smear = toe_hold.copy()
  toe_smear[0] = wall.pivot[0] + (wall.outward[2] * toe_hold[2] - (toe_r + recipe.pivot_smear_gap)) / (-wall.outward[0])
  swing_fold_from = {}
  base_seg_start = data.qpos[0:3].copy()
  # The base holds its station (shifted by pivot_out and biased laterally
  # over the left support side), so the high support arm keeps its handoff
  # geometry; the closure does the rising.
  base_seg_end = base_seg_start + recipe.pivot_out * wall.outward + recipe.pivot_rise * wall.up
  base_seg_end[1] = -recipe.lateral_weight_shift
  posture_start = {nm: float(data.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  swing_leg = leg_joints("left")
  fold_u0 = recipe.pivot_fold_u0  # where the guided fold and the stiff leg posture start
  for t in range(b.seg_frames[seg_name]):
    u = t / (b.seg_frames[seg_name] - 1)
    s = float(kin.smootherstep(u))
    s_rot = float(kin.smootherstep(min(u / 0.9, 1.0)))
    # The planted toe's pin fades out over 0.17 of the pivot from
    # pivot_toe_release; the foot leaves the floor just before it is gone.
    rel_end = recipe.pivot_toe_release + 0.17
    b.on_ground[swing_ee] = u < rel_end - 0.04
    tasks = [b.weld_task(ee, weld_anchor[ee]) for ee in wall_welds]
    fade = float(np.clip((rel_end - u) / 0.17, 0.0, 1.0))
    if fade > 0.01:
      # Smear the toe toward the wall base (u 0.05-0.45, x tracked at 150).
      # Left at its standing spot, the shin must tilt ~35 deg to carry the
      # pad to the wall, the knee runs straight before toe-off, and the
      # leg's bistable flip then costs 15-21 rad/s. Dragging the toe toward
      # the base line shortens the chain.
      smear_s = float(kin.smootherstep(np.clip((u - 0.05) / 0.40, 0.0, 1.0)))
      # z at 250: pinning it at 1000 neither keeps the toe down (the knee
      # drive still lifts it ~1 cm) nor relieves the hands.
      tasks.append(
        kin.IKTask(
          kind="site",
          sid=ik.toe_sid[swing_ee],
          pos=toe_hold + smear_s * (toe_smear - toe_hold),
          rot=None,
          w_pos=np.array([150.0 * fade, 5.0, 250.0 * fade]),
          w_rot=0.0,
          track=False,
        )
      )
    tasks.append(kin.toe_task(ik, rf, b.toe_wall_hold[rf], w_pos=(600.0, 3.0, 10.0)))
    relax = 1.0 - 0.85 * float(
      kin.smootherstep(np.clip((u - 0.45) / 0.12, 0.0, 1.0)) - kin.smootherstep(np.clip((u - 0.72) / 0.10, 0.0, 1.0))
    )
    tasks.append(
      kin.IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=b.swing_target(
          t,
          b.seg_frames[seg_name],
          p0,
          p1,
          p2,
          n_rake_seg=b.seg_rake[seg_name],
          rake_vec_seg=b.rake_vec_foot,
        ),
        rot=kin.slerp_matrix(rot0, rot2, s_rot),
        w_pos=np.array([1000.0 * relax, 1000.0 * relax, 1000.0 * max(relax, 0.5)]),
        w_rot=100.0 * relax,
        track=relax > 0.9,
      )
    )
    base_target = (1.0 - s) * base_seg_start + s * base_seg_end
    press = float(np.sin(np.pi * min(u / 0.9, 1.0)))
    base_target = base_target - recipe.swing_press * press * wall.outward + np.array([0.0, -0.03 * press, 0.0])
    tasks.append(
      kin.base_task(
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
        posture_t[nm] = (1.0 - s_rot) * posture_start[nm] + s_rot * b.state1_joint[nm]
      else:
        posture_t[nm] = posture_start[nm]
    if u >= fold_u0:
      if not swing_fold_from:
        for nm in swing_leg:
          swing_fold_from[nm] = float(data.qpos[ik.qpos_addresses[nm]])
      w = float(kin.smootherstep(np.clip((u - fold_u0) / 0.50, 0.0, 1.0)))
      for nm in swing_leg:
        posture_t[nm] = (1.0 - w) * swing_fold_from[nm] + w * b.state1_joint[nm]
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
    # The trunk does not twist to feed the pivot: with the waist free it
    # twists 0.54 rad mid-swing, 0.32 at weight 4.
    overrides["waist_yaw"] = 8.0
    throttled = t < 8
    in_fold = 0.52 <= u <= 0.85
    res = ik.solve_frame(
      tasks,
      b.sel_all,
      posture_t,
      posture_weight=0.4,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.05,
      lam2=2.0 if in_fold else 0.5,
    )
    b.seg_residual[seg_name].merge(res)
    b.note_base(seg_name, base_target)
    b.push(seg_name, b.engaged(lh, rh, rf))
  landed = ik.pad_pose(swing_ee)
  b.anchor_now[swing_ee] = landed[0].copy()
  b.landing_report[swing_ee] = (
    float(np.linalg.norm(landed[0] - p2)),
    kin.rotation_angle(landed[1], rot2),
  )


def brace_hold(b: MountBuilder) -> None:
  """All four welded; the left toe joins the right at the wall."""
  ik, data = b.ik, b.data
  lh, rh, lf, rf = ik.ee_names

  b.dwell(
    "brace_hold",
    b.engaged(*ik.ee_names),
    weld_ees=ik.ee_names,
    hold_feet=(),
    plant_toes=(lf,),
    toe_hold_targets={rf: b.toe_wall_hold[rf]},
    extra_overrides={"waist_yaw": 6.0},
  )
  b.toe_wall_hold[lf] = data.site_xpos[ik.toe_sid[lf]].copy()


def hand_up(
  b: MountBuilder,
  seg_name,
  swing_ee,
  support_hand,
  base_end,
  closing,
  target_override=None,
  report_key=None,
  pin_swing=True,
) -> None:
  """One single-hand regrip: the swing pad arcs to its target while the other three hold their welds."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)
  for ee in ik.ee_names:
    if ee != swing_ee:
      b.anchor_now[ee] = ik.pad_pose(ee)[0].copy()
  weld_anchor = {ee: ik.pad_pose(ee) for ee in ik.ee_names if ee != swing_ee}
  p0, rot0 = ik.pad_pose(swing_ee)
  # ``target_override`` replaces the final rung with a middle one.
  p2 = b.final_anchor_pos[swing_ee] if target_override is None else np.asarray(target_override, dtype=np.float64)
  b.anchor_now[swing_ee] = p0.copy()
  swing_ctl_off = recipe.swing_bulge * wall.outward
  p1 = 0.5 * (p0 + p2) + swing_ctl_off
  engaged_swing = {ee: ee != swing_ee for ee in ik.ee_names}
  base_seg_start = data.qpos[0:3].copy()
  base_rot_start = np.zeros(9)
  mujoco.mju_quat2Mat(base_rot_start, data.qpos[3:7])
  base_rot_start = base_rot_start.reshape(3, 3)
  base_seg_end = np.asarray(base_end, dtype=np.float64).copy()
  posture_start = {nm: float(data.qpos[ik.qpos_addresses[nm]]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES}
  support_arm = arm_joints("right" if support_hand == rh else "left")
  swing_arm = arm_joints("left" if swing_ee == lh else "right")
  ankle_pitches = tuple(nm for nm in PHYBOT_C2_POLICY_JOINT_NAMES if nm.endswith("ankle_pitch"))
  fold_from = {}
  for t in range(b.seg_frames[seg_name]):
    engaged = b.engaged(*ik.ee_names) if t < b.n_lt else engaged_swing
    if t == b.n_lt:
      b.anchor_now[swing_ee] = p2.copy()
    u = t / (b.seg_frames[seg_name] - 1)
    s = float(kin.smootherstep(u))
    n_arc = b.seg_frames[seg_name] - b.n_rake
    s_arc = float(kin.smootherstep(min(t / (n_arc - 1), 1.0)))
    w_fade = min(s_arc / 0.3, 1.0)
    tasks = [b.weld_task(ee, weld_anchor[ee]) for ee in ik.ee_names if ee != swing_ee]
    toe_w = float(np.clip((0.55 - u) / 0.15, 0.0, 1.0)) ** 2 if closing else 1.0
    if toe_w > 1.0e-3:
      for ee in feet:
        tasks.append(
          kin.toe_task(
            ik,
            ee,
            b.toe_wall_hold[ee],
            w_pos=(600.0 * toe_w, 3.0, 10.0),
          )
        )
    tasks.append(
      kin.IKTask(
        kind="site",
        sid=ik.pad_sid[swing_ee],
        pos=b.swing_target(t, b.seg_frames[seg_name], p0, p1, p2),
        rot=rot0,
        w_pos=np.full(3, 500.0 + 1500.0 * max(1.0 - w_fade, s_arc * s_arc)),
        w_rot=50.0 + 450.0 * max(1.0 - w_fade, s_arc * s_arc),
      )
    )
    base_target = (1.0 - s) * base_seg_start + s * base_seg_end
    tasks.append(
      kin.base_task(
        ik,
        base_target,
        (200.0, 100.0, 150.0),
        rot=kin.slerp_matrix(base_rot_start, wall.R, s),
        w_rot=60.0,
      )
    )
    posture_t = {}
    for nm in PHYBOT_C2_POLICY_JOINT_NAMES:
      if nm in swing_arm:
        # A middle-rung step has no pull toward the handoff pose: that pose
        # belongs to the final rung and drags a middle landing 29 mm off.
        target = b.state1_joint[nm] if (closing or pin_swing) else posture_start[nm]
      elif nm in support_arm or nm == "waist_yaw":
        target = posture_start[nm]
      else:
        # The legs walk toward the handoff hang in every step (feet welded,
        # knees extending); the toes stay held.
        target = b.state1_joint[nm]
      posture_t[nm] = (1.0 - s) * posture_start[nm] + s * target
    # The closing step pins its high support arm (weight 4) against
    # hyper-extension onto the shoulder stop. A non-closing step hangs from
    # a low support arm while the body rises: pinned, that arm freezes the
    # shoulder to the hand and the rise never happens (the base ends 60 mm
    # short), so it stays soft.
    overrides = {nm: (4.0 if closing else 1.0) for nm in support_arm}
    if not closing and pin_swing:
      # Pin the stepping arm to its handoff branch, or the closure pays
      # 0.8 rad of shoulder pitch.
      overrides.update({nm: recipe.reach_arm_pin for nm in swing_arm})
    if closing and u >= 0.42:
      for nm in ankle_pitches:
        if nm not in fold_from:
          fold_from[nm] = float(data.qpos[ik.qpos_addresses[nm]])
        w = float(kin.smootherstep(np.clip((u - 0.42) / 0.50, 0.0, 1.0)))
        posture_t[nm] = (1.0 - w) * fold_from[nm] + w * b.state1_joint[nm]
        overrides[nm] = 3.0
    throttled = t < 8
    res = ik.solve_frame(
      tasks,
      b.sel_all,
      posture_t,
      posture_weight=0.6,
      posture_weight_overrides=overrides,
      iters=2 if throttled else 100,
      max_step=0.003 if throttled else 0.12,
    )
    b.seg_residual[seg_name].merge(res)
    b.note_base(seg_name, base_target)
    b.push(seg_name, engaged)
  landed = ik.pad_pose(swing_ee)
  b.anchor_now[swing_ee] = landed[0].copy()
  b.landing_report[report_key or swing_ee] = (
    float(np.linalg.norm(landed[0] - p2)),
    kin.rotation_angle(landed[1], b.state1_pad_rot[swing_ee]),
  )


def hand_up_steps(b: MountBuilder) -> None:
  """The walk-up: the right hand steps to its final rung, then the left hand in two steps."""
  ik, data, wall, recipe = b.ik, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  base_lh = b.final_qpos[0:3] - 0.012 * wall.up + 0.002 * wall.outward
  base_lh[1] = float(data.qpos[1])
  # The right hand steps first, from its middle rung to its final one, on
  # the low left hand and both feet; the left hand then walks up from the
  # high right support.
  hand_up(b, "hand_up_right", rh, lh, base_lh, closing=False)
  b.dwell(
    "hand_up_hold",
    b.engaged(*ik.ee_names),
    weld_ees=ik.ee_names,
    hold_feet=(),
    toe_hold_targets=b.toe_wall_hold,
  )
  # The left hand's 30 cm walk-up splits at a middle rung, so each landing
  # keeps elbow flexion in reserve instead of arriving at the reach limit.
  mid_lh = b.final_anchor_pos[lh] + (recipe.hand_up_left_mid - recipe.hand_z) * wall.up
  mid_lh = mid_lh + np.array([0.0, recipe.hand_up_left_mid_y, 0.0])
  # The base sits ~5 cm toward the support side, and at the middle height
  # the left arm cannot cross back to its lane against the shoulder-roll
  # stop (a +24 mm lateral landing miss): the middle step walks the base
  # halfway back.
  base_mid = np.asarray(base_lh, dtype=np.float64).copy()
  base_mid[1] = 0.5 * base_mid[1]
  hand_up(
    b,
    "hand_up_left_mid",
    lh,
    rh,
    base_mid,
    closing=False,
    target_override=mid_lh,
    report_key=f"{lh}@mid_up",
    pin_swing=False,
  )
  b.dwell(
    "hand_up_left_hold",
    b.engaged(*ik.ee_names),
    weld_ees=ik.ee_names,
    hold_feet=(),
    toe_hold_targets=b.toe_wall_hold,
  )
  hand_up(b, "hand_up_left", lh, rh, b.final_qpos[0:3].copy(), closing=True)


def handoff(b: MountBuilder) -> None:
  """Blend onto the climb cycle's first frame and hold it through the handoff."""
  ik, data, model = b.ik, b.data, b.model

  # The blend spans half of the last hand step.
  closure_seg = "hand_up_left"
  delta = b.final_qpos - b.qpos_frames[-1]
  closure_frames = max(int(round(0.5 * b.seg_frames[closure_seg])), 2)
  for i in range(closure_frames):
    idx = len(b.qpos_frames) - closure_frames + i
    blend = float(kin.smootherstep((i + 1) / closure_frames))
    b.qpos_frames[idx] = b.qpos_frames[idx] + blend * delta
    b.qpos_frames[idx][3:7] /= np.linalg.norm(b.qpos_frames[idx][3:7])
  closure_pos = float(np.linalg.norm(delta[0:3]))
  closure_joint = float(np.abs(delta[7:]).max())
  closure_joint_name = "<none>"
  if delta[7:].size:
    _wj = int(np.argmax(np.abs(delta[7:])))
    for _nm, _adr in ((nm, ik.qpos_addresses[nm]) for nm in PHYBOT_C2_POLICY_JOINT_NAMES):
      if _adr == 7 + _wj:
        closure_joint_name = _nm
        break

  b.closure = (closure_pos, closure_joint, closure_joint_name)
  data.qpos[:] = b.final_qpos
  mujoco.mj_forward(model, data)
  for ee in ik.ee_names:
    b.anchor_now[ee] = ik.pad_pose(ee)[0].copy()
  # handoff_hold lies inside the playable region (the loader drops the
  # end_hold tail): a clock clamped here is a full segment away from the
  # last engage edge, so letting go is billed at full price.
  for _ in range(b.seg_frames["handoff_hold"]):
    b.push("handoff_hold", b.engaged(*ik.ee_names))
  for _ in range(b.seg_frames["end_hold"]):
    b.push("end_hold", b.engaged(*ik.ee_names))
