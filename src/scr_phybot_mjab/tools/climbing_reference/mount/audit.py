"""The mount clip's audit, printed: IK quality, clearances, support margins and joint limits."""

from __future__ import annotations

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tools.climbing_reference import kinematics as kin

from .builder import SEGMENT_NAMES, MountBuilder


def report(b: MountBuilder, clip: dict) -> None:
  """Audit the solved frames in the native tilted world and print the findings.

  Read the printout before committing a regenerated clip: it is the only
  record of weld drift, swing-pad wall penetration, clearances and joint
  limit margins.
  """
  ik, model, data, wall, recipe = b.ik, b.model, b.data, b.wall, b.recipe
  lh, rh, lf, rf = ik.ee_names
  feet = (lf, rf)
  frames = len(b.qpos_frames)
  engaged_arr = clip["ee_engaged"]
  ground_arr = clip["foot_on_ground"]
  label_arr = clip["phase_segment"]
  joint_positions = clip["joint_positions"]
  root_position = clip["root_position"]
  anchor_arr = np.stack(b.anchor_frames).astype(np.float32)
  pad_positions = np.zeros((frames, 4, 3), np.float32)
  # Clearance sweeps against the wall plane and the floor, over COLLISION
  # geoms only (visual meshes have contype=conaffinity=0 and never contact
  # anything in sim; their conservative AABBs read ~1 cm through the wall
  # even in the climb cycle's own stance). The FOOT (ankle_roll) geoms are
  # exempt from the wall sweep: whenever a toe touches the wall the foot
  # mesh AABB corner reads ~3 cm "through", while the true mesh tip stops
  # 2 mm short of the toe sphere by design (the sphere protrudes exactly so
  # wall contact lands on the pair'd sphere — phybot_c2_constants note).
  pad_or_toe_geoms = set()
  visual_geoms = set()
  # Shank (ankle_pitch) geoms are exempt with the feet: the spine pads
  # stand off the shin by their 12 mm sphere radius, so a welded pad puts
  # the shank's AABB corner within ~1 mm of the plane by construction.
  shank_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_pitch") for side in ("left", "right")}
  all_foot_geoms = set()
  for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "_spine_c_" in name or "toe" in name:
      pad_or_toe_geoms.add(g)
    if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
      visual_geoms.add(g)
    if model.geom_bodyid[g] in b.foot_body_bid.values() or model.geom_bodyid[g] in shank_bids:
      all_foot_geoms.add(g)
  foot_geoms = {ee: {g for g in range(model.ngeom) if model.geom_bodyid[g] == b.foot_body_bid[ee]} for ee in feet}
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
  mass = model.body_mass.sum()
  com_single: dict[str, list[float]] = {}  # per segment: [min lateral toward stance, max ahead of toe]

  for t, q in enumerate(b.qpos_frames):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    for i, ee in enumerate(ik.ee_names):
      pad_positions[t, i] = data.site_xpos[ik.pad_sid[ee]]
      if engaged_arr[t, i]:
        max_weld_drift = max(
          max_weld_drift,
          float(np.linalg.norm(pad_positions[t, i] - anchor_arr[t, i])),
        )
      else:
        for bid_s in ik.sphere_bids[ee]:
          # Penetration past the pad-touch offset plane (one sphere
          # radius short of the tilted surface).
          pen = (kin.WALL_SURFACE_X - kin.PAD_TOUCH_X) - wall.wall_distance(data.xpos[bid_s])
          if pen > max_wall_pen:
            max_wall_pen_at = (t, ee)
          max_wall_pen = max(max_wall_pen, pen)
          if pen > 0.0:
            wall_pen_hits.append((float(pen), int(t), ee))

    seg_name_t = SEGMENT_NAMES[label_arr[t]]
    if seg_name_t in ("hand_up_left", "hand_up_hold", "hand_up_right"):
      for ee in feet:
        gap = wall.wall_distance(data.site_xpos[ik.toe_sid[ee]]) - ik.toes[ee].radius
        key = f"{ee}@{seg_name_t}"
        b.toe_gap_report[key] = max(b.toe_gap_report.get(key, -np.inf), gap)
    grounded_geoms = set()
    for i, ee in enumerate(feet):
      # A foot on the floor legitimately touches it; a foot whose pad is
      # ENGAGED on the wall stands 4 cm over the floor at the wall base —
      # its proximity is a stance fact, not a swing scrape.
      if ground_arr[t, i] or engaged_arr[t, 2 + i]:
        grounded_geoms |= foot_geoms[ee]
    for g in range(model.ngeom):
      if model.geom_bodyid[g] == 0 or g in visual_geoms:
        continue
      center = model.geom_aabb[g, :3]
      half = model.geom_aabb[g, 3:]
      world = data.geom_xpos[g] + (center + half * corner_signs) @ data.geom_xmat[g].reshape(3, 3).T
      if g not in pad_or_toe_geoms and g not in all_foot_geoms:
        clear = float(((world - wall.pivot) @ wall.outward).min())
        if clear < wall_clear_min[0]:
          wall_clear_min = (
            clear,
            t,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}",
          )
      if g not in grounded_geoms:
        clear = float(world[:, 2].min()) - b.ground_z
        if clear < ground_clear_min[0]:
          ground_clear_min = (
            clear,
            t,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}",
          )
    if int(ground_arr[t].sum()) == 1:
      # Single-foot floor support: the CoM's lateral offset toward the stance
      # side (+ = over or beyond the foot) and its lead over the toe front
      # (+ = tipping over the toe).
      com_w = (model.body_mass[:, None] * data.xipos).sum(axis=0) / mass
      i_st = int(np.flatnonzero(ground_arr[t])[0])
      sole = data.site_xpos[ik.foot_sid[i_st]]
      dy = float(com_w[1] - sole[1]) * (1.0 if i_st == 0 else -1.0)
      tw = -np.array([wall.outward[0], wall.outward[1], 0.0])
      tw = tw / max(np.linalg.norm(tw), 1.0e-9)
      toe_front = float(data.site_xpos[ik.toe_sid[feet[i_st]]] @ tw + ik.toes[feet[i_st]].radius)
      dx = float(com_w @ tw) - toe_front
      rec = com_single.setdefault(SEGMENT_NAMES[label_arr[t]], [np.inf, -np.inf])
      rec[0] = min(rec[0], dy)
      rec[1] = max(rec[1], dx)
    if ground_arr[t].all() and not engaged_arr[t].any():
      com_x = float((model.body_mass[:, None] * data.xipos).sum(axis=0)[0] / mass)
      toe_x = max(float(data.site_xpos[ik.toe_sid[ee]][0] + ik.toes[ee].radius) for ee in feet)
      heel_x = min(float(data.xpos[b.foot_body_bid[ee]][0] - 0.06) for ee in feet)
      margin = min(toe_x - com_x, com_x - heel_x)
      if margin < com_margin_min[0]:
        com_margin_min = (margin, t)

  closure_pos, closure_joint, closure_joint_name = b.closure
  print(
    f"mount clip: {frames} frames, {frames / recipe.fps:.2f}s @ {recipe.fps:.1f} fps, "
    f"dz={b.dz:+.4f} m: hands {recipe.hand_z_low:.3f} -> {recipe.hand_z:.3f}, feet "
    f"{b.final_anchor_pos[lf][2]:.3f}"
  )
  for name in SEGMENT_NAMES:
    r = b.seg_residual[name]
    if r.pos > 0.0 or r.rot > 0.0:
      print(f"  {name:18s} IK residual: pos {r.pos * 1000.0:6.2f} mm, rot {np.degrees(r.rot):5.2f} deg")
  for key, (pos_err, rot_err) in b.landing_report.items():
    print(f"  {key:22s} landing vs scheduled anchor: {pos_err * 1000.0:6.2f} mm, {np.degrees(rot_err):5.2f} deg")
  for ee, gap in b.toe_gap_report.items():
    print(
      f"  toe {ee:18s} wall-touch gap {gap * 1000.0:+6.2f} mm "
      "(must be within +0.5 mm for the static audit to credit the brace)"
    )
  joint_speed = np.abs(np.diff(joint_positions, axis=0)) * recipe.fps
  peak_frame, peak_joint = np.unravel_index(np.argmax(joint_speed), joint_speed.shape)
  base_speed = float(np.abs(np.diff(root_position, axis=0)).max() * recipe.fps)
  print(
    f"  peak joint speed {joint_speed.max():.2f} rad/s "
    f"({PHYBOT_C2_POLICY_JOINT_NAMES[int(peak_joint)]} @f{int(peak_frame)} "
    f"{SEGMENT_NAMES[label_arr[int(peak_frame)]]}), peak base speed "
    f"{base_speed:.3f} m/s"
  )
  if joint_speed.max() > 8.0:
    print(
      "    note: a one-frame reference step this fast usually sits at the"
      " toe-pivot leg's bistable fold; check the guided fold before"
      " accepting it."
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
    frames_in_seg = np.flatnonzero(label_arr == b.seg_id[seg_name])
    last = int(frames_in_seg[-1])
    nr = b.seg_rake[seg_name]
    drop = float(clip["pad_positions"][last - nr, i, 2] - clip["pad_positions"][last, i, 2])
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
  for seg_name_b, err in b.base_track.items():
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
      "    note: with one hand high and one low, the wristless 4-DoF arm "
      "puts the low arm's shoulder roll on its stop; no base pose relieves "
      "it, so a short transient is expected."
    )
  # Handoff contract: the final frame is the climb cycle's first frame shifted by dz.
  handoff_joint = float(
    np.abs(joint_positions[-1] - np.array([b.state1_joint[nm] for nm in PHYBOT_C2_POLICY_JOINT_NAMES])).max()
  )
  print(f"  handoff joint delta vs state-1  {handoff_joint:.2e} rad (must be ~0)")
  if max_weld_drift > 5.0e-3 or max_wall_pen > 2.0e-3:
    print("  WARNING: IK quality below tolerance; inspect the render.")
