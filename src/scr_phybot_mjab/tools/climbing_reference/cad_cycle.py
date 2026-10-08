"""The CAD climbing reference: the four-pad stance, its cycle, and the projection onto the pad faces.

``solve_initial_stance`` solves the climbing stance on the CAD pads,
``cycle.generate_clip`` builds the cycle from it, and
``project_to_pad_faces`` re-solves every frame so the pads sit exactly on
their flat faces, with the passive gimbals free. This is offline inverse
kinematics, not policy optimization or a rollout.
"""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_XML, get_spec
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry
from scr_phybot_mjab.tools.climbing_reference.kinematics import (
  ClimbIK,
  IKTask,
  base_task,
  orientation_error_world,
  toe_task,
)

__all__ = ("project_to_pad_faces", "solve_initial_stance", "with_variant_metadata")


def solve_initial_stance() -> dict:
  """The four-pad climbing stance on the CAD pads, as the initial-state payload.

  The solve starts from a fixed seed, so it depends on neither a previously
  generated stance nor the current reference clip.
  """
  ik = ClimbIK()
  ik.init_joint_pos = {n: v for n, v in ik.init_joint_pos.items() if "_gimbal_" not in n}
  ik.init_base_pos = np.array([1.327, 0.0, 1.0])
  for side, sign in (("left", 1), ("right", -1)):
    for joint, angle in dict(
      hip_pitch=-0.4592,
      hip_roll=0.1607 * sign,
      hip_yaw=-0.1584 * sign,
      knee=0.4198,
      ankle_pitch=1.256,
      ankle_roll=0.0,
      shoulder_pitch=-2.5999,
      shoulder_roll=-0.166 * sign,
      shoulder_yaw=0.0,
      elbow_pitch=-0.5386,
    ).items():
      ik.init_joint_pos[f"{side}_{joint}"] = angle
  for name in ("waist_yaw", "neck_yaw", "neck_pitch"):
    ik.init_joint_pos[name] = 0.0
  ik.set_state1()
  m, d = ik.model, ik.data
  names = list(ik.init_joint_pos)
  adr = np.array([ik.qpos_addresses[n] for n in names])
  q0 = d.qpos.copy()
  poses = {ee: ik.pad_pose(ee) for ee in ik.ee_names}
  # The CAD B-to-C offset loads the lower pitch stop under gravity and
  # seating. Model that unilateral load path explicitly: a freely rotating
  # ball at zero cannot carry the stance's moments.
  passive_initial = {}
  for ee in ik.ee_names:
    for axis in ("x", "y", "z"):
      name = f"{ee}_gimbal_{axis}"
      value = float(m.joint(name).range[0]) if axis == "x" else 0.0
      q0[ik.qpos_addresses[name]] = value
      passive_initial[name] = value
  targets = {ee: p.copy() for ee, (p, _) in poses.items()}
  rotations = {}
  for ee, p in targets.items():
    p[0] = 1.5
    if "hand" in ee:
      p[2] = 1.83  # soft ergonomic target; exact face/plane constraints dominate
    z = poses[ee][1][:, 2].copy()
    z[0] = 0
    z /= np.linalg.norm(z)
    normal = np.array([-1.0, 0, 0])
    rotations[ee] = np.stack((np.cross(normal, z), normal, z), axis=1)
  x0 = np.r_[1.345, q0[adr]]
  lower = np.r_[1.30, [ik.joint_ranges[n][0] + 0.02 for n in names]]
  upper = np.r_[1.40, [ik.joint_ranges[n][1] - 0.02 for n in names]]

  def residual(x):
    d.qpos[:] = q0
    d.qpos[0] = x[0]
    d.qpos[adr] = x[1:]
    mujoco.mj_forward(m, d)
    error = []
    for ee in ik.ee_names:
      p, rot = ik.pad_pose(ee)
      error.extend((p - targets[ee]) * [10000, 20, 20])
      error.extend((rot[:, 1] - [-1.0, 0, 0]) * 10000)
      error.extend(orientation_error_world(rot, rotations[ee]) * 2)
    error.extend((x - x0) * 0.2)
    error.append((x[0] - 1.345) * 30)
    return error

  sol = least_squares(
    residual, np.clip(x0, lower, upper), bounds=(lower, upper), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-9
  )
  residual(sol.x)
  centers = {ee: ik.pad_pose(ee)[0].tolist() for ee in ik.ee_names}
  max_gap = max(abs(p[0] - 1.5) for p in centers.values())
  if not sol.success or max_gap > 0.00015:
    raise RuntimeError(f"CAD stance failed: success={sol.success}, gap={max_gap}")
  payload = dict(
    schema_version=1,
    reference_point="C",
    wall_surface_x=1.5,
    root_position=d.qpos[:3].tolist(),
    root_quaternion_wxyz=d.qpos[3:7].tolist(),
    joint_positions={**dict(zip(names, sol.x[1:].tolist())), **passive_initial},
    pad_centers_world_m=centers,
    max_center_wall_error_m=max_gap,
    gimbal_angles="x at lower -3 degree stop; y,z zero; passive RSI state, never actuator targets",
    generator="scripts/reference/gen_cad_climb_reference.py",
    protocol_note=(
      "Four seeded mature contacts with lower-stop support. "
      "Stop strength, seating and dynamic stability are not hardware validated"
    ),
  )
  return payload


def project_to_pad_faces(clip: dict, initial_state: dict | None = None) -> dict:
  """Enforce actual flat box faces and fixed support poses with passive DOFs.

  Raw weighted IK can leave millimetre corner penetration and its final
  joint-space closure can move support anchors. Re-solve geometric targets,
  including passive gimbals, instead of translating the collision geometry.
  """
  # Fresh arrays, as a saved and reloaded clip has: the projection writes into them.
  clip = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in clip.items()}
  ik = ClimbIK(limit_margin_frac=0.08, limit_weight=3.0, initial_state=initial_state)
  ik.set_state1()
  m, d = ik.model, ik.data
  passive = tuple(n for n in ik.qpos_addresses if "_gimbal_" in n)
  fixed_passive = tuple(n for n in passive if n.endswith("_gimbal_x"))
  all_names = tuple(clip["joints_list"]) + tuple(n for n in passive if n not in fixed_passive)
  sel = ik.dof_selection(all_names)
  for n in passive:
    ik.joint_ranges[n] = np.radians([-3.0, 3.0])  # solver retains a 0.005 rad stop margin
    ik.limit_weight_overrides[n] = 50.0
  q_initial = d.qpos.copy()
  base_centers = {ee: ik.pad_pose(ee)[0] for ee in ik.ee_names}
  rotations = {ee: ik.pad_pose(ee)[1] for ee in ik.ee_names}
  for ee in ik.ee_names:
    base_centers[ee][0] = 1.5
    # Exact flat face; retain each pad's natural in-plane rake orientation.
    z = rotations[ee][:, 2].copy()
    z[0] = 0
    z /= np.linalg.norm(z)
    n = np.array([-1.0, 0, 0])
    rotations[ee] = np.stack((np.cross(n, z), n, z), axis=1)
  fps, height = float(clip["fps"]), float(clip["step_height"])
  phases = np.asarray(clip["phase_segment"])
  segment_names = list(clip["segment_names"])
  bounds = {}
  for i, ee in enumerate(ik.ee_names):
    # A hand-swing phase begins with all-four-contact load transfer. Its
    # moving target starts at the actual OFF edge, not at the phase label.
    indices = np.flatnonzero(~clip["ee_engaged"][:, i].astype(bool))
    bounds[ee] = int(indices[0]), int(indices[-1])
  passive_out, qframes = [], []
  max_error, max_angle = 0.0, 0.0
  for frame in range(len(phases)):
    raw = q_initial.copy()
    raw[:3] = clip["root_position"][frame]
    raw[3:7] = clip["root_quaternion"][frame, [3, 0, 1, 2]]
    for j, name in enumerate(clip["joints_list"]):
      raw[ik.qpos_addresses[name]] = clip["joint_positions"][frame, j]
    # Use the previous solved pose as a continuous warm start; first/last
    # dwell remain exactly the generated initial pose and its height shift.
    if not qframes:
      d.qpos[:] = raw
    target_positions, tasks = [], []
    for i, ee in enumerate(ik.ee_names):
      first, last = bounds[ee]
      p = base_centers[ee].copy()
      if frame >= last:
        p[2] += height
      elif frame >= first:
        n_rake = round(0.2 * fps)
        arc_end = last - n_rake
        if frame <= arc_end:
          t = (frame - first) / max(arc_end - first, 1)
          s = t * t * t * (10 + t * (-15 + 6 * t))
          p[0] -= 0.005 * np.sin(np.pi * s) ** 2
          p[2] += (height + 0.002) * s
        else:
          t = (frame - arc_end) / n_rake
          s = t * t * t * (10 + t * (-15 + 6 * t))
          p[2] += height + 0.002 * (1 - s)
      target_positions.append(p)
      tasks.append(
        IKTask(
          kind="site", sid=ik.pad_sid[ee], pos=p, rot=rotations[ee], w_pos=np.full(3, 1000.0), w_rot=1000.0, hard=True
        )
      )
      anchor = base_centers[ee].copy()
      if frame >= first:
        anchor[2] += height
      clip["ee_anchor_positions"][frame, i] = anchor
    posture = {n: float(raw[ik.qpos_addresses[n]]) for n in all_names}
    tasks.append(base_task(ik, raw[:3], (30.0, 30.0, 30.0), w_rot=15.0))
    # Toe pressing is a unilateral support target, never a hidden pad weld.
    for i, (ee, toe) in enumerate(ik.toes.items()):
      if clip["toe_braced"][frame, i]:
        saved = d.qpos.copy()
        d.qpos[:] = raw
        mujoco.mj_forward(m, d)
        target = d.site_xpos[ik.toe_sid[ee]].copy()
        target[0] = 1.5 - toe.radius
        d.qpos[:] = saved
        tasks.append(toe_task(ik, ee, target, w_pos=(1000.0, 1.0, 1.0)))
    if segment_names[int(phases[frame])] in ("hold_start", "end_hold"):
      # The cycle's weighted IK moves even unselected gimbals off their stop.
      # Hold the solved stance exactly here; only the kinematically
      # independent ankles perform the toe handoff. Copying the raw pose
      # would leave a 0.43 mm gap before the first swing starts.
      d.qpos[:] = q_initial
      if segment_names[int(phases[frame])] == "end_hold":
        d.qpos[2] += height
      for name in clip["joints_list"]:
        if "_ankle_" in name:
          d.qpos[ik.qpos_addresses[name]] = raw[ik.qpos_addresses[name]]
      mujoco.mj_forward(m, d)
    else:
      ik.solve_frame(
        tasks,
        sel,
        posture,
        posture_weight=0.8,
        iters=140,
        tol_pos=1e-6,
        tol_rot=1e-5,
        tol_dq=1e-5,
        lam2=0.01,
        max_step=0.025,
      )
      # Project with joint bounds INSIDE the linear solve. Clipping an
      # unconstrained damped step at a gimbal stop leaks into support rows.
      for _ in range(20):
        mujoco.mj_forward(m, d)
        jacobians, errors = [], []
        for ee, p in zip(ik.ee_names, target_positions):
          now_p, now_r = ik.pad_pose(ee)
          jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
          mujoco.mj_jacSite(m, d, jp, jr, ik.pad_sid[ee])
          jacobians.extend((jp[:, sel], jr[:, sel]))
          errors.extend((p - now_p, orientation_error_world(now_r, rotations[ee])))
        error = np.concatenate(errors)
        if np.max(np.abs(error)) < 5e-7:
          break
        lo, hi = np.full(len(sel), -0.05), np.full(len(sel), 0.05)
        for j, name in enumerate(all_names, start=6):
          low, high = ik.joint_ranges[name]
          current = d.qpos[ik.qpos_addresses[name]]
          lo[j] = max(lo[j], low + 0.005 - current)
          hi[j] = min(hi[j], high - 0.005 - current)
        step = lsq_linear(np.vstack(jacobians), error, bounds=(lo, hi), method="bvls", tol=1e-11, max_iter=100)
        velocity = np.zeros(m.nv)
        velocity[sel] = step.x
        mujoco.mj_integratePos(m, d.qpos, velocity, 1.0)
      mujoco.mj_forward(m, d)
    max_error = max(
      max_error, max(np.linalg.norm(ik.pad_pose(ee)[0] - p) for ee, p in zip(ik.ee_names, target_positions))
    )
    qframes.append(d.qpos.copy())
    passive_out.append([d.qpos[ik.qpos_addresses[n]] for n in passive])
    max_angle = max(max_angle, max(abs(v) for v in passive_out[-1]))
    clip["root_position"][frame] = d.qpos[:3]
    clip["root_quaternion"][frame] = d.qpos[[4, 5, 6, 3]]
    clip["joint_positions"][frame] = [d.qpos[ik.qpos_addresses[n]] for n in clip["joints_list"]]
    for i, ee in enumerate(ik.ee_names):
      bid = ik.ee_bid[ee]
      clip["ee_positions"][frame, i] = d.xpos[bid]
      clip["ee_quaternions"][frame, i] = d.xquat[bid, [1, 2, 3, 0]]
      clip["pad_positions"][frame, i] = d.site_xpos[ik.pad_sid[ee]]
    for i, sid in enumerate(ik.foot_sid):
      clip["foot_positions"][frame, i] = d.site_xpos[sid]
    if frame % 100 == 0:
      print(f"CAD face projection {frame}/{len(phases)}, max target error {max_error * 1000:.4f} mm", flush=True)
  clip["schema_version"] = 2
  clip["passive_joint_names"] = passive
  clip["passive_joint_positions"] = np.asarray(passive_out, dtype=np.float32)
  clip["contact_reference_point"] = "C"
  clip["gimbal_stop_support"] = "lower x stop throughout; unilateral stop strength unvalidated"
  clip["generator"] = "scripts/reference/gen_cad_climb_reference.py"
  clip["offline_projection_max_error_m"] = max_error
  clip["asset_xml_sha256"] = hashlib.sha256(PHYBOT_C2_XML.read_bytes()).hexdigest()
  clip["engage_unit_preload_n"] = 10.0
  clip["preload_validation"] = "Kinematic reference; contact-wrench preload must be verified separately"
  print(f"Refined: max target error {max_error * 1000:.5f} mm, gimbal {np.degrees(max_angle):.3f} deg", flush=True)
  if max_error > 0.00015:
    raise RuntimeError("Reference target tolerance failed; candidate is not geometrically accepted")
  return clip


def with_variant_metadata(clip: dict, wall_angle: float, unit_layout: str) -> dict:
  """Keep the canonical trajectory; the runtime/render/audit rotate the wall.

  Unit activation changes load sharing, not CAD poses or hardware mass.
  Feasibility must be audited for each selected angle/layout separately.
  """
  geometry = matching_pad_geometry(get_spec(), clip["ee_list"], unit_layout=unit_layout)
  clip["wall_angle"] = float(wall_angle)
  clip["unit_layout"] = unit_layout
  clip["active_units_per_pad"] = geometry.count
  clip["active_unit_sites"] = geometry.unit_names
  clip["engage_unit_preload_n"] = 10.0
  clip["preload_validation"] = "Kinematic reference; separately audit the selected wall angle and active units"
  return clip
