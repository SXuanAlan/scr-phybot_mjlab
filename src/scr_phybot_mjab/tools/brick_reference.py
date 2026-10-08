"""Robot-native, planted-foot brick reference synthesis and its geometric audit.

The IK is a geometric guide, not a claim that PD replay supplies preload or grip.
No reference frame switches a physical attachment on or off.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_DEPLOY_TORQUE_LIMITS,
  PHYBOT_C2_PAD_GEOMETRY,
  get_spec,
)
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES as JOINTS,
)
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_STANDING_JOINT_POS as STANDING,
)
from scr_phybot_mjab.tasks.brick.geometry import DEFAULT_UNIT_LAYOUT, active_unit_count, brick_robot_collision
from scr_phybot_mjab.tasks.brick.reference import PHASE_NAMES, REFERENCE_PATH, load_reference

__all__ = ("BrickIK", "audit_reference", "generate_reference")

DURATIONS = np.array([5.0, 2.0, 3.0, 4.0, 3.0, 2.0, 4.0])


def smooth(t):
  t = np.clip(t, 0.0, 1.0)
  return t**3 * (10 + t * (-15 + 6 * t))


class BrickIK:
  """Coupled whole-body IK, retaining the actual +/-3 degree pad travel."""

  def __init__(self, brick=BrickCfg(), unit_layout=DEFAULT_UNIT_LAYOUT):
    self.brick = brick
    self.unit_layout = unit_layout
    self.preload_force = 10.0 * active_unit_count(unit_layout)
    spec = get_spec()
    brick_robot_collision().edit_spec(spec)
    spec.worldbody.add_geom(name="reference_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.1])
    block = spec.worldbody.add_body(name="reference_brick")
    block.add_freejoint()
    block.add_geom(
      name="reference_brick_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=np.asarray(brick.size) / 2,
      mass=brick.mass,
      contype=0,
      conaffinity=0,
      rgba=[0.62, 0.30, 0.16, 1.0],
    )
    self.model = m = spec.compile()
    self.data = d = mujoco.MjData(m)
    self.joints = tuple(JOINTS)
    self.gimbals = tuple(f"{s}_hand_spine_gimbal_{a}" for s in ("left", "right") for a in "xyz")
    self.joint_ids = [m.joint(n).id for n in self.joints + self.gimbals]
    self.arm_names = tuple(n for n in JOINTS if "shoulder" in n or "elbow" in n)
    self.arm_ids = np.array([m.joint(n).id for n in self.arm_names])
    self.arm_limits = np.array([PHYBOT_C2_DEPLOY_TORQUE_LIMITS[n.split("_", 1)[1]] for n in self.arm_names])
    self.qids = np.array([m.jnt_qposadr[i] for i in self.joint_ids])
    self.foot_ids = [m.site(s + "_foot").id for s in ("left", "right")]
    self.hand_ids = [m.body(s + "_hand_spine").id for s in ("left", "right")]
    self.block_id = m.body("reference_brick").id
    self.block_qadr = int(m.jnt_qposadr[m.body_jntadr[self.block_id]])
    self.torso_id = m.body("torso").id
    self.ground_geom = m.geom("reference_ground").id
    self.brick_geom = m.geom("reference_brick_geom").id
    self.brick_obstacles = [
      g
      for g in range(m.ngeom)
      if (m.geom_contype[g] or m.geom_conaffinity[g])
      and g != self.ground_geom
      and "hand_spine" not in m.geom(g).name
      and "foot_spine" not in m.geom(g).name
    ]
    self.pad_center = np.asarray(PHYBOT_C2_PAD_GEOMETRY["contact_center_in_pad_m"])
    self.standing = np.r_[0, 0, 0.726, 0, 0, 0, [STANDING[n] for n in self.joints], np.zeros(6)]
    self.lower = np.r_[[-0.45, -0.25, 0.22, -0.3, -0.15, -0.7], m.jnt_range[self.joint_ids, 0] + 1e-4]
    self.upper = np.r_[[0.4, 0.25, 0.85, 0.3, 1.55, 0.7], m.jnt_range[self.joint_ids, 1] - 1e-4]
    self.set_pose(self.standing)
    self.feet = d.site_xpos[self.foot_ids].copy()
    self.foot_rot = d.site_xmat[self.foot_ids].reshape(2, 3, 3).copy()
    self.home_hands, self.home_normals, self.home_rakes = self.hands()
    self.robot_mass = float(m.body_subtreemass[self.torso_id])

  def set_pose(self, x, block_pos=None):
    d = self.data
    q = Rotation.from_euler("xyz", x[3:6]).as_quat()
    d.qpos[:3], d.qpos[3:7], d.qpos[self.qids] = x[:3], q[[3, 0, 1, 2]], x[6:]
    if block_pos is not None:
      d.qpos[self.block_qadr : self.block_qadr + 7] = np.r_[block_pos, 1.0, 0, 0, 0]
    mujoco.mj_kinematics(self.model, d)
    mujoco.mj_comPos(self.model, d)

  def hands(self):
    d = self.data
    r = d.xmat[self.hand_ids].reshape(2, 3, 3)
    return d.xpos[self.hand_ids] + np.einsum("nij,j->ni", r, self.pad_center), -r[:, :, 1], -r[:, :, 2]

  def arm_force_torques(self, forces):
    points = np.repeat(self.hands()[0], 4, axis=0)
    moments = np.cross(points - self.data.xanchor[self.arm_ids], np.repeat(forces, 4, axis=0))
    return (moments * self.data.xaxis[self.arm_ids]).sum(-1)

  def collision_residual(self):
    m, d = self.model, self.data
    mujoco.mj_collision(m, d)
    depth = np.zeros(m.ngeom)
    for contact in d.contact:
      if self.ground_geom in contact.geom and any("ankle_roll_collision" in m.geom(int(g)).name for g in contact.geom):
        continue
      np.maximum.at(depth, contact.geom, max(0.001 - contact.dist, 0.0))
    for g in self.brick_obstacles:
      depth[g] = max(depth[g], 0.002 - mujoco.mj_geomDistance(m, d, g, self.brick_geom, 0.01, None))
    return depth * 250.0

  def solve(self, positions, normals, rakes, block, seed, *, attempts=1, preload_weight=0.0, refine=False):
    def residual(x):
      self.set_pose(x, block)
      p, n, rake = self.hands()
      frot = self.data.site_xmat[self.foot_ids].reshape(2, 3, 3)
      ferr = Rotation.from_matrix(np.einsum("nij,nkj->nik", frot, self.foot_rot)).as_rotvec()
      com = (self.robot_mass * self.data.subtree_com[self.torso_id, :2] + self.brick.mass * block[:2]) / (
        self.robot_mass + self.brick.mass
      )
      outside = np.maximum(com - [0.12, 0.17], 0) + np.minimum(com - [-0.035, -0.17], 0)
      # Necessary active-arm effort check for active units x 10 N, reserving
      # 3 N*m for gravity. This does not certify dynamic preload.
      press_torque = self.arm_force_torques(np.array([[0.0, self.preload_force, 0.0], [0.0, -self.preload_force, 0.0]]))
      effort_excess = np.maximum(np.abs(press_torque) - (self.arm_limits - 3.0), 0)
      foot_weight, hand_weight, normal_weight = (240, 200, 12) if refine else (120, 100, 24)
      return np.r_[
        (self.data.site_xpos[self.foot_ids] - self.feet).ravel() * foot_weight,
        ferr.ravel() * 10,
        (p - positions).ravel() * hand_weight,
        (n - normals).ravel() * normal_weight,
        np.minimum((rake * rakes).sum(-1) - 0.7, 0) * 8,
        outside * 80,
        preload_weight * effort_excess * 0.5,
        self.collision_residual(),
        (x[6:27] - seed[6:27]) * 0.002,
        x[27:] * 0.02,
      ]

    rng = np.random.default_rng(17)
    best = None
    for i in range(attempts):
      initial = seed if i == 0 else seed + rng.normal(0, 0.08, len(seed))
      result = least_squares(
        residual,
        np.clip(initial, self.lower + 1e-6, self.upper - 1e-6),
        bounds=(self.lower, self.upper),
        max_nfev=180,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
      )
      if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
        best = result
    self.set_pose(best.x, block)
    return best.x

  def audit(self, poses, blocks, desired_hands, desired_normals):
    errors, foot_errors, normal_errors, contact_errors, collisions, margins = [], [], [], [], [], []
    contact_hand_errors = []
    ground_depth, self_depth = 0.0, 0.0
    worst_self_pair = []
    arm_ratios = []
    m, d = self.model, self.data
    # Check brick overlap with all non-hand robot collision geoms. Visuals have
    # zero masks. Intended hand contact and inactive foot plates are excluded.
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    brick_geom = m.geom("reference_brick_geom").id
    ground_geom = m.geom("reference_ground").id
    for frame, (x, block, hp, hn) in enumerate(zip(poses, blocks, desired_hands, desired_normals)):
      self.set_pose(x, block)
      mujoco.mj_forward(m, d)
      for contact in d.contact:
        names = [m.geom(int(i)).name for i in contact.geom]
        if ground_geom in contact.geom:
          if any("ankle_roll_collision" in name for name in names):
            continue
          ground_depth = max(ground_depth, -float(contact.dist))
        elif -contact.dist > self_depth:
          self_depth = -float(contact.dist)
          worst_self_pair = names
      p, n, _ = self.hands()
      errors.append(float(np.max(np.linalg.norm(p - hp, axis=1))))
      foot_errors.append(float(np.max(np.linalg.norm(d.site_xpos[self.foot_ids] - self.feet, axis=1))))
      normal_errors.append(float(np.max(np.arccos(np.clip((n * hn).sum(-1), -1, 1)))))
      intended_gap = (hp[:, 1] - block[1]) * np.array([1.0, -1.0]) - self.brick.size[1] / 2
      if np.max(intended_gap) <= 0.005:
        contact_errors.append(normal_errors[-1])
        contact_hand_errors.append(errors[-1])
      phase = frame // (len(poses) // len(PHASE_NAMES))
      if phase == 1 and np.max(intended_gap) <= 0.003:
        forces = np.array([[0.0, self.preload_force, 0.0], [0.0, -self.preload_force, 0.0]])
        torque = d.qfrc_bias[m.jnt_dofadr[self.arm_ids]] - self.arm_force_torques(forces)
        arm_ratios.append(float(np.max(np.abs(torque) / self.arm_limits)))
      com = (self.robot_mass * d.subtree_com[self.torso_id, :2] + self.brick.mass * block[:2]) / (
        self.robot_mass + self.brick.mass
      )
      margins.append(float(min(com[0] + 0.05, 0.14 - com[0], 0.20 - abs(com[1]))))
      depth = 0.0
      for gid in geoms:
        name = m.geom(gid).name or ""
        if "hand_spine" in name or "foot_spine" in name or gid == ground_geom:
          continue
        depth = max(depth, -float(mujoco.mj_geomDistance(m, d, gid, brick_geom, 0.01, None)))
      collisions.append(depth)
    report = {
      "max_hand_error_m": max(errors),
      "max_foot_error_m": max(foot_errors),
      "max_contact_hand_error_m": max(contact_hand_errors, default=0.0),
      "max_face_error_deg": float(np.rad2deg(max(normal_errors))),
      "max_brick_penetration_m": max(collisions),
      "max_contact_face_error_deg": float(np.rad2deg(max(contact_errors, default=0.0))),
      "min_support_margin_m": min(margins),
      "frames": len(poses),
      "max_illegal_ground_penetration_m": ground_depth,
      "max_self_penetration_m": self_depth,
      "worst_self_pair": worst_self_pair,
      "max_preload_arm_effort_ratio": max(arm_ratios, default=0.0),
      "unit_layout": self.unit_layout,
      "preload_force_per_pad_n": self.preload_force,
      "preload_audit": (
        "Quasi-static active-arm torque with gravity and configured normal preload per pad; "
        "zero pad moment. Necessary check only."
      ),
      "kind": "kinematic_only",
      "dynamic_preload_validated": False,
    }
    report["passed"] = (
      report["max_hand_error_m"] < 0.005
      and report["max_contact_hand_error_m"] < 0.005
      and report["max_foot_error_m"] < 0.003
      and report["max_contact_face_error_deg"] < 3
      and report["max_brick_penetration_m"] < 0.002
      and ground_depth < 0.0005
      and self_depth < 0.002
      and report["max_preload_arm_effort_ratio"] < 1.0
      and report["min_support_margin_m"] > 0
    )
    return report


def generate_reference(output=REFERENCE_PATH, brick=BrickCfg(), knots_per_phase=21, unit_layout=DEFAULT_UNIT_LAYOUT):
  if knots_per_phase < 2:
    raise ValueError("At least two knots per phase are required")
  ik = BrickIK(brick, unit_layout)
  start = np.array([*brick.start_xy, brick.size[2] / 2 + 0.001])
  raised = start + [0, 0, brick.lift_height]
  goal = start + [*brick.goal_offset_xy, 0.0]
  raised_goal = goal + [0, 0, brick.lift_height]
  face = np.array([[0.0, -1, 0], [0.0, 1, 0]])
  rake = np.array([[0.0, 0, 1], [0.0, 0, 1]])
  offsets = np.array([[0, brick.size[1] / 2, 0], [0, -brick.size[1] / 2, 0]])
  contact = start + offsets
  pre = contact - 0.025 * face
  # Each phase is an object/hand program. The preload target lies 2 mm into
  # the surface; the policy must generate the actual measured per-unit preload.
  hand_keys = [
    ik.home_hands,
    pre,
    contact + 0.002 * face,
    raised + offsets,
    raised_goal + offsets,
    goal + offsets,
    goal + offsets - 0.035 * face + [0, 0, -0.006],
    ik.home_hands,
  ]
  block_keys = [start, start, start, raised, raised_goal, goal, goal, goal]
  normal_keys = [ik.home_normals, face, face, face, face, face, face, ik.home_normals]
  rake_keys = [ik.home_rakes, rake, rake, rake, rake, rake, rake, ik.home_rakes]
  phases, roots, gimbals, joints, blocks, targets, normals = [], [], [], [], [], [], []
  x = ik.standing.copy()
  # A deep forward reach seed avoids the straight-arm local IK minimum.
  reach_seed = x.copy()
  reach_seed[:6] = [-0.12, 0.0, 0.39, 0.0, 0.95, 0.0]
  for side, sign in (("left", 1), ("right", -1)):
    for suffix, value in (
      ("hip_pitch", -2.0),
      ("knee", 2.3),
      ("ankle_pitch", -1.0),
      ("shoulder_pitch", -0.6),
      ("shoulder_roll", 0.3 * sign),
      ("shoulder_yaw", 1.5 * sign),
      ("elbow_pitch", -0.3),
    ):
      reach_seed[6 + JOINTS.index(side + "_" + suffix)] = value
  grasp_pose = ik.solve(pre, face, rake, start, reach_seed, attempts=4, preload_weight=1.0)
  for phase in range(len(PHASE_NAMES)):
    for k in range(knots_per_phase):
      u = k / (knots_per_phase - 1)
      s = smooth(u)
      hp = (1 - s) * hand_keys[phase] + s * hand_keys[phase + 1]
      if phase == 5:
        hp = goal + offsets + [0, 0, -0.006 * smooth(u / 0.35)] - 0.035 * smooth((u - 0.35) / 0.65) * face
      hn = (1 - s) * normal_keys[phase] + s * normal_keys[phase + 1]
      hn /= np.linalg.norm(hn, axis=-1, keepdims=True)
      hr = (1 - s) * rake_keys[phase] + s * rake_keys[phase + 1]
      hr /= np.linalg.norm(hr, axis=-1, keepdims=True)
      bp = (1 - s) * block_keys[phase] + s * block_keys[phase + 1]
      if phase == 0:
        seed = (1 - s) * ik.standing + s * grasp_pose
      elif phase == 6:
        seed = (1 - s) * x + s * ik.standing
      else:
        seed = x
      preload_weight = s**4 if phase == 0 else 1.0 if phase == 1 else 1.0 - s if phase == 2 else 0.0
      x = ik.solve(hp, hn, hr, bp, seed, preload_weight=preload_weight)
      if phase == 0 and k == 0 or phase == 6 and k == knots_per_phase - 1:
        x = ik.standing.copy()
      phases.append(x.copy())
      quat = Rotation.from_euler("xyz", x[3:6]).as_quat()[[3, 0, 1, 2]]
      roots.append(np.r_[x[:3], quat])
      joints.append(x[6:27])
      gimbals.append(x[27:])
      blocks.append(bp)
      targets.append(hp)
      normals.append(hn)
    print(f"reference: {PHASE_NAMES[phase]} solved", flush=True)
  # Refine each established IK branch. Restart the local solve when a frame
  # still misses a positional tolerance; contact-set changes can terminate an
  # otherwise converged least-squares pass before the position has settled.
  for j, x in enumerate(phases):
    phase, k = divmod(j, knots_per_phase)
    s = smooth(k / (knots_per_phase - 1))
    hr = (1 - s) * rake_keys[phase] + s * rake_keys[phase + 1]
    hr /= np.linalg.norm(hr, axis=-1, keepdims=True)
    weight = s**4 if phase == 0 else 1.0 if phase == 1 else 1.0 - s if phase == 2 else 0.0
    for _ in range(3):
      x = ik.solve(targets[j], normals[j], hr, blocks[j], x, preload_weight=weight, refine=True)
      hand_error = np.linalg.norm(ik.hands()[0] - targets[j], axis=-1).max()
      foot_error = np.linalg.norm(ik.data.site_xpos[ik.foot_ids] - ik.feet, axis=-1).max()
      if hand_error < 0.0045 and foot_error < 0.002:
        break
    if j in (0, len(phases) - 1):
      x = ik.standing.copy()
    phases[j] = x.copy()
    roots[j] = np.r_[x[:3], Rotation.from_euler("xyz", x[3:6]).as_quat()[[3, 0, 1, 2]]]
    joints[j], gimbals[j] = x[6:27], x[27:]
  report = ik.audit(phases, blocks, targets, normals)
  output = Path(output)
  if not report["passed"]:
    output = output.with_name(output.stem + ".failed.npz")
    print(f"Reference failed audit; preserving any existing reference. Candidate: {output}", flush=True)
  output.parent.mkdir(parents=True, exist_ok=True)
  shape = (len(PHASE_NAMES), knots_per_phase)
  np.savez_compressed(
    output,
    schema_version=1,
    joint_names=np.array(JOINTS),
    phase_names=np.array(PHASE_NAMES),
    durations=DURATIONS,
    joint_pos=np.asarray(joints, np.float32).reshape(*shape, 21),
    root_pose=np.asarray(roots, np.float32).reshape(*shape, 7),
    gimbal_pos=np.asarray(gimbals, np.float32).reshape(*shape, 6),
    brick_pos=np.asarray(blocks, np.float32).reshape(*shape, 3),
    hand_pos=np.asarray(targets, np.float32).reshape(*shape, 2, 3),
    hand_normal=np.asarray(normals, np.float32).reshape(*shape, 2, 3),
    brick_size=brick.size,
    brick_mass=brick.mass,
    unit_layout=unit_layout,
    unit_preload_n=10.0,
    audit_json=np.array(json.dumps(report)),
  )
  output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
  return report


def audit_reference(path=REFERENCE_PATH):
  """Recompute the audit from stored poses, without running IK again."""
  ref = load_reference(path, require_audit=False)
  start, goal = ref["brick_pos"][0, 0], ref["brick_pos"][-1, -1]
  brick = BrickCfg(
    size=tuple(ref["brick_size"]),
    mass=float(ref["brick_mass"]),
    start_xy=tuple(start[:2]),
    goal_offset_xy=tuple(goal[:2] - start[:2]),
    lift_height=float(ref["brick_pos"][2, -1, 2] - start[2]),
  )
  ik = BrickIK(brick, str(ref["unit_layout"]))
  root = ref["root_pose"].reshape(-1, 7)
  angles = Rotation.from_quat(root[:, [4, 5, 6, 3]]).as_euler("xyz")
  poses = np.c_[root[:, :3], angles, ref["joint_pos"].reshape(-1, 21), ref["gimbal_pos"].reshape(-1, 6)]
  report = ik.audit(
    poses, ref["brick_pos"].reshape(-1, 3), ref["hand_pos"].reshape(-1, 2, 3), ref["hand_normal"].reshape(-1, 2, 3)
  )
  ref["audit_json"] = np.array(json.dumps(report))
  np.savez_compressed(path, **ref)
  Path(path).with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
  return report
