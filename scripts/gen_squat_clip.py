"""Generate a PHYBot C2 squat motion clip (with arm front-raise) via lower-body IK.

Choreography: stand -> raise arms to a forward horizontal front-raise -> descend ->
hold -> ascend -> lower arms back to the sides -> stand. The default clip pads the
motion with three seconds of the robot's zero-command standing pose at both ends.
This gives a newly selected policy time to absorb observation/state mismatch before
the squat begins and leaves a stationary tail for deployment to hold after standing
back up.

The reference is produced directly on the robot's own kinematics (no human mocap /
retargeting): the two foot sites are pinned to their standing footprints and kept
flat, a smooth base-height curve drives the pelvis down and back up, per-frame
damped least-squares IK solves the 12 leg joints, and the arms (shoulder_pitch /
elbow) follow a scripted raise/lower curve. The arm pose is set before the leg IK
so the CoM balance accounts for the arms-forward mass shift. The squat RL task
tracks both the legs and the arms so the policy reproduces this choreography.

The output format matches the AMP/motion `.npy` schema (same as
``gen_stand_clip.py`` and ``scripts/validation/check_amp_dataset.py``):

    joints_list       list[str]        21 policy joints, PHYBOT_C2 order
    joint_positions   (T, 21) float32  rad
    root_position     (T, 3)  float32  world xyz (standing base z ~= 0.726)
    root_quaternion   (T, 4)  float32  xyzw (SciPy convention)
    foot_sites_list   list[str]        left/right foot site order
    foot_positions    (T, 2, 3) float32 fixed motion-local world xyz
    fps               float32

Validate/preview the result with:
    python scripts/validation/check_amp_dataset.py --data-dir <output dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Initialize MJLab task entry points before importing PHYBot constants. Otherwise
# MJLab discovers this package while the constants module is only half initialized.
import mjlab  # noqa: F401
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_FOOT_SITES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
  get_spec,
)

# Standing base height: the lowest sole point sits on the floor in the -0.295
# ankle pose (see DEFAULT_KEYFRAME / _WALK_STANDING_KEYFRAME, base z 0.726).
_STANDING_BASE_Z = 0.726
# Leg joints are the first 12 policy joints (2 legs x 6). The IK solves these.
_LEG_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[:12]
# The remaining 9 policy joints (waist + 8 arm) are prescribed per frame by the
# arm-raise choreography (not solved by IK).
_UPPER_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[12:]
_FOOT_SITE_NAMES = tuple(site for _, site in PHYBOT_C2_FOOT_SITES)

# Forearms held out in front and parallel to the ground, with bent elbows. The
# forearm is the elbow_pitch link (its mesh runs along that body's local -z). FK
# shows that with the elbow bent to _ARM_RAISE_ELBOW, shoulder_pitch
# _ARM_RAISE_SHOULDER makes the forearm level and pointing forward. shoulder_roll/
# yaw and waist stay at their standing values.
_ARM_RAISE_SHOULDER = -0.6
_ARM_RAISE_ELBOW = -0.9


def _standing_upper_pose() -> dict[str, float]:
  return {name: PHYBOT_C2_STANDING_JOINT_POS[name] for name in _UPPER_JOINT_NAMES}


def _front_raise_pose(base_pitch: float = 0.0) -> dict[str, float]:
  """Forearms out in front, horizontal in the WORLD frame, with bent elbows.

  ``shoulder_pitch`` is relative to the (forward-pitching) torso, so subtract the
  base pitch to keep the forearms parallel to the ground as the torso leans forward
  during the squat. shoulder_pitch and elbow_pitch rotate about parallel axes, so
  offsetting shoulder_pitch by ``-base_pitch`` holds the forearm orientation.
  """
  pose = _standing_upper_pose()
  for side in ("left", "right"):
    pose[f"{side}_shoulder_pitch"] = _ARM_RAISE_SHOULDER - base_pitch
    pose[f"{side}_elbow_pitch"] = _ARM_RAISE_ELBOW
  return pose


def _smootherstep(x: np.ndarray) -> np.ndarray:
  """C2-continuous 0->1 ramp with zero end velocity and acceleration."""
  x = np.clip(x, 0.0, 1.0)
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _build_curves(
  stand_frames: int, reach_frames: int, descent_frames: int, hold_frames: int
) -> tuple[np.ndarray, np.ndarray]:
  """Squat-depth and arm-reach fractions (both in [0, 1]) over the timeline.

  Timeline: stand | raise arms | descend | hold | ascend | lower arms | stand.
  The legs stay standing (depth 0) while the arms raise/lower, and the arms stay
  raised (reach 1) through the whole squat. Both curves start and end at 0 with
  zero velocity (smootherstep) so the clip loops cleanly.
  """
  ramp = _smootherstep(np.arange(descent_frames) / max(descent_frames - 1, 1))
  reach_ramp = _smootherstep(np.arange(reach_frames) / max(reach_frames - 1, 1))
  depth = np.concatenate(
    [
      np.zeros(stand_frames),
      np.zeros(reach_frames),
      ramp,
      np.ones(hold_frames),
      ramp[::-1],
      np.zeros(reach_frames),
      np.zeros(stand_frames),
    ]
  )
  reach = np.concatenate(
    [
      np.zeros(stand_frames),
      reach_ramp,
      np.ones(descent_frames),
      np.ones(hold_frames),
      np.ones(descent_frames),
      reach_ramp[::-1],
      np.zeros(stand_frames),
    ]
  )
  return depth, reach


def _pitch_quat_wxyz(pitch: float) -> np.ndarray:
  """Quaternion (w, x, y, z) for a forward torso pitch about +y."""
  half = 0.5 * pitch
  return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


def _orientation_error_w(rot_cur: np.ndarray, rot_tar: np.ndarray) -> np.ndarray:
  """World-frame small-angle error rotating ``rot_cur`` toward ``rot_tar``.

  Consistent with ``mj_jacSite``'s rotational Jacobian (dq -> world angular
  velocity), so it can be stacked with the positional error in one IK solve.
  """
  return 0.5 * (
    np.cross(rot_cur[:, 0], rot_tar[:, 0])
    + np.cross(rot_cur[:, 1], rot_tar[:, 1])
    + np.cross(rot_cur[:, 2], rot_tar[:, 2])
  )


class _SquatIK:
  """Per-frame leg IK against fixed, flat foot targets."""

  def __init__(self, damping: float = 0.05, max_iters: int = 40, tol: float = 1e-4):
    self.model = get_spec().compile()
    self.data = mujoco.MjData(self.model)
    self.damping = damping
    self.max_iters = max_iters
    self.tol = tol

    m = self.model
    self._qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _all_joint_names(m)}
    self._range = {
      n: np.array(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in _all_joint_names(m)
    }
    self._leg_dof = np.array(
      [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _LEG_JOINT_NAMES]
    )
    self._site_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in _FOOT_SITE_NAMES]

    # Free-joint base body (for whole-robot CoM used in balancing).
    free_jid = int(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0])
    self._root_body = int(m.jnt_bodyid[free_jid])

    self._set_standing()
    mujoco.mj_forward(m, self.data)
    # Freeze the standing footprints and (flat) foot orientations as IK targets.
    self.foot_pos_tar = [self.data.site_xpos[i].copy() for i in self._site_ids]
    self.foot_rot_tar = [self.data.site_xmat[i].reshape(3, 3).copy() for i in self._site_ids]
    self.foot_center_x = float(np.mean([p[0] for p in self.foot_pos_tar]))

  def _set_standing(self) -> None:
    d = self.data
    d.qpos[:] = 0.0
    d.qpos[0:3] = (0.0, 0.0, _STANDING_BASE_Z)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
      d.qpos[self._qadr[name]] = angle

  def _set_base(self, x: float, z: float, pitch: float) -> None:
    self.data.qpos[0:3] = (x, 0.0, z)
    self.data.qpos[3:7] = _pitch_quat_wxyz(pitch)

  def _foot_error(self) -> np.ndarray:
    m, d = self.model, self.data
    err = np.zeros(12)
    for k, sid in enumerate(self._site_ids):
      pos = d.site_xpos[sid]
      rot = d.site_xmat[sid].reshape(3, 3)
      err[6 * k : 6 * k + 3] = self.foot_pos_tar[k] - pos
      err[6 * k + 3 : 6 * k + 6] = _orientation_error_w(rot, self.foot_rot_tar[k])
    return err

  def _solve_legs(self) -> float:
    m, d = self.model, self.data
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(self.max_iters):
      mujoco.mj_forward(m, d)
      err = self._foot_error()
      if np.linalg.norm(err) < self.tol:
        break
      rows = []
      for sid in self._site_ids:
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        rows.append(jacp[:, self._leg_dof].copy())
        rows.append(jacr[:, self._leg_dof].copy())
      jac = np.vstack(rows)  # (12, 12)
      # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e.
      lam2 = self.damping**2
      dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(12), err)
      for i, name in enumerate(_LEG_JOINT_NAMES):
        adr = self._qadr[name]
        lo, hi = self._range[name]
        d.qpos[adr] = np.clip(d.qpos[adr] + dq[i], lo, hi)
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(self._foot_error()))

  def solve_frame(
    self,
    base_x: float,
    base_z: float,
    pitch: float,
    upper_angles: dict[str, float],
    balance: bool,
    balance_iters: int,
  ) -> tuple[np.ndarray, float, float]:
    """Solve one frame; returns (policy joint angles (21,), base_x used, residual).

    ``upper_angles`` prescribes the non-leg policy joints (waist + arms) for this
    frame; they are set before the leg IK so the CoM balance sees the arms-forward
    mass shift.
    """
    for name, angle in upper_angles.items():
      self.data.qpos[self._qadr[name]] = angle
    x = base_x
    residual = 0.0
    for _ in range(max(balance_iters, 1)):
      self._set_base(x, base_z, pitch)
      residual = self._solve_legs()
      if not balance:
        break
      com_x = float(self.data.subtree_com[self._root_body][0])
      err_x = com_x - self.foot_center_x
      if abs(err_x) < 1e-3:
        break
      # Shifting the base back moves the CoM back; damp to keep it stable.
      x -= 0.8 * err_x
    joints = np.array([self.data.qpos[self._qadr[n]] for n in PHYBOT_C2_POLICY_JOINT_NAMES], dtype=np.float32)
    return joints, x, residual


def _all_joint_names(model: mujoco.MjModel) -> list[str]:
  names = []
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
  return names


def generate_clip(
  output: Path,
  depth_m: float,
  stand_s: float,
  descent_s: float,
  hold_s: float,
  reach_s: float,
  fps: float,
  torso_lean: float,
  balance: bool,
) -> None:
  stand_frames = int(round(stand_s * fps))
  reach_frames = max(int(round(reach_s * fps)), 2)
  descent_frames = max(int(round(descent_s * fps)), 2)
  hold_frames = int(round(hold_s * fps))
  frames = stand_frames + reach_frames + descent_frames + hold_frames + descent_frames + reach_frames + stand_frames
  if frames < 2:
    raise ValueError("clip must have at least two frames")

  depth, reach = _build_curves(stand_frames, reach_frames, descent_frames, hold_frames)
  base_z = _STANDING_BASE_Z - depth_m * depth
  pitch = torso_lean * depth

  stand_upper = _standing_upper_pose()

  ik = _SquatIK()
  joint_positions = np.zeros((frames, len(PHYBOT_C2_POLICY_JOINT_NAMES)), dtype=np.float32)
  root_position = np.zeros((frames, 3), dtype=np.float32)
  root_quaternion = np.zeros((frames, 4), dtype=np.float32)

  # ``balance_x`` is the fully CoM-balanced pelvis shift.  Fade that shift in with
  # the C2-continuous motion curves so frame zero remains the robot's exact
  # zero-command standing pose instead of jumping immediately to a different IK
  # solution.  A second IK solve at the faded x keeps both feet pinned.
  balance_x = 0.0
  max_residual = 0.0
  for t in range(frames):
    r = float(reach[t])
    # Compensate the torso lean so the raised forearms stay world-horizontal.
    raise_upper = _front_raise_pose(float(pitch[t]))
    upper_angles = {n: (1.0 - r) * stand_upper[n] + r * raise_upper[n] for n in _UPPER_JOINT_NAMES}
    if balance:
      _, balance_x, balance_residual = ik.solve_frame(
        balance_x,
        float(base_z[t]),
        float(pitch[t]),
        upper_angles,
        balance=True,
        balance_iters=6,
      )
      activity = max(float(depth[t]), r)
      frame_x = activity * balance_x
      joints, _, residual = ik.solve_frame(
        frame_x,
        float(base_z[t]),
        float(pitch[t]),
        upper_angles,
        balance=False,
        balance_iters=1,
      )
      residual = max(residual, balance_residual)
    else:
      frame_x = 0.0
      joints, _, residual = ik.solve_frame(
        frame_x,
        float(base_z[t]),
        float(pitch[t]),
        upper_angles,
        balance=False,
        balance_iters=1,
      )
    joint_positions[t] = joints
    root_position[t] = (frame_x, 0.0, base_z[t])
    q_wxyz = _pitch_quat_wxyz(float(pitch[t]))
    # Store xyzw (SciPy convention) to match the AMP/motion .npy schema.
    root_quaternion[t] = (q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0])
    max_residual = max(max_residual, residual)

  # Make every stationary frame exact, rather than relying on the IK tolerance.
  # Even a converged standing IK solution differs from the deployed zero-command
  # pose by about 1e-3 rad, which would turn a nominal hold into a small command
  # step at task selection. The neck is intentionally absent because it is not a
  # policy/motion joint.
  zero_pos = np.asarray(
    [PHYBOT_C2_STANDING_JOINT_POS[name] for name in PHYBOT_C2_POLICY_JOINT_NAMES],
    dtype=np.float32,
  )
  standing = (depth == 0.0) & (reach == 0.0)
  joint_positions[standing] = zero_pos
  root_position[standing] = (0.0, 0.0, _STANDING_BASE_Z)
  root_quaternion[standing] = (0.0, 0.0, 0.0, 1.0)

  data = {
    "joints_list": list(PHYBOT_C2_POLICY_JOINT_NAMES),
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "foot_sites_list": list(_FOOT_SITE_NAMES),
    "foot_positions": np.repeat(np.asarray(ik.foot_pos_tar, dtype=np.float32)[None, :, :], frames, axis=0),
    "fps": np.float32(fps),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data)

  knee = PHYBOT_C2_POLICY_JOINT_NAMES.index("left_knee")
  shoulder = PHYBOT_C2_POLICY_JOINT_NAMES.index("left_shoulder_pitch")
  print(
    f"wrote {output} ({frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps)\n"
    f"  depth={depth_m:.3f} m, base_z [{base_z.min():.3f}, {base_z.max():.3f}], "
    f"left_knee [{joint_positions[:, knee].min():.3f}, {joint_positions[:, knee].max():.3f}] rad\n"
    f"  left_shoulder_pitch [{joint_positions[:, shoulder].min():.3f}, "
    f"{joint_positions[:, shoulder].max():.3f}] rad (arm raise)\n"
    f"  max foot IK residual={max_residual:.2e} m (large -> depth beyond leg reach)"
  )
  if max_residual > 1e-2:
    print("  WARNING: foot IK did not converge; reduce --depth or check foot targets.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=_REPO / "src/scr_phybot_mjab/assets/motions/squat.npy",
  )
  parser.add_argument("--depth", type=float, default=0.22, help="pelvis drop at squat bottom (m)")
  parser.add_argument(
    "--stand-s",
    type=float,
    default=3.0,
    help="standing hold padded before and after the squat (s)",
  )
  parser.add_argument("--descent-s", type=float, default=2.0, help="descend (and ascend) duration (s)")
  parser.add_argument("--hold-s", type=float, default=1.0, help="hold at squat bottom (s)")
  parser.add_argument("--reach-s", type=float, default=1.0, help="arm raise (and lower) duration (s)")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--torso-lean", type=float, default=0.35, help="forward torso pitch at squat bottom (rad)")
  parser.add_argument(
    "--no-balance",
    dest="balance",
    action="store_false",
    help="disable CoM-over-feet pelvis balancing",
  )
  args = parser.parse_args()

  generate_clip(
    output=args.output,
    depth_m=args.depth,
    stand_s=args.stand_s,
    descent_s=args.descent_s,
    hold_s=args.hold_s,
    reach_s=args.reach_s,
    fps=args.fps,
    torso_lean=args.torso_lean,
    balance=args.balance,
  )


if __name__ == "__main__":
  main()
