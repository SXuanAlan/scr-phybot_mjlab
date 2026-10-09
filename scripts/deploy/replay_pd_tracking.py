"""Replay a squat motion clip under real-PD control and measure the sim2real gap.

Unlike ``check_torque_saturation.py`` (which needs a trained policy), this drives
the robot open-loop by feeding the clip's joint angles as position targets to the
deploy PD law ``tau = kp*(q_target - q) - kd*qd`` (torque clamped to the real
robot limits), then steps physics against a ground plane.

To isolate the *actuation* question from the *balance* question, the base runs on
a virtual vertical gantry: its horizontal position and orientation are held on the
reference each step (so an open-loop, balance-free policy cannot simply tip over),
while its height is left free. The legs therefore bear the full body weight through
foot contact and vertical PD sag is real -- exactly the situation of a real-robot
squat test on a gantry/harness. It reports, per policy joint:

  * torque saturation vs the deploy limit (knee 80 N*m, hip_pitch 200 N*m, ...),
  * PD tracking error (finite stiffness sags under gravity at the squat bottom),
  * base-height sag vs the reference.

This front-loads the depth-vs-feasibility question before any RL training: sweep
``gen_squat_clip.py --depth`` and replay each clip to find the deepest squat the
robot can hold without saturating.

Usage:
    python scripts/deploy/replay_pd_tracking.py --motion-file src/scr_phybot_mjab/assets/motions/squat.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from scr_phybot_mjab.assets.motions import SQUAT_MOTION
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_DEPLOY_TORQUE_LIMITS,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
  get_spec,
)
from scr_phybot_mjab.tools.paths import ARTIFACTS_DIR

# Real-PD gains: nominal sim gains with the hip/knee damping override that the
# deployed walk task uses (PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP, kd 5 -> 10). Keyed
# by joint group (joint name with the left_/right_ prefix stripped).
_PD_GAINS_BY_GROUP = {
  "hip_pitch": (100.0, 10.0),
  "hip_roll": (100.0, 10.0),
  "hip_yaw": (100.0, 10.0),
  "knee": (100.0, 10.0),
  "ankle_pitch": (50.0, 5.0),
  "ankle_roll": (50.0, 5.0),
  "waist_yaw": (20.0, 10.0),
  "shoulder_pitch": (50.0, 5.0),
  "shoulder_roll": (50.0, 5.0),
  "shoulder_yaw": (50.0, 5.0),
  "elbow_pitch": (50.0, 5.0),
}


def _group_of(joint_name: str) -> str:
  return joint_name.removeprefix("left_").removeprefix("right_")


def _load_clip(path: Path) -> dict:
  data = np.load(path, allow_pickle=True).item()
  if list(data["joints_list"]) != list(PHYBOT_C2_POLICY_JOINT_NAMES):
    raise ValueError("clip joints_list does not match PHYBOT_C2_POLICY_JOINT_NAMES")
  return data


def _build_model() -> mujoco.MjModel:
  """Compile the C2 spec with a ground plane (get_spec drops the embedded floor)."""
  spec = get_spec()
  spec.worldbody.add_geom(
    name="pd_replay_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    pos=[0.0, 0.0, 0.0],
    friction=[1.0, 0.005, 0.0001],
  )
  return spec.compile()


def replay(motion_file: Path, output_dir: Path, settle_s: float, sat_threshold: float) -> None:
  clip = _load_clip(motion_file)
  ref_qpos = np.asarray(clip["joint_positions"], dtype=np.float64)  # (T, 21)
  ref_root_pos = np.asarray(clip["root_position"], dtype=np.float64)  # (T, 3)
  ref_root_quat_xyzw = np.asarray(clip["root_quaternion"], dtype=np.float64)  # (T, 4)
  # Gantry holds base orientation on the reference: convert xyzw -> wxyz.
  ref_root_quat = ref_root_quat_xyzw[:, [3, 0, 1, 2]]
  fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
  n_frames = ref_qpos.shape[0]

  model = _build_model()
  data = mujoco.MjData(model)
  dt = model.opt.timestep
  substeps = max(int(round((1.0 / fps) / dt)), 1)

  jname = PHYBOT_C2_POLICY_JOINT_NAMES
  qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jname])
  dadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jname])
  kp = np.array([_PD_GAINS_BY_GROUP[_group_of(n)][0] for n in jname])
  kd = np.array([_PD_GAINS_BY_GROUP[_group_of(n)][1] for n in jname])
  tau_lim = np.array([PHYBOT_C2_DEPLOY_TORQUE_LIMITS[_group_of(n)] for n in jname])

  # Initialize at the clip's first frame (standing).
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = ref_root_pos[0]
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = angle
  data.qpos[qadr] = ref_qpos[0]
  mujoco.mj_forward(model, data)

  settle_frames = int(round(settle_s * fps))
  total_frames = settle_frames + n_frames

  tau_log = np.zeros((total_frames, len(jname)))
  q_log = np.zeros((total_frames, len(jname)))
  qref_log = np.zeros((total_frames, len(jname)))
  base_z = np.zeros(total_frames)
  base_z_ref = np.zeros(total_frames)

  def _apply_gantry(idx: int) -> None:
    """Hold base xy + orientation on the reference; leave height (z) free."""
    data.qpos[0:2] = ref_root_pos[idx, 0:2]
    data.qpos[3:7] = ref_root_quat[idx]
    data.qvel[0:2] = 0.0
    data.qvel[3:6] = 0.0

  for f in range(total_frames):
    # Hold the first frame during the settle window, then play the clip.
    idx = 0 if f < settle_frames else f - settle_frames
    ref = ref_qpos[idx]
    q = data.qpos[qadr].copy()
    qd = data.qvel[dadr].copy()
    tau = np.clip(kp * (ref - q) - kd * qd, -tau_lim, tau_lim)
    for _ in range(substeps):
      data.qfrc_applied[:] = 0.0
      data.qfrc_applied[dadr] = tau
      mujoco.mj_step(model, data)
      _apply_gantry(idx)

    tau_log[f] = tau
    q_log[f] = data.qpos[qadr]
    qref_log[f] = ref
    base_z[f] = data.qpos[2]
    base_z_ref[f] = ref_root_pos[idx, 2]

  _report(
    jname,
    tau_log,
    q_log,
    qref_log,
    tau_lim,
    base_z,
    base_z_ref,
    settle_frames,
    sat_threshold,
    motion_file,
    output_dir,
    fps,
  )


def _report(
  jname,
  tau_log,
  q_log,
  qref_log,
  tau_lim,
  base_z,
  base_z_ref,
  settle_frames,
  sat_threshold,
  motion_file,
  output_dir,
  fps,
) -> None:
  ratio = np.abs(tau_log) / tau_lim  # (T, 21)
  track_err = np.abs(q_log - qref_log)

  print(f"\nmotion: {motion_file}")
  print(f"frames: {tau_log.shape[0]} @ {fps:.1f} fps  (settle {settle_frames} frames)")
  print(f"{'joint':<20}{'tau_lim':>9}{'max|t|/lim':>12}{'sat%':>7}{'track_rms(deg)':>16}")
  order = np.argsort(ratio.max(axis=0))[::-1]
  for i in order:
    sat_pct = 100.0 * np.mean(ratio[:, i] >= sat_threshold)
    print(
      f"{jname[i]:<20}{tau_lim[i]:>9.0f}{ratio[:, i].max():>12.2f}{sat_pct:>7.1f}"
      f"{np.degrees(np.sqrt(np.mean(track_err[:, i] ** 2))):>16.2f}"
    )
  worst = np.unravel_index(np.argmax(ratio), ratio.shape)
  print(f"\npeak torque ratio {ratio[worst]:.2f} at {jname[worst[1]]} ({'SATURATED' if ratio[worst] >= 1.0 else 'ok'})")
  ref_z_min = float(base_z_ref.min())
  achieved_min = float(base_z[settle_frames:].min())
  print(
    f"base height: reference bottom {ref_z_min:.3f} m, achieved bottom "
    f"{achieved_min:.3f} m  (PD sag {ref_z_min - achieved_min:+.3f} m)"
  )

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("(matplotlib unavailable; skipping plots)")
    return

  output_dir.mkdir(parents=True, exist_ok=True)
  t = np.arange(tau_log.shape[0]) / fps
  fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
  for i in order[:6]:
    axes[0].plot(t, ratio[:, i], label=jname[i])
    axes[1].plot(t, np.degrees(track_err[:, i]), label=jname[i])
  axes[0].axhline(1.0, color="r", ls="--", lw=1)
  axes[0].set_ylabel("|tau| / limit")
  axes[0].legend(fontsize=8, ncol=3)
  axes[1].set_ylabel("tracking error (deg)")
  axes[1].set_xlabel("time (s)")
  fig.suptitle(f"PD tracking replay: {motion_file.name}")
  out = output_dir / f"{motion_file.stem}_pd_tracking.png"
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"saved {out}")


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion-file",
    type=Path,
    default=SQUAT_MOTION,
  )
  parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "pd_tracking")
  parser.add_argument("--settle-s", type=float, default=1.0, help="hold frame 0 before playback (s)")
  parser.add_argument("--sat-threshold", type=float, default=0.95)
  args = parser.parse_args(argv)

  replay(args.motion_file, args.output_dir, args.settle_s, args.sat_threshold)


if __name__ == "__main__":
  main()
