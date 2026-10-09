"""Overlap a real-robot walk_amp run against an open-loop MuJoCo replay.

Reads a bdml_phybot_deploy_c2 run log (data.lcmlog, exported to state.csv via
the deploy repo's log2csv), takes the 500 Hz stream of PD position targets
(``m*_pos_des``) that the policy actually commanded on the robot, replays
those targets open-loop through the deploy PD law
``tau = kp*(q_target - q) - kd*qd`` (clamped to the deploy torque limits)
on the C2 MuJoCo model, and overlays sim q(t) with the measured real q(t)
per joint. Reports per-joint RMSE and divergence time, plus dq/torque pages.

The sim is initialized from the real state at the window start (joint pos/vel
from the log, base orientation/angular velocity from the IMU, base height
fitted so the lowest collider touches the floor, base linear velocity
estimated by pinning the stance-foot contacts). Walking is unstable: sim and
real fork after the first
out-of-sync contact, so expect meaningful overlap for the first ~0.5-2 s of a
walk window (much longer for stand/squat windows) -- the divergence time
itself is the sim2real metric.

Usage:
    python scripts/deploy/compare_real_sim.py --run-dir <robot_logs/20260712_..._real>
    python scripts/deploy/compare_real_sim.py --csv state.csv --t-start 2 --duration 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_DEPLOY_TORQUE_LIMITS,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
  get_spec,
)
from scr_phybot_mjab.tools import deploy_logs
from scr_phybot_mjab.tools.paths import ARTIFACTS_DIR, DEPLOY_REPO_DIR

# Real-robot PD gains (motor firmware, backends/real/config/phybot_mini_*.yaml
# PDKp/PDKd) keyed by joint group -- same table as replay_pd_tracking.py.
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

_LEG_IDX = list(range(12))
_UPPER_IDX = list(range(12, 21))


def _group_of(joint_name: str) -> str:
  return joint_name.removeprefix("left_").removeprefix("right_")


def build_model(friction: float) -> mujoco.MjModel:
  spec = get_spec()
  spec.worldbody.add_geom(
    name="replay_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    pos=[0.0, 0.0, 0.0],
    friction=[friction, 0.005, 0.0001],
  )
  return spec.compile()


def quat_wxyz_from_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
  """R = Rz(yaw) Ry(pitch) Rx(roll) as a wxyz quaternion."""
  cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
  cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
  cr, sr = np.cos(roll / 2), np.sin(roll / 2)
  return np.array(
    [
      cy * cp * cr + sy * sp * sr,
      cy * cp * sr - sy * sp * cr,
      cy * sp * cr + sy * cp * sr,
      sy * cp * cr - cy * sp * sr,
    ]
  )


def fit_base_height(model: mujoco.MjModel, data: mujoco.MjData, floor_gid: int) -> None:
  """Shift base z so the lowest colliding geom exactly touches the floor."""
  mujoco.mj_forward(model, data)
  dists = []
  for gid in range(model.ngeom):
    if gid == floor_gid:
      continue
    if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
      continue
    dists.append(mujoco.mj_geomDistance(model, data, floor_gid, gid, 1.0, None))
  data.qpos[2] += 1e-4 - min(dists)
  mujoco.mj_forward(model, data)


def pin_base_linvel_to_contacts(model: mujoco.MjModel, data: mujoco.MjData, floor_gid: int) -> bool:
  """Set base lin vel so the mean world velocity of floor contacts is zero.

  The log has no base velocity, but whatever touches the ground is (nearly)
  stationary on the real robot too, so the stance-foot kinematic pin is the
  same estimate a deploy state estimator would make. Returns False (and
  leaves the zero init) when nothing touches the floor.
  """
  mujoco.mj_forward(model, data)
  jacp = np.zeros((3, model.nv))
  vels = []
  for i in range(data.ncon):
    c = data.contact[i]
    if floor_gid not in (c.geom1, c.geom2):
      continue
    other = c.geom2 if c.geom1 == floor_gid else c.geom1
    mujoco.mj_jac(model, data, jacp, None, c.pos, model.geom_bodyid[other])
    vels.append(jacp @ data.qvel)
  if not vels:
    return False
  data.qvel[0:3] -= np.mean(vels, axis=0)  # jacp[:, 0:3] is identity
  mujoco.mj_forward(model, data)
  return True


def replay(
  log: deploy_logs.RealLog, w0: int, w1: int, physics_dt: float, friction: float, settle_ticks: int
) -> tuple[np.ndarray, np.ndarray]:
  """Open-loop PD replay of q_des rows [w0, w1); returns (q_sim, tau_sim)."""
  model = build_model(friction)
  model.opt.timestep = physics_dt
  substeps = max(int(round(deploy_logs.TICK_DT / physics_dt)), 1)
  data = mujoco.MjData(model)

  jname = PHYBOT_C2_POLICY_JOINT_NAMES
  jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jname]
  qadr = np.array([model.jnt_qposadr[j] for j in jid])
  dadr = np.array([model.jnt_dofadr[j] for j in jid])
  kp = np.array([_PD_GAINS_BY_GROUP[_group_of(n)][0] for n in jname])
  kd = np.array([_PD_GAINS_BY_GROUP[_group_of(n)][1] for n in jname])
  tau_lim = np.array([PHYBOT_C2_DEPLOY_TORQUE_LIMITS[_group_of(n)] for n in jname])

  # Initial state = real state at the window start. Yaw is irrelevant to the
  # dynamics (flat floor) and drifts arbitrarily on the IMU, so zero it.
  mujoco.mj_resetData(model, data)
  for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[j]] = angle
  data.qpos[qadr] = log.q[w0]
  data.qvel[dadr] = log.dq[w0]
  data.qpos[0:3] = (0.0, 0.0, 0.726)
  _, pitch, roll = log.imu_zyx[w0]
  data.qpos[3:7] = quat_wxyz_from_zyx(0.0, pitch, roll)
  data.qvel[0:3] = 0.0  # base lin vel is not observable from the log
  data.qvel[3:6] = log.imu_gyro[w0]  # free-joint ang vel is body-local
  floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "replay_floor")
  fit_base_height(model, data, floor_gid)

  # A loaded stance penetrates the soft contact by ~load/stiffness, so the
  # zero-penetration placement above starts the feet unloaded and the ensuing
  # drop transient alone is enough to fork a balancing policy. Let base z
  # settle under gravity while clamping every state we actually know from the
  # log (joints, base attitude and angular rate) to the window-start row.
  quat0 = data.qpos[3:7].copy()
  for _ in range(settle_ticks):
    tau = np.clip(
      kp * (log.q_des[w0] - data.qpos[qadr]) - kd * data.qvel[dadr],
      -tau_lim,
      tau_lim,
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dadr] = tau
    mujoco.mj_step(model, data)
    data.qpos[qadr] = log.q[w0]
    data.qvel[dadr] = log.dq[w0]
    data.qpos[0:2] = 0.0
    data.qpos[3:7] = quat0
    data.qvel[0:2] = 0.0
    data.qvel[3:6] = log.imu_gyro[w0]
  data.qvel[0:3] = 0.0
  if not pin_base_linvel_to_contacts(model, data, floor_gid):
    print("warning: no floor contact at window start; base lin vel starts at 0")

  n = w1 - w0
  q_sim = np.zeros((n, deploy_logs.N_DOFS))
  tau_sim = np.zeros((n, deploy_logs.N_DOFS))
  for s in range(n):
    q_sim[s] = data.qpos[qadr]  # state at row time, before this tick's torque
    target = log.q_des[w0 + s]
    for _ in range(substeps):
      tau = np.clip(
        kp * (target - data.qpos[qadr]) - kd * data.qvel[dadr],
        -tau_lim,
        tau_lim,
      )
      data.qfrc_applied[:] = 0.0
      data.qfrc_applied[dadr] = tau
      mujoco.mj_step(model, data)
    tau_sim[s] = tau
  return q_sim, tau_sim


def divergence_time(err: np.ndarray, thr: float, sustain_ticks: int = 25) -> float:
  """First time |err| stays above thr for sustain_ticks consecutive rows."""
  over = np.abs(err) > thr
  if len(over) < sustain_ticks:
    return np.nan
  runs = np.convolve(over.astype(int), np.ones(sustain_ticks, int), "valid")
  idx = np.flatnonzero(runs == sustain_ticks)
  return float(idx[0] * deploy_logs.TICK_DT) if len(idx) else np.nan


def report(
  log: deploy_logs.RealLog,
  w0: int,
  w1: int,
  q_sim: np.ndarray,
  tau_sim: np.ndarray,
  thr_deg: float,
  out_dir: Path,
  run_name: str,
) -> None:
  jname = PHYBOT_C2_POLICY_JOINT_NAMES
  t = np.arange(w1 - w0) * deploy_logs.TICK_DT
  q_real = log.q[w0:w1]
  err = q_sim - q_real
  thr = np.radians(thr_deg)
  one_s = min(len(t), int(round(1.0 / deploy_logs.TICK_DT)))

  rmse_1s = np.degrees(np.sqrt(np.mean(err[:one_s] ** 2, axis=0)))
  rmse_full = np.degrees(np.sqrt(np.mean(err**2, axis=0)))
  t_div = np.array([divergence_time(err[:, i], thr) for i in range(err.shape[1])])

  print(f"\nwindow: rows [{w0}, {w1}) = {t[-1] + deploy_logs.TICK_DT:.2f} s @ 500 Hz")
  print(f"command |vx| mean/max: {np.abs(log.js[w0:w1, 0]).mean():.2f}/{np.abs(log.js[w0:w1, 0]).max():.2f} m/s")
  print(f"{'joint':<20}{'rmse@1s(deg)':>14}{'rmse(deg)':>11}{'max(deg)':>10}{'t_div(s)':>10}")
  for i, name in enumerate(jname):
    div = f"{t_div[i]:.2f}" if np.isfinite(t_div[i]) else "-"
    print(f"{name:<20}{rmse_1s[i]:>14.2f}{rmse_full[i]:>11.2f}{np.degrees(np.abs(err[:, i]).max()):>10.2f}{div:>10}")
  legs_div = t_div[_LEG_IDX]
  first = np.nanmin(legs_div) if np.isfinite(legs_div).any() else np.nan
  print(
    f"\nlegs: mean rmse@1s {rmse_1s[_LEG_IDX].mean():.2f} deg; earliest "
    f"divergence (> {thr_deg:.0f} deg for 50 ms): "
    + (f"{first:.2f} s ({jname[_LEG_IDX[int(np.nanargmin(legs_div))]]})" if np.isfinite(first) else "none in window")
  )

  out_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out_dir / "aligned.npz",
    t=t,
    q_real=q_real,
    dq_real=log.dq[w0:w1],
    q_des=log.q_des[w0:w1],
    tau_real=log.tau[w0:w1],
    q_sim=q_sim,
    tau_sim=tau_sim,
    joint_names=np.array(jname),
    row0=w0,
    tick0=log.tick[w0],
  )
  print(f"saved {out_dir / 'aligned.npz'}")

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("(matplotlib unavailable; skipping plots)")
    return

  def overlay_page(idx: list[int], nrow: int, ncol: int, fname: str) -> None:
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), sharex=True, squeeze=False)
    for k, i in enumerate(idx):
      ax = axes[k // ncol][k % ncol]
      ax.plot(t, np.degrees(q_real[:, i]), lw=1.2, label="real")
      ax.plot(t, np.degrees(q_sim[:, i]), lw=1.2, ls="--", label="sim")
      ax.plot(t, np.degrees(log.q_des[w0:w1, i]), lw=0.7, ls=":", color="0.4", label="target")
      if np.isfinite(t_div[i]):
        ax.axvline(t_div[i], color="r", lw=0.7, alpha=0.5)
      ax.set_title(f"{jname[i]}  rmse@1s {rmse_1s[i]:.2f} deg", fontsize=9)
      ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=8)
    for k in range(len(idx), nrow * ncol):
      axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"real vs sim joint position (deg) -- {run_name}")
    fig.supxlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / fname}")

  overlay_page(_LEG_IDX, 4, 3, "overlap_legs.png")
  overlay_page(_UPPER_IDX, 3, 3, "overlap_upper.png")

  fig, ax = plt.subplots(figsize=(10, 5))
  for i in _LEG_IDX:
    ax.plot(t, np.degrees(np.abs(err[:, i])), lw=0.9, label=jname[i])
  ax.axhline(thr_deg, color="r", ls="--", lw=1)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("|q_sim - q_real| (deg)")
  ax.legend(fontsize=7, ncol=4)
  fig.tight_layout()
  fig.savefig(out_dir / "error_legs.png", dpi=120)
  plt.close(fig)
  print(f"saved {out_dir / 'error_legs.png'}")

  fig, axes = plt.subplots(4, 3, figsize=(12.6, 10.4), sharex=True)
  for k, i in enumerate(_LEG_IDX):
    ax = axes[k // 3][k % 3]
    ax.plot(t, log.tau[w0:w1, i], lw=1.0, label="real")
    ax.plot(t, tau_sim[:, i], lw=1.0, ls="--", label="sim")
    ax.set_title(jname[i], fontsize=9)
    ax.tick_params(labelsize=8)
  axes[0][0].legend(fontsize=8)
  fig.suptitle("joint torque (N*m), legs")
  fig.supxlabel("time (s)")
  fig.tight_layout()
  fig.savefig(out_dir / "torque_legs.png", dpi=120)
  plt.close(fig)
  print(f"saved {out_dir / 'torque_legs.png'}")


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--run-dir", type=Path, help="robot_logs run dir (data.lcmlog)")
  src.add_argument("--csv", type=Path, help="pre-exported state.csv")
  parser.add_argument(
    "--deploy-repo",
    type=Path,
    default=DEPLOY_REPO_DIR,
    help="bdml_phybot_deploy_c2 checkout (for log2csv)",
  )
  parser.add_argument("--task-code", type=int, default=None, help="pin the walk_amp select_code instead of auto-detect")
  parser.add_argument("--t-start", type=float, default=0.0, help="window start, seconds into the walk segment")
  parser.add_argument(
    "--duration", type=float, default=np.inf, help="window length in seconds (default: to segment end)"
  )
  parser.add_argument(
    "--physics-dt", type=float, default=0.002, help="sim integration step (default 0.002 = one real tick)"
  )
  parser.add_argument("--friction", type=float, default=1.0)
  parser.add_argument("--settle-ticks", type=int, default=100, help="vertical contact-loading settle before the replay")
  parser.add_argument("--div-threshold-deg", type=float, default=5.0)
  parser.add_argument("--output-dir", type=Path, default=None, help="default: artifacts/sim2real/<run name>")
  args = parser.parse_args(argv)
  # The log columns are in policy joint order; the replay relies on it.
  assert PHYBOT_C2_POLICY_JOINT_NAMES == deploy_logs.JOINT_NAMES

  csv = deploy_logs.ensure_state_csv(args.run_dir or args.csv, args.deploy_repo)
  log = deploy_logs.load_state_csv(csv)
  default_dof_pos = np.array([PHYBOT_C2_STANDING_JOINT_POS[n] for n in PHYBOT_C2_POLICY_JOINT_NAMES])
  seg = deploy_logs.find_walk_segment(log, default_dof_pos, task_code=args.task_code)
  print(
    f"walk segment: rows [{seg.start}, {seg.end}) = {seg.duration_s:.1f} s, "
    f"task_code={seg.task_code}, first inference row {seg.infer_rows[0]}"
    + ("" if seg.true_task_start else " (log starts mid-task)")
  )

  # Snap the window start to a policy step so it begins on a fresh target.
  w0_min = seg.infer_rows[0] + int(round(args.t_start / deploy_logs.TICK_DT))
  later = seg.infer_rows[seg.infer_rows >= w0_min]
  if len(later) == 0:
    raise SystemExit(f"--t-start {args.t_start} is past the segment end")
  w0 = int(later[0])
  w1 = seg.end
  if np.isfinite(args.duration):
    w1 = min(w1, w0 + int(round(args.duration / deploy_logs.TICK_DT)))
  if w1 - w0 < 2 * deploy_logs.DECIMATION:
    raise SystemExit("comparison window is shorter than two policy steps")

  q_sim, tau_sim = replay(log, w0, w1, args.physics_dt, args.friction, args.settle_ticks)

  run_name = csv.parent.name if csv.name == "state.csv" else csv.stem
  out_dir = args.output_dir or (ARTIFACTS_DIR / "sim2real" / run_name)
  report(log, w0, w1, q_sim, tau_sim, args.div_threshold_deg, out_dir, run_name)


if __name__ == "__main__":
  main()
