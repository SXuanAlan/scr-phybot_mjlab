"""Replay a real-robot log through the walk_amp obs builder + policy, offline.

Rebuilds the 720-dim observation for every policy step exactly as the deploy
stack builds it on the robot (tasks/walk_amp/walk_amp_obs.h: raw gyro,
gravity-from-ZYX-euler, q - default_dof_pos, dq, previous raw action,
quantized command; 10-frame term-major history, first frame back-filled),
runs the deployed TorchScript policy on it, and checks the produced actions
against the actions the robot actually executed, recovered from the logged PD
targets: action = (m*_pos_des - default_dof_pos) / action_scale.

Agreement within ~1e-3 raw action units (float32 logging round-trip noise)
proves the C++ obs pipeline, the exported policy, and this reconstruction all
match; a real layout/scale/order bug shows up as O(0.1) mismatches. Run this
gate on a fresh robot log BEFORE trusting any dynamics-level sim2real
comparison (scripts/deploy/compare_real_sim.py).

Usage:
    python scripts/deploy/check_obs_replay.py --run-dir <robot_logs/..._real>
    python scripts/deploy/check_obs_replay.py --csv state.csv --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from scr_phybot_mjab.tools import deploy_logs
from scr_phybot_mjab.tools.paths import DEPLOY_REPO_DIR, REPOSITORY_ROOT

_TERMS = (("ang_vel", 3), ("gravity", 3), ("pos", 21), ("vel", 21), ("action", 21), ("command", 3))


def load_walk_amp_params(path: Path) -> dict:
  with open(path) as f:
    p = yaml.safe_load(f)
  assert p["decimation"] == deploy_logs.DECIMATION
  assert p["history_len"] == 10 and p["num_observations"] == 720
  assert p["num_of_dofs"] == deploy_logs.N_DOFS
  p["action_scale"] = np.asarray(p["action_scale"], dtype=np.float64)
  p["default_dof_pos"] = np.asarray(p["default_dof_pos"], dtype=np.float64)
  return p


def quantize_command(vx: float, wz: float, p: dict) -> np.ndarray:
  """Port of walk_amp_obs.h QuantizeWalkAmpCommand (vy is ignored there)."""
  q_vx = 0.0 if abs(vx) < p["external_deadband"] else vx
  q_wz = 0.0 if abs(wz) < p["external_deadband"] else wz
  if abs(q_wz) >= p["turn_threshold"] and abs(q_vx) < p["walk_threshold"]:
    return np.array([p["turn_forward_speed"], 0.0, np.copysign(p["turn_speed"], q_wz)])
  if q_vx >= p["walk_threshold"]:
    return np.array([p["walk_speed"], 0.0, 0.0])
  if q_vx >= p["slow_threshold"]:
    return np.array([p["slow_speed"], 0.0, 0.0])
  return np.zeros(3)


def gravity_from_zyx(zyx: np.ndarray) -> np.ndarray:
  """Port of core/include/phybot/gravity.h (yaw-independent)."""
  a1, a2 = zyx[1], zyx[2]
  return np.array([np.sin(a1), -np.cos(a1) * np.sin(a2), -np.cos(a1) * np.cos(a2)])


class WalkAmpObsBuilder:
  """float32 history buffers with deploy's Fill/ShiftPush semantics."""

  def __init__(self, history_len: int = 10):
    self.h = history_len
    self.buf = {name: np.zeros((self.h, dim), np.float32) for name, dim in _TERMS}
    self.initialized = False

  def push(self, frames: dict[str, np.ndarray]) -> None:
    for name, _ in _TERMS:
      frame = frames[name].astype(np.float32)
      if not self.initialized:
        self.buf[name][:] = frame
      else:
        self.buf[name][:-1] = self.buf[name][1:]
        self.buf[name][-1] = frame
    self.initialized = True

  def obs(self) -> np.ndarray:
    return np.concatenate([self.buf[name].reshape(-1) for name, _ in _TERMS])


def build_observations(log: deploy_logs.RealLog, rows: np.ndarray, actions: np.ndarray, p: dict) -> np.ndarray:
  builder = WalkAmpObsBuilder(p["history_len"])
  obs = np.zeros((len(rows), 720), np.float32)
  for k, r in enumerate(rows):
    builder.push(
      {
        "ang_vel": log.imu_gyro[r],
        "gravity": gravity_from_zyx(log.imu_zyx[r]),
        "pos": log.q[r] - p["default_dof_pos"],
        "vel": log.dq[r],
        # obs carries the PREVIOUS step's raw action (zero right after entry).
        "action": actions[k - 1] if k > 0 else np.zeros(deploy_logs.N_DOFS),
        "command": quantize_command(log.js[r, 0], log.js[r, 2], p),
      }
    )
    obs[k] = builder.obs()
  return obs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--run-dir", type=Path, help="robot_logs run dir (data.lcmlog)")
  src.add_argument("--csv", type=Path, help="pre-exported state.csv")
  parser.add_argument("--deploy-repo", type=Path, default=DEPLOY_REPO_DIR)
  parser.add_argument("--policy", type=Path, default=None, help="default: <deploy-repo>/tasks/walk_amp/model/policy.pt")
  parser.add_argument(
    "--params", type=Path, default=None, help="default: <deploy-repo>/tasks/walk_amp/config/walk_amp.yaml"
  )
  parser.add_argument("--task-code", type=int, default=None)
  parser.add_argument(
    "--warmup",
    type=int,
    default=None,
    help="policy steps excluded from the pass/fail stats; "
    "default 0 when the log captured the task entry, else 10 "
    "(history is unknown until 10 in-log frames)",
  )
  parser.add_argument("--max-steps", type=int, default=0, help="0 = all")
  parser.add_argument("--atol", type=float, default=1e-3, help="max |predicted - executed| action to PASS")
  parser.add_argument("--plot", action="store_true")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args()

  deploy = args.deploy_repo
  policy_path = args.policy or deploy / "tasks/walk_amp/model/policy.pt"
  params_path = args.params or deploy / "tasks/walk_amp/config/walk_amp.yaml"
  p = load_walk_amp_params(params_path)

  csv = deploy_logs.ensure_state_csv(args.run_dir or args.csv, deploy)
  log = deploy_logs.load_state_csv(csv)
  seg = deploy_logs.find_walk_segment(log, p["default_dof_pos"], task_code=args.task_code)
  rows = seg.infer_rows
  if args.max_steps:
    rows = rows[: args.max_steps]
  warmup = args.warmup if args.warmup is not None else (0 if seg.true_task_start else 10)
  print(
    f"walk segment: rows [{seg.start}, {seg.end}), task_code={seg.task_code}, "
    f"{len(rows)} policy steps" + ("" if seg.true_task_start else " (log starts mid-task)")
  )

  # Executed actions, recovered from the logged PD targets.
  actions = (log.q_des[rows] - p["default_dof_pos"]) / p["action_scale"]

  obs = build_observations(log, rows, actions, p)
  model = torch.jit.load(str(policy_path), map_location="cpu")
  model.eval()
  with torch.no_grad():
    pred = model(torch.from_numpy(obs)).numpy().astype(np.float64)

  delta = pred - actions
  # A robot-side all-zero action means the non-finite guard fired (or the row
  # predates the first inference); the policy output is deliberately ignored
  # on those steps, so exclude them from the match statistics.
  guard = np.max(np.abs(actions), axis=1) == 0.0
  eval_mask = ~guard
  eval_mask[:warmup] = False
  n_eval = int(eval_mask.sum())
  if n_eval == 0:
    raise SystemExit("no evaluable policy steps (all guard rows or warmup)")

  d = np.abs(delta[eval_mask])
  per_joint_max = d.max(axis=0)
  worst = np.argsort(per_joint_max)[::-1][:5]
  print(f"\nevaluated {n_eval} steps (skipped {warmup} warmup, {int(guard.sum())} guard/zero-action rows)")
  print(
    f"max |pred - executed| action: {d.max():.3e}   "
    f"p99.9: {np.quantile(d, 0.999):.3e}   rms: {np.sqrt((d**2).mean()):.3e}"
  )
  print("worst joints:")
  for i in worst:
    print(f"  {deploy_logs.JOINT_NAMES[i]:<20} max {per_joint_max[i]:.3e}")

  ok = d.max() <= args.atol
  print(f"\n{'PASS' if ok else 'FAIL'}: max action mismatch {d.max():.3e} {'<=' if ok else '>'} atol {args.atol:g}")

  if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_name = csv.parent.name if csv.name == "state.csv" else csv.stem
    out_dir = args.output_dir or (REPOSITORY_ROOT / "results" / "sim2real" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = (rows - rows[0]) * deploy_logs.TICK_DT
    show = [int(worst[0]), 3]  # worst joint + left_knee
    fig, axes = plt.subplots(len(show) + 1, 1, figsize=(10, 7), sharex=True)
    for ax, i in zip(axes, show):
      ax.plot(t, actions[:, i], lw=1.0, label="executed (from pos_des)")
      ax.plot(t, pred[:, i], lw=1.0, ls="--", label="policy(rebuilt obs)")
      ax.set_ylabel(deploy_logs.JOINT_NAMES[i], fontsize=8)
    axes[0].legend(fontsize=8)
    axes[-1].plot(t[eval_mask], d.max(axis=1), lw=0.8, color="r")
    axes[-1].set_ylabel("max |delta|", fontsize=8)
    axes[-1].set_yscale("log")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("walk_amp obs replay: executed vs re-predicted action")
    fig.tight_layout()
    fig.savefig(out_dir / "check_obs_replay.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'check_obs_replay.png'}")

  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
