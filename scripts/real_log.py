"""Load and segment bdml_phybot_deploy_c2 run logs (log2csv ``state.csv``).

Pure numpy + stdlib (no mjlab/torch imports) so both the sim-replay and the
obs-replay tools can share it. The column layout mirrors ``apps/log2csv.cpp``
in the deploy repo: one row per 500 Hz control tick, per-motor grouped columns
``m00..m20 x {pos,vel,pos_des,...}`` in policy joint order.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_DOFS = 21
TICK_DT = 0.002  # robot_main control period (config/robot.yaml)
DECIMATION = 10  # walk_amp policy fires every 10th tick (50 Hz)

# Index -> joint name, identical on the real robot (panel/shared/joints.json)
# and in sim (PHYBOT_C2_POLICY_JOINT_NAMES).
JOINT_NAMES = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow_pitch",
)


def default_deploy_repo(repo_root: Path) -> Path:
  return repo_root.parent / "bdml_phybot_deploy_c2"


def ensure_state_csv(source: Path, deploy_repo: Path) -> Path:
  """Resolve *source* (a state.csv or a robot_logs run dir) to a state.csv.

  For a run dir, exports data.lcmlog with the deploy repo's log2csv binary
  the first time (the CSV is cached next to the log).
  """
  source = Path(source)
  if source.is_file():
    return source
  if not source.is_dir():
    raise FileNotFoundError(f"no such run dir or csv: {source}")
  csv = source / "state.csv"
  if csv.is_file():
    return csv
  exe = Path(deploy_repo) / "build" / "log2csv"
  if not exe.is_file():
    raise FileNotFoundError(
      f"{csv} does not exist and {exe} is not built; build the deploy repo "
      "or export state.csv manually (log2csv <run_dir>)"
    )
  subprocess.run([str(exe), str(source), "-o", str(csv)], check=True)
  return csv


@dataclass
class RealLog:
  path: Path
  tick: np.ndarray  # (N,) int64, robot_main loop counter
  current_task: np.ndarray  # (N,) int64
  motors_enabled: np.ndarray  # (N,) int64
  q: np.ndarray  # (N, 21) measured joint pos (sim joint frame)
  dq: np.ndarray  # (N, 21) measured joint vel
  q_des: np.ndarray  # (N, 21) commanded PD target (default + scale*action)
  tau: np.ndarray  # (N, 21) measured joint torque
  imu_zyx: np.ndarray  # (N, 3) ZYX euler (yaw, pitch, roll)
  imu_gyro: np.ndarray  # (N, 3) body angular velocity
  js: np.ndarray  # (N, 3) post-watchdog command (vx, vy, wz)


def load_state_csv(path: Path) -> RealLog:
  path = Path(path)
  with open(path) as f:
    header = f.readline().strip().split(",")
    data = np.loadtxt(f, delimiter=",", ndmin=2)
  col = {name: i for i, name in enumerate(header)}

  def block(names: list[str]) -> np.ndarray:
    return np.ascontiguousarray(data[:, [col[n] for n in names]])

  def motor(field: str) -> np.ndarray:
    return block([f"m{i:02d}_{field}" for i in range(N_DOFS)])

  return RealLog(
    path=path,
    tick=data[:, col["tick"]].astype(np.int64),
    current_task=data[:, col["current_task"]].astype(np.int64),
    motors_enabled=data[:, col["motors_enabled"]].astype(np.int64),
    q=motor("pos"),
    dq=motor("vel"),
    q_des=motor("pos_des"),
    tau=motor("torque"),
    imu_zyx=block(["imu_zyx_0", "imu_zyx_1", "imu_zyx_2"]),
    imu_gyro=block(["imu_gyro_0", "imu_gyro_1", "imu_gyro_2"]),
    js=block(["js_vx", "js_vy", "js_wz"]),
  )


@dataclass
class WalkSegment:
  start: int  # first row of the gap-free constant-task run
  end: int  # one past the last row
  infer_rows: np.ndarray  # rows where the policy produced a new action
  task_code: int
  true_task_start: bool  # log captured task entry (decimation-1 held rows)

  @property
  def duration_s(self) -> float:
    return (self.end - self.start) * TICK_DT


def find_walk_segment(
  log: RealLog,
  default_dof_pos: np.ndarray,
  task_code: int | None = None,
  decimation: int = DECIMATION,
) -> WalkSegment:
  """Locate the walk_amp stretch of the log.

  A qualifying segment is a gap-free (tick increments of 1), constant
  current_task run whose PD targets update every ``decimation`` ticks
  (policy zero-order hold) and, unless ``task_code`` pins the task, whose
  first held target is the default pose (walk_amp holds default_dof_pos for
  decimation-1 ticks after entry, before the first inference). The longest
  qualifying run wins. Policy (inference) rows are every ``decimation``-th
  row from the first target change.
  """
  n = len(log.tick)
  if n < 3 * decimation:
    raise ValueError(f"log too short ({n} rows)")
  boundary = np.flatnonzero((np.diff(log.tick) != 1) | (np.diff(log.current_task) != 0))
  starts = np.concatenate(([0], boundary + 1))
  ends = np.concatenate((boundary + 1, [n]))

  candidates = []
  for a, b in zip(starts, ends):
    if b - a < 3 * decimation:
      continue
    if task_code is not None and log.current_task[a] != task_code:
      continue
    changed = np.any(log.q_des[a + 1 : b] != log.q_des[a : b - 1], axis=1)
    ch = a + 1 + np.flatnonzero(changed)
    if len(ch) < 3:
      continue
    if abs(float(np.median(np.diff(ch))) - decimation) > 0.5:
      continue
    if task_code is None and not np.allclose(log.q_des[a], default_dof_pos, atol=2e-3):
      continue
    candidates.append((a, b, ch))
  if not candidates:
    raise ValueError(
      "no walk_amp-like segment found (constant task, targets updating every "
      f"{decimation} ticks); pass an explicit task code if the run is unusual"
    )
  a, b, ch = max(candidates, key=lambda c: c[1] - c[0])
  first = int(ch[0])
  return WalkSegment(
    start=a,
    end=b,
    infer_rows=np.arange(first, b, decimation),
    task_code=int(log.current_task[a]),
    true_task_start=(first - a) == decimation - 1,
  )
