"""Check PHYBot C2 walk policy metadata against deploy and MotorList configs.

This is an offline gate for the real-PD AMP walk policy. It does not compile
or run the C++ deploy stack, and it does not talk to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TASK = "Mjlab-Walk-Amp-Phybot-C2"
EXPECTED_ACTION_RATE_WEIGHT = -0.25
EXPECTED_CONTROL_FREQ_HZ = 50.0
EXPECTED_DECIMATION = 10
EXPECTED_HISTORY_LEN = 10
EXPECTED_OBS_DIM = 720
EXPECTED_ACTION_DIM = 21
EXPECTED_ACTION_SCALE = 0.25
EXPECTED_OBS_LAYOUT = (
  ("base_ang_vel", 0, 30),
  ("projected_gravity", 30, 60),
  ("joint_pos", 60, 270),
  ("joint_vel", 270, 480),
  ("actions", 480, 690),
  ("command", 690, 720),
)
EXPECTED_MINI_MOTOR_NAMES = (
  "01_BAN_2_CAN_1_2",
  "02_BAN_2_CAN_1_3",
  "03_BAN_2_CAN_1_4",
  "04_BAN_2_CAN_1_5",
  "05_BAN_2_CAN_1_6",
  "06_BAN_2_CAN_1_7",
  "07_BAN_2_CAN_2_1",
  "08_BAN_2_CAN_2_2",
  "09_BAN_2_CAN_2_3",
  "10_BAN_2_CAN_2_4",
  "11_BAN_2_CAN_2_5",
  "12_BAN_2_CAN_2_6",
  "13_BAN_2_CAN_1_1",
  "14_BAN_1_CAN_1_1",
  "15_BAN_1_CAN_1_2",
  "16_BAN_1_CAN_1_3",
  "17_BAN_1_CAN_1_4",
  "18_BAN_1_CAN_2_1",
  "19_BAN_1_CAN_2_2",
  "20_BAN_1_CAN_2_3",
  "21_BAN_1_CAN_2_4",
)
HEAD_MOTOR_IDS = {11, 21}
REAL_PD_DAMPING_BY_TARGET_EXPR = {
  (".*_hip_pitch",): 10.0,
  (".*_hip_roll",): 10.0,
  (".*_hip_yaw",): 10.0,
  (".*_knee",): 10.0,
}


@dataclass
class Issue:
  severity: str
  message: str


class CheckResult:
  def __init__(self) -> None:
    self.issues: list[Issue] = []

  def error(self, message: str) -> None:
    self.issues.append(Issue("ERROR", message))

  def warn(self, message: str) -> None:
    self.issues.append(Issue("WARN", message))

  @property
  def errors(self) -> list[Issue]:
    return [issue for issue in self.issues if issue.severity == "ERROR"]

  def print_and_exit(self, report_only: bool) -> None:
    if not self.issues:
      print("PASS: walk policy, deploy params, MotorList, and real-PD task are aligned.")
      return
    for issue in self.issues:
      print(f"{issue.severity}: {issue.message}")
    if self.errors and not report_only:
      raise SystemExit(1)
    if self.errors:
      print("REPORT-ONLY: errors above were not treated as fatal.")


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as stream:
    return yaml.safe_load(stream)


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as stream:
    return json.load(stream)


def _as_float_list(value: Any, expected_len: int) -> list[float]:
  if isinstance(value, (int, float)):
    return [float(value)] * expected_len
  return [float(item) for item in value]


def _nearly_equal(a: float, b: float, tol: float = 1.0e-6) -> bool:
  return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _check_scalar(result: CheckResult, name: str, actual: float, expected: float, tol: float = 1.0e-6) -> None:
  if not _nearly_equal(float(actual), float(expected), tol):
    result.error(f"{name}: got {actual}, expected {expected}")


def _check_list(
  result: CheckResult,
  name: str,
  actual: list[Any],
  expected: list[Any] | tuple[Any, ...],
  *,
  tol: float = 1.0e-6,
) -> None:
  if len(actual) != len(expected):
    result.error(f"{name}: length {len(actual)}, expected {len(expected)}")
    return
  for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
    if isinstance(e_item, float):
      if not _nearly_equal(float(a_item), e_item, tol):
        result.error(f"{name}[{idx}]: got {a_item}, expected {e_item}")
    elif a_item != e_item:
      result.error(f"{name}[{idx}]: got {a_item!r}, expected {e_item!r}")


def _policy_obs_layout(meta: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
  return tuple((row["term"], int(row["start"]), int(row["end"])) for row in meta.get("obs_layout", ()))


def _load_policy_motors(motor_yaml_paths: list[Path]) -> list[dict[str, Any]]:
  motors = []
  for path in motor_yaml_paths:
    data = _load_yaml(path)
    for item in data.get("Motors", ()):
      motor = dict(item["Motor"])
      motor["_source"] = str(path)
      motors.append(motor)

  policy_motors = [motor for motor in motors if int(motor["Id"]) not in HEAD_MOTOR_IDS]
  return sorted(policy_motors, key=lambda motor: str(motor["Name"]))


def _check_motorlist(result: CheckResult, motors: list[dict[str, Any]], joint_names: list[str]) -> None:
  if len(motors) != EXPECTED_ACTION_DIM:
    result.error(f"MotorList policy motor count: got {len(motors)}, expected {EXPECTED_ACTION_DIM}")
    return
  _check_list(result, "MotorList sorted policy names", [motor["Name"] for motor in motors], EXPECTED_MINI_MOTOR_NAMES)
  if len(joint_names) == EXPECTED_ACTION_DIM:
    print("MotorList policy order:")
    for idx, (joint_name, motor) in enumerate(zip(joint_names, motors)):
      print(
        f"  {idx:02d} {joint_name:22s} <- {motor['Name']:18s} "
        f"kp={float(motor['PDKp']):g} kd={float(motor['PDKd']):g} dir={int(motor['Direction'])}"
      )


def _check_meta(result: CheckResult, meta: dict[str, Any], task_id: str) -> None:
  if meta.get("task_id") != task_id:
    result.error(f"policy_meta.task_id: got {meta.get('task_id')!r}, expected {task_id!r}")
  _check_scalar(result, "policy_meta.control_freq_hz", float(meta.get("control_freq_hz", -1.0)), EXPECTED_CONTROL_FREQ_HZ)
  if int(meta.get("obs_dim", -1)) != EXPECTED_OBS_DIM:
    result.error(f"policy_meta.obs_dim: got {meta.get('obs_dim')}, expected {EXPECTED_OBS_DIM}")
  if int(meta.get("obs_history_length", -1)) != EXPECTED_HISTORY_LEN:
    result.error(
      f"policy_meta.obs_history_length: got {meta.get('obs_history_length')}, expected {EXPECTED_HISTORY_LEN}"
    )
  if int(meta.get("action_dim", -1)) != EXPECTED_ACTION_DIM:
    result.error(f"policy_meta.action_dim: got {meta.get('action_dim')}, expected {EXPECTED_ACTION_DIM}")
  _check_list(result, "policy_meta.obs_layout", list(_policy_obs_layout(meta)), EXPECTED_OBS_LAYOUT)
  _check_list(
    result,
    "policy_meta.action_scale",
    _as_float_list(meta.get("action_scale"), EXPECTED_ACTION_DIM),
    [EXPECTED_ACTION_SCALE] * EXPECTED_ACTION_DIM,
  )


def _check_deploy_params(result: CheckResult, params: dict[str, Any], meta: dict[str, Any]) -> None:
  if int(params.get("decimation", -1)) != EXPECTED_DECIMATION:
    result.error(f"deploy decimation: got {params.get('decimation')}, expected {EXPECTED_DECIMATION}")
  if int(params.get("num_observations", -1)) != EXPECTED_OBS_DIM:
    result.error(f"deploy num_observations: got {params.get('num_observations')}, expected {EXPECTED_OBS_DIM}")
  if int(params.get("history_len", -1)) != EXPECTED_HISTORY_LEN:
    result.error(f"deploy history_len: got {params.get('history_len')}, expected {EXPECTED_HISTORY_LEN}")
  if int(params.get("num_of_dofs", -1)) != EXPECTED_ACTION_DIM:
    result.error(f"deploy num_of_dofs: got {params.get('num_of_dofs')}, expected {EXPECTED_ACTION_DIM}")

  for key, meta_key in (
    ("action_scale", "action_scale"),
    ("default_dof_pos", "default_dof_pos"),
    ("rl_p", "kp"),
    ("rl_d", "kd"),
    ("torque_limits", "torque_limits"),
  ):
    actual = _as_float_list(params.get(key), EXPECTED_ACTION_DIM)
    expected = _as_float_list(meta.get(meta_key), EXPECTED_ACTION_DIM)
    _check_list(result, f"deploy {key} vs policy_meta.{meta_key}", actual, expected, tol=1.0e-5)


def _check_pd_against_motorlist(
  result: CheckResult,
  meta: dict[str, Any],
  params: dict[str, Any],
  motors: list[dict[str, Any]],
) -> None:
  if len(motors) != EXPECTED_ACTION_DIM:
    return
  motor_kp = [float(motor["PDKp"]) for motor in motors]
  motor_kd = [float(motor["PDKd"]) for motor in motors]
  _check_list(result, "policy_meta.kp vs MotorList.PDKp", _as_float_list(meta.get("kp"), EXPECTED_ACTION_DIM), motor_kp)
  _check_list(result, "policy_meta.kd vs MotorList.PDKd", _as_float_list(meta.get("kd"), EXPECTED_ACTION_DIM), motor_kd)
  _check_list(result, "deploy rl_p vs MotorList.PDKp", _as_float_list(params.get("rl_p"), EXPECTED_ACTION_DIM), motor_kp)
  _check_list(result, "deploy rl_d vs MotorList.PDKd", _as_float_list(params.get("rl_d"), EXPECTED_ACTION_DIM), motor_kd)


def _check_task_cfg(result: CheckResult, repo_root: Path, task_id: str) -> None:
  src_dir = repo_root / "src"
  if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

  try:
    import scr_phybot_mjab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  except Exception as exc:
    result.error(f"cannot import mjlab task registry for {task_id}: {exc}")
    return

  try:
    env_cfg = load_env_cfg(task_id, play=False)
    rl_cfg = load_rl_cfg(task_id)
  except Exception as exc:
    result.error(f"cannot load task {task_id}: {exc}")
    return

  actor = env_cfg.observations["actor"]
  if actor.history_length != EXPECTED_HISTORY_LEN:
    result.error(f"{task_id} actor history_length: got {actor.history_length}, expected {EXPECTED_HISTORY_LEN}")
  if list(actor.terms) != [name for name, _, _ in EXPECTED_OBS_LAYOUT]:
    result.error(f"{task_id} actor obs terms: got {list(actor.terms)}, expected {[x[0] for x in EXPECTED_OBS_LAYOUT]}")

  action = env_cfg.actions["joint_pos"]
  _check_scalar(result, f"{task_id} action scale", float(action.scale), EXPECTED_ACTION_SCALE)
  _check_scalar(
    result,
    f"{task_id} action_rate_l2.weight",
    float(env_cfg.rewards["action_rate_l2"].weight),
    EXPECTED_ACTION_RATE_WEIGHT,
  )
  if getattr(rl_cfg, "experiment_name", None) != "phybot_c2_walk_amp":
    result.error(f"{task_id} experiment_name: got {getattr(rl_cfg, 'experiment_name', None)!r}")

  robot_cfg = env_cfg.scene.entities["robot"]
  for actuator in robot_cfg.articulation.actuators:
    target_expr = tuple(actuator.target_names_expr)
    if target_expr in REAL_PD_DAMPING_BY_TARGET_EXPR:
      expected = REAL_PD_DAMPING_BY_TARGET_EXPR[target_expr]
      _check_scalar(result, f"{task_id} actuator {target_expr} damping", float(actuator.damping), expected)


def main() -> None:
  repo_root = _repo_root()
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument(
    "--policy-meta",
    type=Path,
    default=repo_root / "deploy/PhybotSoftware_c2/RL_deploy_walk_preset_amp/model/policy_meta.json",
  )
  parser.add_argument(
    "--deploy-params",
    type=Path,
    default=repo_root / "deploy/PhybotSoftware_c2/RL_deploy_walk_preset_amp/config/rl_params.yaml",
  )
  parser.add_argument(
    "--motor-yaml",
    action="append",
    type=Path,
    default=None,
  )
  parser.add_argument("--skip-task-cfg", action="store_true")
  parser.add_argument("--report-only", action="store_true", help="Print errors but exit 0.")
  args = parser.parse_args()
  motor_yaml_paths = args.motor_yaml or [
    repo_root / "deploy/PhybotSoftware_c2/MotorList/config/phybot_mini_1.yaml",
    repo_root / "deploy/PhybotSoftware_c2/MotorList/config/phybot_mini_2.yaml",
  ]

  result = CheckResult()
  meta = _load_json(args.policy_meta)
  params = _load_yaml(args.deploy_params)
  motors = _load_policy_motors(motor_yaml_paths)

  _check_meta(result, meta, args.task)
  _check_deploy_params(result, params, meta)
  _check_motorlist(result, motors, list(meta.get("joint_names", ())))
  _check_pd_against_motorlist(result, meta, params, motors)
  if not args.skip_task_cfg:
    _check_task_cfg(result, repo_root, args.task)

  result.print_and_exit(report_only=args.report_only)


if __name__ == "__main__":
  main()
