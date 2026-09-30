"""Persistently examine the Corners-A60 run and hold unmastered curriculum stages.

Run this in a separate tmux session from training::

  python scripts/supervise_climb60.py RUN_DIR --gpu 1 --training-log RUN_DIR/train.log
  python scripts/supervise_climb60.py RUN_DIR --once

The training process must already consume RUN_DIR/curriculum_control.json.
This process only writes that control file and monitoring artifacts; it never
starts/restarts training or changes rewards, contacts, or physics. Checkpoints
are evaluated after their modification time has settled. A stage stays locked
until two consecutive complete exams pass and a person/agent has reviewed the
actual engage rasters and rollout videos for both checkpoints. Review templates
are deliberately unsigned and cannot approve themselves.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK = "Mjlab-Climb-Wall-Phybot-C2-Corners-A60"
PROFILES = ("canonical", "training-rsi")
CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
GATES = {
  "engage_match_min": 0.95,
  "first_episode_falls": 0,
  "pad_occupancy_excess_max_per_pad": 0.04,
  "progress_ratio_min": 0.8,
  "true_regrip_cycles_min": 2,
}


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
  with path.open() as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as stream:
      json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


@contextmanager
def file_lock(path: Path, *, nonblocking: bool = False):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as stream:
    mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    fcntl.flock(stream.fileno(), mode)
    try:
      yield
    finally:
      fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_control(control: dict[str, Any]) -> None:
  if control.get("schema_version") != 1:
    raise ValueError("curriculum_control.json needs schema_version=1")
  stages = control.get("angle_stages", [])
  if len(stages) != 3:
    raise ValueError("Corners-A60 needs exactly three angle stages")
  expected = ((45.0, 5.0), (55.0, 5.0), (60.0, 0.0))
  previous = -1
  for stage, (angle, jitter), nominal_iter in zip(stages, expected, (0, 1500, 3300), strict=True):
    iteration = stage.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= previous:
      raise ValueError("stage iterations must be strictly increasing integers")
    if iteration < nominal_iter:
      raise ValueError("future stages cannot precede the original curriculum")
    if (stage.get("angle_deg"), stage.get("jitter_deg")) != (angle, jitter):
      raise ValueError("supervisor is restricted to 45±5, 55±5, and 60 degrees")
    previous = iteration
  if stages[0]["iter"] != 0:
    raise ValueError("first stage must start at iteration zero")
  allowed = control.get("max_stage")
  if isinstance(allowed, bool) or not isinstance(allowed, int) or not 0 <= allowed < len(stages):
    raise ValueError("max_stage must be an integer from zero to two")


def active_stage(control: dict[str, Any], iteration: int) -> int:
  stage = 0
  for index, value in enumerate(control["angle_stages"]):
    if index <= control["max_stage"] and iteration >= value["iter"]:
      stage = index
  return stage


def band_angles(stage: dict[str, Any]) -> list[float]:
  angle, jitter = float(stage["angle_deg"]), float(stage["jitter_deg"])
  return sorted({angle - jitter, angle, angle + jitter})


def stable_checkpoints(run_dir: Path, stable_seconds: float, now: float | None = None) -> dict[int, Path]:
  now = time.time() if now is None else now
  checkpoints = {}
  for path in run_dir.glob("model_*.pt"):
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
      continue
    stat = path.stat()
    if stat.st_size and now - stat.st_mtime >= stable_seconds:
      checkpoints[int(match[1])] = path
  return dict(sorted(checkpoints.items()))


def due_iterations(checkpoints: dict[int, Path], control: dict[str, Any]) -> list[int]:
  """Use saved checkpoints at/after each milestone, without demanding exact filenames."""
  if not checkpoints:
    return []
  latest = max(checkpoints)
  targets = {0, 25, 100, *range(250, latest + 1, 250)}
  for stage in control["angle_stages"][1:]:
    targets.update(stage["iter"] - offset for offset in (250, 100, 50))
  result = set()
  for target in sorted(targets):
    candidates = [value for value in checkpoints if value >= target]
    if target >= 0 and candidates:
      result.add(min(candidates))
  return sorted(result)


def finite_number(value: Any) -> bool:
  return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def numeric_verdict(summary: dict[str, Any]) -> dict[str, Any]:
  """Fail closed on missing/non-finite observations, including individual pad results."""
  metrics = summary.get("metrics", summary)
  failures = []
  for key in ("engage_match_min", "progress_ratio_min", "true_regrip_cycles_min"):
    value = metrics.get(key)
    if not finite_number(value) or value < GATES[key]:
      failures.append(f"{key}={value!r} < {GATES[key]}")
  falls = metrics.get("first_episode_falls")
  if not finite_number(falls) or falls != 0:
    failures.append(f"first_episode_falls={falls!r} != 0")
  for key, minimum in (("full_30s_survival_rate", 1.0), ("reference_cycles_completed_min", 2)):
    value = metrics.get(key)
    if not finite_number(value) or value < minimum:
      failures.append(f"{key}={value!r} < {minimum}")
  excess = metrics.get("pad_occupancy_excess_max_per_pad")
  if not isinstance(excess, list) or len(excess) != 4:
    failures.append("pad_occupancy_excess_max_per_pad must have four entries")
  else:
    for pad, value in enumerate(excess):
      if not finite_number(value) or value >= GATES["pad_occupancy_excess_max_per_pad"]:
        failures.append(f"pad[{pad}] occupancy excess={value!r} >= 0.04")
  keys = {*GATES, "full_30s_survival_rate", "reference_cycles_completed_min", "engage_match_mean"}
  return {"passed": not failures, "failures": failures, "metrics": {key: metrics.get(key) for key in keys}}


def expected_case_keys(stage: dict[str, Any], profiles=PROFILES) -> set[tuple[str, float]]:
  return {(profile, angle) for profile in profiles for angle in band_angles(stage)}


def exam_numeric_pass(exam: dict[str, Any]) -> bool:
  if exam.get("status") != "complete" or exam.get("smoke"):
    return False
  stage = exam.get("stage", {})
  cases = exam.get("cases", [])
  actual = {(item.get("profile"), item.get("angle_deg")) for item in cases}
  if actual != expected_case_keys(stage) or len(cases) != len(actual):
    return False
  if exam.get("num_envs", 0) < 32 or exam.get("duration_s", 0) < 30:
    return False
  return all(item.get("verdict", {}).get("passed") is True for item in cases)


def visual_verdict(exam: dict[str, Any], exam_dir: Path) -> dict[str, Any]:
  path = exam_dir / "visual_review.json"
  if not path.exists():
    return {"approved": False, "reason": "等待同一 checkpoint 的 engage raster 与 rollout 视频审阅"}
  try:
    review = read_json(path)
  except (OSError, ValueError) as exc:
    return {"approved": False, "reason": f"视觉审阅文件无效: {exc}"}
  identity = ("checkpoint_sha256", "checkpoint_iteration", "stage_index")
  if any(review.get(key) != exam.get(key) for key in identity):
    return {"approved": False, "reason": "视觉审阅的 checkpoint / stage 与本次考试不一致"}
  if any(review.get(key) is not True for key in ("raster_reviewed", "video_reviewed")):
    return {"approved": False, "reason": "raster / video 尚未同时完成实际审阅"}
  if review.get("approved") is not True:
    return {
      "approved": False,
      "reviewed": True,
      "path": str(path),
      "reviewer": review.get("reviewer"),
      "reason": "已审阅，未通过：" + review.get("notes", "未填写观察记录"),
      "notes": review.get("notes"),
    }
  if not review.get("reviewer") or not review.get("notes") or not review.get("reviewed_at"):
    return {"approved": False, "reason": "视觉审阅缺少审阅者、时间或观察记录"}
  for case in exam.get("cases", []):
    for key in ("engage_schedule", "video"):
      artifact = case.get("artifacts", {}).get(key)
      if not artifact or not Path(artifact).is_file() or not Path(artifact).stat().st_size:
        return {"approved": False, "reason": f"考试 {case.get('profile')}/{case.get('angle_deg')} 缺少 {key} 实物"}
  return {"approved": True, "reviewer": review["reviewer"], "path": str(path), "notes": review["notes"]}


def consecutive_passes(exams: list[dict[str, Any]], stage_index: int) -> list[dict[str, Any]]:
  candidates = sorted(
    (exam for exam in exams if exam.get("stage_index") == stage_index and not exam.get("smoke")),
    key=lambda exam: exam["checkpoint_iteration"],
  )
  recent = candidates[-2:]
  if len(recent) != 2:
    return []
  if all(exam_numeric_pass(exam) and exam.get("visual_review", {}).get("approved") for exam in recent):
    return recent
  return []


def align_runtime_schedule(
  control: dict[str, Any],
  runtime: dict[str, Any],
  *,
  now: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Match future dates to the training gate's minimum duration after a late entry."""
  updated = copy.deepcopy(control)
  if not runtime:
    return updated, None
  current, started = runtime.get("stage_index"), runtime.get("stage_started_iter")
  if type(current) is not int or not 0 <= current <= control["max_stage"]:
    raise ValueError("Runtime stage must already be permitted by curriculum control")
  if type(started) is not int or started < control["angle_stages"][current]["iter"]:
    raise ValueError("Runtime stage_started_iter must not precede its scheduled start")
  following = current + 1
  if following >= len(control["angle_stages"]):
    return updated, None
  minimum_due = started + (1500, 1800)[current]
  old_due = updated["angle_stages"][following]["iter"]
  if old_due >= minimum_due:
    return updated, None
  delay = minimum_due - old_due
  for stage in updated["angle_stages"][following:]:
    stage["iter"] += delay
  previous_permission = updated["max_stage"]
  # A permission tied to the old date cannot survive a moved evidence window.
  updated["max_stage"] = current
  event = {
    "kind": "minimum_stage_duration_aligned",
    "at": now or utc_now(),
    "stage_index": following,
    "active_stage_index": current,
    "stage_started_iter": started,
    "old_iteration": old_due,
    "new_iteration": minimum_due,
    "delay_iterations": delay,
    "previous_max_stage": previous_permission,
    "max_stage": current,
    "reason": "按实际入课迭代补足原始阶段最小时长，并同步平移未来课程；审批证据窗口随之更新",
  }
  updated.setdefault("supervisor_audit", []).append(event)
  updated["updated_at"] = event["at"]
  updated["note"] = event["reason"]
  return updated, event


def curriculum_decision(
  control: dict[str, Any],
  iteration: int,
  exams: list[dict[str, Any]],
  *,
  now: str | None = None,
  current_stage_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Pure, independently testable decision; mutations only affect future iteration gates."""
  updated = copy.deepcopy(control)
  current = active_stage(control, iteration) if current_stage_index is None else current_stage_index
  next_stage = current + 1
  if next_stage >= len(control["angle_stages"]) or next_stage <= control["max_stage"]:
    return updated, None
  at = now or utc_now()
  passes = consecutive_passes(exams, current)
  due = control["angle_stages"][next_stage]["iter"]
  # Early success is diagnostic. Recheck immediately before changing the band,
  # because an approval made at iter 500 can become stale before iter 1500.
  passes = (
    passes if iteration >= due - 250 and all(exam["checkpoint_iteration"] >= due - 250 for exam in passes) else []
  )
  if passes:
    updated["max_stage"] = next_stage
    event = {
      "kind": "stage_approved",
      "at": at,
      "observed_iteration": iteration,
      "stage_index": next_stage,
      "effective_iteration": updated["angle_stages"][next_stage]["iter"],
      "evidence_iterations": [exam["checkpoint_iteration"] for exam in passes],
      "evidence_sha256": [exam["checkpoint_sha256"] for exam in passes],
      "reason": "连续两次当前课程完整数值考试及对应 raster/video 审阅通过；按原定或已延后门槛升课",
    }
  else:
    if iteration < due - 50:
      return updated, None
    delay = max(250, 250 * math.ceil((iteration + 51 - due) / 250))
    for stage in updated["angle_stages"][next_stage:]:
      stage["iter"] += delay
    event = {
      "kind": "stage_delayed",
      "at": at,
      "observed_iteration": iteration,
      "stage_index": next_stage,
      "old_iteration": due,
      "new_iteration": updated["angle_stages"][next_stage]["iter"],
      "delay_iterations": delay,
      "reason": "未获得连续两次完整数值通过且已审阅 raster/video 的证据；保留当前课程并同步平移后续节点",
    }
  updated.setdefault("supervisor_audit", []).append(event)
  updated["updated_at"] = at
  updated["note"] = event["reason"]
  return updated, event


def training_metrics(run_dir: Path, log: Path | None) -> dict[str, Any]:
  try:
    try:
      from monitor_climb60_metrics import digest, read_run
    except ImportError:
      from scripts.monitor_climb60_metrics import digest, read_run
    rows, source = read_run(run_dir, log)
    return digest(rows, run_dir, source, window=10)
  except Exception as exc:  # Scalar decoding must not crash the checkpoint guard.
    return {"latest_iteration": None, "alerts": [f"训练 metrics 读取失败: {exc}"]}


def checkpoint_stage(checkpoint: Path) -> tuple[int, int]:
  """Read the stage actually used to train this checkpoint, independent of later delays."""
  import torch

  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  monitor = saved.get("infos", {}).get("climb_monitor", {})
  stage, started = monitor.get("active_stage"), monitor.get("stage_started_iter")
  if type(stage) is not int or not 0 <= stage < 3 or type(started) is not int or started < 0:
    raise ValueError("Checkpoint lacks the monitored training curriculum metadata")
  return stage, started


def decode_result(summary: dict[str, Any], exam: dict[str, Any], profile: str, angle: float) -> dict[str, Any]:
  identity = summary.get("checkpoint", {})
  if summary.get("complete") is not True:
    raise ValueError("Evaluator did not finish its summary")
  if identity.get("sha256") != exam["checkpoint_sha256"] or identity.get("iteration") != exam["checkpoint_iteration"]:
    raise ValueError("Evaluator summary refers to another checkpoint")
  results = [
    item for item in summary.get("results", []) if item.get("profile") == profile and item.get("angle_deg") == angle
  ]
  if len(results) != 1:
    raise ValueError("Evaluator summary must have exactly one result for this profile and angle")
  result = results[0]
  if result.get("actor_state_verified") is not True or result.get("auto_reset") is not False:
    raise ValueError("Evaluation must verify actor loading and disable resets")
  if result.get("num_envs") != exam["num_envs"] or result.get("duration_s", 0) < exam["duration_s"] - 0.021:
    raise ValueError("Evaluator did not use the requested environment count or horizon")
  return result


class Supervisor:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.run_dir = args.run_dir.resolve()
    self.monitor_dir = self.run_dir / "monitor"
    self.control_path = self.run_dir / "curriculum_control.json"
    self.status_path = self.monitor_dir / "status.json"
    self.monitor_dir.mkdir(parents=True, exist_ok=True)
    self.state = (
      read_json(self.status_path)
      if self.status_path.exists()
      else {
        "schema_version": 1,
        "task": TASK,
        "run_dir": str(self.run_dir),
        "created_at": utc_now(),
        "exams": {},
        "errors": [],
      }
    )
    # Recover exam artifacts even if the prior process died before writing status.
    for path in sorted((self.run_dir / "exams").glob("iter_*/exam.json")):
      try:
        exam = read_json(path)
        self.state["exams"][str(exam["checkpoint_iteration"])] = exam
      except (OSError, ValueError, KeyError) as exc:
        self.state["errors"].append(f"{path}: {exc}")
    for exam in self.state["exams"].values():
      if exam.get("status") == "running":
        exam.update(
          status="error",
          error="Previous supervisor exited before completing this exam",
          retry_after=time.time() + self.args.error_retry_seconds,
        )
    self.state.pop("active_evaluation", None)

  def save(self) -> None:
    self.state["updated_at"] = utc_now()
    self.state["pid"] = os.getpid()
    self.state["errors"] = self.state.get("errors", [])[-20:]
    atomic_json(self.status_path, self.state)
    self.write_feedback()

  def write_feedback(self) -> None:
    control = self.state.get("curriculum_control", {})
    lines = [f"训练监督更新：{utc_now()}", "", f"任务：`{TASK}`；训练环境 4096。", ""]
    iteration = self.state.get("latest_iteration")
    lines.append(f"最新训练/存档迭代：{iteration if iteration is not None else '等待首个 checkpoint'}。")
    if control:
      stage = self.state.get("curriculum_runtime", {}).get("stage_index", active_stage(control, iteration or 0))
      value = control["angle_stages"][stage]
      lines.append(
        f"当前课程：stage {stage}，{value['angle_deg']}° ± {value['jitter_deg']}°；允许到 stage {control['max_stage']}。"
      )
      if stage + 1 < len(control["angle_stages"]):
        following = control["angle_stages"][stage + 1]
        lines.append(f"下个门槛：iter {following['iter']}，{following['angle_deg']}° ± {following['jitter_deg']}°。")
      if control.get("supervisor_audit"):
        event = control["supervisor_audit"][-1]
        lines.append(f"最近课程决定：{event['reason']}（{event['at']}）。")
    metrics = self.state.get("training_metrics", {})
    rows = metrics.get("rows", [])
    if rows:
      values = rows[-1].get("metrics", {})
      shown = [
        (label, values.get(key))
        for label, key in (
          ("reward", "Train/mean_reward"),
          ("episode steps", "Train/mean_episode_length"),
          ("action std", "Policy/mean_std"),
          ("末帧 engage match", "Metrics/climb_motion/engage_match"),
          ("新挂接事件 Hz", "Episode_Metrics/microspine_engage_event_rate_hz"),
        )
        if values.get(key) is not None
      ]
      lines += ["", "训练日志：" + "；".join(f"{name}={value:.4g}" for name, value in shown) + "。"]
      lines.append("训练平均值与末帧 match 只用于诊断；升课依据连续无重置考试及图像/视频审阅。")
    exams = sorted(self.state["exams"].values(), key=lambda item: item["checkpoint_iteration"])
    if exams:
      exam = exams[-1]
      lines += ["", f"最近考试：iter {exam['checkpoint_iteration']}，状态 {exam['status']}。"]
      if exam.get("smoke"):
        lines.append("本次为首次冒烟检查，不作为升课凭据。")
      else:
        lines.append(f"完整数值判据：{'通过' if exam_numeric_pass(exam) else '未通过或证据不完整'}。")
      lines.append(
        "视觉审阅："
        + exam.get("visual_review", {}).get(
          "reason", "已记录审阅" if exam.get("visual_review", {}).get("approved") else "待审阅"
        )
        + "。"
      )
      lines += [
        "",
        "| Profile / 角度 | Match 最小 | 首个 episode 失败 | 最大过占空比 | 进度比最小 | 真重挂循环最小 | 结果 |",
        "|---|---:|---:|---:|---:|---:|---|",
      ]
      for case in exam.get("cases", []):
        verdict = case.get("verdict", {})
        values = verdict.get("metrics", {})
        excess = values.get("pad_occupancy_excess_max_per_pad", [])
        worst_excess = (
          max(excess) if isinstance(excess, list) and excess and all(finite_number(v) for v in excess) else "—"
        )
        cells = [
          values.get("engage_match_min", "—"),
          values.get("first_episode_falls", "—"),
          worst_excess,
          values.get("progress_ratio_min", "—"),
          values.get("true_regrip_cycles_min", "—"),
        ]
        lines.append(
          f"| {case['profile']} / {case['angle_deg']:g}° | "
          + " | ".join(f"{v:.4g}" if finite_number(v) else str(v) for v in cells)
          + f" | {'通过' if verdict.get('passed') else '未通过'} |"
        )
      lines += ["", f"考试记录和 engage raster / rollout 视频：`{exam['output_dir']}`。"]
    active = self.state.get("active_evaluation")
    if active:
      lines += [
        "",
        f"正在评估：iter {active['checkpoint_iteration']}，{active['profile']} / {active['angle_deg']:g}°。",
      ]
    alerts = [*metrics.get("alerts", []), *self.state.get("errors", [])[-3:]]
    if alerts:
      lines += ["", "诊断提示：", "", *(f"- {alert}" for alert in alerts)]
    lines += [
      "",
      "升课门槛：各角度及 canonical / training-rsi 均须 match ≥ 0.95、首个 episode 零失败、每 pad 过占空比 < 0.04、最小进度比 ≥ 0.8、每环境各 pad 至少 2 次真实重挂；连续两次且对应 raster/video 审阅通过。",
      "",
    ]
    (self.monitor_dir / "feedback.md").write_text("\n".join(lines))

  def refresh(self) -> tuple[dict[str, Any], dict[int, Path]]:
    control = read_json(self.control_path)
    validate_control(control)
    checkpoints = stable_checkpoints(self.run_dir, self.args.checkpoint_stable_seconds)
    metrics = training_metrics(self.run_dir, self.args.training_log)
    runtime_path = self.run_dir / "curriculum_runtime.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {}
    runtime_stage = runtime.get("stage_index")
    if runtime_stage is not None and (type(runtime_stage) is not int or not 0 <= runtime_stage < 3):
      raise ValueError("Invalid stage_index in curriculum_runtime.json")
    current = max(max(checkpoints, default=0), metrics.get("latest_iteration") or 0, runtime.get("iteration", 0))
    self.state.update(curriculum_control=control, training_metrics=metrics, latest_iteration=current)
    self.state["curriculum_runtime"] = runtime
    for exam in self.state["exams"].values():
      if exam.get("status") == "complete":
        exam["visual_review"] = visual_verdict(exam, Path(exam["output_dir"]))
    with file_lock(self.run_dir / ".curriculum_control.lock"):
      control = read_json(self.control_path)
      validate_control(control)
      aligned, alignment_event = align_runtime_schedule(control, runtime)
      updated, event = curriculum_decision(
        aligned, current, list(self.state["exams"].values()), current_stage_index=runtime_stage
      )
      if alignment_event or event:
        atomic_json(self.control_path, updated)
        for decision in (alignment_event, event):
          if decision:
            logging.info("Curriculum decision: %s", json.dumps(decision, ensure_ascii=False))
      self.state["curriculum_control"] = updated
    self.save()
    return updated, checkpoints

  def run_case(self, checkpoint: Path, exam: dict[str, Any], profile: str, angle: float) -> dict[str, Any]:
    output = Path(exam["output_dir"]) / profile / f"angle_{angle:g}"
    output.mkdir(parents=True, exist_ok=True)
    command = [
      self.args.python,
      str(self.args.eval_script),
      "--checkpoint",
      str(checkpoint),
      "--output-dir",
      str(output),
      "--task",
      TASK,
      "--angle-deg",
      str(angle),
      "--profile",
      profile,
      "--num-envs",
      str(self.args.num_envs),
      "--duration-s",
      str(self.args.duration_s),
      "--device",
      "cuda:0",
      "--video",
    ]
    self.state["active_evaluation"] = {
      "checkpoint_iteration": exam["checkpoint_iteration"],
      "profile": profile,
      "angle_deg": angle,
      "started_at": utc_now(),
      "command": command,
      "output_dir": str(output),
    }
    self.save()
    logging.info("Evaluate iter=%d profile=%s angle=%g", exam["checkpoint_iteration"], profile, angle)
    env = dict(
      os.environ,
      CUDA_VISIBLE_DEVICES=str(self.args.gpu),
      MUJOCO_GL="egl",
      OPENBLAS_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
    )
    with (output / "evaluation.log").open("a") as stream:
      process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
      deadline = time.monotonic() + self.args.eval_timeout_seconds
      try:
        while True:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
            raise subprocess.TimeoutExpired(command, self.args.eval_timeout_seconds)
          try:
            code = process.wait(timeout=min(30, remaining))
          except subprocess.TimeoutExpired:
            self.refresh()
            continue
          if code:
            raise subprocess.CalledProcessError(code, command)
          break
      finally:
        if process.poll() is None:
          process.terminate()
          try:
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    summary = decode_result(read_json(output / "summary.json"), exam, profile, angle)
    result = {
      "profile": profile,
      "angle_deg": angle,
      "output_dir": str(output),
      "summary_path": str(output / "summary.json"),
      "verdict": numeric_verdict(summary),
      "artifacts": summary.get("artifacts", {}),
    }
    return result

  def examine(self, checkpoint: Path, iteration: int, control: dict[str, Any]) -> None:
    stage_index = active_stage(control, iteration)
    stage = control["angle_stages"][stage_index]
    output = self.run_dir / "exams" / f"iter_{iteration:06d}"
    output.mkdir(parents=True, exist_ok=True)
    exam = {
      "schema_version": 1,
      "checkpoint_iteration": iteration,
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": None,
      "stage_index": stage_index,
      "stage": copy.deepcopy(stage),
      "stage_started_iter": None,
      "num_envs": self.args.num_envs,
      "duration_s": self.args.duration_s,
      "output_dir": str(output),
      "started_at": utc_now(),
      "status": "running",
      "smoke": iteration == 0,
      "cases": [],
      "numeric_thresholds": GATES,
      "attempts": self.state["exams"].get(str(iteration), {}).get("attempts", 0) + 1,
    }
    self.state["exams"][str(iteration)] = exam
    self.state["last_exam_iteration"] = iteration
    atomic_json(output / "exam.json", exam)
    self.save()
    try:
      stage_index, stage_started_iter = checkpoint_stage(checkpoint)
      stage = control["angle_stages"][stage_index]
      exam.update(
        checkpoint_sha256=sha256(checkpoint),
        stage_index=stage_index,
        stage=copy.deepcopy(stage),
        stage_started_iter=stage_started_iter,
      )
      angles = [float(stage["angle_deg"])] if exam["smoke"] else band_angles(stage)
      for angle in angles:
        exam["cases"].append(self.run_case(checkpoint, exam, "canonical", angle))
        atomic_json(output / "exam.json", exam)
        self.refresh()  # Keep curriculum guarded while a multi-case suite is running.
      next_stage = stage_index + 1
      next_iteration = (
        control["angle_stages"][next_stage]["iter"] if next_stage < len(control["angle_stages"]) else None
      )
      near_gate = next_iteration is not None and iteration >= next_iteration - 250
      canonical_passed = all(case["verdict"]["passed"] for case in exam["cases"])
      if not exam["smoke"] and (near_gate or canonical_passed):
        for angle in angles:
          exam["cases"].append(self.run_case(checkpoint, exam, "training-rsi", angle))
          atomic_json(output / "exam.json", exam)
          self.refresh()
      if sha256(checkpoint) != exam["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation; discard this exam")
      exam["status"] = "complete"
      exam["numeric_passed"] = exam_numeric_pass(exam)
      exam["visual_review"] = visual_verdict(exam, output)
      template = {
        "schema_version": 1,
        "checkpoint_iteration": iteration,
        "checkpoint_sha256": exam["checkpoint_sha256"],
        "stage_index": stage_index,
        "approved": False,
        "raster_reviewed": False,
        "video_reviewed": False,
        "reviewer": "",
        "reviewed_at": "",
        "notes": "",
        "instructions": "实际打开本目录各 profile/angle 的 engage raster 及 rollout 视频；记录观察后另存 visual_review.json。未看或失败保持 approved=false。",
      }
      atomic_json(output / "visual_review.template.json", template)
      logging.info(
        "Exam complete iter=%d numeric_passed=%s; awaiting actual visual review", iteration, exam["numeric_passed"]
      )
    except Exception as exc:
      exam["status"] = "error"
      exam["error"] = f"{type(exc).__name__}: {exc}"
      exam["retry_after"] = time.time() + self.args.error_retry_seconds
      self.state["errors"].append(f"iter {iteration}: {exam['error']}")
      logging.exception("Exam failed at iter %d", iteration)
    finally:
      exam["finished_at"] = utc_now()
      atomic_json(output / "exam.json", exam)
      self.state.pop("active_evaluation", None)
      self.save()

  def tick(self) -> bool:
    control, checkpoints = self.refresh()
    pending = []
    for iteration in due_iterations(checkpoints, control):
      previous = self.state["exams"].get(str(iteration))
      if previous and previous.get("status") == "complete":
        continue
      if previous and previous.get("status") == "error":
        if previous.get("attempts", 1) >= self.args.max_eval_attempts or time.time() < previous.get("retry_after", 0):
          continue
      if iteration not in self.state.get("skipped_stale_iterations", []):
        pending.append(iteration)
    if not pending:
      return False
    early = [iteration for iteration in pending if iteration in (0, 25, 100)]
    iteration = min(early) if early else max(pending)
    # If evaluation falls behind, use the freshest policy and explicitly record
    # missed old periodic exams instead of spending hours draining a stale queue.
    stale = [value for value in pending if 100 < value < iteration]
    if stale:
      self.state.setdefault("skipped_stale_iterations", []).extend(stale)
      logging.warning("Coalescing stale periodic exams %s into iter %d", stale, iteration)
    self.examine(checkpoints[iteration], iteration, control)
    self.refresh()
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--gpu", default="1", help="Physical GPU assigned only to evaluation subprocesses")
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--eval-script", type=Path, default=Path(__file__).with_name("evaluate_climb_checkpoint.py"))
  parser.add_argument("--training-log", type=Path)
  parser.add_argument("--poll-seconds", type=float, default=30)
  parser.add_argument("--checkpoint-stable-seconds", type=float, default=20)
  parser.add_argument("--eval-timeout-seconds", type=float, default=1800)
  parser.add_argument("--error-retry-seconds", type=float, default=600)
  parser.add_argument("--max-eval-attempts", type=int, default=2)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--duration-s", type=float, default=30)
  parser.add_argument("--once", action="store_true", help="Refresh once and evaluate at most one pending checkpoint")
  args = parser.parse_args(argv)
  if args.poll_seconds <= 0 or args.checkpoint_stable_seconds < 0 or args.eval_timeout_seconds <= 0:
    parser.error("poll/timeout must be positive and checkpoint stability interval nonnegative")
  if args.num_envs < 1 or args.duration_s <= 0:
    parser.error("num-envs and duration-s must be positive")
  if args.error_retry_seconds < 60 or args.max_eval_attempts < 1:
    parser.error("error-retry-seconds must be at least 60 and max-eval-attempts positive")
  args.eval_script = args.eval_script.resolve()
  if args.training_log is None:
    args.training_log = args.run_dir / "train.log"
  return args


def main() -> None:
  args = parse_args()
  monitor_dir = args.run_dir / "monitor"
  monitor_dir.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
      logging.StreamHandler(),
      logging.FileHandler(monitor_dir / "supervisor.log"),
    ],
  )
  try:
    with file_lock(monitor_dir / "supervisor.lock", nonblocking=True):
      supervisor = Supervisor(args)
      logging.info("Watching %s; evaluation GPU=%s; poll=%gs", args.run_dir, args.gpu, args.poll_seconds)
      while True:
        try:
          examined = supervisor.tick()
          if args.once and examined:
            latest_exam = supervisor.state["exams"][str(supervisor.state["last_exam_iteration"])]
            if latest_exam["status"] == "error":
              raise RuntimeError(latest_exam["error"])
        except Exception as exc:
          supervisor.state["errors"].append(f"{utc_now()}: {type(exc).__name__}: {exc}")
          supervisor.save()
          logging.exception("Supervisor tick failed; existing curriculum lock remains in force")
          if args.once:
            raise
        if args.once:
          break
        time.sleep(min(args.poll_seconds, 60))
  except BlockingIOError:
    raise SystemExit(f"Another supervisor already holds {monitor_dir / 'supervisor.lock'}")
  except KeyboardInterrupt:
    logging.info("Supervisor stopped by operator; training and curriculum control left in place")


if __name__ == "__main__":
  main()
