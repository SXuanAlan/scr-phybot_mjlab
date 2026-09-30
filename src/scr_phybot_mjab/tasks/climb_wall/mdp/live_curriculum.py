"""Run-local, file-controlled permission to advance the 60-degree curriculum.

The nominal ladder stays iteration based. A supervisor may postpone future
boundaries and approve the next stage after checkpoint/visual evaluation.
Missing, malformed or stale control cannot silently advance the wall.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .curriculums import CORNERS60_WALL_ANGLE_STAGES, wall_angle


def initial_control() -> dict:
  return {
    "schema_version": 1, "max_stage": 0,
    "angle_stages": [dict(s) for s in CORNERS60_WALL_ANGLE_STAGES],
    "note": "Hold the current stage until checkpoint metrics and visual review pass.",
  }


def atomic_json(path: Path, value: dict) -> None:
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
  temporary.replace(path)


def validate_control(value: dict) -> dict:
  if value.get("schema_version") != 1:
    raise ValueError("Unknown curriculum control schema")
  allowed = value.get("max_stage")
  if type(allowed) is not int or not 0 <= allowed < 3:
    raise ValueError("max_stage must be 0, 1 or 2")
  stages = value.get("angle_stages")
  if not isinstance(stages, list) or len(stages) != 3:
    raise ValueError("The four-corner task requires its three-stage ladder")
  previous = -1
  for index, (stage, nominal) in enumerate(zip(stages, CORNERS60_WALL_ANGLE_STAGES, strict=True)):
    start = stage.get("iter")
    if type(start) is not int or start <= previous or start < nominal["iter"]:
      raise ValueError("Stage iterations must increase and cannot precede the original ladder")
    if index == 0 and start != 0:
      raise ValueError("The first stage must start at iteration zero")
    if stage.get("angle_deg") != nominal["angle_deg"] or stage.get("jitter_deg") != nominal["jitter_deg"]:
      raise ValueError("Runtime curriculum control may postpone stages, not alter the angle bands")
    previous = start
  return value


class LiveCurriculumControl:
  def __init__(self, path: str | Path):
    self.path = Path(path)
    self.control = initial_control()
    self.active_stage = 0
    self.last_iteration = -1
    self.last_error = None
    self.last_report = None
    self.stage_started_iter = 0

  def resolve(self, iteration: int) -> dict:
    if iteration == self.last_iteration:
      return dict(self.control["angle_stages"][self.active_stage])
    self.last_iteration = iteration
    try:
      candidate = validate_control(json.loads(self.path.read_text()))
      if candidate["max_stage"] < self.active_stage:
        raise ValueError("Cannot withdraw a stage that is already active")
      if candidate["angle_stages"][:self.active_stage + 1] != self.control["angle_stages"][:self.active_stage + 1]:
        raise ValueError("Cannot retime stages that have already started")
      self.control = candidate
      self.last_error = None
    except (OSError, ValueError, TypeError, AttributeError) as error:
      message = str(error)
      if message != self.last_error:
        print(f"[curriculum] retaining last valid stage: {message}", flush=True)
      self.last_error = message

    scheduled = max(i for i, stage in enumerate(self.control["angle_stages"]) if iteration >= stage["iter"])
    desired = min(scheduled, self.control["max_stage"])
    # Each harder stage retains at least its original training duration even
    # if approval arrives after a boundary while the supervisor is offline.
    if desired > self.active_stage:
      nominal = CORNERS60_WALL_ANGLE_STAGES
      minimum_duration = nominal[self.active_stage + 1]["iter"] - nominal[self.active_stage]["iter"]
      if iteration - self.stage_started_iter >= minimum_duration and self.last_error is None:
        self.active_stage += 1
        self.stage_started_iter = iteration
    stage = self.control["angle_stages"][self.active_stage]
    report = {
      "iteration": iteration, "stage_index": self.active_stage,
      "stage_started_iter": self.stage_started_iter,
      "angle_deg": stage["angle_deg"], "jitter_deg": stage["jitter_deg"],
      "max_stage": self.control["max_stage"], "scheduled_stage_index": scheduled,
      "waiting_for_review": scheduled > self.active_stage,
      "control_error": self.last_error,
    }
    key = (report["stage_index"], report["max_stage"], report["waiting_for_review"], self.last_error)
    if key != self.last_report:
      print("[curriculum] " + json.dumps(report), flush=True)
      self.last_report = key
      atomic_json(self.path.with_name("curriculum_runtime.json"), report)
    elif iteration % 25 == 0:
      atomic_json(self.path.with_name("curriculum_runtime.json"), report)
    return dict(stage)


def file_controlled_wall_angle(env, env_ids, *, control_file: str,
                              angle_stages, steps_per_iter=96,
                              angle_jitter_deg=0., jitter_below_deg=90., wall_entity_name="wall"):
  del angle_stages  # The run-local control contains the validated full ladder.
  controller = getattr(env, "_live_climb_curriculum", None)
  if controller is None:
    controller = env._live_climb_curriculum = LiveCurriculumControl(control_file)
  stage = controller.resolve(env.common_step_counter // steps_per_iter)
  return wall_angle(
    env, env_ids, angle_stages=[{**stage, "iter": 0}], steps_per_iter=steps_per_iter,
    angle_jitter_deg=angle_jitter_deg, jitter_below_deg=jitter_below_deg,
    wall_entity_name=wall_entity_name)
