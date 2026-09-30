"""Curriculum supervision must fail closed and preserve evidence/checkpoint identity."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "supervise_climb60.py"
SPEC = importlib.util.spec_from_file_location("climb60_supervisor", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def control():
  return {"schema_version": 1, "max_stage": 0, "angle_stages": [
    {"iter": 0, "angle_deg": 45, "jitter_deg": 5},
    {"iter": 1500, "angle_deg": 55, "jitter_deg": 5},
    {"iter": 3300, "angle_deg": 60, "jitter_deg": 0},
  ]}


def metrics():
  return {
    "engage_match_min": .96, "first_episode_falls": 0,
    "pad_occupancy_excess_max_per_pad": [.02, .01, -.1, .039],
    "progress_ratio_min": .85, "true_regrip_cycles_min": 2,
    "full_30s_survival_rate": 1., "reference_cycles_completed_min": 3,
  }


def exam(iteration, *, passed=True, reviewed=True, stage_index=0):
  stage = control()["angle_stages"][stage_index]
  return {
    "checkpoint_iteration": iteration, "checkpoint_sha256": f"hash-{iteration}",
    "stage_index": stage_index, "stage": stage, "num_envs": 32, "duration_s": 30,
    "status": "complete", "smoke": False,
    "cases": [{"profile": profile, "angle_deg": angle, "verdict": {"passed": passed}}
              for profile, angle in supervisor.expected_case_keys(stage)],
    "visual_review": {"approved": reviewed},
  }


@pytest.mark.parametrize("key,value", [("max_stage", True), ("max_stage", 3), ("schema_version", 2)])
def test_control_validation_rejects_bad_fields(key, value):
  value_control = control()
  value_control[key] = value
  with pytest.raises(ValueError):
    supervisor.validate_control(value_control)


def test_control_cannot_advance_nominal_time_or_change_hardware_bands():
  value = control()
  value["angle_stages"][1]["iter"] = 1400
  with pytest.raises(ValueError, match="original curriculum"):
    supervisor.validate_control(value)
  value = control()
  value["angle_stages"][1]["angle_deg"] = 60
  with pytest.raises(ValueError, match="restricted"):
    supervisor.validate_control(value)


def test_stable_checkpoints_and_milestones_include_guard_nodes(tmp_path):
  for iteration in (0, 25, 100, 250, 1250, 1400, 1450, 1500):
    path = tmp_path / f"model_{iteration}.pt"
    path.write_bytes(b"checkpoint")
    os.utime(path, (100, 100 if iteration < 1500 else 195))
  checkpoints = supervisor.stable_checkpoints(tmp_path, 20, now=200)
  assert 1500 not in checkpoints
  assert supervisor.due_iterations(checkpoints, control()) == [0, 25, 100, 250, 1250, 1400, 1450]


@pytest.mark.parametrize("key,value", [
  ("engage_match_min", float("nan")), ("progress_ratio_min", None),
  ("true_regrip_cycles_min", 1), ("first_episode_falls", 1),
  ("full_30s_survival_rate", .99), ("reference_cycles_completed_min", 1.99),
  ("pad_occupancy_excess_max_per_pad", [.04, 0, 0, 0]),
])
def test_missing_nonfinite_failed_or_one_bad_pad_cannot_pass(key, value):
  values = metrics()
  assert supervisor.numeric_verdict(values)["passed"]
  values[key] = value
  assert not supervisor.numeric_verdict(values)["passed"]


def test_gate_requires_two_recent_complete_exams_and_visuals():
  value = control()
  for exams in (
    [exam(1250)], [exam(1250), exam(1400, reviewed=False)],
    [exam(1250), exam(1400, passed=False)], [exam(250), exam(500)],
    [exam(1250), exam(1300, passed=False), exam(1400)],
  ):
    updated, event = supervisor.curriculum_decision(value, 1400, exams)
    assert updated == value and event is None
  updated, event = supervisor.curriculum_decision(value, 1400, [exam(1250), exam(1400)])
  assert updated["max_stage"] == 1
  assert updated["angle_stages"] == value["angle_stages"]
  assert event["evidence_iterations"] == [1250, 1400]
  assert value["max_stage"] == 0


def test_canonical_only_or_tiny_test_cannot_approve():
  first, second = exam(1250), exam(1400)
  first["cases"] = [item for item in first["cases"] if item["profile"] == "canonical"]
  assert not supervisor.exam_numeric_pass(first)
  first = exam(1250)
  first["num_envs"] = 4
  assert not supervisor.exam_numeric_pass(first)
  first["num_envs"] = 32
  first["duration_s"] = 1
  assert not supervisor.exam_numeric_pass(first)
  assert not supervisor.consecutive_passes([first, second], 0)


def test_missing_evidence_delays_all_future_stages_and_preserves_current():
  original = control()
  updated, event = supervisor.curriculum_decision(original, 1450, [])
  assert [item["iter"] for item in updated["angle_stages"]] == [0, 1750, 3550]
  assert updated["max_stage"] == 0
  assert event["delay_iterations"] == 250
  assert original == control()
  # A late/offline monitor catches up without granting permission or changing angles.
  updated, event = supervisor.curriculum_decision(original, 2250, [])
  assert updated["angle_stages"][1]["iter"] == 2500
  assert event["delay_iterations"] == 1000


def test_runtime_stage_is_authoritative_and_never_retimes_started_stage():
  value = control()
  value["max_stage"] = 1
  # Late stage-1 approval has not entered stage 1 yet, despite its nominal date.
  updated, event = supervisor.curriculum_decision(value, 4000, [], current_stage_index=0)
  assert event is None and updated == value
  updated, event = supervisor.curriculum_decision(value, 4000, [], current_stage_index=1)
  assert updated["angle_stages"][:2] == value["angle_stages"][:2]
  assert updated["angle_stages"][2]["iter"] > 4050
  assert updated["max_stage"] == 1


def test_late_stage_entry_retimes_future_gate_and_invalidates_old_approval():
  value = control()
  value["max_stage"] = 2
  runtime = {"stage_index": 1, "stage_started_iter": 2200}
  aligned, event = supervisor.align_runtime_schedule(value, runtime)
  assert aligned["angle_stages"][:2] == value["angle_stages"][:2]
  assert aligned["angle_stages"][2]["iter"] == 4000
  assert aligned["max_stage"] == 1
  assert event["previous_max_stage"] == 2
  # Checks near the obsolete 3300 boundary must not unlock 60 degrees.
  old_exams = [exam(3050, stage_index=1), exam(3200, stage_index=1)]
  updated, event = supervisor.curriculum_decision(aligned, 3500, old_exams, current_stage_index=1)
  assert event is None and updated["max_stage"] == 1
  recent = [exam(3750, stage_index=1), exam(3900, stage_index=1)]
  updated, event = supervisor.curriculum_decision(aligned, 3900, recent, current_stage_index=1)
  assert updated["max_stage"] == 2 and event["effective_iteration"] == 4000
  assert supervisor.align_runtime_schedule(aligned, runtime) == (aligned, None)


def test_visual_review_binds_hash_stage_and_real_artifacts(tmp_path):
  record = exam(1250)
  for case in record["cases"]:
    case["artifacts"] = {"engage_schedule": str(tmp_path / "raster.png"), "video": str(tmp_path / "rollout.mp4")}
  review = {key: record[key] for key in ("checkpoint_iteration", "checkpoint_sha256", "stage_index")}
  review.update(approved=True, raster_reviewed=True, video_reviewed=True,
                reviewer="fixture-reviewer", reviewed_at="test-time", notes="Fixture observation")
  supervisor.atomic_json(tmp_path / "visual_review.json", review)
  assert not supervisor.visual_verdict(record, tmp_path)["approved"]
  (tmp_path / "raster.png").write_bytes(b"fixture")
  (tmp_path / "rollout.mp4").write_bytes(b"fixture")
  assert supervisor.visual_verdict(record, tmp_path)["approved"]
  review["checkpoint_sha256"] = "other-checkpoint"
  supervisor.atomic_json(tmp_path / "visual_review.json", review)
  assert not supervisor.visual_verdict(record, tmp_path)["approved"]


def test_evaluator_identity_actor_and_horizon_are_checked():
  record = exam(1250)
  result = {"profile": "canonical", "angle_deg": 45, "actor_state_verified": True,
            "auto_reset": False, "num_envs": 32, "duration_s": 30}
  summary = {"complete": True, "checkpoint": {"sha256": record["checkpoint_sha256"], "iteration": 1250}, "results": [result]}
  assert supervisor.decode_result(summary, record, "canonical", 45) is result
  for mutate in (lambda row: row.update(complete=False),
                 lambda row: row["checkpoint"].update(sha256="other"),
                 lambda row: row["results"][0].update(auto_reset=True),
                 lambda row: row["results"][0].update(duration_s=1)):
    broken = copy.deepcopy(summary)
    mutate(broken)
    with pytest.raises(ValueError):
      supervisor.decode_result(broken, record, "canonical", 45)


def test_once_fixture_persists_results_feedback_and_keeps_stage_locked(tmp_path, monkeypatch):
  fake = tmp_path / "fake_evaluator.py"
  fake.write_text('''import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
for arg in ('checkpoint','output-dir','task','angle-deg','profile','num-envs','duration-s','device'):
  p.add_argument('--'+arg)
p.add_argument('--video', action='store_true')
a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
for name in ('raster.png','video.mp4'): (out/name).write_bytes(b'fixture artifact')
m=METRICS
m.update(profile=a.profile,angle_deg=float(a.angle_deg),actor_state_verified=True,auto_reset=False,
  num_envs=int(a.num_envs),duration_s=float(a.duration_s),
  artifacts={'engage_schedule':str(out/'raster.png'),'video':str(out/'video.mp4')})
ckpt=Path(a.checkpoint)
(out/'summary.json').write_text(json.dumps({'complete':True,'results':[m],
 'checkpoint':{'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'iteration':int(ckpt.stem.split('_')[1])}}))
'''.replace("METRICS", repr(metrics())))
  supervisor.atomic_json(tmp_path / "curriculum_control.json", control())
  checkpoint = tmp_path / "model_1250.pt"
  checkpoint.write_bytes(b"fixture checkpoint")
  monkeypatch.setattr(supervisor, "checkpoint_stage", lambda path: (0, 0))
  monkeypatch.setattr(supervisor, "training_metrics", lambda run, log: {"latest_iteration": 1250, "alerts": []})
  args = supervisor.parse_args([str(tmp_path), "--once", "--checkpoint-stable-seconds", "0", "--eval-script", str(fake), "--python", sys.executable])
  runner = supervisor.Supervisor(args)
  assert runner.tick()
  status = json.loads((tmp_path / "monitor/status.json").read_text())
  record = status["exams"]["1250"]
  assert record["status"] == "complete"
  assert record["numeric_passed"] and len(record["cases"]) == 6
  assert not record["visual_review"]["approved"]
  assert (tmp_path / "monitor/feedback.md").exists()
  assert (tmp_path / "exams/iter_001250/visual_review.template.json").exists()
  assert json.loads((tmp_path / "curriculum_control.json").read_text())["max_stage"] == 0
  assert not runner.tick()  # Persistent checkpoint accounting avoids duplicate evaluation.


def test_broken_tool_has_cooldown_and_bounded_attempts(tmp_path, monkeypatch):
  fake = tmp_path / "broken.py"
  fake.write_text("raise SystemExit(2)\n")
  supervisor.atomic_json(tmp_path / "curriculum_control.json", control())
  (tmp_path / "model_0.pt").write_bytes(b"fixture")
  monkeypatch.setattr(supervisor, "checkpoint_stage", lambda path: (0, 0))
  monkeypatch.setattr(supervisor, "training_metrics", lambda run, log: {"latest_iteration": 0, "alerts": []})
  args = supervisor.parse_args([str(tmp_path), "--once", "--checkpoint-stable-seconds", "0", "--eval-script", str(fake), "--python", sys.executable])
  runner = supervisor.Supervisor(args)
  assert runner.tick()
  record = runner.state["exams"]["0"]
  assert record["status"] == "error" and record["attempts"] == 1
  assert not runner.tick()  # No full retry every 30-second poll.
  record["retry_after"] = 0
  assert runner.tick()
  record = runner.state["exams"]["0"]
  assert record["status"] == "error" and record["attempts"] == 2
  record["retry_after"] = 0
  assert not runner.tick()  # Maximum attempts survives the cooldown expiry.
  assert json.loads((tmp_path / "curriculum_control.json").read_text())["max_stage"] == 0
