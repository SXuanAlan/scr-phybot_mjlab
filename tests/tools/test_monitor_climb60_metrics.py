"""Guard metric interpretation while reading concurrent/resumed logs."""

import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
  "climb60_digest", Path(__file__).resolve().parents[2] / "scripts/monitor_climb60_metrics.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_stdout_requires_complete_iteration_and_preserves_metric_units():
  rows = monitor.parse_stdout("""\x1b[1m Learning iteration 25/1500 \x1b[0m
     Mean reward: -2.3
     Mean episode length: 20.0
     Mean surrogate loss: -0.034
     Episode_Termination/lost_support: 15.5
     Episode_Metrics/microspine_engage_event_rate_hz: 0.0
     Metrics/climb_motion/engage_match: 0.1
     Curriculum/wall_angle: 45.0
     Iteration time: 16.70s
Learning iteration 26/1500
     Mean reward: -2.0
""")
  assert len(rows) == 1
  assert rows[0]["iteration"] == 25
  assert rows[0]["metrics"]["Train/mean_episode_length"] == 20
  assert rows[0]["metrics"]["Episode_Termination/lost_support"] == 15.5
  assert rows[0]["metrics"]["Loss/surrogate"] == -0.034


def test_latest_resumed_iteration_wins():
  template = "Learning iteration 25/100\nMean reward: {reward}\nIteration time: 1s\n"
  rows = monitor.parse_stdout(template.format(reward=2) + template.format(reward=3))
  assert len(rows) == 1
  assert rows[0]["metrics"]["Train/mean_reward"] == 3


def test_digest_alerts_on_no_latches_and_nonfinite_without_claiming_mastery(tmp_path):
  rows = [
    {
      "iteration": 100,
      "metrics": {
        "Episode_Metrics/microspine_engage_event_rate_hz": 0.0,
        "Train/mean_reward": math.nan,
        "Episode_Termination/lost_support": 2,
        "Episode_Termination/bad_tracking": 4,
      },
    }
  ]
  (tmp_path / "model_25.pt").touch()
  (tmp_path / "model_100.pt").touch()
  report = monitor.digest(rows, tmp_path, "test")
  assert report["latest_checkpoint"] == "model_100.pt"
  assert report["rows"][0]["metrics"]["Train/mean_reward"] is None
  assert report["window_statistics"]["Train/mean_reward"]["nonfinite"] == 1
  assert len(report["alerts"]) == 2
  assert report["mastery"] == "requires_rollout_and_visual_review"
  assert report["window_statistics"]["Episode_Termination/lost_support"]["mean"] == 2
  json.dumps(report, allow_nan=False)
