"""Read-only digest of a Climb Corners-A60 run's local training metrics.

Examples::

  python scripts/monitor_climb60_metrics.py RUN_DIR --log RUN_DIR/train.log
  python scripts/monitor_climb60_metrics.py RUN_DIR --json
  python scripts/monitor_climb60_metrics.py RUN_DIR --watch 60

The script never changes a curriculum or claims mastery from training scalars.
Mastery needs uninterrupted rollouts, real relatches, progress, schedule plots,
and video review. TensorBoard is local even when the training logger is W&B.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
SCALAR = re.compile(rf"^\s*([^:\n]+):\s*({NUMBER})(?:s)?\s*$", re.MULTILINE | re.IGNORECASE)
ALIASES = {
  "Mean reward": "Train/mean_reward",
  "Mean episode length": "Train/mean_episode_length",
  "Mean action std": "Policy/mean_std",
  "Steps per second": "Perf/total_fps",
  "Collection time": "Perf/collection_time",
  "Learning time": "Perf/learning_time",
  "Iteration time": "Perf/iteration_time",
}
DISPLAY = {
  "reward": "Train/mean_reward",
  "ep_steps": "Train/mean_episode_length",
  "std": "Policy/mean_std",
  "angle": "Curriculum/wall_angle",
  "end_match": "Metrics/climb_motion/engage_match",
  "eng_hz": "Episode_Metrics/microspine_engage_event_rate_hz",
  "rel_hz": "Episode_Metrics/microspine_forced_release_rate_hz",
  "mean_rise_m": "Episode_Metrics/motion_height_gained_m",
  "LH": "Episode_Metrics/left_hand_engagement_occupancy",
  "RH": "Episode_Metrics/right_hand_engagement_occupancy",
  "LF": "Episode_Metrics/left_foot_engagement_occupancy",
  "RF": "Episode_Metrics/right_foot_engagement_occupancy",
  "samp_H": "Metrics/climb_motion/sampling_entropy",
  "top1": "Metrics/climb_motion/sampling_top1_prob",
  "it_s": "Perf/iteration_time",
}
SEMANTICS = [
  "Train/mean_episode_length is policy steps (50 Hz), from the recent completed-episode buffer.",
  "Episode_Reward/* is completed-episode cumulative reward divided by fixed 30 s, not reward per lived second.",
  "Episode_Metrics/* averages each completed episode over its lived steps; motion_height_gained_m is mean rise, not terminal net progress.",
  "Metrics/climb_motion/error_* and engage_match describe the last command update before reset, not the whole rollout.",
  "Episode_Termination/* is mean per-step reset-batch count; causes can overlap and are not failure probabilities.",
  "action_saturation_fraction counts |raw action| >= 1; current policy actions are not clipped at 1.",
  "joint_soft_limit_violation covers all robot joints, including passive gimbals; the dof_pos_limits reward covers the 21 policy joints only.",
  "Training scalars do not establish mastery; require schedule and video review with uninterrupted progress and real new latches.",
]


def parse_stdout(contents: str) -> list[dict[str, Any]]:
  """Read complete iteration blocks, including non-finite numerical values."""
  clean = ANSI.sub("", contents)
  headers = list(ITERATION.finditer(clean))
  rows = {}
  for i, header in enumerate(headers):
    end = headers[i + 1].start() if i + 1 < len(headers) else len(clean)
    block = clean[header.end():end]
    # A concurrent read can encounter half of the newest print().
    if not re.search(r"Iteration time:\s*", block):
      continue
    metrics = {}
    for key, value in SCALAR.findall(block):
      key = key.strip()
      if key in ALIASES:
        tag = ALIASES[key]
      elif "/" in key:
        tag = key
      elif key.startswith("Mean ") and key.endswith(" loss"):
        tag = "Loss/" + key[5:-5]
      else:
        continue
      metrics[tag] = float(value)
    if "Perf/iteration_time" not in metrics:
      continue
    iteration = int(header.group(1))
    rows[iteration] = {"iteration": iteration, "total_iterations": int(header.group(2)), "metrics": metrics}
  return [rows[key] for key in sorted(rows)]


def load_tensorboard(run: Path) -> list[dict[str, Any]]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
  accumulator.Reload()
  rows: dict[int, dict[str, Any]] = {}
  last_times: dict[tuple[int, str], float] = {}
  for tag in accumulator.Tags().get("scalars", []):
    if tag.endswith("/time"):
      continue  # These use wall time, not learning iteration, as the x axis.
    for point in accumulator.Scalars(tag):
      key = (point.step, tag)
      if point.wall_time < last_times.get(key, -math.inf):
        continue
      last_times[key] = point.wall_time
      row = rows.setdefault(point.step, {"iteration": point.step, "metrics": {}, "wall_time": point.wall_time})
      row["wall_time"] = max(row["wall_time"], point.wall_time)
      row["metrics"][tag] = point.value
  complete = []
  for iteration in sorted(rows):
    row = rows[iteration]
    metrics = row["metrics"]
    # Logged near the end of a block; avoid declaring a newly opened scalar
    # step complete when only some episode extras have reached disk.
    if "Perf/learning_time" not in metrics:
      continue
    metrics["Perf/iteration_time"] = metrics.get("Perf/collection_time", 0) + metrics["Perf/learning_time"]
    complete.append(row)
  return complete


def window_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
  tags = sorted({tag for row in rows for tag in row["metrics"]})
  result = {}
  for tag in tags:
    values = [row["metrics"][tag] for row in rows if tag in row["metrics"]]
    finite = [value for value in values if math.isfinite(value)]
    result[tag] = {
      "n": len(values), "nonfinite": len(values) - len(finite),
      "mean": statistics.fmean(finite) if finite else None,
      "min": min(finite) if finite else None, "max": max(finite) if finite else None,
      "latest": values[-1] if values and math.isfinite(values[-1]) else None,
    }
  return result


def digest(rows: list[dict[str, Any]], run: Path, source: str, window: int = 10) -> dict[str, Any]:
  selected = rows[-window:]
  stats = window_stats(selected)
  checkpoints = [(int(match.group(1)), path.name) for path in run.glob("model_*.pt") if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))]
  alerts = []
  nonfinite = [tag for tag, value in stats.items() if value["nonfinite"]]
  if nonfinite:
    alerts.append("Non-finite scalars: " + ", ".join(nonfinite))
  engage = stats.get(DISPLAY["eng_hz"], {}).get("mean")
  if engage is not None and engage <= 1e-6:
    alerts.append("No new physical latch events in this window; reset anchors are excluded from this metric.")
  age = time.time() - rows[-1]["wall_time"] if rows and "wall_time" in rows[-1] else None
  if age is not None and age > 600:
    alerts.append("Latest complete TensorBoard iteration is over 10 minutes old; check process and stdout.")
  if not rows:
    alerts.append("No complete training iteration available yet.")
  result = {
    "schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "run": str(run.resolve()), "source": source,
    "latest_iteration": rows[-1]["iteration"] if rows else None,
    "latest_checkpoint": max(checkpoints)[1] if checkpoints else None,
    "metric_age_s": round(age, 1) if age is not None else None,
    "window_first_iteration": selected[0]["iteration"] if selected else None,
    "window_count": len(selected), "rows": selected, "window_statistics": stats,
    "alerts": alerts, "mastery": "requires_rollout_and_visual_review", "metric_semantics": SEMANTICS,
  }
  # RFC-compliant JSON uses null, with nonfinite counts preserved above.
  def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
      return None
    if isinstance(value, dict):
      return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
      return [finite_json(item) for item in value]
    return value
  return finite_json(result)


def read_run(run: Path, log: Path | None) -> tuple[list[dict[str, Any]], str]:
  sources = []
  if any(run.glob("events.out.tfevents*")):
    rows = load_tensorboard(run)
    sources.append("tensorboard")
  else:
    rows = []
  if log and log.exists():
    stdout_rows = parse_stdout(log.read_text(errors="replace"))
    sources.append(str(log))
    # Keep TensorBoard precision where available, and use completed stdout
    # iterations for the few seconds before the event writer flushes.
    merged = {row["iteration"]: row for row in stdout_rows}
    for row in rows:
      if row["iteration"] in merged:
        previous = merged[row["iteration"]]
        row = {**previous, **row, "metrics": {**previous["metrics"], **row["metrics"]}}
      merged[row["iteration"]] = row
    rows = [merged[key] for key in sorted(merged)]
  return rows, "+".join(sources) or "none"


def print_digest(result: dict[str, Any]) -> None:
  print(f"[{result['observed_at_utc']}] {result['run']}")
  print(f"source={result['source']} latest={result['latest_iteration']} checkpoint={result['latest_checkpoint']}")
  print(" ".join(f"{label:>10}" for label in ("iteration", *DISPLAY)))
  for row in result["rows"]:
    values = [f"{row['iteration']:10d}"]
    for tag in DISPLAY.values():
      value = row["metrics"].get(tag)
      values.append(f"{value:10.4g}" if value is not None else f"{'--':>10}")
    print(" ".join(values))
  for prefix in ("Episode_Termination/", "Metrics/climb_motion/error_", "Episode_Reward/", "Curriculum/"):
    selected = [(tag.removeprefix(prefix), value["mean"]) for tag, value in result["window_statistics"].items() if tag.startswith(prefix)]
    if selected:
      print(prefix + " window mean: " + " ".join(f"{name}={value:.4g}" if value is not None else f"{name}=null" for name, value in selected))
  for alert in result["alerts"]:
    print("ALERT: " + alert)
  print("Training scalars alone cannot establish mastery; review continuous progress, schedule plots, and rollout video.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run", type=Path)
  parser.add_argument("--log", type=Path, help="Optional stdout log, including iterations not yet flushed to TensorBoard")
  parser.add_argument("--last", type=int, default=10)
  parser.add_argument("--json", action="store_true", help="Print one JSON object per snapshot")
  parser.add_argument("--watch", type=float, default=0, help="Repeat at this interval in seconds; zero reads once")
  args = parser.parse_args()
  if args.last < 1 or args.watch < 0:
    parser.error("--last must be positive and --watch must be nonnegative")
  while True:
    rows, source = read_run(args.run, args.log)
    result = digest(rows, args.run, source, args.last)
    if args.json:
      print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    else:
      print_digest(result)
    if not args.watch:
      break
    time.sleep(args.watch)


if __name__ == "__main__":
  main()
