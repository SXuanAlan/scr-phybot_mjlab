"""Continuously monitor the latest PHYBot C2 AMP training run."""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_RUN_ROOT = Path("logs/rsl_rl/phybot_c2_walk_amp")
TRAIN_MARKERS = (
  "scripts/train.py Mjlab-Walk-",
  "scripts/train_warmstart.py",
)
WATCH_TAGS = (
  "Loss/amp_loss",
  "Loss/grad_pen_loss",
  "Loss/accuracy_policy",
  "Loss/accuracy_expert",
  "Train/mean_style_reward",
  "Train/mean_task_reward",
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Episode_Termination/fell_over",
  "Episode_Termination/illegal_ground_contact",
  "Metrics/twist/error_vel_xy",
  "Metrics/twist/error_vel_yaw",
  "Metrics/stand_foot_speed_mean",
  "Metrics/stand_missing_contact_mean",
  "Metrics/preset_forward_actual",
  "Metrics/preset_yaw_actual",
  "Curriculum/stand_feet_stationary_weight",
  "Curriculum/stand_still_weight",
)


def _now() -> str:
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _latest_run(run_root: Path) -> Path | None:
  if not run_root.exists():
    return None
  runs = [
    path
    for path in run_root.iterdir()
    if path.is_dir() and any(path.glob("events.out.tfevents*"))
  ]
  if not runs:
    return None
  return max(runs, key=lambda path: path.stat().st_mtime)


def _active_train_processes() -> list[str]:
  result = subprocess.run(
    ["ps", "-eo", "pid,etimes,pcpu,pmem,args"],
    check=False,
    text=True,
    capture_output=True,
  )
  lines = []
  for line in result.stdout.splitlines():
    if any(marker in line for marker in TRAIN_MARKERS) and "monitor_amp_training.py" not in line:
      lines.append(line.strip())
  return lines


def _gpu_summary() -> str:
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=index,utilization.gpu,memory.used,memory.total",
      "--format=csv,noheader,nounits",
    ],
    check=False,
    text=True,
    capture_output=True,
  )
  if result.returncode != 0:
    return "gpu=unavailable"
  chunks = []
  for line in result.stdout.splitlines():
    idx, util, used, total = [item.strip() for item in line.split(",")]
    chunks.append(f"gpu{idx}:{util}% {used}/{total}MiB")
  return " ".join(chunks)


def _load_scalars(run: Path) -> tuple[dict[str, tuple[int, float]], int]:
  accumulator = EventAccumulator(str(run))
  accumulator.Reload()
  tags = accumulator.Tags().get("scalars", [])
  latest: dict[str, tuple[int, float]] = {}
  nan_count = 0
  for tag in tags:
    values = accumulator.Scalars(tag)
    for item in values:
      if not math.isfinite(item.value):
        nan_count += 1
        break
    if tag in WATCH_TAGS and values:
      item = values[-1]
      latest[tag] = (item.step, item.value)
  return latest, nan_count


def _latest_checkpoint(run: Path) -> str:
  checkpoints = sorted(run.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
  if not checkpoints:
    return "none"
  return checkpoints[-1].name


def _format_value(metrics: dict[str, tuple[int, float]], tag: str) -> str:
  if tag not in metrics:
    return "missing"
  _, value = metrics[tag]
  return f"{value:.4g}"


def print_status(run_root: Path) -> None:
  run = _latest_run(run_root)
  processes = _active_train_processes()
  print(f"\n[{_now()}] active_train={len(processes)} {_gpu_summary()}", flush=True)
  if processes:
    for process in processes:
      print(f"  process: {process}", flush=True)
  if run is None:
    print(f"  no TensorBoard run found under {run_root}", flush=True)
    return

  metrics, nan_count = _load_scalars(run)
  steps = [step for step, _ in metrics.values()]
  latest_step = max(steps) if steps else -1
  print(
    f"  run={run.name} step={latest_step} ckpt={_latest_checkpoint(run)} nan_tags={nan_count}",
    flush=True,
  )
  print(
    "  "
    f"style={_format_value(metrics, 'Train/mean_style_reward')} "
    f"task={_format_value(metrics, 'Train/mean_task_reward')} "
    f"reward={_format_value(metrics, 'Train/mean_reward')} "
    f"len={_format_value(metrics, 'Train/mean_episode_length')} "
    f"amp_loss={_format_value(metrics, 'Loss/amp_loss')} "
    f"grad_pen={_format_value(metrics, 'Loss/grad_pen_loss')}",
    flush=True,
  )
  print(
    "  "
    f"acc_policy={_format_value(metrics, 'Loss/accuracy_policy')} "
    f"acc_expert={_format_value(metrics, 'Loss/accuracy_expert')} "
    f"fell={_format_value(metrics, 'Episode_Termination/fell_over')} "
    f"illegal={_format_value(metrics, 'Episode_Termination/illegal_ground_contact')} "
    f"err_xy={_format_value(metrics, 'Metrics/twist/error_vel_xy')} "
    f"err_yaw={_format_value(metrics, 'Metrics/twist/error_vel_yaw')} "
    f"stand_foot={_format_value(metrics, 'Metrics/stand_foot_speed_mean')} "
    f"stand_miss={_format_value(metrics, 'Metrics/stand_missing_contact_mean')}",
    flush=True,
  )
  print(
    "  "
    f"stand_weight={_format_value(metrics, 'Curriculum/stand_feet_stationary_weight')} "
    f"still_weight={_format_value(metrics, 'Curriculum/stand_still_weight')}",
    flush=True,
  )

  warnings = []
  style = metrics.get("Train/mean_style_reward", (-1, float("nan")))[1]
  acc_policy = metrics.get("Loss/accuracy_policy", (-1, float("nan")))[1]
  acc_expert = metrics.get("Loss/accuracy_expert", (-1, float("nan")))[1]
  if nan_count:
    warnings.append("non-finite TensorBoard scalar detected")
  if math.isfinite(style) and latest_step >= 200 and style < 0.25:
    warnings.append("style_reward below 0.25 after warmup")
  if (
    math.isfinite(acc_policy)
    and math.isfinite(acc_expert)
    and latest_step >= 200
    and acc_policy > 0.98
    and acc_expert > 0.98
  ):
    warnings.append("discriminator accuracy pinned high")
  if warnings:
    print("  WARN: " + "; ".join(warnings), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
  parser.add_argument("--interval-s", type=float, default=60.0)
  parser.add_argument("--once", action="store_true")
  args = parser.parse_args()

  while True:
    print_status(args.run_root)
    if args.once:
      return
    time.sleep(args.interval_s)


if __name__ == "__main__":
  main()
