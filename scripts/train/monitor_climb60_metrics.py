"""Read-only digest of a Climb Corners-A60 run's local training metrics.

Examples::

  python scripts/train/monitor_climb60_metrics.py RUN_DIR --log RUN_DIR/train.log
  python scripts/train/monitor_climb60_metrics.py RUN_DIR --json
  python scripts/train/monitor_climb60_metrics.py RUN_DIR --watch 60

The script never changes a curriculum or claims mastery from training scalars.
Mastery needs uninterrupted rollouts, real relatches, progress, schedule plots,
and video review. TensorBoard is local even when the training logger is W&B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scr_phybot_mjab.tools.training_monitor.metrics import digest, print_digest, read_run


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run", type=Path)
  parser.add_argument(
    "--log", type=Path, help="Optional stdout log, including iterations not yet flushed to TensorBoard"
  )
  parser.add_argument("--last", type=int, default=10)
  parser.add_argument("--json", action="store_true", help="Print one JSON object per snapshot")
  parser.add_argument("--watch", type=float, default=0, help="Repeat at this interval in seconds; zero reads once")
  args = parser.parse_args(argv)
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
