"""Digest an rsl_rl stdout log of the Transition80 task into compact lane tables.

Usage:
  python scripts/transition80_train_digest.py LOG [--last N] [--every K] [--since ITER]

Parses every "Learning iteration i/N" block, prints (a) a per-iteration table
over the selected rows, (b) the death-cause shares over the last window, and
(c) the latest curriculum / sampling lanes. ANSI colour codes are stripped.
"""

from __future__ import annotations

import argparse
import re
import sys

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_KEYS = {
  "rew": "Mean reward",
  "eplen": "Mean episode length",
  "std": "Mean action std",
  "vloss": "Mean value loss",
  "t_out": "Episode_Termination/time_out",
  "cone": "Episode_Termination/engage_constraint",
  "bpos": "Episode_Termination/body_pos",
  "apos": "Episode_Termination/anchor_pos",
  "aori": "Episode_Termination/anchor_ori",
  "closs": "Episode_Termination/contact_loss",
  "rip": "Episode_Termination/anchor_rip",
  "wall": "Episode_Termination/wall_contact",
  "fall": "Episode_Termination/base_height",
  "grind": "Episode_Metrics/constraint_grind_violation",
  "tilt": "Episode_Metrics/constraint_tilt_violation",
  "force": "Episode_Metrics/constraint_force_violation",
  "slam": "Episode_Metrics/constraint_slam_violation",
  "unsch": "Episode_Metrics/constraint_unscheduled_violation",
  "delta": "Episode_Metrics/constraint_delta",
  "phi": "Episode_Reward/engage_progress",
  "pad_r": "Episode_Reward/motion_pad_pos",
  "bpos_r": "Episode_Reward/motion_body_pos",
  "LH": "Episode_Metrics/left_hand_engagement_occupancy",
  "RH": "Episode_Metrics/right_hand_engagement_occupancy",
  "LF": "Episode_Metrics/left_foot_engagement_occupancy",
  "RF": "Episode_Metrics/right_foot_engagement_occupancy",
  "eng_hz": "Episode_Metrics/microspine_engage_event_rate_hz",
  "rel_hz": "Episode_Metrics/microspine_forced_release_rate_hz",
  "handoff": "Episode_Metrics/mount_handoff_ready",
  "e_pad": "Metrics/climb_motion/error_pad_pos",
  "e_anc": "Metrics/climb_motion/error_anchor_pos",
  "sampH": "Metrics/climb_motion/sampling_entropy",
  "top1": "Metrics/climb_motion/sampling_top1_bin",
}
_DEATHS = ("t_out", "cone", "bpos", "apos", "aori", "closs", "rip", "wall", "fall")


def parse(path: str) -> list[dict]:
  text = _ANSI.sub("", open(path, errors="ignore").read())
  parts = re.split(r"Learning iteration (\d+)/(\d+)", text)
  rows = []
  for i in range(1, len(parts) - 2, 3):
    it = int(parts[i])
    body = parts[i + 2]
    row = {"it": it, "total": int(parts[i + 1])}
    for short, key in _KEYS.items():
      m = re.search(re.escape(key) + r":\s*([-+\d.eE]+)", body)
      row[short] = float(m.group(1)) if m else float("nan")
    row["curriculum"] = dict(
      re.findall(r"Curriculum/([\w/]+):\s*([-+\d.eE]+)", body)
    )
    m = re.search(r"Iteration time:\s*([\d.]+)s", body)
    row["it_s"] = float(m.group(1)) if m else float("nan")
    rows.append(row)
  return rows


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("--last", type=int, default=12)
  ap.add_argument("--every", type=int, default=1)
  ap.add_argument("--since", type=int, default=None)
  args = ap.parse_args()
  rows = parse(args.log)
  if not rows:
    print("no iterations parsed yet")
    sys.exit(0)
  sel = rows if args.since is None else [r for r in rows if r["it"] >= args.since]
  sel = sel[-args.last * args.every :][:: args.every] if args.every > 1 else sel[-args.last :]
  cols = [
    "it", "rew", "eplen", "std", "cone", "bpos", "closs", "rip", "wall", "t_out",
    "delta", "grind", "tilt", "force", "phi", "pad_r", "LH", "RH", "LF", "RF",
    "eng_hz", "handoff", "e_pad", "sampH", "top1",
  ]
  print(" ".join(f"{c:>7s}" for c in cols))
  for r in sel:
    out = []
    for c in cols:
      v = r[c]
      out.append(f"{int(v):7d}" if c == "it" else f"{v:7.3f}")
    print(" ".join(out))
  last = rows[-1]
  window = rows[-min(len(rows), 10):]
  totals = {k: sum(r[k] for r in window if r[k] == r[k]) for k in _DEATHS}
  denom = sum(totals.values()) or 1.0
  print(
    f"\nlatest iter {last['it']}/{last['total']}  it_time {last['it_s']:.1f}s  "
    f"rew {last['rew']:.2f}  eplen {last['eplen']:.1f}  std {last['std']:.2f}"
  )
  print("death shares (last 10 iters): " + "  ".join(
    f"{k} {totals[k] / denom:.0%}" for k in _DEATHS if totals[k] > 0
  ))
  if last["curriculum"]:
    print("curriculum lanes: " + "  ".join(f"{k}={v}" for k, v in last["curriculum"].items()))


if __name__ == "__main__":
  main()
