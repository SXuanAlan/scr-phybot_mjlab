"""Passive-hold feasibility gate + tenable-birth mask for a mount clip.

Runs ``transition80_seed_diag.py`` (hold-birth-pose mode, seed guard = the
task's birth freeze) on every Nth frame of the clip, parses the release
log, and reports per frame the fraction of envs that lose a SCHEDULED seed
after the guard lifts. Two outputs:

* GATE: every seeded frame must keep >= (1 - fail_frac) of its seeds
  (default 95 %) — the reference-validator rule that closes the gap between
  the kinematic validator and the spring-field plant (2026-08-22);
* MASK: frames losing >= mask_frac (default 20 %) are merged into
  ``(start, end)`` ranges (end exclusive, padded to the sampling stride) to
  paste into ``TRANSITION80_BIRTH_EXCLUDE_FRAMES`` until the clip is fixed.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/transition80_birth_mask.py run  [--stride 10] [--guard 25] [--steps 60] [--clip PATH] [--out FILE]
  python scripts/transition80_birth_mask.py parse FILE [--clip PATH]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PY = sys.executable
_DIAG = _REPO / "scripts" / "transition80_seed_diag.py"
_DEFAULT_CLIP = _REPO / "src/scr_phybot_mjab/assets/motions/climb_mount_prologue_a80.npy"
_EE = ("LH", "RH", "LF", "RF")


def parse(text: str, nenv: int = 128) -> list[dict]:
  """Per-frame records: frame, ref_engaged, post-guard losses per EE."""
  out = []
  for block in re.split(r"\n=== frame ", text)[1:]:
    head = re.match(r"(\d+) \(clock \d+\) ref_engaged (\{.*?\})", block)
    if not head:
      continue
    frame = int(head.group(1))
    ref = eval(head.group(2))  # noqa: S307 - our own print format
    guard = re.search(r"guard=(\d+)", block)
    guard_steps = int(guard.group(1)) if guard else 0
    lost = {ee: 0 for ee in _EE}
    reasons: dict[str, int] = {}
    for ee, reason, n, med, mn in re.findall(
      r"releases (\w+) ([\w-]+): n=(\d+) first-step med (\d+) min (\d+)", block
    ):
      if int(mn) >= max(0, guard_steps - 1):  # post-guard (the lift step counts)
        lost[ee] += int(n)
        reasons[f"{ee} {reason}"] = reasons.get(f"{ee} {reason}", 0) + int(n)
    sched = [ee for ee in _EE if ref.get(ee, False)]
    frac = max((lost[ee] / nenv for ee in sched), default=0.0)
    out.append({"frame": frame, "ref": ref, "loss": frac, "reasons": reasons})
  return out


def ranges_from(records: list[dict], stride: int, threshold: float) -> list[tuple[int, int]]:
  bad = sorted(r["frame"] for r in records if r["loss"] >= threshold)
  ranges: list[list[int]] = []
  for f in bad:
    if ranges and f <= ranges[-1][1]:
      ranges[-1][1] = f + stride
    else:
      ranges.append([f, f + stride])
  return [(a, b) for a, b in ranges]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("run", "parse"))
  ap.add_argument("file", nargs="?", help="seed_diag output to parse (parse mode)")
  ap.add_argument("--clip", default=str(_DEFAULT_CLIP))
  ap.add_argument("--stride", type=int, default=10)
  ap.add_argument("--guard", type=int, default=25)
  ap.add_argument("--steps", type=int, default=60)
  ap.add_argument("--out", default=None)
  ap.add_argument("--fail-frac", type=float, default=0.05, help="gate: max seed loss per frame")
  ap.add_argument("--mask-frac", type=float, default=0.20, help="mask: loss that excludes a frame")
  args = ap.parse_args()

  data = np.load(args.clip, allow_pickle=True).item()
  engaged = np.asarray(data["ee_engaged"]).astype(bool)
  names = list(data["segment_names"])
  seg = np.asarray(data["phase_segment"])
  seeded_frames = [f for f in range(0, engaged.shape[0], args.stride) if engaged[f].any()]

  if args.mode == "run":
    env = dict(os.environ, SEED_DIAG_GUARD=str(args.guard), MUJOCO_GL="egl", SEED_DIAG_CLIP=str(Path(args.clip).resolve()))
    env.setdefault("CUDA_VISIBLE_DEVICES", "2")
    cmd = [_PY, str(_DIAG), ",".join(map(str, seeded_frames)), str(args.steps)]
    text = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_REPO).stdout
    if args.out:
      Path(args.out).write_text(text)
  else:
    text = Path(args.file).read_text()

  records = parse(text)
  worst = 0.0
  print(f"{'frame':>5} {'segment':18} {'sched':6} {'loss':>6}  post-guard releases")
  for r in records:
    fr = r["frame"]
    flag = "  FAIL" if r["loss"] >= args.fail_frac else ""
    print(
      f"{fr:5d} {names[seg[fr]]:18} {''.join('1' if r['ref'][e] else '0' for e in _EE):6} "
      f"{r['loss']:6.0%}  " + "; ".join(f"{k} {v}" for k, v in r["reasons"].items()) + flag
    )
    worst = max(worst, r["loss"])
  mask = ranges_from(records, args.stride, args.mask_frac)
  print(f"\nGATE ({1 - args.fail_frac:.0%} of seeds kept on every seeded frame): "
        f"{'PASS' if worst < args.fail_frac else 'FAIL'}  (worst loss {worst:.0%})")
  print(f"MASK (loss >= {args.mask_frac:.0%}): TRANSITION80_BIRTH_EXCLUDE_FRAMES = {tuple(mask)}")
  return 0 if worst < args.fail_frac else 1


if __name__ == "__main__":
  raise SystemExit(main())
