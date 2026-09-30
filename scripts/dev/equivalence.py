"""Record and compare behaviour snapshots of the kept tasks.

  python scripts/dev/equivalence.py record artifacts/equivalence/G1
  python scripts/dev/equivalence.py record artifacts/equivalence/G1b    # same code again
  python scripts/dev/equivalence.py floor artifacts/equivalence/G1 artifacts/equivalence/G1b
  python scripts/dev/equivalence.py compare artifacts/equivalence/G1 artifacts/equivalence/NEW

MuJoCo Warp is not bitwise deterministic, so rollouts are judged against the
same-code noise floor stored in ``<base>/floor.npz``; cfg, model and clips
must match exactly.

A behaviour-preserving commit must compare clean on every tier. When a
function or class only moves, add ``"old.qualname" = "new.qualname"`` to
``equivalence_renames.toml`` in the same commit. To snapshot another checkout,
run it with ``PYTHONPATH=<checkout>/src``; the editable install otherwise
imports the main tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RENAMES = Path(__file__).with_name("equivalence_renames.toml")
TIERS = ("cfg", "model", "rollout", "clips")
_ENV_KEYS = ("PHYBOT_C2_PAD_GIMBAL", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "PYTHONPATH")


def main(argv: list[str] | None = None) -> int:
  _pin_hash_seed()
  args = _parse_args(argv)
  _check_source_tree()
  if args.command == "record":
    return record(args)
  if args.command == "floor":
    return floor(args)
  return compare(args)


def record(args: argparse.Namespace) -> int:
  out = args.snapshot
  if out.exists() and any(out.iterdir()) and not args.force:
    raise SystemExit(f"{out} is not empty (use --force to add to it)")
  out.mkdir(parents=True, exist_ok=True)
  (out / "manifest.json").write_text(json.dumps(_manifest(args), indent=1) + "\n")
  failures = []
  if {"cfg", "model"} & set(args.tiers):
    _record_static(out, args.tiers, strict=args.strict_tasks)
  if "rollout" in args.tiers:
    failures += _record_rollouts(out, args)
  if "clips" in args.tiers:
    failures += _record_clips(out)
  for failure in failures:
    print(f"FAILED {failure}", flush=True)
  return 1 if failures else 0


def compare(args: argparse.Namespace) -> int:
  from scr_phybot_mjab.tools.equivalence import compare as cmp

  renames = tomllib.loads(RENAMES.read_text()).get("renames", {})
  findings = []
  if "cfg" in args.tiers:
    findings += cmp.compare_cfg(args.base, args.new, renames)
  if "model" in args.tiers:
    findings += cmp.compare_model(args.base, args.new)
  if "rollout" in args.tiers:
    floor_file = args.base / "floor.npz"
    noise = dict(np.load(floor_file)) if floor_file.exists() else {}
    if not noise:
      print("no floor.npz in base: rollouts must match exactly", flush=True)
    findings += cmp.compare_rollouts(args.base, args.new, noise)
  if "clips" in args.tiers:
    findings += cmp.compare_clips(args.base, args.new)
  _print_findings(findings)
  return 0 if all(f.ok for f in findings) else 1


def floor(args: argparse.Namespace) -> int:
  from scr_phybot_mjab.tools.equivalence.compare import noise_floor

  noise = noise_floor([args.base, *args.others])
  np.savez_compressed(args.base / "floor.npz", **noise)
  noisy = sum(1 for values in noise.values() if np.any(values > 0))
  print(f"floor.npz: {noisy}/{len(noise)} rollout signals differ between same-code runs")
  return 0


def _record_static(out: Path, tiers: list[str], *, strict: bool) -> None:
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  from scr_phybot_mjab.tools.equivalence.canonical import canonical_lines
  from scr_phybot_mjab.tools.equivalence.fingerprint import model_fingerprint
  from scr_phybot_mjab.tools.equivalence.tasks import KEPT_TASKS, check_task_set

  check_task_set(strict=strict)
  for task in KEPT_TASKS:
    for mode in ("train", "play"):
      stem = f"{task}__{mode}"
      if "cfg" in tiers:
        _write_lines(out / "cfg" / f"{stem}.txt", canonical_lines(load_env_cfg(task, play=mode == "play"), "env"))
      if "model" in tiers:
        fingerprint, xml = model_fingerprint(load_env_cfg(task, play=mode == "play"))
        _write_text(out / "model" / f"{stem}.json", json.dumps(fingerprint, indent=1) + "\n")
        _write_text(out / "model" / f"{stem}.xml", xml or "")
    if "cfg" in tiers:
      runner = {"rl": load_rl_cfg(task), "runner_cls": load_runner_cls(task)}
      _write_lines(out / "cfg" / f"{task}__rl.txt", canonical_lines(runner, "task"))
    print(f"static  {task}", flush=True)


def _record_rollouts(out: Path, args: argparse.Namespace) -> list[str]:
  from scr_phybot_mjab.tools.equivalence.tasks import KEPT_TASKS, MOTION_TASKS

  jobs = [
    (task, mode, actions)
    for task in (args.tasks or KEPT_TASKS)
    for mode in args.modes
    for actions in args.actions
    if actions != "reference" or task in MOTION_TASKS
  ]
  env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "MJLAB_WARP_QUIET": "1"}

  def run(job: tuple[str, str, str]) -> str | None:
    task, mode, actions = job
    target = out / "rollout" / f"{task}__{mode}__{actions}.npz"
    num_envs = args.num_envs if mode == "train" else args.play_num_envs
    argv = [
      sys.executable, "-m", "scr_phybot_mjab.tools.equivalence.rollout", task,
      "--mode", mode, "--actions", actions, "--steps", str(args.steps),
      "--num-envs", str(num_envs), "--seed", str(args.seed),
      "--segment-steps", str(args.segment_steps), "--out", str(target),
    ]
    started = time.time()
    result = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
    if result.returncode != 0:
      _write_text(target.with_suffix(".log"), result.stdout + result.stderr)
      return f"rollout {target.name} (see {target.with_suffix('.log').name})"
    print(f"rollout {target.stem}  {time.time() - started:.0f}s", flush=True)
    return None

  with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    return [failure for failure in pool.map(run, jobs) if failure]


def _record_clips(out: Path) -> list[str]:
  from scr_phybot_mjab.tools.equivalence.clips import clip_jobs, run_clip_job

  jobs = clip_jobs()
  with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
    codes = list(pool.map(lambda job: run_clip_job(job, out / "clips", sys.executable), jobs))
  for job, code in zip(jobs, codes):
    print(f"clip    {job.name}  exit {code}", flush=True)
  return [f"clip {job.name} (see clips/{job.name}.log)" for job, code in zip(jobs, codes) if code != 0]


def _print_findings(findings: list) -> None:
  for tier in TIERS:
    tier_findings = [f for f in findings if f.tier == tier]
    if not tier_findings:
      continue
    failed = [f for f in tier_findings if not f.ok]
    print(f"{tier:8s} {len(tier_findings) - len(failed)}/{len(tier_findings)} identical")
    for finding in failed:
      print(f"  DIFF {finding.item}")
      for line in finding.detail.splitlines():
        print(f"       {line}")


def _manifest(args: argparse.Namespace) -> dict:
  import importlib.metadata as metadata

  import scr_phybot_mjab

  def git(*cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True).stdout.strip()

  versions = {name: metadata.version(name) for name in ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab")}
  gpu = subprocess.run(
    ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
  ).stdout.strip()
  return {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "package": scr_phybot_mjab.__file__,
    "argv": sys.argv,
    "versions": versions,
    "gpu": gpu,
    "env": {key: os.environ.get(key) for key in _ENV_KEYS},
  }


def _check_source_tree() -> None:
  import scr_phybot_mjab

  imported = Path(scr_phybot_mjab.__file__).resolve().parents[1]
  if imported != REPO / "src":
    raise SystemExit(f"Imported {imported}, not {REPO / 'src'}; run with PYTHONPATH={REPO / 'src'}")


def _pin_hash_seed() -> None:
  """Re-exec with a fixed hash seed so set/dict-of-set orders are reproducible."""
  if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _write_lines(path: Path, lines: list[str]) -> None:
  _write_text(path, "\n".join(lines) + "\n")


def _write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  rec = sub.add_parser("record", help="snapshot the current tree")
  rec.add_argument("snapshot", type=Path)
  rec.add_argument("--tiers", type=lambda s: s.split(","), default=list(TIERS))
  rec.add_argument("--tasks", nargs="*", help="limit rollouts to these task ids")
  rec.add_argument("--modes", type=lambda s: s.split(","), default=["train", "play"])
  rec.add_argument("--actions", type=lambda s: s.split(","), default=["zero", "random", "reference"])
  rec.add_argument("--steps", type=int, default=150)
  rec.add_argument("--num-envs", type=int, default=8)
  rec.add_argument("--play-num-envs", type=int, default=2)
  rec.add_argument("--seed", type=int, default=1234)
  rec.add_argument("--segment-steps", type=int, default=5, help="reset all envs every N steps; 0 = never")
  rec.add_argument("--gpu", default="1", help="single GPU for every rollout (results are GPU-model specific)")
  rec.add_argument("--jobs", type=int, default=3, help="concurrent rollout processes")
  rec.add_argument("--strict-tasks", action="store_true", help="also fail on unexpected project tasks")
  rec.add_argument("--force", action="store_true")
  cmp = sub.add_parser("compare", help="compare two snapshots")
  cmp.add_argument("base", type=Path)
  cmp.add_argument("new", type=Path)
  cmp.add_argument("--tiers", type=lambda s: s.split(","), default=list(TIERS))
  flo = sub.add_parser("floor", help="measure the same-code rollout noise floor into <base>/floor.npz")
  flo.add_argument("base", type=Path)
  flo.add_argument("others", type=Path, nargs="+")
  return parser.parse_args(argv)


if __name__ == "__main__":
  sys.exit(main())
