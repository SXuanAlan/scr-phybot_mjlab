"""Regenerate the committed motion clips from their recipes into a scratch dir.

Each job runs the real generator script single-threaded on CPU, so two runs
of the same code must produce identical arrays.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scr_phybot_mjab.tools.equivalence.canonical import REPO_ROOT

CLIMB_MOTIONS_DIR = REPO_ROOT / "src/scr_phybot_mjab/tasks/climb_wall/motions"
MOUNT_MOTIONS_DIR = REPO_ROOT / "src/scr_phybot_mjab/tasks/climb_transition80/motions"
_SINGLE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@dataclass(frozen=True)
class ClipJob:
  name: str
  script: str
  args: tuple[str, ...]  # "{out}" is replaced by the output directory


def clip_jobs() -> list[ClipJob]:
  cad_climb = str(CLIMB_MOTIONS_DIR / "climb_cad_empirical.npy")
  jobs = [
    ClipJob(
      "climb_cad_empirical",
      "scripts/reference/gen_cad_climb_reference.py",
      (
        "--step-height",
        "0.02",
        "--output",
        "{out}/climb_cad_empirical.npy",
        "--initial-output",
        "{out}/climb_cad_empirical_initial.json",
      ),
    ),
  ]
  for angle in (60, 80):
    name = f"climb_cad_empirical_corners_a{angle}"
    jobs.append(
      ClipJob(
        name,
        "scripts/reference/gen_cad_climb_reference.py",
        (
          "--source-motion",
          cad_climb,
          "--output",
          f"{{out}}/{name}.npy",
          "--initial-output",
          f"{{out}}/{name}_unused_initial.json",
          "--wall-angle",
          str(angle),
          "--unit-layout",
          "corners",
        ),
      )
    )
  jobs.append(
    ClipJob(
      "climb_mount_prologue_a80",
      "scripts/reference/gen_mount_clip.py",
      (
        *recipe_args(MOUNT_MOTIONS_DIR / "climb_mount_prologue_a80.recipe.txt", "scripts/reference/gen_mount_clip.py"),
        "--output",
        "{out}/climb_mount_prologue_a80.npy",
      ),
    )
  )
  jobs.append(
    ClipJob(
      "brick_pick_place",
      "scripts/reference/gen_brick_reference.py",
      ("--output", "{out}/brick_pick_place.npz"),
    )
  )
  return jobs


def recipe_args(recipe: Path, script: str) -> tuple[str, ...]:
  """Generator arguments recorded in a clip's recipe file, minus its ``--output``."""
  for line in recipe.read_text().splitlines():
    tokens = shlex.split(line)
    if script in tokens:
      args = list(tokens[tokens.index(script) + 1 :])
      if "--output" in args:
        del args[args.index("--output") : args.index("--output") + 2]
      return tuple(args)
  raise ValueError(f"No '{script}' command in {recipe}")


def run_clip_job(job: ClipJob, out_dir: Path, python: str) -> int:
  """Run one generator; its stdout/stderr go to ``<out_dir>/<name>.log``."""
  out_dir.mkdir(parents=True, exist_ok=True)
  argv = [python, job.script, *(arg.replace("{out}", str(out_dir)) for arg in job.args)]
  env = {**os.environ, **_SINGLE_THREAD, "CUDA_VISIBLE_DEVICES": ""}
  with (out_dir / f"{job.name}.log").open("w") as log:
    log.write(" ".join(shlex.quote(a) for a in argv) + "\n")
    log.flush()
    result = subprocess.run(argv, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
  return result.returncode
