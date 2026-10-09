"""Which tasks the snapshots cover, and deterministic process setup."""

from __future__ import annotations

import os

KEPT_TASKS: tuple[str, ...] = (
  "Mjlab-Climb-Wall-Phybot-C2",
  "Mjlab-Climb-Wall-Phybot-C2-Corners-A60",
  "Mjlab-Climb-Wall-Phybot-C2-Corners-A80",
  "Mjlab-Climb-Transition80-Phybot-C2",
  "Mjlab-Walk-Phybot-C2",
  "Mjlab-Stand-Default-Phybot-C2",
  "Mjlab-Walk-Amp-Phybot-C2",
  "Mjlab-Squat-Phybot-C2",
)

# Tasks driven by a motion clip; they also get the reference-tracking rollout.
MOTION_TASKS: frozenset[str] = frozenset(
  {
    "Mjlab-Climb-Wall-Phybot-C2",
    "Mjlab-Climb-Wall-Phybot-C2-Corners-A60",
    "Mjlab-Climb-Wall-Phybot-C2-Corners-A80",
    "Mjlab-Climb-Transition80-Phybot-C2",
  }
)

# mjlab ships its own example tasks; project task ids carry one of these.
_PROJECT_MARKERS = ("Phybot",)


def project_tasks() -> list[str]:
  """Project task ids currently registered with mjlab."""
  import mjlab  # noqa: F401  (entry points import and register the project tasks)
  from mjlab.tasks.registry import list_tasks

  return [t for t in list_tasks() if any(m in t for m in _PROJECT_MARKERS)]


def check_task_set(*, strict: bool) -> None:
  """Fail loudly when a kept task did not register.

  mjlab only prints a warning when a task package fails to import, so a broken
  task would otherwise vanish from every snapshot without notice.
  """
  registered = set(project_tasks())
  missing = [t for t in KEPT_TASKS if t not in registered]
  if missing:
    raise RuntimeError(f"Kept tasks not registered: {missing}")
  extra = sorted(registered - set(KEPT_TASKS))
  if strict and extra:
    raise RuntimeError(f"Unexpected project tasks registered: {extra}")


def configure_determinism() -> None:
  """Pin the CUDA/torch settings that otherwise vary between identical runs."""
  os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
  import torch

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True, warn_only=True)
