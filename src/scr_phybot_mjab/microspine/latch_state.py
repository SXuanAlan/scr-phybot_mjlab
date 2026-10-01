"""Pure tensor bookkeeping of the per-pad latch state machine.

Every function maps (E, H) tensors to new tensors without side effects, so the
state machine can be tested on CPU without a simulation.
"""

from __future__ import annotations

import torch


def count_consecutive(condition: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
  """Steps the condition has held in a row; any failed step restarts at zero."""
  return torch.where(condition, count + 1, torch.zeros_like(count))


def in_hold_guard(
  steps_since_engage: torch.Tensor, seeded: torch.Tensor, guard: int, seed_guard: int | None
) -> torch.Tensor:
  """Latches too young to judge.

  The constraint only produces force one decimation after engaging, so hold
  checks wait out that readout settling. Seeded latches may also wait out
  the reset's whole-body settle (``seed_guard``); a latch stays seeded only
  while continuously engaged since birth, so this never loosens in-play
  engagements.
  """
  if seed_guard is None:
    return steps_since_engage < guard
  return steps_since_engage < torch.where(
    seeded,
    torch.full_like(steps_since_engage, seed_guard),
    torch.full_like(steps_since_engage, guard),
  )


def next_latch_state(engaged: torch.Tensor, engage: torch.Tensor, release: torch.Tensor) -> torch.Tensor:
  """Latch state after this step's engage and release transitions."""
  return torch.where(
    engage,
    torch.ones_like(engaged),
    torch.where(release, torch.zeros_like(engaged), engaged),
  )


def tick_cooldown(cooldown: torch.Tensor, released: torch.Tensor, cooldown_steps: int) -> torch.Tensor:
  """Count the re-engage cooldown down; a release restarts it."""
  cooldown = (cooldown - 1).clamp(min=0)
  if cooldown_steps > 0:
    cooldown = torch.where(released, torch.full_like(cooldown, cooldown_steps), cooldown)
  return cooldown


def advance_engage_age(age: torch.Tensor, engaged: torch.Tensor, engage_transition: torch.Tensor) -> torch.Tensor:
  """Policy steps since each latch engaged: 0 on the engage step and while released.

  Latches that exist at reset never see a transition; they count up from the
  reset value of 0.
  """
  return torch.where(
    engaged,
    torch.where(engage_transition, torch.zeros_like(age), age + 1),
    torch.zeros_like(age),
  )


def clear_where(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
  """Zero ``value`` wherever ``mask`` is set."""
  return torch.where(mask, torch.zeros_like(value), value)
