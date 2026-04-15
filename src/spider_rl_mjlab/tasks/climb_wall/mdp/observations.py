"""Custom observation functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def EMAObsFilter(base_func: Callable, alpha: float = 0.35):
  """Per-term first-order IIR low-pass filter, applied to any obs func.

  Matches the formula from the wall-climbing paper (arXiv:2510.20174):
    o_filtered = (1 - alpha) * o_raw + alpha * o_filtered_prev
  with alpha=0.35 by default — i.e. mostly raw signal, 35% history.

  Reduces obs noise from CCD jitter / contact-induced velocity spikes
  without distorting steady-state values.

  Implemented as a closure (not a class) so tyro's CLI introspection
  doesn't try to expose the captured ``base_func`` as a CLI argument.
  Returns a callable with a ``.reset`` attribute, which is the duck-typed
  interface mjlab's ObservationManager uses to detect class-style terms
  (see observation_manager.py:437).
  """
  alpha = float(alpha)
  state = {"buf": None, "fresh": None}

  def call(env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
    raw = base_func(env, **kwargs)
    if state["buf"] is None or state["buf"].shape != raw.shape:
      state["buf"] = raw.clone()
      state["fresh"] = torch.zeros(raw.shape[0], dtype=torch.bool, device=raw.device)
      return raw
    fresh = state["fresh"]
    if fresh.any():
      state["buf"][fresh] = raw[fresh]
      fresh.zero_()
    state["buf"] = (1.0 - alpha) * raw + alpha * state["buf"]
    return state["buf"]

  def reset(env_ids: torch.Tensor | slice | None = None) -> dict:
    if state["fresh"] is None:
      return {}
    if env_ids is None:
      state["fresh"].fill_(True)
    else:
      state["fresh"][env_ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from the previous step (one step before current `last_action`)."""
  return env.action_manager.prev_action


def prev_prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from two steps ago."""
  return env.action_manager.prev_prev_action


# Gait clock period (seconds). A full cycle = one swing + one stance for each
# hand. Left hand: swing in φ ∈ [0, 0.5), stance in [0.5, 1). Right hand is
# offset by 0.5 so exactly one hand is always swinging.
GAIT_PERIOD_S = 2.0


def gait_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env gait phase ∈ [0, 1). Advances with episode_length_buf · step_dt."""
  t = env.episode_length_buf.float() * env.step_dt
  return (t / GAIT_PERIOD_S) % 1.0


def gait_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Gait clock observation (sin/cos × 2 hands) — shape (E, 4).

  The left hand uses the raw phase; the right hand phase is offset by 0.5 so
  the two hands swing in alternation (trot-style).
  """
  p = gait_phase(env)
  p_left = p
  p_right = (p + 0.5) % 1.0
  two_pi = 2.0 * math.pi
  return torch.stack(
    [
      torch.sin(two_pi * p_left),
      torch.cos(two_pi * p_left),
      torch.sin(two_pi * p_right),
      torch.cos(two_pi * p_right),
    ],
    dim=-1,
  )
