"""Transition-side curricula.

``spawn_bias_window`` is the REVERSE-CURRICULUM RSI schedule: the
biased-spawn window starts at the chain's END (the latch stretch) and
widens BACKWARD one segment at a time as training progresses. Rationale
(v1-v6 ledger): each window teaches exactly the stretch it covers — v2's
early window taught standing, v4's latch window ignited the press — and a
segment learns fastest when its payoff is bridging into a stretch the
policy already values. So: master the ending first, then keep prepending.

The iters are CEILINGS in the curriculum-doctrine sense — drive by lane
mastery via kill/edit/resume. Clock (corrected 08-18): the counter is
``common_step_counter // 96`` and the resume path RESTORES it, so stage
iters are CUMULATIVE learning iterations across the whole lineage (v59b
launched at 12450 and its spawn lane read 10500 -> 12600 — the early
ladder never replays). Only ``train_warmstart`` zeroes the counter.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch

from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
  resolve_segment_window,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "SpawnWindowStage",
  "engage_gate_schedule",
  "gravity_magnitude_schedule",
  "gravity_scale_at",
  "microspine_param_schedule",
  "reward_param_schedule",
  "spawn_bias_window",
  "termination_param_schedule",
)


def gravity_scale_at(stages: list[dict], iteration: int) -> float:
  """Ladder value at a cumulative iteration (pure; the exam-side helper).

  v62e4 exam doctrine (user call, 08-19 evening): probes grade at the
  CURRICULUM's current gravity — a 1 g exam of a 0.6 g-era checkpoint
  manufactures artifacts (the 4425 "LH seed collapse") that say nothing
  about classroom progress. Exam scripts pop the ``gravity_ladder`` term,
  resolve the scale from its stages at the checkpoint's iteration, and
  scale ``cfg.sim.mujoco.gravity`` before the env is built; a
  ``gravity=<scale>`` token forces a specific scale (the graduation exam
  pins 1.0).
  """
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if iteration >= stage["iter"]:
      current = stage["value"]
    else:
      break
  return float(current)


def gravity_magnitude_schedule(
  env: "ManagerBasedRlEnv",
  env_ids: "torch.Tensor",
  stages: list[dict],
  steps_per_iter: int = 96,
) -> "torch.Tensor":
  """Stage the world gravity MAGNITUDE by training iteration.

  v62e (user call, 08-19 afternoon): the seeded-birth settle transient
  scales with gravity — the clip frame carries kinematics but no forces,
  so at 1 g the body's first sag routes ~200-340 N through high-anchored
  hand welds before the policy can act (set_hold RH seeds: med = p90 = 6
  steps = death exactly at guard expiry, 0% eased over 5 h while LH
  graduated 3% -> 82%). Scaling gravity shrinks that transient at the
  source, so the classroom exists and the seed guard can return to the
  climbing-parity 2 steps. Easy -> hard; the final stage returns to the
  honest -9.81 (v11 doctrine, same as the shear ladder).

  Writes the live warp model option in place (zero-copy torch view; the
  next physics step reads it). The host template ``sim._mj_model`` keeps
  the honest cfg value and provides the baseline. Exam scripts pop this
  term and re-apply the scale via :func:`gravity_scale_at` at the
  checkpoint's iteration (curriculum-gravity exams, user call 08-19;
  ``gravity=`` token pins a specific scale, e.g. the 1.0 graduation exam).
  """
  del env_ids
  import torch as _torch
  import warp as _wp

  current_iter = env.common_step_counter // steps_per_iter
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage["value"]
    else:
      break
  base = env.sim._mj_model.opt.gravity  # host template: never scaled
  g = _wp.to_torch(env.sim.wp_model.opt.gravity)
  g[:, 0] = float(base[0]) * current
  g[:, 1] = float(base[1]) * current
  g[:, 2] = float(base[2]) * current
  return _torch.tensor([float(current)], device=env.device)


def microspine_param_schedule(
  env: "ManagerBasedRlEnv",
  env_ids: "torch.Tensor",
  param_name: str,
  stages: list[dict],
  command_name: str = "microspine_affordance",
  steps_per_iter: int = 96,
) -> "torch.Tensor":
  """Stage a live MicrospineCfg scalar by training iteration.

  The engage-gate schedule's generic sibling: writes ``param_name`` on the
  affordance's live ``_mcfg`` (the same write path the gate anneal uses).
  v3 first user: the ``engage_shear_threshold_n`` ignition ladder — the
  10 N touch+drag weld bar demands genuine weight transfer through the
  pad (the PD-armed drag probe measured 2-8 N residual contact shear
  under a 30 N pull), which is a v49-style cliff for a newborn policy.
  Easy -> hard, final stage returns to the honest spec (v11 doctrine).
  """
  del env_ids
  import torch as _torch

  current_iter = env.common_step_counter // steps_per_iter
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage["value"]
    else:
      break
  affordance = env.command_manager.get_term(command_name)
  setattr(affordance._mcfg, param_name, current)
  return _torch.tensor([float(current)], device=env.device)


def termination_param_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  termination_name: str,
  param_name: str,
  stages: list[dict],
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Stage a termination term's PARAM by training iteration.

  The reward ratchet's sibling for the contact law: v54 anneals the
  position-aware radius (loose birth acceptance -> precision by
  graduation) exactly like every other training aid. Same stepwise
  lookup; mutates the live termination cfg's params dict.
  """
  del env_ids
  current_iter = env.common_step_counter // steps_per_iter
  cfg = env.termination_manager.get_term_cfg(termination_name)
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage["value"]
    else:
      break
  cfg.params[param_name] = current
  return torch.tensor([float(current)], device=env.device)


def reward_param_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_name: str,
  stages: list[dict],
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Stage a reward term's PARAM by training iteration (kernel ratchet).

  The v34 use: tighten the wide tracking kernels' std as tracking error
  falls — the wide kernel's job is capture-from-anywhere, but its gentle
  slope also pays half-compliance generously (v33 parked at the half
  raise, the shape's comfort point). Ratcheting the std once the error is
  down shrinks the half-way wage and grows the finish margin without ever
  reopening the dead zone. Same stepwise-lookup semantics as
  ``reward_weight``; mutates the live term cfg's params dict.
  """
  del env_ids
  current_iter = env.common_step_counter // steps_per_iter
  cfg = env.reward_manager.get_term_cfg(reward_name)
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage["value"]
    else:
      break
  cfg.params[param_name] = current
  return torch.tensor([float(current)], device=env.device)


class SpawnWindowStage(TypedDict):
  iter: int
  segments: tuple[str, ...]


def _apply_bias_segments(motion, segments: tuple[str, ...]) -> None:
  """Rewrite the command's biased-spawn window(s) for the given segments.

  Single-clip: ``_bias_window`` (local frames). Band family: per-slot
  absolute windows over each native clip. Unresolvable segments degrade to
  the whole clip (uniform), never to a stale window.
  """
  if getattr(motion, "_family", False):
    lo_list: list[int] = []
    hi_list: list[int] = []
    bases = motion._slot_base.tolist()
    lens = motion._slot_len.tolist()
    for base, length, (_, path) in zip(bases, lens, motion.cfg.angle_clip_files):
      window = resolve_segment_window(path, segments)
      if window is None:
        lo, hi = 0, length
      else:
        lo = int(window[0] * length)
        hi = min(max(int(window[1] * length), lo + 1), length)
      lo_list.append(base + lo)
      hi_list.append(base + hi)
    device = motion._slot_base.device
    motion._slot_bias_lo = torch.tensor(
      lo_list, dtype=torch.long, device=device
    )
    motion._slot_bias_hi = torch.tensor(
      hi_list, dtype=torch.long, device=device
    )
    return
  cycle = motion.clip.cycle_frames
  window = resolve_segment_window(motion.cfg.motion_file, segments)
  if window is None:
    motion._bias_window = (0, cycle)
  else:
    lo = int(window[0] * cycle)
    hi = min(max(int(window[1] * cycle), lo + 1), cycle)
    motion._bias_window = (lo, hi)


def engage_gate_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[dict],
  command_name: str = "microspine_affordance",
  off_threshold: float = 0.25,
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Anneal the engage-quality gate: strict at birth, stock by graduation.

  Writes ``engage_parallel_cos_threshold`` on the affordance's LIVE cfg —
  the same write path the tilt-peel schedule uses for its band. A stage's
  ``max_tilt_deg`` of None restores the stock threshold (the gate is a
  training aid, not a physics claim; the final phase trains under the
  honest weld-anything acceptance). Returns the active threshold (cos).
  """
  if not stages:
    raise ValueError("engage_gate_schedule requires at least one stage.")
  current_iter = env.common_step_counter // steps_per_iter
  current = stages[0]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage
    else:
      break
  max_tilt = current["max_tilt_deg"]
  threshold = (
    off_threshold if max_tilt is None else math.cos(math.radians(max_tilt))
  )
  affordance = env.command_manager.get_term(command_name)
  affordance._mcfg.engage_parallel_cos_threshold = threshold
  return torch.tensor([threshold], device=env.device)


def spawn_bias_window(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[SpawnWindowStage],
  command_name: str = "climb_motion",
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Stage the biased-spawn window by training iteration (see module doc).

  Stepwise lookup like the wall-angle ladder; the window is recomputed
  only on a stage CHANGE (one npy read per transition). Returns the active
  stage's iter threshold for the curriculum log.
  """
  if not stages:
    raise ValueError("spawn_bias_window requires at least one stage.")
  current_iter = env.common_step_counter // steps_per_iter
  current = stages[0]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage
    else:
      break
  segments = tuple(current["segments"])
  motion = env.command_manager.get_term(command_name)
  if getattr(motion, "_bias_segments_active", None) != segments:
    _apply_bias_segments(motion, segments)
    motion._bias_segments_active = segments
  return torch.tensor([float(current["iter"])], device=env.device)
