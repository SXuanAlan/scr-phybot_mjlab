"""Custom termination functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def both_hands_unlatched(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
) -> torch.Tensor:
  """Terminate when neither hand is engaged (airborne).

  Reads the microspine physics term's per-(env, hand) latch state. Triggers
  when both hands are simultaneously off — the robot has no anchor and is
  falling. Combined with the -200 ``termination_penalty`` reward, this
  directly punishes the "both hands off" state the strict single-stance
  reward does not cover.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  engaged = term_cfg.func.engaged  # (E, H) bool
  return engaged.sum(dim=-1) == 0


def gait_curriculum_success(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """Truncate an episode after meeting its current productive re-grip goal."""
  gait = env.command_manager.get_term(command_name)
  return gait.episode_success


def no_progress_stall():
  """Factory for terminating commanded climbs that make no new height.

  The initial settling window continuously follows the base height, so the
  unavoidable reset-pose sag does not consume the stall budget. Afterwards a
  new progress mark must be cleared by ``minimum_progress`` before the timer
  resets. Tiny solver jitter therefore cannot keep a hanging episode alive.
  """
  state: dict[str, torch.Tensor | None] = {
    "progress_mark_z": None,
    "stall_time": None,
    "fresh": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str = "climb_velocity",
    settle_time_s: float = 0.5,
    stall_time_s: float = 5.0,
    minimum_progress: float = 0.01,
    command_threshold: float = 0.01,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["progress_mark_z"] is None or state["progress_mark_z"].shape != z.shape:
      state["progress_mark_z"] = z.clone()
      state["stall_time"] = torch.zeros_like(z)
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      return torch.zeros_like(z, dtype=torch.bool)

    mark = state["progress_mark_z"]
    timer = state["stall_time"]
    fresh = state["fresh"]
    assert mark is not None and timer is not None and fresh is not None

    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    advanced = z >= mark + minimum_progress

    mark = torch.where(initialize | advanced, z, mark)
    timer = torch.where(
      initialize | advanced,
      torch.zeros_like(timer),
      timer + env.step_dt,
    )
    fresh.zero_()
    state["progress_mark_z"] = mark
    state["stall_time"] = timer

    command_z = env.command_manager.get_command(command_name)[:, 2].abs()
    active = command_z > command_threshold
    return active & ~initialize & (timer >= stall_time_s)

  def reset(env_ids=None) -> dict:
    fresh = state["fresh"]
    if fresh is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    fresh[ids] = True
    state["stall_time"][ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def bad_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  max_displacement: float = 10.0,
  max_velocity: float = 50.0,
  max_ang_velocity: float = 200.0,
  max_joint_velocity: float = 200.0,
) -> torch.Tensor:
  """Terminate when simulation diverges (NaN/Inf or magnitude explodes).

  Pure numerical safety — covers every state tensor that feeds actor
  observations so divergence is caught before NaN propagates to the
  policy. Uses env-local position to avoid false positives from env
  spacing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w - env.scene.env_origins
  lin_vel = asset.data.root_link_lin_vel_w
  ang_vel = asset.data.root_link_ang_vel_w
  quat = asset.data.root_link_quat_w
  jpos = asset.data.joint_pos
  jvel = asset.data.joint_vel

  def _bad(t: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    b = (~torch.isfinite(t)).any(dim=1)
    if limit is not None:
      b = b | (t.abs() > limit).any(dim=1)
    return b

  return (
    _bad(pos, max_displacement)
    | _bad(lin_vel, max_velocity)
    | _bad(ang_vel, max_ang_velocity)
    | _bad(quat)
    | _bad(jpos)
    | _bad(jvel, max_joint_velocity)
  )
