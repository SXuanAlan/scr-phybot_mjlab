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


def motion_tracking_divergence(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  threshold: float = 0.7,
) -> torch.Tensor:
  """DeepMimic-style early termination: the pose left the reference entirely.

  Mean absolute clip-joint error above ``threshold`` (rad) means the robot is
  no longer anywhere on the cycle (a clean hang scores ~0.1–0.2 even at the
  wrong phase; a collapsed or inverted robot scores far above). Ending the
  rollout early stops PPO from spending samples on unrecoverable states.
  """
  motion = env.command_manager.get_term(command_name)
  error = (motion.ref_joint_pos - motion.robot_joint_pos).abs().mean(dim=-1)
  return error > threshold


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


def microspine_destructive_release(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  reasons: tuple[int, ...] = (4,),
  min_episode_steps: int = 25,
) -> torch.Tensor:
  """Terminate on destructive limit-surface releases (default: ultimate shear).

  Reason 4 means the pad was torn off its asperities at ultimate rake load —
  on hardware that fails spine tips under full load, damage on par with a
  fall, so it shares the fall's terminal cost instead of pricing in as a
  cheap regrab. Judged from the affordance's ``release_reason`` on the
  release step. The designed reverse-shear unhook (1) and DR-forced drills
  (6) never terminate; extend ``reasons`` only for faces that physically
  destroy hardware.

  ``min_episode_steps`` exempts the spawn window: RSI births carry an
  anchor-settling transient that can fire one ultimate on the first judged
  step (benign historically — release, re-latch, climb on). Terminating it
  turns every spawn into an execution and the episode stream into a 0.2 s
  metronome (measured: 166 spawn kills / 30 s). Inside the window the
  release still happens and still pays ``unsafe_microspine_release``; it
  just is not terminal — same newborn-clemency lesson as the motionref18
  tilt-capture.
  """
  aff = env.command_manager.get_term(command_name)
  released = aff.forced_release_this_step
  reason = aff.release_reason
  mask = torch.zeros_like(released)
  for r in reasons:
    mask = mask | (reason == int(r))
  past_spawn = env.episode_length_buf >= int(min_episode_steps)
  return past_spawn & (released & mask).any(dim=-1)
