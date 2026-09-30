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


def root_below_wall_altitude(
  env: ManagerBasedRlEnv,
  minimum_altitude: float = 0.3,
  motion_command_name: str = "climb_motion",
) -> torch.Tensor:
  """Along-wall analog of ``root_height_below_minimum`` (the fall guard).

  ``(root - wall_base_line) . wall_up < minimum_altitude`` — world z on a
  vertical wall (bit-identical to the mjlab builtin there). The builtin's
  absolute world-z bar is a trap under wall tilt: the RSI spawn's world z
  shrinks with the angle (the whole reference tips toward the base line), so
  a legitimately-placed robot on a 45° wall could terminate at t=0.
  """
  motion = env.command_manager.get_term(motion_command_name)
  robot_pos = env.scene["robot"].data.root_link_pos_w
  altitude = ((robot_pos - motion.wall_pivot_w) * motion.wall_up_w).sum(dim=-1)
  return altitude < minimum_altitude


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


def motion_body_tracking_divergence(
  env,
  command_name="climb_motion",
  position_threshold=0.3,
  orientation_threshold=1.0,
  height_threshold=0.25,
):
  """End large body-pose departures; retain a wall-frame progress check.

  Pose errors use the same XY/yaw alignment as body tracking. Along-wall
  height and global root orientation prevent that alignment hiding a fall.
  These are failure thresholds, independent of reward error scales.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = env.command_manager.get_term(command_name)
  reference_pos, _ = motion.aligned_body_reference()
  position_error = (reference_pos - motion.robot_body_pos_w).norm(dim=-1).amax(-1)
  orientation_error = quat_error_magnitude(motion.ref_root_quat_w, motion.robot_root_quat_w)
  height_error = ((motion.robot.data.root_link_pos_w - motion.ref_root_pos_w) * motion.wall_up_w).sum(-1).abs()
  return (
    (position_error > position_threshold)
    | (orientation_error > orientation_threshold)
    | (height_error > height_threshold)
  )


def prolonged_support_loss():
  """End a continuous no-anchor interval, allowing a short recovery attempt."""
  elapsed = None

  def call(env, command_name="microspine_affordance", timeout_s=0.4):
    nonlocal elapsed
    engaged = env.command_manager.get_term(command_name).engaged
    if elapsed is None:
      elapsed = torch.zeros(env.num_envs, device=env.device)
    elapsed = torch.where(engaged.any(-1), torch.zeros_like(elapsed), elapsed + env.step_dt)
    return elapsed >= timeout_s - 1.0e-7

  def reset(env_ids=None):
    if elapsed is not None:
      elapsed[slice(None) if env_ids is None else env_ids] = 0.0
    return {}

  call.reset = reset
  return call


def no_progress_stall():
  """Factory for terminating commanded climbs that make no new height.

  RETIRED from the shipped cfg (2026-07-30) and NOT wall-frame aware: if
  revived, its world-z progress mark must become an along-wall projection
  (see ``root_below_wall_altitude``).

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
