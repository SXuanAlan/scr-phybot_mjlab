"""Headless full-clip acceptance test for a squat checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import warp as wp

import scr_phybot_mjab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


TASK = "Mjlab-Squat-Phybot-C2"


def _geom_id_with_suffix(model, suffix: str) -> int:
  matches = [
    geom_id
    for geom_id in range(model.ngeom)
    if model.geom(geom_id).name.rsplit("/", 1)[-1] == suffix
  ]
  if len(matches) != 1:
    raise RuntimeError(f"Expected one geom ending in {suffix!r}; found {matches}.")
  return matches[0]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--friction", type=float, choices=(0.4, 0.7), default=0.7)
  args = parser.parse_args()

  configure_torch_backends()
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.events["foot_friction"].params["ranges"] = (args.friction, args.friction)
  # Add one validation-only step to the episode horizon. The loop still sends
  # exactly one action for each of the 350 reference frames, but the final
  # transition remains observable instead of being hidden by timeout auto-reset.
  cfg.episode_length_s += cfg.sim.mujoco.timestep * cfg.decimation
  agent_cfg = load_rl_cfg(TASK)
  base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint,
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  robot = base_env.scene["robot"]
  foot_ids = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
  command = base_env.command_manager.get_term("motion")
  feet_contact = base_env.scene["feet_ground_contact"]
  self_sensor = base_env.scene["self_collision"]
  arm_names = tuple(cfg.rewards["track_arms"].params["joint_names"])
  arm_ref_ids = torch.tensor(
    [command.cfg.joint_names.index(name) for name in arm_names],
    dtype=torch.long,
    device=args.device,
  )
  arm_joint_ids = command.joint_ids[arm_ref_ids]
  action_limit_cfg = cfg.rewards["arm_action_limits"].params
  action_term = base_env.action_manager.get_term(action_limit_cfg["action_name"])
  arm_action_ids = torch.tensor(
    [action_term.target_names.index(name) for name in arm_names],
    dtype=torch.long,
    device=args.device,
  )
  arm_action_limits = torch.tensor(
    action_limit_cfg["max_abs_actions"], device=args.device
  )
  action_reference_cfg = cfg.rewards["arm_reference_action"].params
  arm_action_tolerance = torch.tensor(
    action_reference_cfg["max_abs_deviation"], device=args.device
  )
  arm_action_scale = action_term.scale
  if isinstance(arm_action_scale, torch.Tensor):
    arm_action_scale = arm_action_scale[:, arm_action_ids]
  arm_action_offset = action_term.offset
  if isinstance(arm_action_offset, torch.Tensor):
    arm_action_offset = arm_action_offset[:, arm_action_ids]
  arm_pairs = tuple(
    (arm_names.index(name), arm_names.index(name.replace("left_", "right_")))
    for name in arm_names
    if name.startswith("left_")
  )
  wp_data = base_env.sim._wp_data
  mj_model = base_env.sim.mj_model
  left_ankle_geom = _geom_id_with_suffix(mj_model, "left_ankle_roll_collision")
  right_ankle_geom = _geom_id_with_suffix(mj_model, "right_ankle_roll_collision")
  frames = command.motion.time_step_total

  max_error = torch.zeros(args.num_envs, 2, device=args.device)
  speed_sum = torch.zeros(args.num_envs, device=args.device)
  foot_speed_sum = torch.zeros(args.num_envs, 2, device=args.device)
  contact_sum = torch.zeros(args.num_envs, device=args.device)
  samples = 0
  failed = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
  min_separation = torch.full((args.num_envs,), torch.inf, device=args.device)
  collision_hits = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
  arm_squared_error_sum = torch.zeros(args.num_envs, device=args.device)
  arm_max_error = torch.zeros(args.num_envs, device=args.device)
  arm_symmetry_error_sum = torch.zeros(args.num_envs, device=args.device)
  arm_action_limit_hits = torch.zeros((), device=args.device)
  arm_action_tube_hits = torch.zeros((), device=args.device)
  arm_global_clip_hits = torch.zeros((), device=args.device)
  arm_action_reference_squared_error_sum = torch.zeros(
    args.num_envs, device=args.device
  )
  arm_action_symmetry_error_sum = torch.zeros(args.num_envs, device=args.device)
  arm_action_rate_squared_sum = torch.zeros((), device=args.device)
  previous_arm_action: torch.Tensor | None = None
  ankle_pair_contacts = 0

  base_env.reset()
  with torch.no_grad():
    for frame in range(frames):
      # Before stepping, state and reference are synchronized to this frame.
      foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
      error = torch.norm(foot_xy - command.foot_position_ref_w[:, :, :2], dim=-1)
      max_error = torch.maximum(max_error, error)
      relative = robot.data.site_pos_w[:, foot_ids] - robot.data.root_link_pos_w.unsqueeze(1)
      root_quat = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
      foot_pos_b = quat_apply_inverse(
        root_quat.reshape(-1, 4), relative.reshape(-1, 3)
      ).reshape(args.num_envs, 2, 3)
      separation = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
      min_separation = torch.minimum(min_separation, separation)
      if frame > 0:  # Contact sensors need one physics step after reset.
        speed = torch.norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1)
        speed_sum += speed.mean(dim=1)
        foot_speed_sum += speed
        contact_sum += (feet_contact.data.found > 0).all(dim=1).float()
        samples += 1

      actions = policy(env.get_observations()).clamp(
        -agent_cfg.clip_actions, agent_cfg.clip_actions
      )
      arm_action = actions[:, arm_action_ids]
      arm_reference_action = (
        command.joint_pos_ref[:, arm_ref_ids]
        - arm_action_offset
        + robot.data.encoder_bias[:, arm_joint_ids]
      ) / arm_action_scale
      arm_action_error = arm_action - arm_reference_action
      arm_action_reference_squared_error_sum += torch.sum(
        torch.square(arm_action_error), dim=1
      )
      arm_action_tube_hits += (
        torch.abs(arm_action_error) > arm_action_tolerance
      ).sum()
      action_symmetry_error = torch.stack(
        [
          arm_action_error[:, left] - arm_action_error[:, right]
          for left, right in arm_pairs
        ],
        dim=1,
      )
      arm_action_symmetry_error_sum += torch.sum(
        torch.square(action_symmetry_error), dim=1
      )
      arm_action_limit_hits += (torch.abs(arm_action) > arm_action_limits).sum()
      arm_global_clip_hits += (
        torch.abs(arm_action) >= agent_cfg.clip_actions - 1.0e-3
      ).sum()
      if previous_arm_action is not None:
        arm_action_rate_squared_sum += torch.sum(
          torch.square(arm_action - previous_arm_action)
        )
      previous_arm_action = arm_action.clone()

      env.step(actions)
      arm_position = robot.data.joint_pos[:, arm_joint_ids]
      arm_reference = command.joint_pos_ref[:, arm_ref_ids]
      arm_error = arm_position - arm_reference
      arm_squared_error_sum += torch.sum(torch.square(arm_error), dim=1)
      arm_max_error = torch.maximum(
        arm_max_error, torch.abs(arm_error).max(dim=1).values
      )
      symmetry_error = torch.stack(
        [
          (arm_position[:, left] - arm_position[:, right])
          - (arm_reference[:, left] - arm_reference[:, right])
          for left, right in arm_pairs
        ],
        dim=1,
      )
      arm_symmetry_error_sum += torch.sum(torch.square(symmetry_error), dim=1)

      force_history = self_sensor.data.force_history
      assert force_history is not None
      force = torch.norm(force_history, dim=-1)
      collision_hits += (force > 10.0).any(dim=1).any(dim=-1).long()
      nacon = int(wp.to_torch(wp_data.nacon)[0].item())
      geom_pairs = wp.to_torch(wp_data.contact.geom)[:nacon].long()
      ankle_pair_contacts += int(
        (
          ((geom_pairs[:, 0] == left_ankle_geom) & (geom_pairs[:, 1] == right_ankle_geom))
          | ((geom_pairs[:, 0] == right_ankle_geom) & (geom_pairs[:, 1] == left_ankle_geom))
        ).sum().item()
      )
      for name in base_env.termination_manager.active_terms:
        if name == "time_out":
          continue
        failed |= base_env.termination_manager.get_term(name)

    # Include the physical state produced by the final reference-frame action.
    # The validation-only horizon extension above keeps this from auto-resetting.
    foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
    error = torch.norm(foot_xy - command.foot_position_ref_w[:, :, :2], dim=-1)
    max_error = torch.maximum(max_error, error)
    speed = torch.norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1)
    speed_sum += speed.mean(dim=1)
    foot_speed_sum += speed
    contact_sum += (feet_contact.data.found > 0).all(dim=1).float()
    relative = robot.data.site_pos_w[:, foot_ids] - robot.data.root_link_pos_w.unsqueeze(1)
    root_quat = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    foot_pos_b = quat_apply_inverse(
      root_quat.reshape(-1, 4), relative.reshape(-1, 3)
    ).reshape(args.num_envs, 2, 3)
    separation = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    min_separation = torch.minimum(min_separation, separation)
    samples += 1

  completion_rate = (~failed).float().mean()
  max_per_env = max_error.max(dim=1).values
  error_p95 = torch.quantile(max_per_env, 0.95)
  mean_speed = torch.mean(speed_sum / max(samples, 1))
  worst_foot_mean_speed = torch.max(
    foot_speed_sum / max(samples, 1), dim=1
  ).values
  worst_foot_mean_speed_p95 = torch.quantile(worst_foot_mean_speed, 0.95)
  contact_rate = torch.mean(contact_sum / max(samples, 1))
  separation_p01 = torch.quantile(min_separation, 0.01)
  separation_p05 = torch.quantile(min_separation, 0.05)
  separation_min = torch.min(min_separation)
  arm_rmse = torch.sqrt(
    arm_squared_error_sum / (frames * len(arm_names))
  )
  arm_rmse_mean = torch.mean(arm_rmse)
  arm_rmse_p95 = torch.quantile(arm_rmse, 0.95)
  arm_max_error_p95 = torch.quantile(arm_max_error, 0.95)
  arm_symmetry_rmse = torch.sqrt(
    arm_symmetry_error_sum / (frames * len(arm_pairs))
  )
  arm_symmetry_rmse_p95 = torch.quantile(arm_symmetry_rmse, 0.95)
  arm_action_count = args.num_envs * frames * len(arm_names)
  arm_action_limit_fraction = arm_action_limit_hits / arm_action_count
  arm_action_tube_fraction = arm_action_tube_hits / arm_action_count
  arm_global_clip_fraction = arm_global_clip_hits / arm_action_count
  arm_action_reference_rmse = torch.sqrt(
    arm_action_reference_squared_error_sum / (frames * len(arm_names))
  )
  arm_action_reference_rmse_p95 = torch.quantile(
    arm_action_reference_rmse, 0.95
  )
  arm_action_symmetry_rmse = torch.sqrt(
    arm_action_symmetry_error_sum / (frames * len(arm_pairs))
  )
  arm_action_symmetry_rmse_p95 = torch.quantile(
    arm_action_symmetry_rmse, 0.95
  )
  arm_action_rate_count = args.num_envs * (frames - 1) * len(arm_names)
  arm_action_rate_rms = torch.sqrt(
    arm_action_rate_squared_sum / arm_action_rate_count
  )
  hits = int(collision_hits.sum().item())
  limit = 0.01 if args.friction == 0.7 else 0.02

  metrics = {
    "friction": args.friction,
    "completion_rate": float(completion_rate),
    "max_foot_error_p95_m": float(error_p95),
    "mean_foot_speed_mps": float(mean_speed),
    "worst_foot_mean_speed_p95_mps": float(worst_foot_mean_speed_p95),
    "both_feet_contact": float(contact_rate),
    "foot_separation_p01_m": float(separation_p01),
    "foot_separation_p05_m": float(separation_p05),
    "foot_separation_min_m": float(separation_min),
    "arm_position_rmse_mean_rad": float(arm_rmse_mean),
    "arm_position_rmse_p95_rad": float(arm_rmse_p95),
    "arm_position_max_error_p95_rad": float(arm_max_error_p95),
    "arm_symmetry_rmse_p95_rad": float(arm_symmetry_rmse_p95),
    "arm_action_reference_rmse_p95": float(arm_action_reference_rmse_p95),
    "arm_action_symmetry_rmse_p95": float(arm_action_symmetry_rmse_p95),
    "arm_action_tube_fraction": float(arm_action_tube_fraction),
    "arm_action_limit_fraction": float(arm_action_limit_fraction),
    "arm_global_clip_fraction": float(arm_global_clip_fraction),
    "arm_action_rate_rms": float(arm_action_rate_rms),
    "self_collision_hits": hits,
    "raw_ankle_pair_contacts": ankle_pair_contacts,
  }
  for key, value in metrics.items():
    print(f"{key}: {value}")

  passed = (
    metrics["completion_rate"] >= 0.98
    and metrics["max_foot_error_p95_m"] <= limit
    and metrics["mean_foot_speed_mps"] <= 0.02
    and metrics["worst_foot_mean_speed_p95_mps"] <= 0.025
    and metrics["both_feet_contact"] >= 0.99
    and metrics["foot_separation_p01_m"] >= 0.15
    and metrics["arm_position_rmse_p95_rad"] <= 0.15
    and metrics["arm_position_max_error_p95_rad"] <= 0.35
    and metrics["arm_symmetry_rmse_p95_rad"] <= 0.10
    and metrics["arm_action_reference_rmse_p95"] <= 0.65
    and metrics["arm_action_symmetry_rmse_p95"] <= 0.25
    and metrics["arm_action_tube_fraction"] <= 0.01
    and metrics["arm_action_limit_fraction"] <= 0.01
    and metrics["arm_global_clip_fraction"] == 0.0
    and hits == 0
    and ankle_pair_contacts == 0
  )
  env.close()
  if not passed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
