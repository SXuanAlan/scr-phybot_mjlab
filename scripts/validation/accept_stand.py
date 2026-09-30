"""Headless acceptance test for the stand-default policy.

The first three seconds are the task's recovery window. The foot anchor is
captured at that same boundary, while metrics start at five seconds to give the
rest of the robot two additional seconds to settle. The process exits non-zero
when any acceptance threshold fails.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
)

TASK = "Mjlab-Stand-Default-Phybot-C2"


def _geom_id_with_suffix(model, suffix: str) -> int:
  matches = [geom_id for geom_id in range(model.ngeom) if model.geom(geom_id).name.rsplit("/", 1)[-1] == suffix]
  if len(matches) != 1:
    raise RuntimeError(f"Expected one geom ending in {suffix!r}; found {matches}.")
  return matches[0]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--steps", type=int, default=1000, help="20 s at 50 Hz")
  parser.add_argument("--strict-start-s", type=float, default=5.0)
  parser.add_argument(
    "--nominal",
    action="store_true",
    help="use deterministic play reset instead of the training recovery reset",
  )
  args = parser.parse_args()

  configure_torch_backends()
  cfg = load_env_cfg(TASK, play=args.nominal)
  cfg.scene.num_envs = args.num_envs
  # Keep the last measured state observable instead of letting the normal
  # 20-second timeout auto-reset it inside the final env.step().
  control_dt = cfg.sim.mujoco.timestep * cfg.decimation
  cfg.episode_length_s = max(cfg.episode_length_s, (args.steps + 1) * control_dt)
  recovery_time_s = float(cfg.rewards["stand_feet_lock"].params["recovery_time_s"])
  if args.strict_start_s < recovery_time_s:
    parser.error(f"--strict-start-s must be >= the task recovery time ({recovery_time_s:.1f} s)")
  recovery_step = int(round(recovery_time_s / control_dt))
  strict_start_step = int(round(args.strict_start_s / control_dt))
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
  feet_lock_term = base_env.reward_manager.get_term_cfg("stand_feet_lock").func
  foot_ids = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
  policy_joint_ids = robot.find_joints(
    tuple(PHYBOT_C2_POLICY_JOINT_NAMES),
    preserve_order=True,
  )[0]
  sensor = base_env.scene["self_collision"]
  wp_data = base_env.sim._wp_data
  mj_model = base_env.sim.mj_model
  left_ankle_geom = _geom_id_with_suffix(mj_model, "left_ankle_roll_collision")
  right_ankle_geom = _geom_id_with_suffix(mj_model, "right_ankle_roll_collision")
  anchor_xy: torch.Tensor | None = None
  valid = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
  speed_sum = torch.zeros(args.num_envs, device=args.device)
  contact_sum = torch.zeros(args.num_envs, device=args.device)
  pose_sum = torch.zeros(args.num_envs, device=args.device)
  samples = torch.zeros(args.num_envs, device=args.device)
  max_drift = torch.zeros(args.num_envs, device=args.device)
  max_tilt = torch.zeros(args.num_envs, device=args.device)
  max_height_error = torch.zeros(args.num_envs, device=args.device)
  min_separation = torch.full((args.num_envs,), torch.inf, device=args.device)
  self_hits = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
  ankle_pair_contacts = 0

  base_env.reset()
  with torch.no_grad():
    for step in range(args.steps):
      obs = env.get_observations()
      env.step(policy(obs))

      time_out = base_env.termination_manager.get_term("time_out")
      sample_mask = ~time_out
      for name in base_env.termination_manager.active_terms:
        if name == "time_out":
          continue
        valid &= ~base_env.termination_manager.get_term(name)

      force_history = sensor.data.force_history
      assert force_history is not None
      force = torch.norm(force_history, dim=-1)
      self_hits += ((force > 10.0).any(dim=1).any(dim=-1) & sample_mask).long()

      nacon = int(wp.to_torch(wp_data.nacon)[0].item())
      geom_pairs = wp.to_torch(wp_data.contact.geom)[:nacon].long()
      ankle_pair_contacts += int(
        (
          ((geom_pairs[:, 0] == left_ankle_geom) & (geom_pairs[:, 1] == right_ankle_geom))
          | ((geom_pairs[:, 0] == right_ankle_geom) & (geom_pairs[:, 1] == left_ankle_geom))
        )
        .sum()
        .item()
      )

      foot_pos = robot.data.site_pos_w[:, foot_ids]
      foot_xy = foot_pos[:, :, :2]
      relative = foot_pos - robot.data.root_link_pos_w.unsqueeze(1)
      root_quat = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
      foot_pos_b = quat_apply_inverse(root_quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(args.num_envs, 2, 3)
      separation = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
      min_separation = torch.where(sample_mask, torch.minimum(min_separation, separation), min_separation)

      completed_steps = step + 1
      if anchor_xy is None and completed_steps >= recovery_step:
        if feet_lock_term.anchor_xy is None:
          raise RuntimeError("stand_feet_lock did not freeze its anchor on schedule")
        anchor_xy = feet_lock_term.anchor_xy.clone()

      if completed_steps < strict_start_step:
        continue
      assert anchor_xy is not None
      foot_speed = torch.norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1)
      drift = torch.norm(foot_xy - anchor_xy, dim=-1).max(dim=1).values
      contact = (base_env.scene["feet_ground_contact"].data.found > 0).all(dim=1)
      default = robot.data.default_joint_pos[:, policy_joint_ids]
      pose_rmse = torch.sqrt(torch.mean(torch.square(robot.data.joint_pos[:, policy_joint_ids] - default), dim=1))
      tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))

      sample_f = sample_mask.float()
      speed_sum += foot_speed.mean(dim=1) * sample_f
      contact_sum += contact.float() * sample_f
      pose_sum += pose_rmse * sample_f
      samples += sample_f
      max_drift = torch.where(sample_mask, torch.maximum(max_drift, drift), max_drift)
      max_tilt = torch.where(sample_mask, torch.maximum(max_tilt, tilt), max_tilt)
      height_error = torch.abs(robot.data.root_link_pos_w[:, 2] - 0.713)
      max_height_error = torch.where(sample_mask, torch.maximum(max_height_error, height_error), max_height_error)

  success_rate = valid.float().mean()
  selected = valid & (samples > 0)
  if not selected.any():
    raise SystemExit("No environment completed the strict stand window.")
  mean_speed = torch.mean(speed_sum[selected] / samples[selected])
  contact_rate = torch.mean(contact_sum[selected] / samples[selected])
  mean_pose_rmse = torch.mean(pose_sum[selected] / samples[selected])
  drift_p95 = torch.quantile(max_drift[selected], 0.95)
  tilt_p99 = torch.quantile(max_tilt[selected], 0.99)
  height_p95 = torch.quantile(max_height_error[selected], 0.95)
  separation_p01 = torch.quantile(min_separation[selected], 0.01)
  separation_p05 = torch.quantile(min_separation[selected], 0.05)
  separation_min = torch.min(min_separation[selected])
  collision_hits = int(self_hits.sum().item())

  metrics = {
    "recovery_success": float(success_rate),
    "both_feet_contact": float(contact_rate),
    "mean_foot_speed_mps": float(mean_speed),
    "foot_drift_p95_m": float(drift_p95),
    "zero_pose_rmse_rad": float(mean_pose_rmse),
    "tilt_p99_deg": float(tilt_p99),
    "base_height_error_p95_m": float(height_p95),
    "foot_separation_p01_m": float(separation_p01),
    "foot_separation_p05_m": float(separation_p05),
    "foot_separation_min_m": float(separation_min),
    "self_collision_hits": collision_hits,
    "raw_ankle_pair_contacts": ankle_pair_contacts,
  }
  for key, value in metrics.items():
    print(f"{key}: {value}")

  passed = (
    metrics["recovery_success"] >= 0.95
    and metrics["both_feet_contact"] >= 0.99
    and metrics["mean_foot_speed_mps"] <= 0.02
    and metrics["foot_drift_p95_m"] <= 0.05
    and metrics["zero_pose_rmse_rad"] <= 0.08
    and metrics["tilt_p99_deg"] <= 8.0
    and metrics["base_height_error_p95_m"] <= 0.04
    and metrics["foot_separation_p01_m"] >= 0.15
    and collision_hits == 0
    and ankle_pair_contacts == 0
  )
  env.close()
  if not passed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
