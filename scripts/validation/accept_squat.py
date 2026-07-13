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
  wp_data = base_env.sim._wp_data
  mj_model = base_env.sim.mj_model
  left_ankle_geom = _geom_id_with_suffix(mj_model, "left_ankle_roll_collision")
  right_ankle_geom = _geom_id_with_suffix(mj_model, "right_ankle_roll_collision")
  frames = command.motion.time_step_total

  max_error = torch.zeros(args.num_envs, 2, device=args.device)
  speed_sum = torch.zeros(args.num_envs, device=args.device)
  contact_sum = torch.zeros(args.num_envs, device=args.device)
  samples = 0
  failed = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
  min_separation = torch.full((args.num_envs,), torch.inf, device=args.device)
  collision_hits = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
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
        contact_sum += (feet_contact.data.found > 0).all(dim=1).float()
        samples += 1

      env.step(policy(env.get_observations()))
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
  contact_rate = torch.mean(contact_sum / max(samples, 1))
  separation_p01 = torch.quantile(min_separation, 0.01)
  separation_p05 = torch.quantile(min_separation, 0.05)
  separation_min = torch.min(min_separation)
  hits = int(collision_hits.sum().item())
  limit = 0.01 if args.friction == 0.7 else 0.02

  metrics = {
    "friction": args.friction,
    "completion_rate": float(completion_rate),
    "max_foot_error_p95_m": float(error_p95),
    "mean_foot_speed_mps": float(mean_speed),
    "both_feet_contact": float(contact_rate),
    "foot_separation_p01_m": float(separation_p01),
    "foot_separation_p05_m": float(separation_p05),
    "foot_separation_min_m": float(separation_min),
    "self_collision_hits": hits,
    "raw_ankle_pair_contacts": ankle_pair_contacts,
  }
  for key, value in metrics.items():
    print(f"{key}: {value}")

  passed = (
    metrics["completion_rate"] >= 0.98
    and metrics["max_foot_error_p95_m"] <= limit
    and metrics["mean_foot_speed_mps"] <= 0.02
    and metrics["both_feet_contact"] >= 0.99
    and metrics["foot_separation_p01_m"] >= 0.15
    and hits == 0
    and ankle_pair_contacts == 0
  )
  env.close()
  if not passed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
