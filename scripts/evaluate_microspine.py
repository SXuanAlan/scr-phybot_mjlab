"""Evaluate microspine adhesion: run simulation and output metrics.

Runs the Forearm-Hand microspine test for 10s:
  Phase 1 (0-3s): 20N +x push
  Phase 2 (3s+): 300N -z shear

Outputs JSON metrics for the evaluator.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

import spider_rl_mjlab.tasks  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.config.forearm_hand.env_cfgs import (
    forearm_hand_microspine_test_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv


def run_evaluation(num_envs: int = 4, device: str = "cuda:0") -> dict:
    cfg = forearm_hand_microspine_test_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = 10.0

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset()

    robot = env.scene["robot"]
    dt = env.step_dt
    total_steps = int(cfg.episode_length_s / dt)

    # Track metrics
    initial_pos = robot.data.root_link_pos_w[:, :3].clone()
    phase2_start_step = int(3.0 / dt)

    # Phase 2 tracking
    phase2_positions_z = []
    phase2_positions_x = []
    max_z_drift = torch.zeros(num_envs, device=device)
    max_x_drift = torch.zeros(num_envs, device=device)
    terminated_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    termination_step = torch.full((num_envs,), total_steps, dtype=torch.long, device=device)

    # Run simulation
    for step_i in range(total_steps):
        actions = torch.zeros(num_envs, 1, device=device)
        obs, rewards, terminated, truncated, info = env.step(actions)

        current_pos = robot.data.root_link_pos_w[:, :3]

        if step_i >= phase2_start_step:
            z_drift = (initial_pos[:, 2] - current_pos[:, 2]).abs()
            x_drift = (initial_pos[:, 0] - current_pos[:, 0]).abs()
            max_z_drift = torch.maximum(max_z_drift, z_drift)
            max_x_drift = torch.maximum(max_x_drift, x_drift)

            phase2_positions_z.append(current_pos[:, 2].clone())
            phase2_positions_x.append(current_pos[:, 0].clone())

        # Track terminations
        newly_terminated = (terminated | truncated) & ~terminated_envs
        if newly_terminated.any():
            termination_step[newly_terminated] = step_i
            terminated_envs |= newly_terminated

        if terminated_envs.all():
            break

    # Compute final metrics
    final_pos = robot.data.root_link_pos_w[:, :3]
    z_displacement = (final_pos[:, 2] - initial_pos[:, 2])
    x_displacement = (final_pos[:, 0] - initial_pos[:, 0])

    # Time survived in phase 2 (seconds after phase2 start)
    phase2_survival_steps = termination_step.float() - phase2_start_step
    phase2_survival_s = (phase2_survival_steps.clamp(min=0) * dt)

    # Stability: std of z position in last 2 seconds of phase 2
    if len(phase2_positions_z) > int(2.0 / dt):
        last_2s_z = torch.stack(phase2_positions_z[-int(2.0 / dt):], dim=0)
        z_stability = last_2s_z.std(dim=0)
    else:
        z_stability = torch.full((num_envs,), float('inf'), device=device)

    env.close()

    metrics = {
        "num_envs": num_envs,
        "per_env": {},
        "mean": {},
    }

    for i in range(num_envs):
        metrics["per_env"][f"env_{i}"] = {
            "final_z_displacement_m": z_displacement[i].item(),
            "final_x_displacement_m": x_displacement[i].item(),
            "max_z_drift_m": max_z_drift[i].item(),
            "max_x_drift_m": max_x_drift[i].item(),
            "phase2_survival_s": phase2_survival_s[i].item(),
            "z_stability_std": z_stability[i].item(),
            "terminated_early": terminated_envs[i].item(),
        }

    metrics["mean"] = {
        "final_z_displacement_m": z_displacement.mean().item(),
        "final_x_displacement_m": x_displacement.mean().item(),
        "max_z_drift_m": max_z_drift.mean().item(),
        "max_x_drift_m": max_x_drift.mean().item(),
        "phase2_survival_s": phase2_survival_s.mean().item(),
        "z_stability_std": z_stability.mean().item(),
        "pct_survived_full": (phase2_survival_s >= 6.5).float().mean().item(),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate microspine adhesion")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    metrics = run_evaluation(num_envs=args.num_envs, device=args.device)

    output_text = json.dumps(metrics, indent=2)
    print(output_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text)
        print(f"\nMetrics written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
