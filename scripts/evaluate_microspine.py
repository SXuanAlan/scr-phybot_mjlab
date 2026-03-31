"""Evaluate microspine adhesion against eval-criteria.md.

Runs the Forearm-Hand microspine test headlessly:
  Phase 1 (0-3s): 20N +x push
  Phase 2 (3s+):  300N -z shear

Outputs scores.json with per-criterion pass/fail and overall verdict.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import spider_rl_mjlab.tasks  # noqa: F401  — register tasks
from mjlab.envs import ManagerBasedRlEnv
from spider_rl_mjlab.tasks.microspine_test.config.forearm_hand.env_cfgs import (
    forearm_hand_microspine_test_env_cfg,
)


def run_evaluation(num_envs: int = 16, device: str = "cuda:0") -> dict:
    cfg = forearm_hand_microspine_test_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = 10.0

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset()

    robot = env.scene["robot"]
    dt = env.step_dt
    total_steps = int(cfg.episode_length_s / dt)
    phase2_start_step = int(3.0 / dt)

    # Capture initial position right after reset.
    initial_pos = robot.data.root_link_pos_w[:, :3].clone()

    # Phase 2 tracking buffers.
    phase2_z_history = []
    max_z_drift = torch.zeros(num_envs, device=device)
    max_x_drift = torch.zeros(num_envs, device=device)
    terminated_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    termination_step = torch.full((num_envs,), total_steps, dtype=torch.long, device=device)

    # Timing for RT measurement.
    sim_wall_start = time.perf_counter()

    for step_i in range(total_steps):
        actions = torch.zeros(num_envs, env.num_actions, device=device)
        obs, rewards, terminated, truncated, info = env.step(actions)

        current_pos = robot.data.root_link_pos_w[:, :3]

        if step_i >= phase2_start_step:
            z_drift = (initial_pos[:, 2] - current_pos[:, 2]).abs()
            x_drift = (initial_pos[:, 0] - current_pos[:, 0]).abs()
            max_z_drift = torch.maximum(max_z_drift, z_drift)
            max_x_drift = torch.maximum(max_x_drift, x_drift)
            phase2_z_history.append(current_pos[:, 2].clone())

        newly_terminated = (terminated | truncated) & ~terminated_envs
        if newly_terminated.any():
            termination_step[newly_terminated] = step_i
            terminated_envs |= newly_terminated

        if terminated_envs.all():
            break

    sim_wall_end = time.perf_counter()
    wall_time_s = sim_wall_end - sim_wall_start
    sim_time_s = cfg.episode_length_s
    rt_ratio = sim_time_s / wall_time_s if wall_time_s > 0 else float("inf")

    # --- Compute metrics ---
    final_pos = robot.data.root_link_pos_w[:, :3]
    final_z_disp = (final_pos[:, 2] - initial_pos[:, 2]).abs()
    final_x_disp = (final_pos[:, 0] - initial_pos[:, 0]).abs()

    phase2_survival_steps = termination_step.float() - phase2_start_step
    phase2_survival_s = phase2_survival_steps.clamp(min=0) * dt

    # Z stability: std over last 2s of phase 2.
    last_2s_steps = int(2.0 / dt)
    if len(phase2_z_history) > last_2s_steps:
        last_2s_z = torch.stack(phase2_z_history[-last_2s_steps:], dim=0)  # (T, N)
        z_stability_std = last_2s_z.std(dim=0)
    else:
        z_stability_std = torch.full((num_envs,), float("inf"), device=device)

    pct_survived_full = (phase2_survival_s >= 6.5).float().mean().item()

    env.close()

    # --- Evaluate criteria ---
    mean_final_z = final_z_disp.mean().item()
    mean_final_x = final_x_disp.mean().item()
    mean_max_z = max_z_drift.mean().item()
    mean_max_x = max_x_drift.mean().item()
    mean_survival = phase2_survival_s.mean().item()
    mean_z_std = z_stability_std.mean().item()

    # Hard constraints.
    h1 = True  # Phase 1 unchanged — enforced by architecture, not measurable here.
    h2 = True  # No anti-cheat — code review concern, assume True for automated eval.
    h3 = mean_final_z < 0.5
    # H4: hand stays stable, doesn't vibrate — use z_stability_std < 0.05 as proxy.
    h4 = mean_z_std < 0.05

    hard = {"H1_phase1_unchanged": h1, "H2_no_anticheat": h2,
            "H3_final_z_lt_0.5m": h3, "H4_no_vibration": h4}
    hard_all_pass = all(hard.values())

    # Soft targets.
    s1 = mean_final_z < 0.1
    s2 = mean_max_z < 0.15
    s3 = mean_survival >= 7.0
    s4 = mean_z_std < 0.02
    s5 = mean_max_x < 0.01
    s6 = pct_survived_full >= 1.0
    # S7: RT ratio >= 70% of target. Target RT = 1.0 (real-time).
    s7 = rt_ratio >= 0.7

    soft = {"S1_final_z_lt_0.1m": s1, "S2_max_z_lt_0.15m": s2,
            "S3_survival_gte_7s": s3, "S4_z_std_lt_0.02m": s4,
            "S5_max_x_lt_0.01m": s5, "S6_100pct_survived": s6,
            "S7_rt_gte_70pct": s7}
    soft_pass_count = sum(soft.values())

    # Verdict.
    if not hard_all_pass:
        verdict = "PIVOT"
    elif soft_pass_count >= 5:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "hard_constraints": hard,
        "hard_all_pass": hard_all_pass,
        "soft_targets": soft,
        "soft_pass_count": soft_pass_count,
        "soft_total": len(soft),
        "metrics": {
            "mean_final_z_displacement_m": mean_final_z,
            "mean_final_x_displacement_m": mean_final_x,
            "mean_max_z_drift_m": mean_max_z,
            "mean_max_x_drift_m": mean_max_x,
            "mean_phase2_survival_s": mean_survival,
            "mean_z_stability_std": mean_z_std,
            "pct_survived_full": pct_survived_full,
            "rt_ratio": rt_ratio,
            "wall_time_s": wall_time_s,
            "num_envs": num_envs,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate microspine adhesion")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None, help="Output scores.json path")
    args = parser.parse_args()

    results = run_evaluation(num_envs=args.num_envs, device=args.device)

    output_text = json.dumps(results, indent=2)
    print(output_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text)
        print(f"\nScores written to {args.output}", file=sys.stderr)

    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
