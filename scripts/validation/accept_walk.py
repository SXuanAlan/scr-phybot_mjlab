"""Acceptance test for the walk policy: STAND / WALK / SIDESTEP / STOP-GO phases.

Per phase & segment it reports termination counts, self-collision hit rate,
feet lateral separation (p1/p5/p50), base tilt (p50/p99/max), and base height.
Lessons baked in from runs 1-5:
- standing quality needs a dedicated zero-command phase (training-side
  Episode_Termination metrics cannot distinguish standing pathology), and
- always record tilt (a robot lying flat still shows "normal" feet separation).

Usage:
  python scripts/validation/accept_walk.py \
      --ckpt logs/rsl_rl/phybot_c2_walk/<run>/model_12000.pt \
      [--num-envs 256] [--device cuda:0] [--steps 750] [--phases stand,walk,stopgo]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

_REPO = Path("/home/spider/projects/scr-phybot_mjlab")
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

import scr_phybot_mjab  # noqa: F401  (registers tasks)

DEFAULT_TASK = "Mjlab-Walk-Phybot-C2"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--task", default=DEFAULT_TASK)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--steps", type=int, default=750, help="steps per phase (50 Hz)")
  ap.add_argument("--phases", default="stand,walk,sidestep,stopgo")
  args = ap.parse_args()

  configure_torch_backends()
  device = args.device

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  uenv = env.unwrapped
  twist = uenv.command_manager.get_term("twist")
  robot = uenv.scene["robot"]
  tm = uenv.termination_manager
  sc_params = env_cfg.rewards["self_collisions"].params
  sc_sensor = uenv.scene[sc_params["sensor_name"]]
  sc_threshold = float(sc_params.get("force_threshold", 10.0))
  lin_track_std = float(env_cfg.rewards["track_linear_velocity"].params["std"])
  ang_track_std = float(env_cfg.rewards["track_angular_velocity"].params["std"])

  if hasattr(robot.data, "site_names"):
    foot_ids = [
      robot.data.site_names.index("left_foot"),
      robot.data.site_names.index("right_foot"),
    ]
  else:
    foot_ids = env_cfg.rewards["feet_distance"].params["asset_cfg"].site_ids

  down = torch.tensor([[0.0, 0.0, -1.0]], device=device)

  def feet_sep() -> torch.Tensor:
    pos_w = robot.data.site_pos_w[:, foot_ids, :]
    rel = pos_w - robot.data.root_link_pos_w.unsqueeze(1)
    b, n = rel.shape[:2]
    quat = robot.data.root_link_quat_w.unsqueeze(1).expand(b, n, 4)
    pos_b = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(b, n, 3)
    return (pos_b[:, 0, 1] - pos_b[:, 1, 1]).abs()

  def tilt_deg() -> torch.Tensor:
    quat = robot.data.root_link_quat_w
    g_b = quat_apply_inverse(quat, down.expand(quat.shape[0], 3))
    return torch.rad2deg(torch.acos((-g_b[:, 2]).clamp(-1.0, 1.0)))

  def self_collision_envs() -> torch.Tensor:
    data = sc_sensor.data
    if data.force_history is not None:
      force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
      return (force_mag > sc_threshold).any(dim=1).any(dim=-1)  # [B]
    assert data.found is not None
    return data.found.reshape(args.num_envs, -1).any(dim=-1)

  def run_phase(name: str, schedule: list[tuple[tuple[float, float, float], int, str]]) -> None:
    uenv.reset()
    term_counts: dict[tuple[str, str], int] = defaultdict(int)
    seps: dict[str, list[torch.Tensor]] = defaultdict(list)
    tilts: dict[str, list[torch.Tensor]] = defaultdict(list)
    heights: dict[str, list[torch.Tensor]] = defaultdict(list)
    vels: dict[str, list[torch.Tensor]] = defaultdict(list)
    vel_errs: dict[str, list[torch.Tensor]] = defaultdict(list)
    foot_disps: dict[str, list[torch.Tensor]] = defaultdict(list)
    lin_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    ang_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    sc_steps: dict[str, int] = defaultdict(int)
    seg_steps: dict[str, int] = defaultdict(int)
    seg_order: list[str] = []

    with torch.no_grad():
      for cmd, n_steps, seg in schedule:
        cmd_t = torch.tensor(cmd, device=device, dtype=torch.float32)
        if seg not in seg_order:
          seg_order.append(seg)
        start_foot_xy = robot.data.site_pos_w[:, foot_ids, :2].clone()
        twist.set_gamepad_source(lambda c=cmd: c)
        for _ in range(n_steps):
          obs = env.get_observations()
          env.step(policy(obs))
          lin_b = robot.data.root_link_lin_vel_b
          ang_b = robot.data.root_link_ang_vel_b
          lin_error_sq = torch.sum(torch.square(cmd_t[:2] - lin_b[:, :2]), dim=1)
          lin_error_sq = lin_error_sq + torch.square(lin_b[:, 2])
          ang_error_sq = torch.square(cmd_t[2] - ang_b[:, 2])
          ang_error_sq = ang_error_sq + torch.sum(torch.square(ang_b[:, :2]), dim=1)
          for term_name in tm.active_terms:
            term_counts[(seg, term_name)] += int(tm.get_term(term_name).sum().item())
          seps[seg].append(feet_sep())
          tilts[seg].append(tilt_deg())
          heights[seg].append(robot.data.root_link_pos_w[:, 2].clone())
          vels[seg].append(torch.stack((lin_b[:, 0], lin_b[:, 1], ang_b[:, 2]), dim=1).clone())
          foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
          foot_disps[seg].append(torch.norm(foot_xy - start_foot_xy, dim=-1).max(dim=1).values)
          vel_errs[seg].append(
            torch.stack(
              (
                torch.abs(lin_b[:, 0] - cmd_t[0]),
                torch.abs(lin_b[:, 1] - cmd_t[1]),
                torch.abs(ang_b[:, 2] - cmd_t[2]),
              ),
              dim=1,
            )
          )
          lin_scores[seg].append(torch.exp(-lin_error_sq / lin_track_std**2))
          ang_scores[seg].append(torch.exp(-ang_error_sq / ang_track_std**2))
          sc_steps[seg] += int(self_collision_envs().sum().item())
          seg_steps[seg] += args.num_envs

    print(f"\n=== {name} ({args.num_envs} envs) ===")
    for seg in seg_order:
      total = seg_steps[seg]
      sep = torch.cat(seps[seg]).float()
      tilt = torch.cat(tilts[seg]).float()
      height = torch.cat(heights[seg]).float()
      vel = torch.cat(vels[seg]).float()
      vel_err = torch.cat(vel_errs[seg]).float()
      foot_disp = torch.cat(foot_disps[seg]).float()
      lin_score = torch.cat(lin_scores[seg]).float()
      ang_score = torch.cat(ang_scores[seg]).float()
      sq = torch.quantile(sep, torch.tensor([0.01, 0.05, 0.5], device=sep.device))
      tq = torch.quantile(tilt, torch.tensor([0.5, 0.99], device=tilt.device))
      eq = torch.quantile(vel_err, torch.tensor([0.5, 0.9], device=vel_err.device), dim=0)
      dq = torch.quantile(foot_disp, torch.tensor([0.5, 0.99], device=foot_disp.device))
      print(f"  [{seg}] {total} env-steps ({total // args.num_envs} steps)")
      fired = {t: c for (s, t), c in term_counts.items() if s == seg and c}
      if fired:
        for t, c in sorted(fired.items()):
          print(f"    termination {t:28s} {c:6d}  ({c / total:.5f}/env-step)")
      else:
        print("    terminations: none")
      print(
        f"    self-collision(>{sc_threshold:.0f}N): {sc_steps[seg]} env-steps"
        f" ({100.0 * sc_steps[seg] / total:.2f}%)"
      )
      print(
        f"    feet sep  p1={sq[0]:.3f} p5={sq[1]:.3f} p50={sq[2]:.3f}"
        f"  <0.15: {100.0 * (sep < 0.15).float().mean():.2f}%"
      )
      print(
        f"    tilt(deg) p50={tq[0]:.1f} p99={tq[1]:.1f} max={tilt.max():.1f}"
        f"  >20deg: {100.0 * (tilt > 20.0).float().mean():.2f}%"
      )
      print(
        f"    actual mean vx={vel[:, 0].mean():.3f} vy={vel[:, 1].mean():.3f}"
        f" wz={vel[:, 2].mean():.3f}"
      )
      print(
        f"    foot xy disp p50={dq[0]:.3f} p99={dq[1]:.3f} max={foot_disp.max():.3f}"
      )
      print(
        f"    abs err p50 vx={eq[0, 0]:.3f} vy={eq[0, 1]:.3f} wz={eq[0, 2]:.3f}"
        f"  p90 vx={eq[1, 0]:.3f} vy={eq[1, 1]:.3f} wz={eq[1, 2]:.3f}"
      )
      print(f"    tracking score linear={lin_score.mean():.3f} angular={ang_score.mean():.3f}")
      print(f"    base height mean={height.mean():.3f} min={height.min():.3f}")

  wanted = [p.strip() for p in args.phases.split(",") if p.strip()]
  zero = (0.0, 0.0, 0.0)
  if "stand" in wanted:
    run_phase("STAND cmd=0", [(zero, args.steps, "stand")])
  if "slow" in wanted:
    run_phase("SLOW vx=0.35", [((0.35, 0.0, 0.0), args.steps, "slow")])
  if "walk" in wanted:
    run_phase("WALK vx=0.6", [((0.6, 0.0, 0.0), args.steps, "walk")])
  if "turn" in wanted:
    run_phase("TURN vx=0.2 wz=0.35", [((0.2, 0.0, 0.35), args.steps, "turn")])
  if "walk1" in wanted or "walk1.0" in wanted:
    run_phase("WALK vx=1.0", [((1.0, 0.0, 0.0), args.steps, "walk1")])
  if "sidestep" in wanted:
    run_phase("SIDESTEP vy=0.5", [((0.0, 0.5, 0.0), 250, "side")])
  if "stopgo" in wanted:
    cycle = [((0.6, 0.0, 0.0), 150, "go"), (zero, 250, "stop")]
    run_phase("STOP-GO x3 (3s walk / 5s stand)", cycle * 3)

  env.close()


if __name__ == "__main__":
  main()
