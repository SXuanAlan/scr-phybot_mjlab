"""Measure per-world peak contact & constraint counts for climb_wall.

Runs the task for a number of steps with random actions, records per-env
contact count (from contact.worldid) and nefc, then reports percentiles.
Use output to tune SimulationCfg.nconmax and njmax.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import warp as wp

import mjlab.tasks  # noqa: F401  register builtin tasks
import scr_phybot_mjab.tasks  # noqa: F401  register spider tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Mjlab-Climb-Wall-Phybot-C2")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--action-scale", type=float, default=0.3)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  # Generous limits so we measure true needs, not truncated.
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 1500

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()

  sim = env.sim
  wp_data = sim._wp_data  # mjwarp Data

  action_dim = sum(env.action_manager.action_term_dim)

  ncon_max_per_step = np.zeros(args.steps, dtype=np.int32)
  nefc_max_per_step = np.zeros(args.steps, dtype=np.int32)
  ncon_sum_per_step = np.zeros(args.steps, dtype=np.int32)
  nefc_sum_per_step = np.zeros(args.steps, dtype=np.int32)

  for step in range(args.warmup + args.steps):
    act = (torch.rand((args.num_envs, action_dim), device=args.device) * 2.0 - 1.0) * args.action_scale
    env.step(act)
    if step < args.warmup:
      continue

    nefc_t = wp.to_torch(wp_data.nefc)  # (nworld,) int
    nacon = int(wp.to_torch(wp_data.nacon)[0].item())
    worldid = wp.to_torch(wp_data.contact.worldid)[:nacon].long()
    ncon_per_world = torch.bincount(worldid, minlength=args.num_envs)

    i = step - args.warmup
    ncon_max_per_step[i] = int(ncon_per_world.max().item())
    nefc_max_per_step[i] = int(nefc_t.max().item())
    ncon_sum_per_step[i] = nacon
    nefc_sum_per_step[i] = int(nefc_t.sum().item())

  def pct(arr, p):
    return int(np.percentile(arr, p))

  print(f"\nTask: {args.task}")
  print(f"num_envs={args.num_envs}, steps measured={args.steps}, action_scale={args.action_scale}")
  print()
  print(f"{'metric':<20} {'max':>6} {'p99':>6} {'p95':>6} {'p50':>6} {'mean':>8}")
  print("-" * 60)
  print(f"{'ncon (per world)':<20} {ncon_max_per_step.max():>6d} {pct(ncon_max_per_step,99):>6d} {pct(ncon_max_per_step,95):>6d} {pct(ncon_max_per_step,50):>6d} {ncon_max_per_step.mean():>8.2f}")
  print(f"{'nefc (per world)':<20} {nefc_max_per_step.max():>6d} {pct(nefc_max_per_step,99):>6d} {pct(nefc_max_per_step,95):>6d} {pct(nefc_max_per_step,50):>6d} {nefc_max_per_step.mean():>8.2f}")
  print()
  print(f"Total across worlds (for reference):")
  print(f"  nacon total  max={ncon_sum_per_step.max()}, mean={ncon_sum_per_step.mean():.1f}  (limit = nconmax*num_envs = {2000*args.num_envs})")
  print(f"  nefc total   max={nefc_sum_per_step.max()}, mean={nefc_sum_per_step.mean():.1f}")
  print()
  print("Suggested cfg.sim values (peak × 1.3 margin):")
  print(f"  nconmax = {int(ncon_max_per_step.max() * 1.3)}")
  print(f"  njmax   = {int(nefc_max_per_step.max() * 1.3)}")


if __name__ == "__main__":
  main()
