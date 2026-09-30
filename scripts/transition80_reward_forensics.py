"""Reward forensics: does WELDING actually pay better than HOVERING?

Rolls the policy in the unmodified training cfg (curriculum clock pinned to
the checkpoint's iteration), splits every step into "all scheduled pads
welded" vs ">=1 scheduled pad missing", and prints each reward term's income
per second for both groups — the ledger behind the dare-to-touch question.
Also measures the EDGE-TO-WELD latency per EE (scheduled rising edge ->
first weld; never = inf), the standing courage metric.

Usage: python scripts/transition80_reward_forensics.py <ckpt.pt> [envs] [steps]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import scr_phybot_mjab.tasks  # noqa: F401

CKPT = Path(sys.argv[1])
NENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
ITER = int("".join(ch for ch in CKPT.stem if ch.isdigit()) or "0")
TASK = "Mjlab-Climb-Transition80-Phybot-C2"
device = "cuda:0"

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NENVS
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), "/tmp/.t80_forensics_runner", device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
u = env.unwrapped
u.common_step_counter = ITER * 96
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
rm = u.reward_manager
terms = list(rm.active_terms)
EE = ["LH", "RH", "LF", "RF"]
DT = float(u.step_dt)

env.reset()
obs = wrapped.get_observations()
sum_welded = torch.zeros(len(terms), dtype=torch.float64, device=device)
sum_hover = torch.zeros_like(sum_welded)
n_welded = 0
n_hover = 0
# edge-to-weld bookkeeping
pending = torch.full((NENVS, 4), -1, dtype=torch.long, device=device)  # step the edge rose
latencies: list[list[int]] = [[] for _ in range(4)]
infs = [0, 0, 0, 0]
prev_sched = motion.ref_engaged.clone()
for step in range(STEPS):
  with torch.no_grad():
    act = policy(obs)
  obs, _, dones, _ = wrapped.step(act)
  step_rew = rm._step_reward  # (E, T) this step's per-term reward
  sched = motion.ref_engaged
  eng = aff.engaged
  missing = (sched & ~eng).any(dim=-1)
  full = sched.any(dim=-1) & ~missing
  if int(full.sum()):
    sum_welded += step_rew[full].sum(dim=0).double()
    n_welded += int(full.sum())
  if int(missing.sum()):
    sum_hover += step_rew[missing].sum(dim=0).double()
    n_hover += int(missing.sum())
  # edges: schedule rising and not already welded
  rise = sched & ~prev_sched
  newly = rise & ~eng
  pending = torch.where(newly, torch.full_like(pending, step), pending)
  # welds resolve pending
  resolved = (pending >= 0) & eng
  for i in range(4):
    for e in torch.nonzero(resolved[:, i]).flatten().tolist():
      latencies[i].append(step - int(pending[e, i]))
    pending[:, i] = torch.where(resolved[:, i], torch.full_like(pending[:, i], -1), pending[:, i])
  # deaths drop pending as inf
  if dones.any():
    d = dones.flatten().bool()
    for i in range(4):
      infs[i] += int(((pending[:, i] >= 0) & d).sum())
    pending[d] = -1
  prev_sched = sched.clone()

print(f"=== forensics {CKPT.name} iter {ITER} envs {NENVS} steps {STEPS} ===")
print(f"steps in group: all-scheduled-WELDED {n_welded}  |  >=1 scheduled MISSING (hover) {n_hover}")
print(f"{'term':22s} {'welded /s':>12s} {'hover /s':>12s} {'diff /s':>12s}")
tot_w = tot_h = 0.0
for i, t in enumerate(terms):
  w = float(sum_welded[i]) / max(n_welded, 1) / DT
  h = float(sum_hover[i]) / max(n_hover, 1) / DT
  tot_w += w
  tot_h += h
  print(f"{t:22s} {w:12.3f} {h:12.3f} {w - h:+12.3f}")
print(f"{'TOTAL':22s} {tot_w:12.3f} {tot_h:12.3f} {tot_w - tot_h:+12.3f}")
print("edge-to-weld latency (steps; scheduled rising edges only):")
for i in range(4):
  arr = np.array(latencies[i])
  if arr.size:
    print(f"  {EE[i]}: n={arr.size} median {np.median(arr):.0f} p90 {np.percentile(arr, 90):.0f} | never-welded (died pending) {infs[i]}")
  else:
    print(f"  {EE[i]}: n=0 | never-welded (died pending) {infs[i]}")
