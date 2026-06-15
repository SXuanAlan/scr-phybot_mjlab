"""Headless A/B check: do the new resting self-collision excludes kill the
spurious self_collision terminations when the trained walk policy stands still?

Usage:
  python validate_walk_excludes.py            # current config (new excludes)
  STRIP_NEW_EXCLUDES=1 python validate_walk_excludes.py   # old behavior arm
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

_REPO = Path("/home/spider/projects/scr-phybot_mjlab")
_REPO_SRC = _REPO / "src"
if str(_REPO_SRC) not in sys.path:
  sys.path.insert(0, str(_REPO_SRC))

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import scr_phybot_mjab  # noqa: F401  (registers tasks)
from scr_phybot_mjab.assets.phybot_c2 import phybot_c2_constants as c2

TASK = "Mjlab-Walk-Phybot-C2"
CKPT = _REPO / "logs/rsl_rl/phybot_c2_walk/2026-06-12_11-05-55/model_11999.pt"
NUM_ENVS = 256
PHASE_STEPS = 750  # 15 s at 50 Hz control


def _old_excludes_only(spec) -> None:
  """Pre-fix behavior: only the hand-spine-vs-hip pairs are excluded."""
  hand_spine_bodies = (
    "hand_spine",
    *(f"hand_spine_c_{i}_{j}" for i in range(2) for j in range(2)),
  )
  for side in ("left", "right"):
    for spine_body in hand_spine_bodies:
      spec.add_exclude(
        bodyname1=f"{side}_{spine_body}",
        bodyname2=f"{side}_hip_roll",
      )


def main() -> None:
  strip = os.environ.get("STRIP_NEW_EXCLUDES") == "1"
  if strip:
    c2._add_resting_self_collision_excludes = _old_excludes_only
    print("[ARM] OLD excludes (hand-spine only)")
  else:
    print("[ARM] NEW excludes (hand-spine + elbow/hip + shoulder/torso)")

  configure_torch_backends()
  device = "cuda:0"

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  twist = env.unwrapped.command_manager.get_term("twist")

  def run_phase(name: str, cmd: tuple[float, float, float]) -> None:
    twist.set_gamepad_source(lambda: cmd)
    tm = env.unwrapped.termination_manager
    counts: dict[str, int] = defaultdict(int)
    env.unwrapped.reset()
    with torch.no_grad():
      for _ in range(PHASE_STEPS):
        obs = env.get_observations()
        actions = policy(obs)
        env.step(actions)
        for term_name in tm.active_terms:
          counts[term_name] += int(tm.get_term(term_name).sum().item())
    total_env_steps = PHASE_STEPS * NUM_ENVS
    print(f"\n=== phase {name} cmd={cmd} ({PHASE_STEPS} steps x {NUM_ENVS} envs)")
    for k in sorted(counts):
      rate = counts[k] / total_env_steps
      print(f"  {k:25s} {counts[k]:6d}  ({rate:.5f}/env-step)")

  run_phase("STAND", (0.0, 0.0, 0.0))
  run_phase("WALK", (0.6, 0.0, 0.0))

  env.close()


if __name__ == "__main__":
  main()
