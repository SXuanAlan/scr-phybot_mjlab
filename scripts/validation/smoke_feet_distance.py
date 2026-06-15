"""Smoke test the feet_distance reward term + measure foot separation stats."""

from __future__ import annotations

import sys
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

import scr_phybot_mjab  # noqa: F401

TASK = "Mjlab-Walk-Phybot-C2"
CKPT = _REPO / "logs/rsl_rl/phybot_c2_walk/2026-06-12_11-05-55/model_11999.pt"
NUM_ENVS = 128
STEPS = 250


def main() -> None:
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  assert "feet_distance" in env_cfg.rewards, "feet_distance term missing from cfg"
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  uenv = env.unwrapped
  twist = uenv.command_manager.get_term("twist")
  robot = uenv.scene["robot"]
  site_ids = [robot.data.site_names.index("left_foot"), robot.data.site_names.index("right_foot")] if hasattr(robot.data, "site_names") else None

  def separation() -> torch.Tensor:
    if site_ids is not None:
      pos_w = robot.data.site_pos_w[:, site_ids, :]
    else:
      # Fall back: resolve via reward term's asset cfg
      term_cfg = uenv.reward_manager.get_term_cfg("feet_distance")
      ids = term_cfg.params["asset_cfg"].site_ids
      pos_w = robot.data.site_pos_w[:, ids, :]
    rel = pos_w - robot.data.root_link_pos_w.unsqueeze(1)
    B, N = rel.shape[:2]
    quat = robot.data.root_link_quat_w.unsqueeze(1).expand(B, N, 4)
    pos_b = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(B, N, 3)
    return (pos_b[:, 0, 1] - pos_b[:, 1, 1]).abs()

  # 1) default stance: reset, zero steps -> separation & term value
  uenv.reset()
  uenv.sim.forward()
  sep0 = separation()
  print(f"default-stance separation: mean={sep0.mean():.4f} min={sep0.min():.4f}")

  def run_phase(name, cmd):
    twist.set_gamepad_source(lambda: cmd)
    seps = []
    penalty_steps = 0
    total = 0
    with torch.no_grad():
      for _ in range(STEPS):
        obs = env.get_observations()
        env.step(policy(obs))
        s = separation()
        seps.append(s)
        penalty_steps += int((s < 0.15).sum().item())
        total += s.numel()
    s = torch.cat(seps)
    q = torch.quantile(s.float(), torch.tensor([0.01, 0.05, 0.5], device=s.device))
    print(
      f"{name}: sep p1={q[0]:.4f} p5={q[1]:.4f} p50={q[2]:.4f} "
      f"below-0.15: {100.0 * penalty_steps / total:.2f}% of env-steps"
    )

  run_phase("STAND(old narrow policy)", (0.0, 0.0, 0.0))
  run_phase("WALK 0.6", (0.6, 0.0, 0.0))
  run_phase("SIDESTEP vy=0.5", (0.0, 0.5, 0.0))

  env.close()


if __name__ == "__main__":
  main()
