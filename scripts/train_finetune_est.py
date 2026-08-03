"""Route-B phase 3: fine-tune the teacher policy WITH the phi estimator in
the loop.

phase 2b closed the estimator gap to 0.904-in-loop vs 0.963-with-truth; the
residual is the estimator's stable error signature at latch transitions
(feet churn, occasional falls). This script lets the POLICY adapt to that
signature: ordinary PPO (full rewards, TRUE-privileged critic), except the
actor's privileged slices are substituted by phi's estimates during rollout
— exactly the deploy-time observation channel. The policy starts from the
full-strength teacher, so there is no weld-basin risk (the polish-v1
failure needed a fall-prone init).

Wiring: the env is the -Distill variant (carries the student group phi
reads, drills off, curricula pinned to exact-90/cal); the runner cfg is the
BASE task's PPO cfg (actor<-actor 150, critic<-critic 170); a thin
RslRlVecEnvWrapper subclass substitutes the actor slices per step. Teacher
actor+critic load strict from the PPO checkpoint; counters reset.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

ENV_TASK = "Mjlab-Climb-Wall-Phybot-C2-Distill"
RL_TASK = "Mjlab-Climb-Wall-Phybot-C2"


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--estimator", type=Path, required=True)
  p.add_argument("--checkpoint", type=Path, required=True,
                 help="teacher PPO checkpoint (actor+critic load strict)")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--max-iterations", type=int, default=600)
  p.add_argument("--run-name", default="est-finetune")
  p.add_argument("--gpu-id", type=int, default=1)
  p.add_argument("--lr", type=float, default=1.0e-4)
  p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  for f in (args.estimator, args.checkpoint):
    if not f.exists():
      raise FileNotFoundError(f)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0"

  import dataclasses

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wandb import add_wandb_tags

  configure_torch_backends()

  phi = torch.jit.load(str(args.estimator), map_location=device).eval()
  meta = json.loads(args.estimator.with_suffix(".json").read_text())
  target_sl = [(t["actor_start"], t["actor_end"]) for t in meta["targets"]]
  zero_sl = [(z["actor_start"], z["actor_end"]) for z in meta["zero_slices"]]

  class EstimatorObsWrapper(RslRlVecEnvWrapper):
    """Substitute the actor's privileged slices with phi estimates."""

    def _substitute(self, obs):
      with torch.no_grad():
        est = phi(obs["student"])
      actor = obs["actor"].clone()
      ofs = 0
      for s, e in target_sl:
        actor[:, s:e] = est[:, ofs : ofs + (e - s)]
        ofs += e - s
      for s, e in zero_sl:
        actor[:, s:e] = 0.0
      obs = obs.clone()
      obs["actor"] = actor
      return obs

    def get_observations(self):
      return self._substitute(super().get_observations())

    def step(self, actions):
      obs, rew, dones, extras = super().step(actions)
      return self._substitute(obs), rew, dones, extras

  env_cfg = load_env_cfg(ENV_TASK)
  agent_cfg = load_rl_cfg(RL_TASK)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.run_name = args.run_name
  agent_cfg.logger = args.logger
  agent_cfg.upload_model = False
  agent_cfg.experiment_name = "phybot_c2_climb_wall_estft"
  agent_cfg.wandb_tags = ("phybot_c2", "est-finetune")
  # BC-adjacent fine-tune: fixed low LR (the adaptive-KL controller fights a
  # warm-started policy's own rollouts).
  agent_cfg.algorithm = dataclasses.replace(
    agent_cfg.algorithm, learning_rate=args.lr, schedule="fixed"
  )

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_dir = log_root / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
  )
  log_dir.mkdir(parents=True, exist_ok=False)
  print(f"[estft] run: {log_dir}")
  print(f"[estft] estimator: {args.estimator}")
  print(f"[estft] teacher ckpt: {args.checkpoint}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = EstimatorObsWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True, "critic": True},
    strict=True,
    map_location=device,
  )
  runner.current_learning_iteration = 0
  env.unwrapped.common_step_counter = 0

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


if __name__ == "__main__":
  main()
