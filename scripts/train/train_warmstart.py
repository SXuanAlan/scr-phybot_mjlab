"""Train a task from an actor-only warm-start checkpoint."""

from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

DEFAULT_TASK = "Mjlab-Walk-Amp-Phybot-C2"
DEFAULT_WARMSTART = "logs/rsl_rl/phybot_c2_walk_amp/2026-06-13_10-08-48_autoresearch_4096x12000_s1_gpu0/model_11999.pt"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--warmstart-checkpoint", type=Path, default=Path(DEFAULT_WARMSTART))
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=200)
  parser.add_argument("--run-name", default="preset_amp_warmstart")
  parser.add_argument("--gpu-id", type=int, default=0)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
  parser.add_argument(
    "--load-mode",
    choices=("actor", "policy", "full"),
    default="actor",
    help=("actor: load only actor; policy: actor+critic+discriminator; full: policy plus optimizer"),
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if not args.warmstart_checkpoint.exists():
    raise FileNotFoundError(args.warmstart_checkpoint)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  device = "cuda:0"

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.run_name = args.run_name
  agent_cfg.logger = args.logger
  agent_cfg.upload_model = False
  if args.seed is not None:
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  log_dir = log_root / log_dir_name
  log_dir.mkdir(parents=True, exist_ok=False)

  print(f"[INFO] Training with: device={device}, seed={agent_cfg.seed}, rank=0")
  print(f"[INFO] Logging experiment in directory: {log_dir}")
  print(f"[INFO] Warm-start checkpoint: {args.warmstart_checkpoint}")
  print(f"[INFO] Warm-start load mode: {args.load_mode}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  load_policy = args.load_mode in {"policy", "full"}
  load_optimizer = args.load_mode == "full"
  load_cfg = {
    "actor": True,
    "critic": load_policy,
    "optimizer": load_optimizer,
  }
  # The AMP runner's load() takes extra kwargs (and its algorithm knows a
  # discriminator); the plain mjlab/PPO path accepts neither.
  load_kwargs: dict = {}
  if "load_optimizer" in inspect.signature(runner.load).parameters:
    load_kwargs["load_optimizer"] = load_optimizer
    load_cfg["discriminator"] = load_policy
  runner.load(
    str(args.warmstart_checkpoint),
    load_cfg=load_cfg,
    strict=True,
    map_location=device,
    **load_kwargs,
  )
  runner.current_learning_iteration = 0
  env.unwrapped.common_step_counter = 0

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
