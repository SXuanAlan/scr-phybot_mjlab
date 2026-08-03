"""Phase-3 polish: PPO on the student obs group from a two-source warm start.

BC-init + RL polish (2026-08-03): the distilled student only imitates states
its all-knowing teacher visits, so active information gathering (probe-loading
a pad before trusting it) can only emerge under RL, where the acting policy
pays for its own engage-state uncertainty. This launcher assembles that run:

  actor  <- distill checkpoint's ``student_state_dict``   (890-dim student)
  critic <- teacher PPO checkpoint's ``critic_state_dict`` (170-dim privileged)

Both load strict — the polish runner cfg clones the distill student model cfg
field-for-field, and the critic cfg is untouched from the teacher's. Iteration
and curriculum counters reset to zero AFTER the loads (MjlabOnPolicyRunner.load
restores common_step_counter from the checkpoint's env_state; the polish env
pins its curricula, so this is hygiene, not a fix).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

DEFAULT_TASK = "Mjlab-Climb-Wall-Phybot-C2-Polish"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--distill-checkpoint", type=Path, required=True)
  parser.add_argument("--teacher-checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--run-name", default="polish")
  parser.add_argument("--gpu-id", type=int, default=0)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument(
    "--logger", choices=("tensorboard", "wandb"), default="wandb"
  )
  parser.add_argument(
    "--init-std",
    type=float,
    default=None,
    help=(
      "Overwrite distribution.std_param AFTER loading (the loaded distill "
      "value is 0.3; the cfg's init_std is discarded by load_state_dict)."
    ),
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  for ckpt in (args.distill_checkpoint, args.teacher_checkpoint):
    if not ckpt.exists():
      raise FileNotFoundError(ckpt)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  device = "cuda:0"

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wandb import add_wandb_tags

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

  print(f"[INFO] Polish run: {log_dir}")
  print(f"[INFO] actor  <- {args.distill_checkpoint} [student_state_dict]")
  print(f"[INFO] critic <- {args.teacher_checkpoint} [critic_state_dict]")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  distilled = torch.load(
    args.distill_checkpoint, weights_only=False, map_location=device
  )
  if "student_state_dict" not in distilled:
    raise KeyError(
      f"{args.distill_checkpoint} has no student_state_dict — is it a "
      "distillation checkpoint? (PPO checkpoints carry actor_state_dict.)"
    )
  runner.alg._raw_actor.load_state_dict(distilled["student_state_dict"], strict=True)
  # Critic (and ONLY the critic) from the privileged teacher; do this second —
  # runner.load also restores common_step_counter, reset right after.
  runner.load(
    str(args.teacher_checkpoint),
    load_cfg={"critic": True},
    strict=True,
    map_location=device,
  )
  runner.current_learning_iteration = 0
  env.unwrapped.common_step_counter = 0
  if args.init_std is not None:
    with torch.no_grad():
      runner.alg._raw_actor.distribution.std_param.fill_(args.init_std)
    print(f"[INFO] distribution.std_param overwritten to {args.init_std}")

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


if __name__ == "__main__":
  main()
