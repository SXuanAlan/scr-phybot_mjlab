"""L-D unified mount+climb student: PPO + segment-routed annealed KL.

The full-chain policy (standing spawn -> mount -> seam -> cycles) trained
per docs/plans/2026-08-10_unified_mount_climb_policy_plan.md §6:

* warm-start from the CLIMB teacher checkpoint (the hardest-won asset is
  the starting point; the mount skill flows in through the KL);
* PPO on the unified env's climb-parity rewards (no press bootstrap — the
  ignition wars are encapsulated in the mount teacher);
* an annealed imitation pass per iteration: mount-segment samples imitate
  the mount teacher, cycle-segment samples the climb teacher, and a fade
  band around the handoff frame (which neither teacher has seen) leaves
  the seam to RL alone.

Both teachers consume the student's actor contract verbatim (one obs
feeds all three networks); routing/fade are captured at rollout time by
the wrapper below into a DistillBuffer.

Example:
  python scripts/train_unified_kl.py \\
    --mount-teacher logs/.../transition-vX/model_NNNN.pt \\
    --climb-teacher logs/.../deploypd-v4-rung85/model_5050.pt \\
    --num-envs 2048 --run-name unified-v1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

ENV_TASK = "Mjlab-Climb-Unified-Phybot-C2"
RL_TASK = "Mjlab-Climb-Transition-Phybot-C2"


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--mount-teacher", type=Path, required=True,
                 help="mount specialist PPO checkpoint (frozen KL teacher)")
  p.add_argument("--climb-teacher", type=Path, required=True,
                 help="cycle teacher PPO checkpoint (frozen KL teacher AND "
                 "the student's warm-start unless --init overrides)")
  p.add_argument("--init", type=Path, default=None,
                 help="student init checkpoint (default: --climb-teacher)")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--max-iterations", type=int, default=4000)
  p.add_argument("--run-name", default="unified-v1")
  p.add_argument("--gpu-id", type=int, default=0)
  p.add_argument("--lr", type=float, default=1.0e-4)
  p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
  # Distillation knobs (plan §6.3: tune at 90, bands inherit).
  p.add_argument("--lambda0", type=float, default=1.0)
  p.add_argument("--lambda-final", type=float, default=0.0)
  p.add_argument("--anneal-iters", type=int, default=2000)
  p.add_argument("--seam-fade-s", type=float, default=0.4,
                 help="half-width (s) of the zero-KL band around handoff")
  p.add_argument("--distill-mode", choices=("kl", "mse"), default="kl")
  p.add_argument("--distill-epochs", type=int, default=1)
  p.add_argument("--distill-minibatches", type=int, default=4)
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  init_ckpt = args.init or args.climb_teacher
  for f in (args.mount_teacher, args.climb_teacher, init_ckpt):
    if not f.exists():
      raise FileNotFoundError(f)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0"

  import dataclasses

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wandb import add_wandb_tags

  from scr_phybot_mjab.rl.kl_distill import (
    DistillBuffer,
    DistillCfg,
    KLDistillPPO,
    load_frozen_actor,
  )

  configure_torch_backends()

  env_cfg = load_env_cfg(ENV_TASK)
  agent_cfg = load_rl_cfg(RL_TASK)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.run_name = args.run_name
  agent_cfg.logger = args.logger
  agent_cfg.upload_model = False
  agent_cfg.experiment_name = "phybot_c2_climb_unified"
  agent_cfg.wandb_tags = ("phybot_c2", "unified-kl")
  # Warm-started policy on its own rollouts: fixed low LR (the adaptive-KL
  # controller fights it — same call as the estimator finetune).
  agent_cfg.algorithm = dataclasses.replace(
    agent_cfg.algorithm,
    learning_rate=args.lr,
    schedule="fixed",
    class_name="scr_phybot_mjab.rl.kl_distill:KLDistillPPO",
  )

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_dir = log_root / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
  )
  log_dir.mkdir(parents=True, exist_ok=False)
  print(f"[unified] run: {log_dir}")
  print(f"[unified] mount teacher: {args.mount_teacher}")
  print(f"[unified] climb teacher: {args.climb_teacher}")
  print(f"[unified] student init:  {init_ckpt}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  buffer = DistillBuffer()
  live_motion = env.command_manager.get_term("climb_motion")
  fade_frames = max(1, int(round(args.seam_fade_s * live_motion.clip.fps)))
  print(
    f"[unified] handoff frame {live_motion._handoff_frame}, "
    f"seam fade +-{fade_frames} frames, episode mix "
    f"{env_cfg.commands['climb_motion'].episode_mix}"
  )

  class RouterCaptureWrapper(RslRlVecEnvWrapper):
    """Capture (actor obs, segment route, seam fade) every step."""

    def _capture(self, obs):
      motion = self.unwrapped.command_manager.get_term("climb_motion")
      dist = (motion.time_steps - motion._handoff_frame).abs()
      buffer.append(
        obs["actor"].detach().clone(),
        motion.in_cycle_region.clone(),
        (dist >= fade_frames).float(),
      )
      return obs

    def get_observations(self):
      return self._capture(super().get_observations())

    def step(self, actions):
      obs, rew, dones, extras = super().step(actions)
      return self._capture(obs), rew, dones, extras

  wrapped = RouterCaptureWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), str(log_dir), device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  runner.load(
    str(init_ckpt),
    load_cfg={"actor": True, "critic": True},
    strict=True,
    map_location=device,
  )
  runner.current_learning_iteration = 0
  env.unwrapped.common_step_counter = 0

  alg = runner.alg
  if not isinstance(alg, KLDistillPPO):
    raise RuntimeError(
      f"Runner built {type(alg).__name__}, expected KLDistillPPO — check "
      "algorithm.class_name plumbing."
    )
  mount_teacher = load_frozen_actor(alg._raw_actor, args.mount_teacher, device)
  cycle_teacher = load_frozen_actor(alg._raw_actor, args.climb_teacher, device)
  num_actions = int(wrapped.num_actions)
  alg.attach_distillation(
    buffer,
    mount_teacher,
    cycle_teacher,
    DistillCfg(
      lambda0=args.lambda0,
      lambda_final=args.lambda_final,
      anneal_iters=args.anneal_iters,
      epochs=args.distill_epochs,
      num_minibatches=args.distill_minibatches,
      mode=args.distill_mode,
    ),
    num_actions=num_actions,
  )

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


if __name__ == "__main__":
  main()
