"""Train the registered four-corner 60-degree task with a live curriculum gate.

Same PPO/task recipe as the standard CLI. The episode clock starts at zero
because random phase RSI already diversifies births, and climb event rewards
use actual episode age for their initial settling/anti-farming guards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch
import scr_phybot_mjab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from scr_phybot_mjab.tasks.climb_wall.mdp.live_curriculum import (
  LiveCurriculumControl,
  atomic_json,
  file_controlled_wall_angle,
  initial_control,
)

TASK = "Mjlab-Climb-Wall-Phybot-C2-Corners-A60"


class MonitoredRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    sampler = env.command_manager.get_term("climb_motion").phase_sampler
    controller = env._live_climb_curriculum
    infos = {
      **(infos or {}),
      "climb_monitor": {
        "phase_failed": sampler.failed.detach().cpu(),
        "active_stage": controller.active_stage,
        "stage_started_iter": controller.stage_started_iter,
        "control": controller.control,
      },
    }
    # Publish complete checkpoint files so concurrent evaluations never load
    # a partially serialized optimizer/model.
    destination = Path(path)
    temporary = destination.with_name("." + destination.name + ".tmp")
    upload = self.cfg["upload_model"]
    self.cfg["upload_model"] = False
    try:
      super().save(str(temporary), infos)
      temporary.replace(destination)
    finally:
      self.cfg["upload_model"] = upload
    if upload:
      self.logger.save_model(str(destination), self.current_learning_iteration)

  def load(self, *args, **kwargs):
    infos = super().load(*args, **kwargs)
    env = self.env.unwrapped
    saved = (infos or {}).get("climb_monitor")
    if saved:
      sampler = env.command_manager.get_term("climb_motion").phase_sampler
      sampler.failed.copy_(saved["phase_failed"].to(env.device))
      controller: LiveCurriculumControl = env._live_climb_curriculum
      controller.active_stage = saved["active_stage"]
      controller.stage_started_iter = saved["stage_started_iter"]
      controller.control = saved["control"]
      controller.last_iteration = -1
    # Saved model_N already completed update N; continue at N+1.
    self.current_learning_iteration += 1
    env.reset()
    return infos


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iterations", type=int, default=12000)
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
  args = parser.parse_args()
  run_dir = args.run_dir.resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  control_path = run_dir / "curriculum_control.json"
  if not control_path.exists():
    atomic_json(control_path, initial_control())
  configure_torch_backends()
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
  cfg = load_env_cfg(TASK)
  agent = load_rl_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = agent.seed
  agent.max_iterations = args.max_iterations
  agent.logger = args.logger
  agent.run_name = run_dir.name
  curriculum = cfg.curriculum["wall_angle"]
  curriculum.func = file_controlled_wall_angle
  curriculum.params["control_file"] = str(control_path)
  assert cfg.commands["microspine_affordance"].microspine_cfg.limit_surface.unit_layout == "corners"
  assert len(cfg.rewards) == 16
  print(f"[train] task={TASK} envs={args.num_envs} run_dir={run_dir}", flush=True)
  print(
    "[train] live curriculum gate; init_at_random_ep_len=False; physical capacity and rewards unchanged", flush=True
  )
  dump_yaml(run_dir / "params" / "env.yaml", dataclasses.asdict(cfg))
  dump_yaml(run_dir / "params" / "agent.yaml", dataclasses.asdict(agent))
  atomic_json(
    run_dir / "launch.json",
    {
      "task": TASK,
      "num_envs": args.num_envs,
      "max_iterations": args.max_iterations,
      "seed": agent.seed,
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
      "control_file": str(control_path),
      "init_at_random_ep_len": False,
      "resume": str(args.resume) if args.resume else None,
      "pid": os.getpid(),
    },
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MonitoredRunner(wrapper, dataclasses.asdict(agent), str(run_dir), "cuda:0")
    add_wandb_tags((*agent.wandb_tags, "bm16", "live_curriculum", "envs4096"))
    runner.add_git_repo_to_log(__file__)
    if args.resume:
      runner.load(str(args.resume), map_location="cuda:0")
    remaining = args.max_iterations - runner.current_learning_iteration
    if remaining <= 0:
      raise ValueError("Requested iteration limit is already completed by the checkpoint")
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    atomic_json(
      run_dir / "training_finished.json",
      {
        "last_iteration": runner.current_learning_iteration,
        "stage_index": env._live_climb_curriculum.active_stage,
        "target_stage_reached": env._live_climb_curriculum.active_stage == 2,
      },
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
