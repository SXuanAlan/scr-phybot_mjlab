"""Run a policy for N steps and report per-actuator torque saturation.

Examples:
  python scripts/check_torque_saturation.py Mjlab-Climb-Wall-Phybot-C2 \
      --agent zero --num-envs 8 --steps 200
  python scripts/check_torque_saturation.py Mjlab-Climb-Wall-Phybot-C2 \
      --agent trained --checkpoint-file logs/.../model.pt
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

import spider_rl_mjlab.tasks  # noqa: F401


@dataclass(frozen=True)
class Cfg:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  num_envs: int = 16
  steps: int = 500
  device: str | None = None
  sat_threshold: float = 0.95
  """Ratio |tau|/limit at or above which a sample counts as saturated."""


def run(task_id: str, cfg: Cfg):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  resume_path: Path | None = None
  if cfg.agent == "trained":
    log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(resume_path)
    elif cfg.wandb_run_path:
      resume_path, _ = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
    else:
      raise ValueError(
        "trained agent requires --checkpoint-file or --wandb-run-path"
      )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  asset = env.unwrapped.scene["robot"]
  ctrl_ids = asset.indexing.ctrl_ids
  names = list(asset.actuator_names)

  if cfg.agent == "zero":
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    def policy(_obs):
      return torch.zeros(action_shape, device=device)
  elif cfg.agent == "random":
    action_shape = env.unwrapped.action_space.shape

    def policy(_obs):
      return 2 * torch.rand(action_shape, device=device) - 1
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

  obs, _ = env.reset()

  fr = env.unwrapped.sim.model.actuator_forcerange
  if fr.ndim == 3:
    limit_hi = torch.as_tensor(fr[:, ctrl_ids, 1], device=device).abs()
    limit_lo = torch.as_tensor(fr[:, ctrl_ids, 0], device=device).abs()
    limit = torch.maximum(limit_hi, limit_lo)  # (num_envs, n_act)
    limit_per_actuator = limit[0]
  else:
    limit_per_actuator = (
      torch.as_tensor(fr[ctrl_ids], device=device).abs().amax(dim=-1)
    )
    limit = limit_per_actuator.unsqueeze(0).expand(cfg.num_envs, -1)

  n_act = len(names)
  abs_max = torch.zeros(n_act, device=device)
  count_sat = torch.zeros(n_act, device=device)
  sum_abs = torch.zeros(n_act, device=device)
  total = 0

  with torch.inference_mode():
    for _ in range(cfg.steps):
      action = policy(obs)
      obs, _, _, _ = env.step(action)
      tau = asset.data.actuator_force.abs()  # (num_envs, n_act)
      ratio = tau / limit.clamp(min=1e-6)
      abs_max = torch.maximum(abs_max, ratio.amax(dim=0))
      count_sat += (ratio >= cfg.sat_threshold).sum(dim=0).float()
      sum_abs += tau.sum(dim=0)
      total += tau.shape[0]

  mean_ratio = (sum_abs / total) / limit_per_actuator.clamp(min=1e-6)

  print(
    f"\n[{cfg.agent} agent | {cfg.num_envs} envs × {cfg.steps} steps "
    f"= {total} samples per actuator]"
  )
  header = (
    f"{'actuator':<40}{'limit':>10}{'mean|τ|/lim':>14}"
    f"{'max|τ|/lim':>14}{'sat≥' + f'{cfg.sat_threshold:.2f}':>14}"
  )
  print(header)
  print("-" * len(header))
  rows = []
  for i, n in enumerate(names):
    rows.append(
      (
        abs_max[i].item(),
        n,
        limit_per_actuator[i].item(),
        mean_ratio[i].item(),
        abs_max[i].item(),
        count_sat[i].item() / total,
      )
    )
  rows.sort(key=lambda r: r[0], reverse=True)
  for _, n, lim, mr, mx, sf in rows:
    print(f"{n:<40}{lim:>10.2f}{mr:>14.3f}{mx:>14.3f}{sf:>14.2%}")

  print(
    f"\nany-actuator any-step max ratio: {abs_max.max().item():.3f}  "
    f"(>=1.0 means hit forcerange)"
  )

  env.close()


def main():
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    Cfg,
    args=remaining,
    default=Cfg(),
    prog=sys.argv[0] + f" {chosen}",
    config=mjlab.TYRO_FLAGS,
  )
  run(chosen, args)


if __name__ == "__main__":
  main()
