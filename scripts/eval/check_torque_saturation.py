"""Run a policy for N steps and report per-actuator torque saturation.

Examples:
  python scripts/eval/check_torque_saturation.py Mjlab-Climb-Wall-Phybot-C2 \
      --agent zero --num-envs 8 --steps 200
  python scripts/eval/check_torque_saturation.py Mjlab-Climb-Wall-Phybot-C2 \
      --agent trained --checkpoint-file logs/.../model.pt
"""

import re
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

import mjlab  # noqa: F401
import torch
import tyro
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
)
from mjlab.utils.torch import configure_torch_backends

import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.utils.runtime import load_policy, make_env, pin_climb_wall_angle, resolve_checkpoint


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
  wall_angle_deg: float | None = None
  """Override the climb wall angle. If omitted, infer from model_<iter>.pt."""
  save_plots: bool = True
  """Save torque-vs-time plots under results/checkpoint by default."""
  output_dir: str | None = None
  """Directory for plots. Defaults to results/checkpoint/<experiment>/<run>/<checkpoint>."""
  plot_env_idx: int = 0
  """Environment index used for signed torque-vs-time traces."""


def _safe_path_part(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _default_output_dir(task_id: str, cfg: Cfg, checkpoint_path: Path | None) -> Path:
  if cfg.output_dir is not None:
    return Path(cfg.output_dir)

  root = Path("results") / "checkpoint"
  if checkpoint_path is not None:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = _safe_path_part(checkpoint_path.stem)
    run = _safe_path_part(checkpoint_path.parent.name)
    experiment = _safe_path_part(checkpoint_path.parent.parent.name)
    return root / experiment / run / checkpoint

  return root / _safe_path_part(task_id) / _safe_path_part(cfg.agent)


def _save_torque_plots(
  *,
  out_dir: Path,
  names: list[str],
  torque_env: torch.Tensor,
  max_abs_ratio: torch.Tensor,
  limits: torch.Tensor,
  step_dt: float,
  sat_threshold: float,
  plot_env_idx: int,
) -> None:
  import matplotlib

  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt

  out_dir.mkdir(parents=True, exist_ok=True)

  time_s = torch.arange(torque_env.shape[0], dtype=torch.float32) * step_dt
  time_np = time_s.numpy()
  limits_np = limits.cpu().numpy()
  n_act = len(names)
  ncols = 4
  nrows = ceil(n_act / ncols)

  fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5.0 * ncols, max(2.1 * nrows, 3.5)),
    sharex=True,
    squeeze=False,
  )
  for i, ax in enumerate(axes.ravel()):
    if i >= n_act:
      ax.axis("off")
      continue
    lim = float(limits_np[i])
    ax.plot(time_np, torque_env[:, i].numpy(), linewidth=0.8, color="#1f77b4")
    ax.axhline(lim, color="#b22222", linestyle="--", linewidth=0.7)
    ax.axhline(-lim, color="#b22222", linestyle="--", linewidth=0.7)
    ax.set_title(names[i], fontsize=8)
    ax.grid(True, alpha=0.25)
    if i % ncols == 0:
      ax.set_ylabel("torque")
  fig.supxlabel("time (s)")
  fig.suptitle(f"Torque vs time (env {plot_env_idx})", fontsize=12)
  fig.tight_layout()
  torque_path = out_dir / "torque_vs_time.png"
  fig.savefig(torque_path, dpi=140, bbox_inches="tight")
  plt.close(fig)

  fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5.0 * ncols, max(2.1 * nrows, 3.5)),
    sharex=True,
    squeeze=False,
  )
  ratio_np = max_abs_ratio.numpy()
  y_max = max(1.05, float(max_abs_ratio.max().item()) * 1.05)
  for i, ax in enumerate(axes.ravel()):
    if i >= n_act:
      ax.axis("off")
      continue
    ax.plot(time_np, ratio_np[:, i], linewidth=0.8, color="#2ca02c")
    ax.axhline(sat_threshold, color="#b22222", linestyle="--", linewidth=0.7)
    ax.axhline(1.0, color="#111111", linestyle=":", linewidth=0.7)
    ax.set_ylim(0.0, y_max)
    ax.set_title(names[i], fontsize=8)
    ax.grid(True, alpha=0.25)
    if i % ncols == 0:
      ax.set_ylabel("max |tau| / limit")
  fig.supxlabel("time (s)")
  fig.suptitle("Torque saturation ratio vs time (max over envs)", fontsize=12)
  fig.tight_layout()
  ratio_path = out_dir / "torque_ratio_vs_time.png"
  fig.savefig(ratio_path, dpi=140, bbox_inches="tight")
  plt.close(fig)

  print(f"[INFO]: Wrote torque plots to {out_dir}")


def run(task_id: str, cfg: Cfg):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  resume_path: Path | None = None
  if cfg.agent == "trained":
    resume_path, _ = resolve_checkpoint(
      agent_cfg.experiment_name, cfg.checkpoint_file, cfg.wandb_run_path, cfg.wandb_checkpoint_name
    )

  if cfg.agent == "trained" or cfg.wall_angle_deg is not None:
    pin_climb_wall_angle(
      env_cfg,
      wall_angle_deg=cfg.wall_angle_deg,
      checkpoint=resume_path,
      checkpoint_hint=cfg.wandb_checkpoint_name or cfg.checkpoint_file,
    )

  env = make_env(task_id, env_cfg, device)

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
    _, policy = load_policy(task_id, env, resume_path, device)

  obs, _ = env.reset()

  fr = env.unwrapped.sim.model.actuator_forcerange
  if fr.ndim == 3:
    limit_hi = torch.as_tensor(fr[:, ctrl_ids, 1], device=device).abs()
    limit_lo = torch.as_tensor(fr[:, ctrl_ids, 0], device=device).abs()
    limit = torch.maximum(limit_hi, limit_lo)  # (num_envs, n_act)
    limit_per_actuator = limit[0]
  else:
    limit_per_actuator = torch.as_tensor(fr[ctrl_ids], device=device).abs().amax(dim=-1)
    limit = limit_per_actuator.unsqueeze(0).expand(cfg.num_envs, -1)

  n_act = len(names)
  abs_max = torch.zeros(n_act, device=device)
  count_sat = torch.zeros(n_act, device=device)
  sum_abs = torch.zeros(n_act, device=device)
  total = 0
  plot_env_idx = min(max(cfg.plot_env_idx, 0), cfg.num_envs - 1)
  if plot_env_idx != cfg.plot_env_idx:
    print(f"[WARN]: plot_env_idx {cfg.plot_env_idx} is out of range for {cfg.num_envs} envs; using {plot_env_idx}.")
  torque_env_history: list[torch.Tensor] = []
  ratio_max_history: list[torch.Tensor] = []

  with torch.inference_mode():
    for _ in range(cfg.steps):
      action = policy(obs)
      obs, _, _, _ = env.step(action)
      tau_signed = asset.data.actuator_force  # (num_envs, n_act)
      tau = tau_signed.abs()
      ratio = tau / limit.clamp(min=1e-6)
      abs_max = torch.maximum(abs_max, ratio.amax(dim=0))
      count_sat += (ratio >= cfg.sat_threshold).sum(dim=0).float()
      sum_abs += tau.sum(dim=0)
      total += tau.shape[0]
      if cfg.save_plots:
        torque_env_history.append(tau_signed[plot_env_idx].detach().cpu())
        ratio_max_history.append(ratio.amax(dim=0).detach().cpu())

  mean_ratio = (sum_abs / total) / limit_per_actuator.clamp(min=1e-6)

  print(f"\n[{cfg.agent} agent | {cfg.num_envs} envs × {cfg.steps} steps = {total} samples per actuator]")
  header = (
    f"{'actuator':<40}{'limit':>10}{'mean|τ|/lim':>14}{'max|τ|/lim':>14}{'sat≥' + f'{cfg.sat_threshold:.2f}':>14}"
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

  print(f"\nany-actuator any-step max ratio: {abs_max.max().item():.3f}  (>=1.0 means hit forcerange)")

  if cfg.save_plots:
    out_dir = _default_output_dir(task_id, cfg, resume_path)
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0))
    _save_torque_plots(
      out_dir=out_dir,
      names=names,
      torque_env=torch.stack(torque_env_history),
      max_abs_ratio=torch.stack(ratio_max_history),
      limits=limit_per_actuator.detach().cpu(),
      step_dt=step_dt,
      sat_threshold=cfg.sat_threshold,
      plot_env_idx=plot_env_idx,
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
