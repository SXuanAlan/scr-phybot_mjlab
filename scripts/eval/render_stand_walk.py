"""Render walk-policy rollouts of a checkpoint to mp4, by default in <run>/eval/<checkpoint>/videos/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from scr_phybot_mjab.tools.paths import checkpoint_eval_dir
from scr_phybot_mjab.utils.runtime import load_policy, make_env


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Walk-Amp-Phybot-C2")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--output-dir", type=Path, help="default: <run>/eval/<checkpoint>/videos/")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--steps", type=int, default=500, help="steps per single-command phase")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--fps", type=int, default=25)
  ap.add_argument(
    "--phases",
    default="stand,walk06,walk10,stopgo",
    help="comma-separated: stand,walk06,walk10,side,stopgo",
  )
  args = ap.parse_args(argv)

  configure_torch_backends()
  device = args.device

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 2
  cfg.viewer.width = args.width
  cfg.viewer.height = args.height
  env = make_env(args.task, cfg, device, render_mode="rgb_array")
  _, policy = load_policy(args.task, env, args.ckpt, device)

  uenv = env.unwrapped
  twist = uenv.command_manager.get_term("twist")
  out_dir = args.output_dir or checkpoint_eval_dir(args.ckpt) / "videos"
  out_dir.mkdir(parents=True, exist_ok=True)

  def record(name: str, schedule: list[tuple[tuple[float, float, float], int]]) -> None:
    uenv.reset()
    frames = []
    with torch.no_grad():
      step = 0
      for cmd, n_steps in schedule:
        twist.set_gamepad_source(lambda c=cmd: c)
        for _ in range(n_steps):
          obs = env.get_observations()
          env.step(policy(obs))
          if step % 2 == 0:  # 25 fps from 50 Hz control
            frame = uenv.render()
            if frame is not None:
              frames.append(frame)
          step += 1
    path = out_dir / f"{name}.mp4"
    imageio.mimsave(path, frames, fps=args.fps)
    print(f"saved {path} ({len(frames)} frames)")

  phases = {p.strip() for p in args.phases.split(",") if p.strip()}
  if "stand" in phases:
    record("amp_stand_zero", [((0.0, 0.0, 0.0), args.steps)])
  if "walk06" in phases:
    record("amp_walk_vx06", [((0.6, 0.0, 0.0), args.steps)])
  if "walk10" in phases:
    record("amp_walk_vx10", [((1.0, 0.0, 0.0), args.steps)])
  if "side" in phases:
    record("amp_sidestep_vy05", [((0.0, 0.5, 0.0), args.steps)])
  if "stopgo" in phases:
    record(
      "amp_stopgo_x3",
      [((0.6, 0.0, 0.0), 150), ((0.0, 0.0, 0.0), 250)] * 3,
    )
  env.close()


if __name__ == "__main__":
  main()
