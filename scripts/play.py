"""Play/evaluate RL agent after registering spider_rl_mjlab tasks.

Adds climb-wall gravity playback controls on top of mjlab's play script:

  python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --checkpoint-file logs/.../model_2000.pt
  python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --gravity-angle-deg 90 --agent zero
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig as MjlabPlayConfig
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from spider_rl_mjlab.tasks.climb_wall.mdp.curriculums import resolve_gravity_angle


@dataclass(frozen=True)
class PlayConfig(MjlabPlayConfig):
  gravity_angle_deg: float | None = None
  """Override climb-wall play gravity angle in degrees. If omitted, infer it from model_<iter>.pt."""


def _display_available() -> bool:
  """Check if a working display is available for the native viewer."""
  display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
  if not display:
    return False
  try:
    import glfw

    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1, 1, "", None, None)
    if window:
      glfw.destroy_window(window)
      glfw.terminate()
      return True
    glfw.terminate()
    return False
  except Exception:
    return False


def _normalize_cli_aliases(argv: list[str]) -> list[str]:
  aliases = {
    "--gravity-angle": "--gravity-angle-deg",
    "--gravity-angle-degrees": "--gravity-angle-deg",
  }
  return [aliases.get(arg, arg) for arg in argv]


def _checkpoint_iteration(path_or_name: str | Path | None) -> int | None:
  if path_or_name is None:
    return None
  name = Path(path_or_name).name
  match = re.search(r"(?:model|checkpoint|ckpt)[_-](\d+)", name)
  if match is None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
  return int(match.group(1)) if match is not None else None


def _apply_climb_wall_play_gravity(
  env_cfg,
  *,
  gravity_angle_deg: float | None,
  checkpoint_path: Path | None,
  checkpoint_hint: str | None = None,
) -> None:
  curriculum = getattr(env_cfg, "curriculum", None)
  if not curriculum or "gravity_angle" not in curriculum:
    return

  term_cfg = curriculum["gravity_angle"]
  stages = term_cfg.params.get("angle_stages")
  if not stages:
    return

  if gravity_angle_deg is not None:
    angle_deg = float(gravity_angle_deg)
    source = "CLI override"
  else:
    checkpoint_iter = _checkpoint_iteration(checkpoint_path)
    if checkpoint_iter is None:
      checkpoint_iter = _checkpoint_iteration(checkpoint_hint)
    if checkpoint_iter is None:
      print(
        "[WARN]: Could not infer gravity angle because checkpoint name does not "
        "look like model_<iter>.pt; keeping play curriculum at its initial angle."
      )
      return
    angle_deg = float(resolve_gravity_angle(checkpoint_iter, stages))
    source = f"checkpoint iter {checkpoint_iter}"

  term_cfg.params["angle_stages"] = [{"iter": 0, "angle_deg": angle_deg}]
  print(f"[INFO]: Play gravity angle set to {angle_deg:.2f} deg ({source})")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  dummy_mode = cfg.agent in {"zero", "random"}
  trained_mode = not dummy_mode

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif dummy_mode:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if trained_mode:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  _apply_climb_wall_play_gravity(
    env_cfg,
    gravity_angle_deg=cfg.gravity_angle_deg,
    checkpoint_path=resume_path,
    checkpoint_hint=cfg.wandb_checkpoint_name or cfg.checkpoint_file,
  )

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  if cfg.video and dummy_mode:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if trained_mode and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if dummy_mode:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  import spider_rl_mjlab.tasks  # noqa: F401

  sys.argv[:] = _normalize_cli_aliases(sys.argv)

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  _ = load_rl_cfg(chosen_task)
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  if "--viewer" not in sys.argv and not _display_available():
    sys.argv.extend(["--viewer", "viser"])
  main()
