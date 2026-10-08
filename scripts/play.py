"""Play/evaluate RL agent after registering scr_phybot_mjab tasks.

Adds climb-wall gravity playback controls on top of mjlab's play script:

  python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --checkpoint-file logs/.../model_2000.pt
  python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --gravity-angle-deg 90 --agent zero
  python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --checkpoint-file ... --disable-microspine

Walk can also be driven by a local physical gamepad (left stick = linear x/y,
right stick = yaw, shoulder buttons cycle slow/medium/fast; medium is the
default):

  python scripts/play.py Mjlab-Walk-Phybot-C2 --checkpoint-file logs/.../model_XXXX.pt --gamepad
"""

from __future__ import annotations

import os
import subprocess
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
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from scr_phybot_mjab.utils.runtime import pin_climb_wall_angle, resolve_checkpoint


@dataclass(frozen=True)
class PlayConfig(MjlabPlayConfig):
  wall_angle_deg: float | None = None
  """Override the climb wall angle; otherwise resolve it from the checkpoint iteration."""
  disable_microspine: bool = False
  """Disable physical latch engagement for a climb-policy ablation rollout."""
  gamepad: bool = False
  """Drive the walk velocity command from a physical gamepad (left stick = linear x/y, right stick = yaw)."""
  gamepad_auto_bridge: bool = True
  """When --gamepad is enabled, also start scripts/deploy/gamepad_bridge.py as a local child process."""
  gamepad_host: str = "127.0.0.1"
  """Host the gamepad receiver binds to (use the SSH-forwarded loopback)."""
  gamepad_port: int = 8766
  """TCP port the gamepad receiver listens on; must match the bridge."""


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
  bool_flags = {"--disable-microspine", "--gamepad", "--gamepad-auto-bridge"}
  normalized: list[str] = []
  for idx, raw_arg in enumerate(argv):
    if raw_arg == "--no-gamepad-auto-bridge":
      normalized.extend(["--gamepad-auto-bridge", "False"])
      continue
    arg = aliases.get(raw_arg, raw_arg)
    normalized.append(arg)
    if arg in bool_flags:
      next_arg = argv[idx + 1] if idx + 1 < len(argv) else None
      if next_arg is None or next_arg.startswith("-"):
        normalized.append("True")
  return normalized


def _setup_gamepad(env, cfg: PlayConfig):
  """Start the gamepad receiver and route it into the velocity command term."""
  from scr_phybot_mjab.tasks.walk.mdp import UniformVelocityCommand
  from scr_phybot_mjab.utils.gamepad import GamepadReceiver

  command_manager = env.unwrapped.command_manager
  if "twist" not in command_manager.active_terms:
    raise RuntimeError("--gamepad requires a velocity ('twist') command; this task has none.")
  term = command_manager.get_term("twist")
  if not isinstance(term, UniformVelocityCommand):
    raise RuntimeError(f"--gamepad expects a UniformVelocityCommand 'twist' term, got {type(term).__name__}.")

  ranges = term.cfg.ranges
  receiver = GamepadReceiver(
    host=cfg.gamepad_host,
    port=cfg.gamepad_port,
    lin_vel_x_range=ranges.lin_vel_x,
    lin_vel_y_range=ranges.lin_vel_y,
    ang_vel_z_range=ranges.ang_vel_z,
  )
  receiver.start()
  term.set_gamepad_source(receiver.command)
  return receiver


def _start_gamepad_bridge(cfg: PlayConfig) -> subprocess.Popen | None:
  """Start the local gamepad bridge process for same-machine play."""
  script = Path(__file__).resolve().parent / "deploy" / "gamepad_bridge.py"
  cmd = [
    sys.executable,
    str(script),
    "--host",
    cfg.gamepad_host,
    "--port",
    str(cfg.gamepad_port),
  ]
  try:
    proc = subprocess.Popen(cmd)
  except OSError as exc:
    print(f"[WARN]: Could not start local gamepad bridge: {exc}")
    print(
      "[WARN]: Gamepad receiver is still listening; start "
      f"`{sys.executable} {script} --host {cfg.gamepad_host} --port {cfg.gamepad_port}` manually."
    )
    return None
  print(f"[INFO]: Started local gamepad bridge process pid={proc.pid}")
  return proc


def _stop_gamepad_bridge(proc: subprocess.Popen | None) -> None:
  if proc is None or proc.poll() is not None:
    return
  proc.terminate()
  try:
    proc.wait(timeout=2.0)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait(timeout=2.0)


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

  is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)

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
          art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if trained_mode:
    resume_path, source = resolve_checkpoint(
      agent_cfg.experiment_name, cfg.checkpoint_file, cfg.wandb_run_path, cfg.wandb_checkpoint_name
    )
    print(f"[INFO]: Loading checkpoint: {source}")
    log_dir = resume_path.parent

  pin_climb_wall_angle(
    env_cfg,
    wall_angle_deg=cfg.wall_angle_deg,
    checkpoint=resume_path,
    checkpoint_hint=cfg.wandb_checkpoint_name or cfg.checkpoint_file,
  )
  if cfg.disable_microspine:
    affordance_cfg = env_cfg.commands.get("microspine_affordance")
    if affordance_cfg is None:
      raise ValueError("--disable-microspine requires a microspine climb task.")
    affordance_cfg.microspine_cfg.grip_enabled = False
    event_cfg = env_cfg.events.get("microspine_grip")
    if event_cfg is not None:
      event_cfg.params["microspine_cfg"].grip_enabled = False
    print("[INFO]: Microspine grip disabled for ablation playback")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  if cfg.video and dummy_mode:
    print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
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
      str(resume_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  gamepad_rx = _setup_gamepad(env, cfg) if cfg.gamepad else None
  gamepad_bridge = _start_gamepad_bridge(cfg) if cfg.gamepad and cfg.gamepad_auto_bridge else None

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    _stop_gamepad_bridge(gamepad_bridge)
    if gamepad_rx is not None:
      gamepad_rx.close()

  env.close()


def main():
  import scr_phybot_mjab.tasks  # noqa: F401

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
