"""Record what the transition80 policy does when BORN inside one clip window.

Every reset lands uniformly inside the named segments (the exam path of the
motion command's bias window); the curriculum clock is pinned to the
checkpoint's iteration so the ignition ladder, the rip bill and the gimbal
travel are the ones the policy trained under. One env, tracking camera,
deterministic policy; several episodes per video (the robot dies and is
re-born in the window). A fate line per episode goes to stdout.

Usage:
  python scripts/transition80_window_video.py <ckpt.pt> segments=hand_place[,..] \\
      [steps=400] [out=logs/v5_windows/name] [iter=<auto from ckpt name>]
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

import scr_phybot_mjab.tasks  # noqa: F401

_argv = [a for a in sys.argv[1:] if "=" not in a]
_tok = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
CKPT = Path(_argv[0])
SEGMENTS = tuple(_tok.get("segments", "hand_place").split(","))
STEPS = int(_tok.get("steps", "400"))
OUT = Path(_tok.get("out", f"logs/v5_windows/{'+'.join(SEGMENTS)}"))
_digits = "".join(ch for ch in CKPT.stem if ch.isdigit())
ITER = int(_tok.get("iter", _digits or "0"))
TASK = "Mjlab-Climb-Transition80-Phybot-C2"
STEPS_PER_ITER = 96
device = "cuda:0"

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = 1
motion_cfg = cfg.commands["climb_motion"]
motion_cfg.sampling_mode = "uniform"
motion_cfg.spawn_early_fraction = 1.0
motion_cfg.spawn_bias_segments = SEGMENTS
motion_cfg.spawn_bias_frame_windows = ()
motion_cfg.spawn_bias_auto_rake_frames = 0
cfg.viewer = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_ROOT,
  entity_name="robot",
  distance=3.0,
  elevation=-12.0,
  azimuth=90.0,
  height=720,
  width=1280,
)
agent_cfg = load_rl_cfg(TASK)
OUT.parent.mkdir(parents=True, exist_ok=True)
video_dir = OUT.parent / f".{OUT.stem}_videotmp"
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(
  env,
  video_folder=video_dir,
  step_trigger=lambda step: step == 0,
  video_length=STEPS,
  disable_logger=True,
)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(
  wrapped, asdict(agent_cfg), str(OUT.parent / ".window_video_runner"), device
)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

u = env.unwrapped
# Pin the curriculum clock to the checkpoint's iteration (ladders, bill,
# gimbal travel) before the first reset.
u.common_step_counter = ITER * STEPS_PER_ITER
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
term = u.termination_manager
seg_d = np.load(motion_cfg.motion_file, allow_pickle=True).item()
names = list(seg_d["segment_names"])
labels = np.asarray(seg_d["phase_segment"])


def seg_of(f: int) -> str:
  return names[labels[min(int(f), len(labels) - 1)]]


print(f"[window video] {CKPT.name} iter {ITER} segments {SEGMENTS} bias window {motion._bias_window}")
env.reset()
obs = wrapped.get_observations()
birth = int(motion._birth_frames[0])
seeds0 = aff.engaged[0].clone()
ep_len = 0
ep_id = 0
rips = 0
for step in range(STEPS):
  with torch.no_grad():
    act = policy(obs)
  obs, _, dones, infos = wrapped.step(act)
  ep_len += 1
  rips += int(aff.forced_release_this_step[0].sum())
  if bool(dones[0]):
    cause = [n for n in term.active_terms if bool(term.get_term(n)[0])]
    eng = aff.engaged[0]
    print(
      f"  ep{ep_id}: born f{birth} ({seg_of(birth)}) seeds {int(seeds0.sum())} -> "
      f"len {ep_len} steps ({ep_len * float(u.step_dt):.2f} s), rips {rips}, "
      f"engaged at end {int(eng.sum())}, cause {cause}"
    )
    ep_id += 1
    ep_len = 0
    rips = 0
    birth = int(motion._birth_frames[0])
    seeds0 = aff.engaged[0].clone()
if ep_len > 0:
  print(f"  ep{ep_id}: born f{birth} ({seg_of(birth)}) still alive after {ep_len} steps, rips {rips}")
env.close()
vids = sorted(video_dir.glob("*.mp4"))
if vids:
  mp4 = OUT.with_suffix(".mp4")
  shutil.copy(vids[-1], mp4)
  shutil.rmtree(video_dir, ignore_errors=True)
  print(f"wrote {mp4}")
