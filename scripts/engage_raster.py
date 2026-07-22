"""Roll out a climb checkpoint; plot the microspine engage raster + video.

One deterministic rollout produces, exactly aligned:
  <out_prefix>.png  — engage raster: policy latches (green) vs the reference
                      clip schedule (blue), forced releases as red ticks,
                      episode resets as dashed lines.
  <out_prefix>.mp4  — 720p side tracking-camera video of the SAME rollout
                      (50 fps sim time, timestamp burned in so video seconds
                      map 1:1 onto the raster time axis).

Also prints occupancy / match / release stats and the limit-surface release
reason histogram (1=reverse shear [free by design], 2=tension, 3=lateral,
4=ultimate).

Usage:
  python scripts/engage_raster.py <ckpt.pt> [angle_deg] [steps] [out_prefix]
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

TASK = "Mjlab-Climb-Wall-Phybot-C2"
CKPT = Path(sys.argv[1]).resolve()
ANGLE = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
PREFIX = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("engage_raster")
REASON_NAMES = {1: "reverse-shear(free)", 2: "tension", 3: "lateral", 4: "ultimate"}

device = "cuda:0"
env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
# Pin the play gravity stage to the requested angle, whichever curriculum
# flavor the config carries.
if "climb_difficulty" in env_cfg.curriculum:
  env_cfg.curriculum["climb_difficulty"].params["stages"] = [
    {"angle_deg": ANGLE, "min_height_gain": 0.18}
  ]
else:
  env_cfg.curriculum["gravity_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": ANGLE}
  ]
env_cfg.viewer = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_ROOT,
  entity_name="robot",
  distance=3.0,
  elevation=-12.0,
  azimuth=90.0,
  height=720,
  width=1280,
)

video_dir = PREFIX.parent / f".{PREFIX.stem}_videotmp"
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(
  env,
  video_folder=video_dir,
  step_trigger=lambda step: step == 0,
  video_length=STEPS,
  disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
dt = float(u.step_dt)

ret = env.get_observations()
obs = ret[0] if isinstance(ret, (tuple, list)) else ret

pol = np.zeros((STEPS, 4), dtype=bool)
ref = np.zeros((STEPS, 4), dtype=bool)
forced = np.zeros((STEPS, 4), dtype=bool)
reason = np.zeros((STEPS, 4), dtype=np.int8)
gain = np.zeros(STEPS, dtype=np.float32)
resets = []
with torch.no_grad():
  for t in range(STEPS):
    step_ret = env.step(policy(obs))
    obs, dones = step_ret[0], step_ret[2]
    pol[t] = motion.robot_engaged[0].bool().cpu().numpy()
    ref[t] = motion.ref_engaged[0].bool().cpu().numpy()
    forced[t] = aff.forced_release_this_step[0].bool().cpu().numpy()
    reason[t] = aff.release_reason[0].cpu().numpy()
    gain[t] = float(motion.height_gained[0].item())
    if bool(dones[0]):
      resets.append(t)
env.close()

names = ["LH", "RH", "LF", "RF"]
match = float((pol == ref).mean())
occ_p, occ_r = pol.mean(0), ref.mean(0)


def _events(mask_col):
  d = np.diff(np.r_[0, mask_col.astype(int)])
  return np.where(d == -1)[0]


print(f"steps={STEPS} ({STEPS * dt:.1f}s) angle={ANGLE:.0f} ckpt={CKPT.name}")
print(f"match(step x EE agreement)={match:.3f}")
rel_tot, forc_tot = 0, 0
for i, n in enumerate(names):
  rel = len(_events(pol[:, i]))
  fr = int(forced[:, i].sum())
  rel_tot += rel
  forc_tot += fr
  print(
    f"  {n}: occ pol={occ_p[i]:.2f} ref={occ_r[i]:.2f} | releases={rel} "
    f"(forced={fr}) | ref releases={len(_events(ref[:, i]))}"
  )
print(f"total releases={rel_tot} forced={forc_tot}")
hist = {
  REASON_NAMES[rid]: int((forced & (reason == rid)).sum()) for rid in (1, 2, 3, 4)
}
print(f"forced-release reasons: {hist}")
print(
  f"height gained (last)={gain[-1]:.3f} m; "
  f"resets at {[round(r * dt, 1) for r in resets]} s"
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True, constrained_layout=True)
panels = [
  (axes[0], pol, f"policy  {CKPT.name} @ {ANGLE:.0f}°   (▾ = forced release, colored by type)", "#2f9e44"),
  (axes[1], ref, "reference clip engage schedule (phase-aligned)", "#3b6fd4"),
]
for ax, data, title, color in panels:
  for i in range(4):
    e = data[:, i].astype(int)
    d = np.diff(np.r_[0, e, 0])
    for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
      ax.fill_between([s0 * dt, e0 * dt], i - 0.38, i + 0.38, color=color, lw=0, alpha=0.9)
  ax.set_yticks(range(4), names)
  ax.set_ylim(-0.6, 3.6)
  ax.invert_yaxis()
  ax.set_title(title, loc="left", fontsize=11)
  ax.grid(axis="x", alpha=0.3)
# Forced releases split by limit-surface failure type. Reverse shear is the
# pad's intended unhook direction (free while supported); the warm colors are
# genuine overload failures.
REASON_STYLE = {
  1: ("#4c9be8", "reverse-shear (free unhook)"),
  2: ("#e03131", "tension pull-off"),
  3: ("#f59f00", "lateral peel"),
  4: ("#9c36b5", "ultimate overload"),
}
for rid, (color, _label) in REASON_STYLE.items():
  for i in range(4):
    ts = np.where(forced[:, i] & (reason[:, i] == rid))[0]
    if len(ts):
      axes[0].plot(
        ts * dt, np.full(len(ts), i - 0.45), "v", color=color, ms=4, ls="none"
      )
from matplotlib.lines import Line2D

legend_handles = [
  Line2D(
    [], [], marker="v", color=color, ls="none", ms=5,
    label=f"{label}  ×{int((forced & (reason == rid)).sum())}",
  )
  for rid, (color, label) in REASON_STYLE.items()
]
axes[0].legend(
  handles=legend_handles, loc="upper right", fontsize=8, ncol=2, framealpha=0.9
)
for r in resets:
  for ax in axes:
    ax.axvline(r * dt, color="k", ls="--", lw=0.9)
axes[1].set_xlabel("time (s)")
fig.suptitle(
  f"microspine engage raster — match {match:.2f} | occupancy pol "
  f"[{', '.join(f'{v:.2f}' for v in occ_p)}] vs ref [{', '.join(f'{v:.2f}' for v in occ_r)}]"
  f" | height +{gain[-1]:.2f} m",
  fontsize=12,
)
png_path = PREFIX.with_suffix(".png")
fig.savefig(png_path, dpi=130)
print(f"wrote {png_path}")

vids = sorted(video_dir.glob("*.mp4"))
if vids:
  import shutil
  import subprocess

  mp4_path = PREFIX.with_suffix(".mp4")
  # Burn a sim-time stamp so video seconds map 1:1 onto the raster x-axis.
  burn = subprocess.run(
    [
      "ffmpeg", "-v", "error", "-y", "-i", str(vids[0]),
      "-vf",
      "drawtext=text='t=%{eif\\:t\\:d}.%{eif\\:mod(t*100,100)\\:d\\:2}s':"
      "x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5",
      "-an", str(mp4_path),
    ],
    capture_output=True,
    text=True,
  )
  if burn.returncode != 0:
    print(f"timestamp burn failed ({burn.stderr.strip()[:200]}); copying raw video")
    shutil.copy(vids[0], mp4_path)
  print(f"wrote {mp4_path}")
  shutil.rmtree(video_dir, ignore_errors=True)
else:
  print(f"WARNING: no video produced under {video_dir}")
