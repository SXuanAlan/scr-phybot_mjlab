"""Record a rollout that SPAWNS INSIDE the biased RSI window.

The standard raster video plays the mount from the standing start — it
shows the from-start chain, not what the policy does where it is actually
training. This probe forces every reset into the bias window (default =
the reverse curriculum's stage 0, the place -> set latch stretch: arms on
the anchors, one press away) and records the tracking-camera video, so
"pressed but never latched" vs "latched but ripped out" is visible.

Usage:
  python scripts/mount_window_probe.py <ckpt.pt> [steps] [out_prefix] \\
      [task=Mjlab-Climb-Transition-Phybot-C2] [segments=hand_place,hand_set_hold]
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import scr_phybot_mjab.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

_argv = [a for a in sys.argv[1:] if "=" not in a]
_tokens = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
CKPT = Path(_argv[0])
STEPS = int(_argv[1]) if len(_argv) > 1 else 600
PREFIX = Path(
  _argv[2] if len(_argv) > 2 else str(CKPT.with_suffix("")) + "_window"
)
TASK = _tokens.get("task", "Mjlab-Climb-Transition-Phybot-C2")
SEGMENTS = tuple(
  _tokens.get("segments", "hand_place,hand_set_hold").split(",")
)
# tiltband=lo,hi pins the tilt-peel band to the TRAINING physics (play
# pins the calibrated band; a probe of an annealing run must match its
# current curriculum stage to measure the physics the policy trains in).
TILTBAND = (
  tuple(float(v) for v in _tokens["tiltband"].split(","))
  if "tiltband" in _tokens
  else None
)
# engagegate=<deg> pins the engage-quality gate to the training value
# (play runs the stock 0.25-cos acceptance).
ENGAGE_GATE_DEG = (
  float(_tokens["engagegate"]) if "engagegate" in _tokens else None
)
# holdstill: replace the policy with "freeze at the spawned pose" (the
# action that PD-holds the RSI joint pose). Decisive control experiment:
# if seeded anchors rip even under perfect stillness, the rips are a
# spawn-transient physics artifact, not policy violence.
HOLD_STILL = "holdstill" in _tokens or "holdstill" in _argv
RL_TASK = "Mjlab-Climb-Transition-Phybot-C2"
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
motion_cfg = env_cfg.commands["climb_motion"]
# Every reset lands uniformly inside the bias window; play keeps zero
# scatter and zero joint noise, so what varies between resets is only the
# window frame.
motion_cfg.sampling_mode = "uniform"
motion_cfg.spawn_early_fraction = 1.0
motion_cfg.spawn_bias_segments = SEGMENTS
# Exams grade at the CURRICULUM's gravity (user call, 08-19 evening);
# `gravity=<scale>` overrides (the graduation exam pins 1.0).
from scr_phybot_mjab.tasks.climb_transition.mdp.curriculums import (
  gravity_scale_at,
)

_glad = env_cfg.curriculum.pop("gravity_ladder", None)
if "gravity" in _tokens:
  _gscale = float(_tokens["gravity"])
elif _glad is not None:
  _digits = "".join(ch for ch in CKPT.stem if ch.isdigit())
  _gscale = (
    gravity_scale_at(_glad.params["stages"], int(_digits)) if _digits else 1.0
  )
else:
  _gscale = 1.0
if _gscale != 1.0:
  env_cfg.sim.mujoco.gravity = tuple(
    c * _gscale for c in env_cfg.sim.mujoco.gravity
  )
print(f"[exam] gravity scale = {_gscale:g}")
agent_cfg = load_rl_cfg(RL_TASK)

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
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = MjlabOnPolicyRunner(
  wrapped, asdict(agent_cfg), str(PREFIX.parent / ".window_probe_runner"), device
)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
# Crutch telltales (the v15/v16 lesson: lifespan lies, contact geometry
# doesn't): min head/torso SURFACE distance to the wall over the rollout
# (geom center minus wall-ward half-extent — link origins under-read by
# up to ~13 cm, the v17 telltale bug).
# Full-body wall-distance table (v20 lesson: the head/torso-only telltale
# certified "no crutch" while the SHOULDER sat inside the wall plane —
# every robot geom is watched now, so nothing hides in a geom list's null
# space). Only pad spheres and toes physically collide with the wall; for
# any other geom a NEGATIVE surface distance is visual penetration (zero
# force, impossible on hardware) — a deploy red flag on its own.
_mj = u.sim.mj_model
_geom_ids = [
  gid for gid in range(_mj.ngeom) if _mj.geom(gid).name.startswith("robot/")
]
_geom_names = [_mj.geom(g).name for g in _geom_ids]
_geom_half = torch.tensor(
  [float(_mj.geom_size[g, 0]) for g in _geom_ids],
  device=aff.engaged.device,
)
_min_geom_dist = torch.full(
  (len(_geom_ids),), float("inf"), device=aff.engaged.device
)
_FENCE_GEOMS = ("robot/neck_pitch_collision", "robot/torso_collision")
_toe_sensor = u.scene["climb_toe_contact"]
_max_toe: torch.Tensor | None = None
_pad_max = torch.zeros(4, device=aff.engaged.device)


def _geom_dists() -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply

  pos = u.sim.data.geom_xpos[:, _geom_ids, :]
  n_local = torch.zeros_like(pos[:, 0])
  n_local[:, 0] = -1.0
  n_w = quat_apply(motion.wall_quat, n_local)
  rel = pos - motion.wall_pivot_w.unsqueeze(1)
  return ((rel * n_w.unsqueeze(1)).sum(dim=-1) - _geom_half)[0]
if TILTBAND is not None:
  aff._ls_cfg.tilt_peel_max_deg_range = TILTBAND
  print(f"[probe] tilt-peel band pinned to {TILTBAND}")
if ENGAGE_GATE_DEG is not None:
  import math as _math

  aff._mcfg.engage_parallel_cos_threshold = _math.cos(
    _math.radians(ENGAGE_GATE_DEG)
  )
  print(f"[probe] engage gate pinned to {ENGAGE_GATE_DEG} deg")

robot = u.scene["robot"]
_act_term = u.action_manager.get_term("joint_pos")


def _hold_action() -> torch.Tensor:
  """Action that PD-holds the CURRENT joint pose (freeze in place).

  SoftJointPositionAction: target = offset + action * scale over
  target_ids — inverted here for the spawned pose.
  """
  q = robot.data.joint_pos[:, _act_term.target_ids]
  return (q - _act_term.offset) / _act_term.scale


obs = wrapped.get_observations()
hold_act = _hold_action() if HOLD_STILL else None
if HOLD_STILL:
  print("[probe] HOLD-STILL control: action freezes the spawned pose")
resets = 0
spawn_frame = int(motion.time_steps[0].item())
latches = 0
prev_engaged = aff.engaged.clone()
# Forced-release reason histogram (1=reverse shear, 2=tension, 3=lateral,
# 4=ultimate, 5=tilt-peel) — the channel that rips anchors decides which
# knob fixes it.
reason_names = {
  1: "reverse-shear",
  2: "tension",
  3: "lateral",
  4: "ultimate",
  5: "tilt-peel",
}
reason_hist = {name: 0 for name in reason_names.values()}
birth_step: dict[int, int] = {}
lifetimes: list[int] = []
hand_press: list[torch.Tensor] = []  # (2,) LH/RH pad press per step [N]
press_at_latch: list[float] = []
# Window-level engage raster traces (user call, 08-19 evening: the
# schedule-vs-actual picture must exist for SMALL windows too, in the
# SAME visual language as the fullrun raster). Rendered to
# <PREFIX>_schedule.png after rollout.
sched_trace: list[torch.Tensor] = []
eng_trace: list[torch.Tensor] = []
margin_trace: list[torch.Tensor] = []
margin_raw_trace: list[torch.Tensor] = []
forced_trace: list[torch.Tensor] = []
reason_trace: list[torch.Tensor] = []
reset_steps: list[int] = []
print(f"[probe] window segments {SEGMENTS}, spawn frame {spawn_frame}")
for step in range(STEPS):
  with torch.no_grad():
    act = hold_act if HOLD_STILL else policy(obs)
  obs, _, dones, _ = wrapped.step(act)
  if HOLD_STILL and bool(dones.any()):
    hold_act = _hold_action()  # re-freeze at the new spawn pose
  now = aff.engaged.clone()
  sched_trace.append(motion.ref_engaged[0].detach().cpu().clone())
  eng_trace.append(now[0].detach().cpu().clone())
  margin_trace.append(aff.hold_margin[0].float().detach().cpu().clone())
  margin_raw_trace.append(
    aff.hold_margin_raw[0].float().detach().cpu().clone()
  )
  forced_trace.append(
    aff.forced_release_this_step[0].bool().detach().cpu().clone()
  )
  reason_trace.append(aff.release_reason[0].detach().cpu().clone())
  _min_geom_dist = torch.minimum(_min_geom_dist, _geom_dists())
  _pad_max = torch.maximum(_pad_max, aff.pad_press_force[0])
  _tf = _toe_sensor.data.force[0, :, 0].clamp(min=0.0)
  _max_toe = _tf if _max_toe is None else torch.maximum(_max_toe, _tf)
  hand_press.append(aff.pad_press_force[0, :2].detach().cpu().clone())
  latches += int((now & ~prev_engaged).sum().item())
  for ee in (now & ~prev_engaged)[0].nonzero().flatten().tolist():
    birth_step[ee] = step
    if ee < 2:
      press_at_latch.append(float(aff.pad_press_force[0, ee].item()))
  released = (prev_engaged & ~now)[0]
  if bool(released.any()):
    for ee in released.nonzero().flatten().tolist():
      if ee in birth_step:
        lifetimes.append(step - birth_step.pop(ee))
      rid = int(aff.release_reason[0][ee].item())
      if rid in reason_names:
        reason_hist[reason_names[rid]] += 1
  prev_engaged = now
  if bool(dones[0]):
    resets += 1
    reset_steps.append(step)
    print(
      f"[probe] t={step / 50.0:6.2f}s reset #{resets} "
      f"(spawned at frame {spawn_frame}, latch events so far {latches})"
    )
    spawn_frame = int(motion.time_steps[0].item())
env.close()

# Window engage raster — the fullrun raster's visual language verbatim
# (user call: the small-window plot must read like engage_raster's png):
# top panel = policy latch bars, two lanes per EE (force-only margin over
# billed margin, RdYlGn), release markers by reason; bottom panel = the
# reference clip's engage schedule in blue; resets as black dashes.
if sched_trace:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  from matplotlib import cm
  from matplotlib.colors import Normalize
  from matplotlib.lines import Line2D

  dt = 1.0 / 50.0
  ref_m = torch.stack(sched_trace).numpy().astype(bool)  # (T, 4)
  pol_m = torch.stack(eng_trace).numpy().astype(bool)
  margin_m = torch.stack(margin_trace).numpy()
  margin_raw_m = torch.stack(margin_raw_trace).numpy()
  forced_m = torch.stack(forced_trace).numpy().astype(bool)
  reason_m = torch.stack(reason_trace).numpy()
  t_total = ref_m.shape[0]
  names = ["LH", "RH", "LF", "RF"]
  fig, axes = plt.subplots(
    2, 1, figsize=(16, 6.5), sharex=True, constrained_layout=True
  )
  margin_cmap = plt.get_cmap("RdYlGn")
  margin_norm = Normalize(vmin=0.0, vmax=1.0)
  for i in range(4):
    for values, y_hi, y_lo in (
      (margin_raw_m[:, i], i - 0.38, i - 0.02),
      (margin_m[:, i], i + 0.02, i + 0.38),
    ):
      rgba = margin_cmap(margin_norm(np.clip(values, 0.0, 1.0)))
      rgba[:, 3] = np.where(pol_m[:, i], 0.95, 0.0)
      axes[0].imshow(
        rgba[None, :, :],
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, t_total * dt, y_lo, y_hi),
        zorder=1,
      )
  axes[0].set_title(
    f"policy  window={','.join(SEGMENTS)}  {CKPT.name}   (top lane = "
    "force-only margin, bottom lane = billed margin; ▾ = forced release)",
    loc="left",
    fontsize=10,
  )
  reason_style = {
    1: ("#4c9be8", "reverse-shear (free unhook)"),
    2: ("#e03131", "tension pull-off"),
    3: ("#f59f00", "lateral peel"),
    4: ("#9c36b5", "ultimate overload"),
    5: ("#0ca678", "tilt-peel (rotated off grip)"),
  }
  for rid, (color, _label) in reason_style.items():
    for i in range(4):
      ts = np.where(forced_m[:, i] & (reason_m[:, i] == rid))[0]
      if len(ts):
        axes[0].plot(
          ts * dt, np.full(len(ts), i - 0.45), "v",
          color=color, ms=4, ls="none",
        )
  # Moment-billed releases (force-only margin still positive at the kill
  # step) — engage_raster's diamond marker, same derivation.
  moment_caused = forced_m & (margin_raw_m > 0.0) & (reason_m < 5)
  for i in range(4):
    ts = np.where(moment_caused[:, i])[0]
    if len(ts):
      axes[0].plot(
        ts * dt, np.full(len(ts), i - 0.45), "D",
        color="#212529", ms=4.5, ls="none",
        markerfacecolor="none", markeredgewidth=1.4,
      )
  legend_handles = [
    Line2D(
      [], [], marker="v", color=color, ls="none", ms=5,
      label=f"{label}  ×{int((forced_m & (reason_m == rid)).sum())}",
    )
    for rid, (color, label) in reason_style.items()
    if int((forced_m & (reason_m == rid)).sum())
  ]
  if int(moment_caused.sum()):
    legend_handles.append(
      Line2D(
        [], [], marker="D", color="#212529", ls="none", ms=5,
        markerfacecolor="none", markeredgewidth=1.4,
        label=f"moment-billed  ×{int(moment_caused.sum())}",
      )
    )
  if legend_handles:
    axes[0].legend(
      handles=legend_handles, loc="upper right", fontsize=8, ncol=2,
      framealpha=0.9,
    )
  for i in range(4):
    e = ref_m[:, i].astype(int)
    d = np.diff(np.r_[0, e, 0])
    for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
      axes[1].fill_between(
        [s0 * dt, e0 * dt], i - 0.38, i + 0.38,
        color="#3b6fd4", lw=0, alpha=0.9,
      )
  axes[1].set_title(
    "reference clip engage schedule (window RSI, resets = black dashes)",
    loc="left",
    fontsize=11,
  )
  for ax in axes:
    ax.set_yticks(range(4), names)
    ax.set_ylim(3.6, -0.6)
    ax.set_xlim(0.0, t_total * dt)
    ax.grid(axis="x", alpha=0.3)
  fig.colorbar(
    cm.ScalarMappable(norm=margin_norm, cmap=margin_cmap),
    ax=list(axes),
    pad=0.01,
    label="hold margin (1 unloaded → 0 release boundary)",
  )
  for r in reset_steps:
    for ax in axes:
      ax.axvline(r * dt, color="k", ls="--", lw=0.9)
  sched_png = str(PREFIX) + "_schedule.png"
  fig.savefig(sched_png, dpi=130)
  print(f"wrote {sched_png}")
print(f"[probe] release reasons: {reason_hist}")
_head_torso = min(
  _min_geom_dist[_geom_names.index(n)].item() for n in _FENCE_GEOMS
)
print(f"[probe] min head/torso wall distance: {_head_torso * 100:.1f} cm")
print("[probe] closest geoms to the wall surface (min over rollout):")
for _k in torch.argsort(_min_geom_dist)[:12].tolist():
  _d = _min_geom_dist[_k].item()
  _flag = "  <-- PENETRATING (link has no wall collision)" if _d < 0 else ""
  print(f"    {_geom_names[_k]:48s} {_d * 100:+7.2f} cm{_flag}")
print(
  "[probe] max pad press LH/RH/LF/RF [N]: "
  f"{[round(v, 2) for v in _pad_max.tolist()]}"
)
if _max_toe is not None:
  print(
    "[probe] max toe-wall normal force [N]: "
    f"{[round(v, 2) for v in _max_toe.tolist()]}"
  )
if lifetimes:
  mean_life = sum(lifetimes) / len(lifetimes)
  print(
    f"[probe] latch lifetimes: n={len(lifetimes)} "
    f"mean={mean_life:.1f} steps ({mean_life / 50.0:.2f}s) "
    f"max={max(lifetimes)}"
  )
if hand_press:
  press = torch.stack(hand_press)  # (T, 2)
  active = press[press > 0.5]  # contact steps only
  if active.numel():
    q = torch.quantile(active, torch.tensor([0.5, 0.95]))
    print(
      f"[probe] hand press while touching: n={active.numel()} steps, "
      f"median={q[0]:.1f} N p95={q[1]:.1f} N max={active.max():.1f} N"
    )
  else:
    print("[probe] hand press: no contact steps above 0.5 N")
  if press_at_latch:
    print(
      "[probe] press at latch moments [N]: "
      f"{[round(v, 1) for v in press_at_latch]}"
    )

vids = sorted(video_dir.glob("*.mp4"))
if vids:
  mp4 = PREFIX.with_suffix(".mp4")
  shutil.copy(vids[-1], mp4)
  shutil.rmtree(video_dir, ignore_errors=True)
  print(f"wrote {mp4}")
print(f"[probe] {resets} resets, {latches} latch rising edges in {STEPS} steps")
