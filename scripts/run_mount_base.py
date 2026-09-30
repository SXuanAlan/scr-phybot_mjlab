"""Drive the scripted WBC+QP mount base in the play env; report the gate.

The B-line offline teacher probe: no policy — actions come from
``MountScriptedBase`` (event-gated acyclic clock, mixed anchor/floor contact
QP, kinematic WBC, force-controlled seat/release). Produces:

  <prefix>.png          — engage raster (scripted vs schedule, floor lanes)
                          + anchor λ tracking + press forces + com/root
  <prefix>_events.log   — per-event lines (engage/release/freeze/retry/...)
  <prefix>.mp4          — timestamped side video of the same rollout
  stdout                — P0-mount acceptance-gate summary

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/run_mount_base.py [steps] [out_prefix]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.tasks.climb_transition.mdp.mount_base import (
  MountBaseCfg,
  MountScriptedBase,
)

TASK = "Mjlab-Climb-Transition-Phybot-C2"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
PREFIX = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
  "logs/eval/mount_base/mount_base"
)
PREFIX.parent.mkdir(parents=True, exist_ok=True)

device = "cuda:0"
env_cfg = load_env_cfg(TASK, play=True)
# The scripted base is a diagnostic controller, not a policy under training:
# its event-gated clock legitimately freezes at transitions while the env's
# own reference clock keeps rolling, so the schedule-synchronized executioners
# would kill healthy runs. Falls still terminate via base_height/bad_state.
env_cfg.terminations.pop("bad_tracking", None)
env_cfg.terminations.pop("scheduled_contact_loss", None)
# The contact-law radius curriculum edits the popped termination's params.
env_cfg.curriculum.pop("contact_law_radius", None)
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

env.reset()
_cfg = MountBaseCfg()
if os.environ.get("MOUNT_PURE_REF"):
  # Trust the re-choreographed clip: its left_foot_mount base trajectory
  # already carries the wall press (swing_press profile) and the lateral
  # weight shift — the controller's hand-rolled single-stance corrections
  # double up on it and fight the choreography.
  _cfg.single_stance_crouch_m = 0.0
  _cfg.single_stance_wall_lean_m = 0.0
  _cfg.single_stance_com_mix = 0.0
  _cfg.hang_force_target_single = _cfg.hang_force_target
  print("[pure-ref] single-stance corrections OFF — following the clip")
if os.environ.get("MOUNT_SYSID"):
  # System-identification probe: kill the balance law, inject a constant
  # ankle bias (rad) from the env var.
  _cfg.ankle_balance_gain = 0.0
  _cfg.ankle_balance_dgain = 0.0
  _cfg.ankle_bias_const = float(os.environ["MOUNT_SYSID"])
  print(f"[sysid] balance law OFF, constant ankle bias {_cfg.ankle_bias_const}")
ctrl = MountScriptedBase(env, cfg=_cfg)
# The action vector is indexed by the env's action-term joint order; the
# controller builds it in PHYBOT_C2_POLICY_JOINT_NAMES order. Assert they
# are the same or every command is scrambled.
_act_cfg = env_cfg.actions["joint_pos"]
_act_joints = tuple(getattr(_act_cfg, "joint_names", ())) or None
if _act_joints is not None and tuple(_act_joints) != tuple(
  ctrl.policy_joint_names
):
  raise SystemExit(
    f"Action joint order mismatch:\n env: {_act_joints}\n "
    f"ctrl: {ctrl.policy_joint_names}"
  )
dt = ctrl.dt
names = ["LH", "RH", "LF", "RF"]
resets: list[int] = []

with torch.no_grad():
  for t in range(STEPS):
    action = ctrl.step_action()
    ret = env.step(action)
    terminated, truncated = ret[2], ret[3]
    done = bool(
      (terminated[0] if torch.is_tensor(terminated) else terminated)
      or (truncated[0] if torch.is_tensor(truncated) else truncated)
    )
    if done:
      resets.append(t)
      ctrl._event(None, "EPISODE_RESET", f"step={t}")
      ctrl.reset()
env.close()

lg = ctrl.log
t_arr = np.arange(len(lg["t"])) * dt
engaged = np.stack(lg["engaged"])  # (T, 4)
sched = np.stack(lg["sched"])
fog = np.stack(lg["fog_sched"])  # (T, 2)
lam_star = np.stack(lg["lambda_star_anchor"])  # (T, 4, 3)
lam_meas = np.stack(lg["lambda_meas"])
margin = np.stack(lg["margin"])
freeze = np.asarray(lg["freeze"], dtype=bool)
frames = np.asarray(lg["frame"])
press = np.stack(lg["press_force"])
frac = np.stack(lg["spine_fraction"])
qp_feasible = np.asarray(lg["qp_feasible"], dtype=bool)
com = np.stack(lg["com"])
com_tgt = np.stack(lg["com_target"])
root_z = np.asarray(lg["root_z"])

# ---------------------------------------------------------------- events ----
np.savez_compressed(
  Path(str(PREFIX) + "_log.npz"),
  t=t_arr,
  frames=frames,
  engaged=engaged,
  sched=sched,
  freeze=freeze,
  com=com,
  com_target=com_tgt,
  root_z=root_z,
  press=press,
  lam_star=lam_star,
  lam_meas=lam_meas,
  qp_feasible=qp_feasible,
  knee_err=np.stack([np.asarray(k) for k in lg["knee_err"]])
  if lg.get("knee_err")
  else np.zeros((0, 2)),
  pad_pos=np.stack([np.asarray(p) for p in lg["pad_pos"]])
  if lg.get("pad_pos")
  else np.zeros((0, 4, 3)),
  seat_delta=np.stack(lg["seat_delta"]),
  spine_fraction=np.stack(lg["spine_fraction"]),
  eff_load=np.stack(lg["eff_load"]),
  moment=np.stack(lg["moment"]),
  deltas=np.stack(lg["deltas"]),
  ankle_bias=np.asarray(lg["ankle_bias"], dtype=float),
  residual_tags=np.array(sorted({k for r in lg["residuals"] for k in r})),
  residuals=np.array(
    [
      [r.get(k, np.nan) for k in sorted({k for rr in lg["residuals"] for k in rr})]
      for r in lg["residuals"]
    ]
  ),
)
events_path = Path(str(PREFIX) + "_events.log")
with events_path.open("w") as fh:
  for ev in ctrl.events:
    fh.write(
      f"t={ev.t:7.2f}s f={ev.frame:4d} {ev.ee:<18s} "
      f"{ev.kind:<16s} {ev.detail}\n"
    )
print(f"wrote {events_path} ({len(ctrl.events)} events)")

# ------------------------------------------------------------- acceptance ---
handoff_frame = 1210  # handoff_hold start in the staggered clip
reached_handoff = bool(
  ((frames >= handoff_frame) & (engaged.sum(axis=1) == 4)).any()
)
all_four_ever = bool((engaged.sum(axis=1) == 4).any())
match = float((engaged == sched).mean())
hold_frac = float(freeze.mean())
final_frame = int(frames[-1]) if len(frames) else 0

kinds = [ev.kind for ev in ctrl.events]
n_free = kinds.count("release_free")
n_forced = kinds.count("release_forced")
n_engage = kinds.count("engage")
n_retry = kinds.count("seat_retry")
n_fail = kinds.count("seat_failed")

rising = engaged[1:] & ~engaged[:-1]
seat_qualities = frac[1:][rising]

# Press-force style: peak per unlatched-contact stretch.
press_active = press[(press > 1.0) & ~engaged]
press_p95 = float(np.percentile(press_active, 95)) if len(press_active) else 0.0
press_max = float(press_active.max()) if len(press_active) else 0.0

lam_err = np.linalg.norm(lam_meas - lam_star, axis=-1)
quiet = engaged & ~freeze[:, None] & qp_feasible[:, None]
per_pad_p90 = [
  float(np.percentile(lam_err[quiet[:, i], i], 90))
  if quiet[:, i].any()
  else float("nan")
  for i in range(4)
]

print("\n================ P0-mount acceptance (90° staggered) ================")
print(f"steps={len(t_arr)} ({len(t_arr) * dt:.1f}s)  final clip frame={final_frame}/1259")
print(f"reached handoff (4 pads @ frame>=1210): {reached_handoff}")
print(f"all four ever latched : {all_four_ever}")
print(f"falls (episode resets): {len(resets)}   (gate: 0)")
print(f"engages               : {n_engage} (retries {n_retry}, failures {n_fail})")
print(f"releases free/forced  : {n_free}/{n_forced}   (regrips should be free)")
print(
  "seat quality          : "
  + (
    f"min {seat_qualities.min():.2f} mean {seat_qualities.mean():.2f}"
    if len(seat_qualities)
    else "n/a"
  )
)
print(f"press force p95/max   : {press_p95:.1f}/{press_max:.1f} N  (gate: max<35)")
print(f"hold (frozen) fraction: {hold_frac:.2f}")
print(
  "anchor λ err p90 (N)  : "
  + ", ".join(f"{n}={v:.1f}" for n, v in zip(names, per_pad_p90))
)
print(f"schedule match        : {match:.3f}")
print(f"qp infeasible steps   : {int((~qp_feasible).sum())}")

# ---------------------------------------------------------------- figure ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
  4,
  1,
  figsize=(16, 13),
  sharex=True,
  constrained_layout=True,
  gridspec_kw={"height_ratios": [2.4, 3.0, 1.6, 1.4]},
)

ax = axes[0]
for i in range(4):
  e = sched[:, i].astype(int)
  d = np.diff(np.r_[0, e, 0])
  for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    ax.fill_between(
      [s0 * dt, e0 * dt], i - 0.38, i + 0.38, color="#3b6fd4", lw=0, alpha=0.35
    )
  e = engaged[:, i].astype(int)
  d = np.diff(np.r_[0, e, 0])
  for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    ax.fill_between(
      [s0 * dt, e0 * dt], i - 0.25, i + 0.25, color="#2b8a3e", lw=0, alpha=0.9
    )
for f_i in range(2):
  lane = 4 + f_i
  e = fog[:, f_i].astype(int)
  d = np.diff(np.r_[0, e, 0])
  for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    ax.fill_between(
      [s0 * dt, e0 * dt],
      lane - 0.3,
      lane + 0.3,
      color="#a9760b",
      lw=0,
      alpha=0.5,
    )
if freeze.any():
  ax.fill_between(
    t_arr, 5.6, -0.6, where=freeze, color="#adb5bd", alpha=0.25, lw=0,
    label="clock frozen",
  )
ax.set_yticks(range(6), names + ["gndL", "gndR"])
ax.set_ylim(5.6, -0.6)
ax.legend(loc="upper right", fontsize=8)
ax.set_title(
  "latches (green) vs schedule (blue); floor stance (amber); grey = frozen",
  loc="left",
  fontsize=11,
)

ax = axes[1]
colors = ["#1f77b4", "#e03131", "#2b8a3e", "#f59f00"]
for i in range(4):
  m = np.where(engaged[:, i], lam_meas[:, i, 0], np.nan)
  s = np.where(engaged[:, i], lam_star[:, i, 0], np.nan)
  ax.plot(t_arr, m, color=colors[i], lw=1.2, label=f"{names[i]} f_t meas")
  ax.plot(t_arr, s, color=colors[i], lw=1.0, ls="--", alpha=0.7)
ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
ax.set_ylabel("N")
ax.legend(loc="upper left", fontsize=7, ncol=4)
ax.set_title(
  "anchor rake load f_t: measured (solid) vs QP target (dashed)",
  loc="left",
  fontsize=11,
)

ax = axes[2]
for i in range(4):
  p = np.where(~engaged[:, i], press[:, i], np.nan)
  ax.plot(t_arr, p, color=colors[i], lw=1.0, label=f"{names[i]} press")
ax.axhline(35.0, color="#e03131", ls="--", lw=0.8, alpha=0.7)
ax.set_ylabel("press N")
ax.legend(loc="upper left", fontsize=7, ncol=4)

ax = axes[3]
ax.plot(t_arr, frames / 50.0, color="#adb5bd", lw=1.0, label="clip time (s)")
ax.plot(t_arr, com[:, 0] * 10 - 12, color="#1f77b4", lw=0.9, label="com x (offset)")
ax.plot(t_arr, root_z, color="#2b8a3e", lw=0.9, label="root z")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlabel("time (s)")

for r in resets:
  for a in axes:
    a.axvline(r * dt, color="k", ls="--", lw=0.9)
fig.suptitle(
  f"scripted mount base @90° — frame {final_frame}/1259, "
  f"handoff={reached_handoff}, falls={len(resets)}, "
  f"press p95 {press_p95:.0f}N, hold {hold_frac:.2f}",
  fontsize=12,
)
png_path = PREFIX.with_suffix(".png")
fig.savefig(png_path, dpi=130)
print(f"wrote {png_path}")

# ---------------------------------------------------------------- video -----
vids = sorted(video_dir.glob("*.mp4"))
if vids:
  import shutil
  import subprocess

  mp4_path = PREFIX.with_suffix(".mp4")
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
    print(f"timestamp burn failed ({burn.stderr.strip()[:200]}); copying raw")
    shutil.copy(vids[0], mp4_path)
  print(f"wrote {mp4_path}")
  shutil.rmtree(video_dir, ignore_errors=True)
else:
  print(f"WARNING: no video produced under {video_dir}")
