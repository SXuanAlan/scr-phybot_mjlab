"""Passive-hold gate for the CLIMB CYCLE reference (Mjlab-Climb-Wall-Phybot-C2):
births pinned to every --stride-th seeded frame of the cycle clip, hold the
birth pose with the deploy PD, judge scheduled-seed loss after the guard.
Same rule as scripts/mount_clip_passive_gate.py, so the two references can
be compared. Frames are pinned by intercepting the command's uniform draw.
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/climb_cycle_passive_gate.py [--angle 90] [--stride 10] [--guard 25] [--steps 40] [--envs 128]
"""

import argparse
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import mjlab  # noqa
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import scr_phybot_mjab.tasks  # noqa
import scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants as K
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import pin_curricula_to_final_stage

ap = argparse.ArgumentParser()
ap.add_argument("--angle", type=float, default=90.0)
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--guard", type=int, default=25)
ap.add_argument("--envs", type=int, default=128)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--fail-frac", type=float, default=0.05)
args = ap.parse_args()
device = "cuda:0"
cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2", play=False)
cfg.scene.num_envs = args.envs
pin_curricula_to_final_stage(cfg)
mc = cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"
mc.height_offset_range = (0.0, 0.0)
mc.joint_pos_noise = 0.0
cfg.commands["microspine_affordance"].microspine_cfg.limit_surface.seed_hold_judge_guard_steps = args.guard
for name in ("encoder_bias", "body_friction", "base_mass"):
  cfg.events.pop(name, None)
# wall angle: the final-stage curriculum pins it; override if a different angle is wanted
if "wall_angle" in cfg.curriculum:
  st = cfg.curriculum["wall_angle"].params.get("angle_stages") or cfg.curriculum["wall_angle"].params.get("stages")
  if st:
    for s_ in st:
      for k in ("angle_deg", "value", "angle"):
        if k in s_:
          s_[k] = args.angle
      if "jitter_deg" in s_:
        s_["jitter_deg"] = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
act = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
term = u.termination_manager
C = int(motion.clip.cycle_frames)
N = args.envs
clip = np.load(mc.motion_file, allow_pickle=True).item()
engaged = np.asarray(clip["ee_engaged"]).astype(bool)
names = list(clip["segment_names"])
labels = np.asarray(clip["phase_segment"])


def seg(f):
  return names[labels[min(int(f), len(labels) - 1)]]


EE = ["LH", "RH", "LF", "RF"]
EEJ = ["left_hand", "right_hand", "left_foot", "right_foot"]
REASON = {
  0: "none",
  1: "rev-shear",
  2: "tension",
  3: "lateral",
  4: "ultimate",
  6: "dr",
  7: "emp-domain",
  8: "emp-early",
}
tnames = list(term.active_terms)
gim_ids = [i for i, n in enumerate(robot.joint_names) if "gimbal" in n]
gim_ee = [[k for k, i in enumerate(gim_ids) if EEJ[e] in robot.joint_names[i]] for e in range(4)]
_orig_randint = torch.randint
PIN = {"frame": 0}


def _pinned_randint(*a, **kw):
  size = kw.get("size", a[2] if len(a) > 2 else None)
  high = a[1] if len(a) > 1 else kw.get("high")
  if high == C and size == (N,):
    return torch.full((N,), PIN["frame"], dtype=torch.long, device=kw.get("device", device))
  return _orig_randint(*a, **kw)


seeded = [f for f in range(0, C, args.stride) if engaged[f].any()]
wa = (
  float(math.degrees(float(torch.atan2(motion.wall_quat[0, 2], motion.wall_quat[0, 0]))) * 2)
  if hasattr(motion, "wall_quat")
  else float("nan")
)
print(
  f"=== climb cycle passive gate: {Path(mc.motion_file).name} cycle {C} frames, angle arg {args.angle}, stride {args.stride}, guard {args.guard}, steps {args.steps}, envs {N}, gimbal stop +-{math.degrees(float(K.PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD)):.1f} deg"
)
print("frame segment            sched | loss% (post-guard, worst EE) per-EE | gimbal p90 deg | deaths | verdict")
rows = []
for f in seeded:
  PIN["frame"] = f
  torch.randint = _pinned_randint
  try:
    env.reset()
  finally:
    torch.randint = _orig_randint
  birth = motion.time_steps.clone()
  if int((birth - f).abs().max()) > 1:
    print(f"{f:5d} pin failed {birth[:3].tolist()}")
    continue
  seed = motion.clip.ee_engaged[birth].bool()
  ids = act._target_ids
  hold = (robot.data.joint_pos[:, ids] - act._offset) / act._scale
  alive = torch.ones(N, dtype=torch.bool, device=device)
  first_rel = torch.full((N, 4), -1, dtype=torch.long, device=device)
  reason = torch.zeros((N, 4), dtype=torch.long, device=device)
  death = torch.full((N,), -1, dtype=torch.long, device=device)
  gim_max = torch.zeros(N, 4, device=device)
  for step in range(args.steps):
    _, _, terminated, truncated, _ = env.step(hold)
    g = robot.data.joint_pos[:, gim_ids].abs()
    for e in range(4):
      if gim_ee[e]:
        gim_max[:, e] = torch.where(
          alive, torch.maximum(gim_max[:, e], g[:, gim_ee[e]].max(dim=1).values), gim_max[:, e]
        )
    rel = aff.forced_release_this_step & seed & alive.unsqueeze(1) & (first_rel < 0)
    first_rel[rel] = step
    reason[rel] = aff.release_reason[rel].long()
    dones = (terminated | truncated).bool()
    d = dones & alive
    for k, n in enumerate(tnames):
      flag = term.get_term(n) & d & (death < 0)
      death[flag] = k
    alive &= ~dones
  sched = [e for e in range(4) if bool(seed[0, e])]
  loss_ee = {e: float(((first_rel[:, e] >= 0) & seed[:, e]).float().sum() / N) for e in sched}
  loss = max(loss_ee.values()) if loss_ee else 0.0
  gp = {e: math.degrees(float(torch.quantile(gim_max[:, e], 0.9))) for e in sched}
  dd = " ".join(f"{tnames[k]} {int((death == k).sum()) / N:.2f}" for k in range(len(tnames)) if int((death == k).sum()))
  rtxt = " ".join(
    f"{EE[e]} {REASON[int(torch.bincount(reason[first_rel[:, e] >= 0, e], minlength=7).argmax())]}"
    for e in sched
    if loss_ee[e] > 0
  )
  ok = loss <= args.fail_frac
  rows.append((f, loss, ok))
  print(
    f"{f:5d} {seg(f):18s} {''.join(str(int(x)) for x in seed[0].tolist())}  | {loss:5.0%} "
    + " ".join(f"{EE[e]} {loss_ee[e]:.2f}" for e in sched)
    + " | "
    + " ".join(f"{EE[e]} {gp[e]:.1f}" for e in sched)
    + f" | {dd} | {'ok' if ok else 'FAIL'} {rtxt}"
  )
fails = [r for r in rows if not r[2]]
print(
  f"\nGATE ({len(rows)} seeded frames): {'PASS' if not fails else 'FAIL'}  failing {len(fails)}  worst {max(r[1] for r in rows):.0%}"
)
