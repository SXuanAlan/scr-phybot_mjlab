"""Schedule-conformance exam for the mount task — the climb exam's sibling.

The climb ladder is judged by MATCH (actual engaged mask vs the reference
schedule, per step); the mount was being judged by reward lanes and event
counts, which can improve while the visible behavior (slide-then-engage,
wrong-position welds) does not. This exam scores the mount the climb way:

  per EE (LH RH LF RF), over mature steps:
    match  = P(actual == scheduled)
    miss   = P(scheduled ON, actually OFF)   — split into rubbing
             (pressing >2 N unwelded) vs absent (not even touching)
    extra  = P(scheduled OFF, actually ON)
  per scheduled rising edge: delay until the robot's actual weld (censored)
  per weld event: |pad - anchor|, SIGNED vertical error (neg = below the
             anchor = slid down before latching), and the slide during the
             pressing approach (peak pad z while pressing - z at weld)
  while holding in-schedule: |pad - anchor| (parked at the wrong spot?)

Usage:
  python scripts/mount_schedule_exam.py <ckpt.pt> [steps=900] [envs=192] \
      [segments=hand_place,hand_set_left,hand_place_right,hand_set_hold] \
      [noise] [task=Mjlab-Climb-Transition-Phybot-C2]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import scr_phybot_mjab.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

_argv = [a for a in sys.argv[1:] if "=" not in a]
_tok = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
CKPT = Path(_argv[0])
STEPS = int(_tok.get("steps", 900))
NENVS = int(_tok.get("envs", 192))
SEGMENTS = tuple(
  _tok.get(
    "segments", "hand_place,hand_set_left,hand_place_right,hand_set_hold"
  ).split(",")
)
NOISE = "noise" in _argv
TASK = _tok.get("task", "Mjlab-Climb-Transition-Phybot-C2")
device = "cuda:0"
EE = ["LH", "RH", "LF", "RF"]

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = NENVS
mc = env_cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"
mc.spawn_early_fraction = 1.0
mc.spawn_bias_segments = SEGMENTS
# Exams grade at the CURRICULUM's gravity (user call, 08-19 evening);
# `gravity=<scale>` overrides (the graduation exam pins 1.0).
from scr_phybot_mjab.tasks.climb_transition.mdp.curriculums import (
  gravity_scale_at,
)

_glad = env_cfg.curriculum.pop("gravity_ladder", None)
if "gravity" in _tok:
  _gscale = float(_tok["gravity"])
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
if NOISE:
  mc.spawn_xy_noise = (0.05, 0.03)
  mc.spawn_yaw_noise = 0.0873
  mc.joint_pos_noise = 0.02
agent_cfg = load_rl_cfg("Mjlab-Climb-Transition-Phybot-C2")

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(
  wrapped, asdict(agent_cfg), str(Path("/tmp/.exam_runner")), device
)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
robot = u.scene["robot"]
ex = aff._get_executor()

_clipd = np.load(mc.motion_file, allow_pickle=True).item()
_names = list(_clipd["segment_names"])
_labels = np.asarray(_clipd["phase_segment"])
_cycles = np.asarray(_clipd["cycle_index"])
_end = _names.index("end_hold")
_cmask = (_cycles == _cycles.min()) & (_labels != _end)
_cyc_labels = _labels[_cmask]


def seg_of(frame: int) -> str:
  f = min(max(int(frame), 0), len(_cyc_labels) - 1)
  return _names[_cyc_labels[f]]


E = NENVS
NEG = torch.finfo(torch.float32).min
obs = wrapped.get_observations()
age = torch.zeros(E, dtype=torch.long, device=device)
birth_frame = motion.time_steps.clone()

# per-episode accumulators (E,4)
z4 = lambda: torch.zeros(E, 4, device=device)
n_mature = torch.zeros(E, device=device)
n_match, n_miss, n_extra, n_rub = z4(), z4(), z4(), z4()
n_hold, d_hold = z4(), z4()  # engaged-in-schedule steps + anchor dist sum
# edge->weld delay: -1 = no pending edge
edge_wait = torch.full((E, 4), -1.0, device=device)
# slide tracking: peak pad z during current pressing-miss stretch
peak_z = torch.full((E, 4), NEG, device=device)

prev_engaged = aff.engaged.clone()
prev_ref = motion.ref_engaged.clone()

episodes = []  # per-episode dicts
welds = []  # (seg, ee, dist, z_err_signed, slide_drop|None)
delays = []  # (ee, steps, censored)

for step in range(STEPS):
  with torch.no_grad():
    act = policy(obs)
  obs, _, dones, _ = wrapped.step(act)
  age += 1
  mature = (age >= 3).unsqueeze(-1)

  ref = motion.ref_engaged
  eng = aff.engaged
  press = aff.pad_press_force
  pad = ex.pad_center_world()
  anchor = motion.ref_anchor_pos_w
  dist = (pad - anchor).norm(dim=-1)

  m = mature.expand(-1, 4).float()
  n_mature += mature.squeeze(-1).float()
  n_match += ((eng == ref) & mature).float()
  miss = ref & ~eng
  n_miss += (miss & mature).float()
  n_extra += ((~ref & eng) & mature).float()
  rubbing = miss & (press > 2.0)
  n_rub += (rubbing & mature).float()
  holding = ref & eng
  n_hold += (holding & mature).float()
  d_hold += dist * (holding & mature).float()

  # edge->weld delay bookkeeping
  ref_rising = ref & ~prev_ref
  edge_wait = torch.where(ref_rising & mature, torch.zeros_like(edge_wait), edge_wait)
  pending = edge_wait >= 0
  edge_wait = torch.where(pending & ~eng, edge_wait + 1, edge_wait)

  # weld events are recorded BEFORE the peak_z reset below — at the weld
  # step miss is already False, so resetting first would wipe the slide.
  rising = (eng & ~prev_engaged) & mature
  if bool(rising.any()):
    for e, k in rising.nonzero(as_tuple=False).tolist():
      d = float(dist[e, k])
      zerr = float(pad[e, k, 2] - anchor[e, k, 2])
      pz = float(peak_z[e, k])
      drop = (pz - float(pad[e, k, 2])) if pz > -1e30 else None
      welds.append((seg_of(int(birth_frame[e])), k, d, zerr, drop))
      if edge_wait[e, k] >= 0:
        delays.append((k, int(edge_wait[e, k]), False))
        edge_wait[e, k] = -1.0

  # slide: peak z while pressing unwelded in-schedule (after weld capture)
  peak_z = torch.where(rubbing, torch.maximum(peak_z, pad[..., 2]), peak_z)
  peak_z = torch.where(~miss, torch.full_like(peak_z, NEG), peak_z)

  done_idx = dones.nonzero(as_tuple=False).flatten().tolist()
  if done_idx:
    for e in done_idx:
      steps_e = float(n_mature[e])
      if steps_e >= 20:
        episodes.append(
          dict(
            seg=seg_of(int(birth_frame[e])),
            steps=int(age[e]),
            match=(n_match[e] / steps_e).tolist(),
            miss=(n_miss[e] / steps_e).tolist(),
            extra=(n_extra[e] / steps_e).tolist(),
            rub=(n_rub[e] / steps_e).tolist(),
            hold_d=[
              (d_hold[e, k] / n_hold[e, k]).item() if n_hold[e, k] > 0 else None
              for k in range(4)
            ],
          )
        )
      for k in range(4):
        if edge_wait[e, k] >= 0:
          delays.append((k, int(edge_wait[e, k]), True))
      age[e] = 0
      birth_frame[e] = motion.time_steps[e]
      n_mature[e] = 0
      for t in (n_match, n_miss, n_extra, n_rub, n_hold, d_hold):
        t[e] = 0
      edge_wait[e] = -1.0
      peak_z[e] = NEG
  prev_engaged = eng.clone()
  prev_ref = ref.clone()

env.close()


def med(v):
  s = sorted(v)
  return s[len(s) // 2] if s else float("nan")


print(
  f"\n=== MOUNT SCHEDULE EXAM: {len(episodes)} episodes | ckpt={CKPT.name} "
  f"segments={SEGMENTS} noise={NOISE}"
)
by_seg = defaultdict(list)
for ep in episodes:
  by_seg[ep["seg"]].append(ep)
for seg in sorted(by_seg):
  eps = by_seg[seg]
  print(f"\n[{seg}] n={len(eps)} survival med={med([e['steps'] for e in eps])}")
  print("        match   miss(rub/absent)  extra   hold-dist")
  for k in range(4):
    mt = med([e["match"][k] for e in eps])
    ms = med([e["miss"][k] for e in eps])
    rb = med([e["rub"][k] for e in eps])
    ex_ = med([e["extra"][k] for e in eps])
    hd = [e["hold_d"][k] for e in eps if e["hold_d"][k] is not None]
    hds = f"{med(hd) * 100:.1f}cm" if hd else "--"
    print(
      f"  {EE[k]}   {mt:.3f}   {ms:.3f} ({rb:.3f}/{ms - rb:.3f})   "
      f"{ex_:.3f}   {hds}"
    )
print("\nOVERALL match per EE (episode-median):")
for k in range(4):
  print(f"  {EE[k]}: {med([e['match'][k] for e in episodes]):.3f}")

if welds:
  print(f"\nWELD events: {len(welds)}")
  for k in range(2):
    ws = [w for w in welds if w[1] == k]
    if not ws:
      print(f"  {EE[k]}: none")
      continue
    dists = [w[2] for w in ws]
    zerrs = [w[3] for w in ws]
    drops = [w[4] for w in ws if w[4] is not None]
    below = sum(1 for z in zerrs if z < -0.02)
    print(
      f"  {EE[k]}: n={len(ws)} dist med={med(dists) * 100:.1f}cm  "
      f"z-err med={med(zerrs) * 100:+.1f}cm (below-anchor>2cm: {below}/{len(ws)})  "
      f"slide-before-weld med={med(drops) * 100:.1f}cm"
      if drops
      else f"  {EE[k]}: n={len(ws)} dist med={med(dists) * 100:.1f}cm "
      f"z-err med={med(zerrs) * 100:+.1f}cm"
    )
if delays:
  print("\nEDGE->WELD delay (steps, 50Hz):")
  for k in range(4):
    ds = [d for (kk, d, c) in delays if kk == k and not c]
    cs = [d for (kk, d, c) in delays if kk == k and c]
    if ds or cs:
      print(
        f"  {EE[k]}: answered n={len(ds)} med={med(ds)} p90="
        f"{sorted(ds)[int(len(ds) * 0.9)] if ds else '--'}  "
        f"UNANSWERED-at-death n={len(cs)} (waited med={med(cs)})"
      )
