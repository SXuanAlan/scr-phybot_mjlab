"""Seat-charge probe — does the mount ever FULL-CHARGE a weld? (v60 gate)

Question 1 (self-welds, RH esp.): do policy-made welds die before the
capacity ramp fills? Records per latch: lifetime, charge fraction at
death, fraction of life with rake shear >= charge threshold, reason.

Question 2 (seeds): does the reference stance PROVIDE charging current —
i.e. do latched-at-birth pads carry >= 10 N rake shear? (Seeds are born
full so their own charge never moves; their shear exposure answers
whether a FRESH weld in the same stance would charge.) Decides the clip
surgery: no shear at set stages -> the standing prefix has no free
gravity charging (the cycle's hidden subsidy) and the clip must add it.

Usage:
  CUDA_VISIBLE_DEVICES=2 python charge_probe.py <ckpt.pt> [steps=900] \
      [envs=192] [segments=hand_place,hand_set_left,hand_place_right,hand_set_hold]
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
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
TASK = _tok.get("task", "Mjlab-Climb-Transition-Phybot-C2")
device = "cuda:0"
EE = ["LH", "RH", "LF", "RF"]

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = NENVS
mc = env_cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"
mc.spawn_early_fraction = 1.0
mc.spawn_bias_segments = SEGMENTS
# Exams grade at the CURRICULUM's gravity (user call, 08-19 evening): a
# 1 g exam of a 0.6 g-era checkpoint manufactures artifacts (the 4425
# "LH seed collapse"). Resolve the ladder at this checkpoint's iteration;
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
# Cross-plant replay tokens (v3 era): `legacy_obs` drops the anchor lane so
# pre-v3 checkpoints (150-dim actors) load; `legacy_plant` rebuilds the
# rigid-pad physics. Old ckpt on the NEW plant = legacy_obs only (the
# plant-effect exam); full reproduction of its own world = both.
if "legacy_obs" in _argv:
  for _g in ("actor", "critic"):
    env_cfg.observations[_g].terms.pop("motion_anchor_relative", None)
if "legacy_plant" in _argv:
  env_cfg.commands[
    "microspine_affordance"
  ].microspine_cfg.spring_field_enabled = False
agent_cfg = load_rl_cfg("Mjlab-Climb-Transition-Phybot-C2")

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(
  wrapped, asdict(agent_cfg), str(Path("/tmp/.charge_probe_runner")), device
)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
ex = aff._get_executor()
ls = aff._ls_cfg
RAMP_T = float(ls.capacity_ramp_time_s)
SHEAR_N = float(ls.capacity_ramp_shear_n)
print(
  f"ramp: floor={ls.capacity_ramp_floor} time={RAMP_T}s shear_n={SHEAR_N}N "
  f"guard={ls.hold_judge_guard_steps} seed_guard={ls.seed_hold_judge_guard_steps}"
)

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
obs = wrapped.get_observations()

active = torch.zeros(E, 4, dtype=torch.bool, device=device)
age = torch.zeros(E, 4, dtype=torch.long, device=device)
was_seed = torch.zeros(E, 4, dtype=torch.bool, device=device)
birth_frame = torch.zeros(E, 4, dtype=torch.long, device=device)
weld_dist = torch.zeros(E, 4, device=device)
shear_steps = torch.zeros(E, 4, dtype=torch.long, device=device)
max_shear = torch.zeros(E, 4, device=device)
hit_full = torch.zeros(E, 4, dtype=torch.bool, device=device)
prev_charge = torch.zeros(E, 4, device=device)
prev_shear_frac = torch.zeros(E, 4, device=device)

records = []  # dicts


def finalize(e: int, k: int, reason: int, censored: bool) -> None:
  a = int(age[e, k])
  records.append(
    dict(
      ee=k,
      seed=bool(was_seed[e, k]),
      seg=seg_of(int(birth_frame[e, k])),
      life=a,
      charge=float(prev_charge[e, k]),
      shear_frac=(float(shear_steps[e, k]) / max(a, 1)),
      max_shear=float(max_shear[e, k]),
      full=bool(hit_full[e, k]),
      reason=int(reason),
      dist=float(weld_dist[e, k]),
      censored=censored,
    )
  )


for step in range(STEPS):
  with torch.no_grad():
    act = policy(obs)
  obs, _, dones, _ = wrapped.step(act)

  eng = aff.engaged
  seeded = aff._seeded
  charge = (aff._seat_charge / RAMP_T).clamp(0.0, 2.0)
  shear = aff.anchor_load_tnl[..., 0]
  reasons = aff.release_reason
  pad = ex.pad_center_world()
  anchor = motion.ref_anchor_pos_w
  dist = (pad - anchor).norm(dim=-1)
  done_mask = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()

  # 1) deaths of whole episodes: censor everything that was active
  for e in done_mask.nonzero(as_tuple=False).flatten().tolist():
    for k in range(4):
      if active[e, k]:
        finalize(e, k, reason=-1, censored=True)
    active[e] = False

  # 2) falling edges (live envs): the release step still carries the reason
  fall = active & ~eng & ~done_mask.unsqueeze(-1)
  for e, k in fall.nonzero(as_tuple=False).tolist():
    finalize(e, k, reason=int(reasons[e, k]), censored=False)
    active[e, k] = False

  # 3) rising edges (incl. post-done seed births: prev cleared above)
  rise = ~active & eng
  if bool(rise.any()):
    idx = rise.nonzero(as_tuple=False)
    for e, k in idx.tolist():
      age[e, k] = 0
      was_seed[e, k] = bool(seeded[e, k])
      birth_frame[e, k] = int(motion.time_steps[e])
      weld_dist[e, k] = dist[e, k]
      shear_steps[e, k] = 0
      max_shear[e, k] = 0.0
      hit_full[e, k] = charge[e, k] >= 1.0
    active |= rise

  # 4) accumulate on active latches
  age += active.long()
  shear_steps += (active & (shear >= SHEAR_N)).long()
  max_shear = torch.where(active, torch.maximum(max_shear, shear), max_shear)
  hit_full |= active & (charge >= 1.0)
  prev_charge = torch.where(active, charge, prev_charge)

env.close()

for e in range(E):
  for k in range(4):
    if active[e, k]:
      finalize(e, k, reason=-1, censored=True)


def med(v):
  s = sorted(v)
  return s[len(s) // 2] if s else float("nan")


def p90(v):
  s = sorted(v)
  return s[int(len(s) * 0.9)] if s else float("nan")


print(f"\n=== SEAT-CHARGE PROBE: {len(records)} latch records | ckpt={CKPT.name}")
print(f"segments={SEGMENTS} envs={NENVS} steps={STEPS}\n")

groups = defaultdict(list)
for r in records:
  groups[(r["ee"], r["seed"])].append(r)

for k in range(4):
  for seed in (True, False):
    rs = groups.get((k, seed), [])
    if not rs:
      continue
    fin = [r for r in rs if not r["censored"]]
    cen = [r for r in rs if r["censored"]]
    tag = "SEED" if seed else "self"
    print(f"[{EE[k]} {tag}] n={len(rs)} (died {len(fin)} / censored {len(cen)})")
    if fin:
      reasons = Counter(r["reason"] for r in fin)
      print(
        f"  died:  life med={med([r['life'] for r in fin])} "
        f"p90={p90([r['life'] for r in fin])} steps | "
        f"charge@death med={med([r['charge'] for r in fin]):.2f} | "
        f"reached-full {sum(r['full'] for r in fin)}/{len(fin)} | "
        f"reasons {dict(sorted(reasons.items()))}"
      )
      print(
        f"         shear>= {SHEAR_N:.0f}N frac med={med([r['shear_frac'] for r in fin]):.2f} "
        f"| max-shear med={med([r['max_shear'] for r in fin]):.1f}N "
        f"| weld-dist med={med([r['dist'] for r in fin]) * 100:.1f}cm"
      )
    if cen:
      print(
        f"  alive: life med={med([r['life'] for r in cen])} | "
        f"reached-full {sum(r['full'] for r in cen)}/{len(cen)} | "
        f"shear-frac med={med([r['shear_frac'] for r in cen]):.2f} | "
        f"max-shear med={med([r['max_shear'] for r in cen]):.1f}N"
      )

print("\nRH self-welds by segment at weld:")
for seg, rs in sorted(
  ((s, [r for r in records if r["ee"] == 1 and not r["seed"] and r["seg"] == s])
   for s in set(r["seg"] for r in records)),
):
  if rs:
    fin = [r for r in rs if not r["censored"]]
    print(
      f"  {seg}: n={len(rs)} life med={med([r['life'] for r in rs])} "
      f"full {sum(r['full'] for r in rs)}/{len(rs)} "
      f"shear-frac med={med([r['shear_frac'] for r in rs]):.2f}"
    )
