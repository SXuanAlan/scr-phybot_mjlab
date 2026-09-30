"""Hold-still pre-flight for Mjlab-Climb-Transition80-Phybot-C2.

Uniform births over the whole a80 clip (scatter off), action frozen at the
birth pose, STEPS steps. Reports per clip-second bin: survival, termination
causes, constraint lanes and reward at birth. A reference-consistent world
must show: zero wall_contact at birth, zero C1-C4 violations at birth,
reward/step > 0 at birth, Phi = 1 on scheduled-engaged births, and seeded
welds surviving the settle transient.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_holdstill_probe.py [NENVS=512] [STEPS=60]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import mjlab  # noqa
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import scr_phybot_mjab.tasks  # noqa

TASK = "Mjlab-Climb-Transition80-Phybot-C2"
NENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
device = "cuda:0"
cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NENVS
mc = cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"
mc.spawn_xy_noise = (0.0, 0.0); mc.spawn_yaw_noise = 0.0; mc.joint_pos_noise = 0.0
cfg.terminations["engage_constraint"].params["mode"] = "off"  # lanes only
for name in ("encoder_bias", "body_friction", "base_mass"):
  cfg.events.pop(name, None)
cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k in ("wall_angle", "engage_shear_ladder")}
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
act = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
C = motion.clip.cycle_frames
nbins = motion._bin_count
print(f"clip frames {C}, bins {nbins}, envs {NENVS}, steps {STEPS}")

obs, _ = env.reset()
birth_frame = motion.time_steps.clone()
ids = act._target_ids if hasattr(act, "_target_ids") else act._joint_ids
offset = act._offset if hasattr(act, "_offset") else robot.data.default_joint_pos[:, ids]
scale = act._scale
def hold_action():
  q = robot.data.joint_pos[:, ids]
  return (q - offset) / scale

# birth snapshot (after the reset forward): constraint lanes need one term call
term = u.termination_manager
rew = u.reward_manager
alive = torch.ones(NENVS, dtype=torch.bool, device=device)
death_cause = {}
first_death_step = torch.full((NENVS,), -1, dtype=torch.long, device=device)
birth_reward = None; birth_terms = None; birth_phi = None; birth_viol = None
for step in range(STEPS):
  a = hold_action()
  obs, r, terminated, timeouts, extras = env.step(a)
  if step == 0:
    birth_reward = r.clone()
    birth_terms = {k: v.clone() for k, v in zip(rew._term_names, rew._step_reward.T)} if hasattr(rew, "_step_reward") else {}
    cone = term.get_term_cfg("engage_constraint").func
    birth_viol = {k: (v > 0).any(dim=-1).clone() for k, v in cone.last_violations.items()}
    birth_phi = rew._term_cfgs[rew._term_names.index("engage_progress")].func(u, **rew._term_cfgs[rew._term_names.index("engage_progress")].params).clone()
    birth_wall = term.get_term("wall_contact").clone()
    birth_any = {n: term.get_term(n).clone() for n in term.active_terms}
  dead_now = terminated & alive
  for n in term.active_terms:
    flag = term.get_term(n) & dead_now
    if flag.any():
      death_cause.setdefault(n, torch.zeros(NENVS, dtype=torch.bool, device=device))
      death_cause[n] |= flag
  first_death_step = torch.where(dead_now & (first_death_step < 0), torch.full_like(first_death_step, step), first_death_step)
  alive &= ~terminated
  # auto-reset re-births dead envs; keep analysing only the original cohort
bins = (birth_frame * nbins) // C
seg_names = list(np.load(mc.motion_file, allow_pickle=True).item()["segment_names"])
labels = np.asarray(np.load(mc.motion_file, allow_pickle=True).item()["phase_segment"])
def seg(f): return seg_names[labels[min(int(f), len(labels)-1)]]
print("\nbirth checks over the whole clip:")
print(f"  reward/step at birth: min {birth_reward.min():.3f} mean {birth_reward.mean():.3f} (x50 = per second)")
print(f"  wall_contact at birth: {int(birth_wall.sum())} envs")
for k, v in birth_viol.items(): print(f"  C-{k} violation at birth: {int(v.sum())} envs")
for n, v in birth_any.items():
  if int(v.sum()): print(f"  termination {n} at step 1: {int(v.sum())} envs")
sched = motion.clip.ee_engaged[birth_frame].any(dim=-1)
print(f"  Phi at birth: scheduled-engaged births mean {birth_phi[sched].mean():.3f} (n={int(sched.sum())}), standing births mean {birth_phi[~sched].mean():.3f}")
print(f"\nper-bin survival over {STEPS} hold-still steps (death causes):")
for b in range(nbins):
  m = bins == b
  if not m.any(): continue
  f0 = int((b * C) // nbins)
  surv = float(alive[m].float().mean())
  causes = {n: int((v & m).sum()) for n, v in death_cause.items() if int((v & m).sum())}
  med = first_death_step[m][first_death_step[m] >= 0]
  med_s = f"{int(med.median())}" if med.numel() else "-"
  print(f"  bin {b:2d} frame~{f0:4d} {seg(f0):17s} n={int(m.sum()):3d} survive {surv:5.2f} first-death-step med {med_s:>3s} {causes}")
