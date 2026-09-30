"""Seeded-birth diagnostic: spawn every env at one clip frame, hold still,
log per-step engaged count, anchor loads (t,l,n), hold margins, release
reasons; plus tracking-only reward at birth. Usage: script FRAME [STEPS] [GRAV]
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_seed_diag.py FRAME[,FRAME..] [STEPS=40] [GRAVITY_SCALE=1.0]
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
FRAMES = [int(x) for x in sys.argv[1].split(",")]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
GRAV = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
NENVS = 128
device = "cuda:0"
cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NENVS
mc = cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"; mc.spawn_early_fraction = 1.0
mc.birth_exclude_frames = ()  # the diagnostic pins frames; the training mask must not redirect them
mc.spawn_bias_auto_rake_frames = 0; mc.spawn_bias_frame_windows = ()  # pinned frames, not the classroom
if os.environ.get("SEED_DIAG_CLIP"):
  mc.motion_file = str(Path(os.environ["SEED_DIAG_CLIP"]).resolve())  # audit a draft clip
mc.spawn_xy_noise = (0.0, 0.0); mc.spawn_yaw_noise = 0.0; mc.joint_pos_noise = 0.0
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "bad_state")}
GUARD = os.environ.get("SEED_DIAG_GUARD")  # override the seed judge guard (steps) to emulate a birth anchor guard
if GUARD is not None:
  cfg.commands["microspine_affordance"].microspine_cfg.limit_surface.seed_hold_judge_guard_steps = int(GUARD)
for name in ("encoder_bias", "body_friction", "base_mass"):
  cfg.events.pop(name, None)
cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k in ("wall_angle", "engage_shear_ladder")}
if GRAV != 1.0:
  cfg.sim.mujoco.gravity = tuple(c * GRAV for c in cfg.sim.mujoco.gravity)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
act = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
rew = u.reward_manager
REASON = {0: "none", 1: "rev-shear", 2: "tension", 3: "lateral", 4: "ultimate", 5: "tilt", 6: "dr"}
EE = ["LH", "RH", "LF", "RF"]
for frame in FRAMES:
  motion._bias_window = (frame, frame + 1)
  env.reset()
  t0 = int(motion.time_steps[0])
  assert abs(t0 - frame) <= 1, (t0, frame)
  ids = act._target_ids
  hold = (robot.data.joint_pos[:, ids] - act._offset) / act._scale
  TRACK = os.environ.get("SEED_DIAG_TRACK", "0") == "1"  # PD target = the MOVING reference each step

  def _action():
    if not TRACK:
      return hold
    q = robot.data.joint_pos.clone()
    q[:, motion._joint_ids] = motion.ref_joint_pos
    return (q[:, ids] - act._offset) / act._scale
  ref_eng = motion.ref_engaged[0].tolist()
  print(f"\n=== frame {frame} (clock {t0}) ref_engaged {dict(zip(EE, ref_eng))} gravity x{GRAV} mode={'track-ref' if TRACK else 'hold-birth-pose'} guard={GUARD or 'cfg'}")
  reasons = {}
  alive_seed = aff.engaged.clone()
  for step in range(STEPS):
    env.step(_action())
    if step == 0:
      vals = {}
      for name, tcfg in zip(rew._term_names, rew._term_cfgs):
        vals[name] = float(tcfg.func(u, **tcfg.params).mean()) * tcfg.weight
      track = sum(v for k, v in vals.items() if k not in ("action_rate_l2", "dof_pos_limits"))
      print(f"  birth reward/s (tracking+Phi, no regularisers): {track:.2f}  terms: " + " ".join(f"{k.replace('motion_','')}={v:.2f}" for k, v in vals.items()))
    rel = aff.forced_release_this_step
    if rel.any():
      for e in range(4):
        for r in aff.release_reason[rel[:, e], e].tolist():
          reasons.setdefault((EE[e], REASON.get(int(r), r)), []).append(step)
    if step in (0, 2, 5, 6, 7, 9, 14, 19, 24, 26, 29, 39, 49, 59):
      eng = aff.engaged.float().mean(0)
      load = aff.anchor_load_tnl.mean(0)  # (4,3) t,l,n
      mg = aff.hold_margin
      print(f"  step {step:2d} engaged " + " ".join(f"{EE[e]}:{eng[e]:.2f}" for e in range(4))
            + " | f_t/f_l/f_n(N) " + " ".join(f"{EE[e]}:{load[e,0]:+.0f}/{load[e,1]:+.0f}/{load[e,2]:+.0f}" for e in range(4) if ref_eng[e])
            + " | margin p10 " + " ".join(f"{EE[e]}:{torch.quantile(mg[:,e],0.1):+.2f}" for e in range(4) if ref_eng[e]))
  for (ee, r), steps in sorted(reasons.items()):
    print(f"  releases {ee} {r}: n={len(steps)} first-step med {int(np.median(steps))} min {min(steps)}")
