"""Sag-compensation probe: can a PD-target offset bring the loaded wrist back
inside the gimbal stop? For each pinned frame, iteration 0 holds the kinematic
birth pose (targets = q_kin); after STEPS steps the settled joint pose q_s is
read (mean over alive envs) and the targets are updated
q_t <- q_t + gain * (q_kin - q_s), then the cohort is re-born and held again.
Reports per iteration: gimbal max p50/p90 per seeded EE, seed loss, root
drift, target offset norm. Zero spawn noise, no DR, real plant.
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/mount_clip_sag_probe.py FRAME[,..] [--iters 3] [--gain 1.0] [--steps 40] [--envs 64] [--guard 25] [--clip PATH]
"""
import argparse, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, torch
import mjlab  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import scr_phybot_mjab.tasks  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("frames"); ap.add_argument("--iters", type=int, default=3); ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--envs", type=int, default=64); ap.add_argument("--guard", type=int, default=25)
ap.add_argument("--clip", default=None)
args = ap.parse_args()
TASK = "Mjlab-Climb-Transition80-Phybot-C2"; device = "cuda:0"
cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = args.envs
mc = cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"; mc.spawn_early_fraction = 1.0
mc.birth_exclude_frames = (); mc.spawn_bias_auto_rake_frames = 0; mc.spawn_bias_frame_windows = ()
if args.clip: mc.motion_file = str(Path(args.clip).resolve())
mc.spawn_xy_noise = (0.0, 0.0); mc.spawn_yaw_noise = 0.0; mc.joint_pos_noise = 0.0
cfg.commands["microspine_affordance"].microspine_cfg.limit_surface.seed_hold_judge_guard_steps = args.guard
for name in ("encoder_bias", "body_friction", "base_mass"): cfg.events.pop(name, None)
cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k in ("wall_angle", "engage_shear_ladder")}
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion"); aff = u.command_manager.get_term("microspine_affordance")
act = u.action_manager.get_term("joint_pos"); robot = u.scene["robot"]
N = args.envs; EE = ["LH", "RH", "LF", "RF"]; EEJ = ["left_hand", "right_hand", "left_foot", "right_foot"]
gim_ids = [i for i, n in enumerate(robot.joint_names) if "gimbal" in n]
gim_ee = [[k for k, i in enumerate(gim_ids) if EEJ[e] in robot.joint_names[i]] for e in range(4)]
ids = act._target_ids
jn = [robot.joint_names[i] for i in ids.tolist()] if hasattr(ids, "tolist") else [robot.joint_names[i] for i in ids]
for f in [int(x) for x in args.frames.split(",")]:
  motion._bias_window = (f, f + 1)
  q_t = None
  print(f"\n=== frame {f} gain {args.gain} steps {args.steps} guard {args.guard}")
  for it in range(args.iters + 1):
    env.reset()
    assert int((motion._birth_frames - f).abs().max()) <= 1
    seed = motion.clip.ee_engaged[motion._birth_frames].bool()
    q_kin = robot.data.joint_pos[:, ids].clone()
    root0 = robot.data.root_link_pos_w.clone()
    if q_t is None: q_t = q_kin.clone()
    hold = (q_t - act._offset) / act._scale
    alive = torch.ones(N, dtype=torch.bool, device=device)
    lost = torch.zeros(N, 4, dtype=torch.bool, device=device)
    gim_max = torch.zeros(N, 4, device=device)
    g_birth = None
    for step in range(args.steps):
      _, _, terminated, truncated, _ = env.step(hold)
      g = robot.data.joint_pos[:, gim_ids].abs()
      if step == 0: g_birth = g.clone()
      for e in range(4):
        if gim_ee[e]:
          gim_max[:, e] = torch.where(alive, torch.maximum(gim_max[:, e], g[:, gim_ee[e]].max(dim=1).values), gim_max[:, e])
      lost |= aff.forced_release_this_step & seed & alive.unsqueeze(1)
      alive &= ~(terminated | truncated).bool()
    sched = [e for e in range(4) if bool(seed[0, e])]
    q_s = robot.data.joint_pos[:, ids]
    drift = float((robot.data.root_link_pos_w - root0).norm(dim=-1)[alive].mean()) if alive.any() else float("nan")
    off = (q_t - q_kin)
    top = sorted(range(off.shape[1]), key=lambda j: -abs(float(off[0, j])))[:4]
    gb = " ".join(f"{EE[e]} {math.degrees(float(g_birth[:, gim_ee[e]].max(dim=1).values.median())):.1f}" for e in sched if gim_ee[e])
    print(f"  iter {it}: alive {float(alive.float().mean()):.2f} | gimbal@step0 {gb} | seed loss " + " ".join(f"{EE[e]} {float(lost[:, e].float().mean()):.2f}" for e in sched)
          + " | gimbal p50/p90 " + " ".join(f"{EE[e]} {math.degrees(float(gim_max[:, e].median())):.1f}/{math.degrees(float(torch.quantile(gim_max[:, e], 0.9))):.1f}" for e in sched)
          + f" | root drift {drift*100:.1f} cm | target offset max " + " ".join(f"{jn[j]} {math.degrees(float(off[0, j])):+.1f}deg" for j in top))
    if not alive.any(): break
    q_settled = q_s[alive].mean(dim=0, keepdim=True).expand_as(q_kin)
    q_t = q_t + args.gain * (q_kin - q_settled)
