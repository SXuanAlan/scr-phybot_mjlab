"""Passive-hold acceptance gate for a mount clip on the REAL plant.

For every seeded frame of the clip (stride) this births NENVS copies at that
frame (zero spawn noise, no DR events), holds the birth pose with the deploy
PD for STEPS steps and judges, after the settle transient (judge guard),
whether the scheduled seeds stay welded, how far the pad gimbals are driven
(wrist-pad misalignment vs the hard stop) and whether a fence geom touches
the wall. This is the rule the kinematic validator lacks: a PASS there only
proves a static load path exists; a passive PD robot must also realise it.

GATE per frame: seed loss <= --fail-frac AND gimbal p90 <= --gimbal-deg AND
no wall-contact / fall deaths. MASK: frames losing >= --mask-frac merged
into birth_exclude_frames ranges (end exclusive; climb_transition80/config/phybot_c2/commands.py).
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/reference/mount_clip_passive_gate.py [--clip PATH] [--stride 10]
       [--guard 25] [--envs 128] [--steps 40] [--fail-frac 0.05] [--mask-frac 0.20] [--gimbal-deg 2.5] [--out FILE]
"""

import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants as K
import scr_phybot_mjab.tasks  # noqa
from scr_phybot_mjab.microspine.release_reason import release_reason_name


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default=None)
  ap.add_argument("--stride", type=int, default=10)
  ap.add_argument("--guard", type=int, default=25)
  ap.add_argument("--envs", type=int, default=128)
  ap.add_argument("--steps", type=int, default=40)
  ap.add_argument("--fail-frac", type=float, default=0.05)
  ap.add_argument("--mask-frac", type=float, default=0.20)
  ap.add_argument("--gimbal-deg", type=float, default=2.5)
  ap.add_argument("--out", default=None)
  ap.add_argument(
    "--gimbal-hard", action="store_true", help="make the gimbal margin a hard FAIL (default: warning only)"
  )
  args = ap.parse_args(argv)
  TASK = "Mjlab-Climb-Transition80-Phybot-C2"
  device = "cuda:0"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = args.envs
  mc = cfg.commands["climb_motion"]
  mc.sampling_mode = "uniform"
  mc.spawn_early_fraction = 1.0
  mc.birth_exclude_frames = ()
  mc.spawn_bias_auto_rake_frames = 0
  mc.spawn_bias_frame_windows = ()
  if args.clip:
    mc.motion_file = str(Path(args.clip).resolve())
  mc.spawn_xy_noise = (0.0, 0.0)
  mc.spawn_yaw_noise = 0.0
  cfg.commands["microspine_affordance"].microspine_cfg.limit_surface.seed_hold_judge_guard_steps = args.guard
  for name in ("encoder_bias", "body_friction", "base_mass"):
    cfg.events.pop(name, None)
  cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k == "wall_angle"}
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
  tnames = list(term.active_terms)

  def tidx(n):
    return tnames.index(n) if n in tnames else -1

  I_WALL, I_FALL, I_RIP = tidx("wall_contact"), tidx("base_height"), tidx("anchor_rip")
  gim_ids = [i for i, n in enumerate(robot.joint_names) if "gimbal" in n]
  gim_ee = [[k for k, i in enumerate(gim_ids) if EEJ[e] in robot.joint_names[i]] for e in range(4)]
  stop_deg = math.degrees(float(K.PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD))
  seeded_frames = [f for f in range(0, C, args.stride) if engaged[f].any()]
  lines = []

  def emit(s):
    print(s)
    lines.append(s)

  emit(
    f"=== passive gate: clip {Path(mc.motion_file).name}  playable {C} frames  stride {args.stride}  guard "
    f"{args.guard}  envs {N}  steps {args.steps}  gimbal stop +-{stop_deg:.1f} deg  (loss<= {args.fail_frac:.0%}, "
    f"gimbal p90 <= {args.gimbal_deg} deg, wall/fall = 0) ==="
  )
  emit(
    "frame segment            sched | loss% (post-guard, worst EE) per-EE | gimbal p90 deg per EE | wall fall rip | "
    "verdict"
  )
  rows = []
  for f in seeded_frames:
    motion._bias_window = (f, f + 1)
    env.reset()
    birth = motion._birth_frames.clone()
    if int((birth - f).abs().max()) > 1:
      emit(f"{f:5d} {seg(f):18s} pin failed (births {birth[:3].tolist()}) — skipped")
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
    gim_p90 = {e: math.degrees(float(torch.quantile(gim_max[:, e], 0.9))) for e in sched}
    gmax = max(gim_p90.values()) if gim_p90 else 0.0
    wall = float((death == I_WALL).float().mean()) if I_WALL >= 0 else 0.0
    fall = float((death == I_FALL).float().mean()) if I_FALL >= 0 else 0.0
    rip = float((death == I_RIP).float().mean()) if I_RIP >= 0 else 0.0
    rtxt = " ".join(
      f"{EE[e]} {release_reason_name(r)}"
      for e in sched
      for r in [torch.bincount(reason[first_rel[:, e] >= 0, e], minlength=7).argmax()]
      if loss_ee[e] > 0
    )
    ok = loss <= args.fail_frac and wall == 0.0 and fall == 0.0 and (gmax <= args.gimbal_deg or not args.gimbal_hard)
    rows.append((f, loss, gmax, wall, fall, ok))
    gw = "" if gmax <= args.gimbal_deg else " gimbal>" + str(args.gimbal_deg)
    emit(
      f"{f:5d} {seg(f):18s} {''.join(str(int(x)) for x in seed[0].tolist())}  | {loss:5.0%} "
      + " ".join(f"{EE[e]} {loss_ee[e]:.2f}" for e in sched)
      + " | "
      + " ".join(f"{EE[e]} {gim_p90[e]:.1f}" for e in sched)
      + f" | {wall:.2f} {fall:.2f} {rip:.2f} | {'ok' if ok else 'FAIL'} {rtxt}{gw}"
    )
  worst = max(rows, key=lambda r: r[1]) if rows else None
  fails = [r for r in rows if not r[5]]
  emit(
    f"\nGATE ({len(rows)} seeded frames): {'PASS' if not fails else 'FAIL'}  failing frames {len(fails)}  worst loss "
    f"{worst[1]:.0%} @ {worst[0]}  max gimbal p90 {max(r[2] for r in rows):.1f} deg  wall-death frames "
    f"{sum(1 for r in rows if r[3] > 0)}"
  )
  mask = []
  for f, loss, *_ in rows:
    if loss >= args.mask_frac:
      if mask and mask[-1][1] == f:
        mask[-1][1] = f + args.stride
      else:
        mask.append([f, f + args.stride])
  emit("MASK (loss >= %.0f%%): birth_exclude_frames = %s" % (args.mask_frac * 100, tuple((a, b) for a, b in mask)))
  gim_fail = [r[0] for r in rows if r[2] > args.gimbal_deg and r[1] < args.mask_frac]
  emit(f"gimbal-only failures (seeds held but wrist past {args.gimbal_deg} deg): {gim_fail}")
  if args.out:
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("written", args.out)


if __name__ == "__main__":
  main()
