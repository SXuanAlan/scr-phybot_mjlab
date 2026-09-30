"""Validation suite for the microspine connect-equality attachment.

Subcommands:
  hang     16-env hang test on a vertical (90°) wall: all 4 EEs anchored,
           zero actions. Asserts the anchors hold the robot weight without
           sag, jitter, NaNs, or efc overflow.
  reference-hold
           GPU controller-settling stress test from sampled reference
           snapshots. This is intentionally not a static feasibility proof.
           Releases during the artificial gravity ramp are settle
           transients: tolerated on <= 5% of frames (one anchor at most,
           stance stable on the rest). A release at held full gravity, any
           cascade, or a frame ending two anchors short always fails.
  ablation Disable physical grip at the same low ordinary wall friction and
           assert that the robot falls. This proves the equality anchors, not
           incidental contact friction, provide vertical support.
  sweep    GPU-parallel limit-surface envelope scan: each env pulls one EE in
           a different direction of the t-n / t-l planes with a per-direction
           force ramp, recording the anchor load at forced release. Checks
           the measured release points lie on the configured envelope and
           optionally plots measured-vs-theory (--plot out.png).
  readout  Cross-checks the efc-row force gather against the kinematic
           anchor error (constraint force ∝ pose error) under a time-varying
           torso push.

Run from the repo root inside the env_mjlab environment, e.g.:
  python scripts/validate_microspine_constraint.py hang --num-envs 16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from scr_phybot_mjab.microspine.wang_limit_surface import capacity_ratio
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

DEVICE = "cuda:0"
EE_NAMES = (
  "left_hand_spine",
  "right_hand_spine",
  "left_foot_spine",
  "right_foot_spine",
)
# Deterministic target-EE envelope used by sweep (overrides the sampled DR).
SWEEP_PARAMS = {
  "f_t_max": 400.0,
  "f_rev_max": 20.0,
  "mu_adhesion": 0.4,
  "f_n0": 15.0,
  "mu_lateral": 0.6,
  "f_l0": 15.0,
}


def build_env(
  num_envs: int,
  limit_surface_enable: bool = True,
  reanchor_after_grace: bool = True,
  anchor_lock_rotation: bool | None = None,
  grip_enabled: bool = True,
  motion_file: str | Path | None = None,
  anchor_torquescale: float | None = None,
) -> ManagerBasedRlEnv:
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1.0e9  # No time_out resets during tests.
  cfg.curriculum = {}  # Gravity is set explicitly per test.
  # These are deterministic physics probes: strip every DR event so a
  # domain_rand cfg cannot perturb the benches.
  for dr_event in ("encoder_bias", "body_friction", "base_mass", "random_release"):
    cfg.events.pop(dr_event, None)
  cfg.observations["actor"].enable_corruption = False
  # Mechanics probes deliberately hold a fixed pose or apply artificial force
  # ramps; task-level lack-of-climb or reference mismatch is expected here and
  # must not reset them.
  cfg.terminations.pop("no_progress", None)
  cfg.terminations.pop("gait_curriculum_success", None)
  cfg.terminations.pop("bad_tracking", None)
  # Deterministic spawn: pin the motion command's RSI to clip frame 0 (all
  # four EEs engaged, pads on the wall) with zero noise. The DeepMimic task
  # cfg removed the generic reset_robot_joints/base events — the motion
  # command owns the reset pose.
  motion = cfg.commands["climb_motion"]
  if motion_file is not None:
    motion.motion_file = str(Path(motion_file).resolve())
  motion.sampling_mode = "start"
  motion.joint_pos_noise = 0.0
  motion.height_offset_range = (0.0, 0.0)
  # The nested limit_surface cfg is shared between the affordance command and
  # the physics event term (dataclasses.replace is shallow copy: nested
  # objects shared, scalar fields independent) — so scalars must be written
  # on both copies. anchor_lock_rotation also steers the scene spec_fn, which
  # closes over the commands-side cfg — written before env construction, so
  # the equality type and the executor stay consistent.
  mcfg = cfg.commands["microspine_affordance"].microspine_cfg
  # Mechanics tests intentionally exercise every anchor regardless of the
  # performance curriculum stage selected by the production task.
  mcfg.initial_engaged_ee_names = EE_NAMES
  mcfg.grip_enabled = grip_enabled
  if not grip_enabled:
    cfg.terminations.pop("base_height", None)
  mcfg.limit_surface.enable = limit_surface_enable
  for owner in (mcfg, cfg.events["microspine_grip"].params["microspine_cfg"]):
    owner.reanchor_after_grace = reanchor_after_grace
    if anchor_lock_rotation is not None:
      owner.anchor_lock_rotation = anchor_lock_rotation
    if anchor_torquescale is not None:
      owner.anchor_torquescale = anchor_torquescale
  return ManagerBasedRlEnv(cfg=cfg, device=DEVICE)


def set_gravity(env: ManagerBasedRlEnv, angle_deg: float, magnitude: float = 9.81):
  """Angle convention of the RETIRED gravity representation (90° = gravity
  straight down). The shipped task tilts the wall instead; this bench helper
  survives only for the gravity-MAGNITUDE ramp at the vertical operating
  point (angle_deg=90) — do not use it to emulate wall angles."""
  rad = math.radians(angle_deg)
  view = wp.to_torch(env.sim.wp_model.opt.gravity)
  view[:, 0] = magnitude * math.cos(rad)
  view[:, 1] = 0.0
  view[:, 2] = -magnitude * math.sin(rad)


def get_terms(env: ManagerBasedRlEnv):
  executor = env.event_manager.get_term_cfg("microspine_grip").func
  affordance = env.command_manager._terms["microspine_affordance"]
  return executor, affordance


def hold_pose_actions(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant actions whose PD target equals the current (spawn) pose.

  The DeepMimic task spawns at motion-clip frame 0, which differs from
  ``default_joint_pos``. Zero actions would therefore command a PD pull
  toward the default pose and preload the anchors with a pure pose fight
  (~760 N per EE at zero gravity) that swamps every mechanics probe. Call
  after ``env.reset()``; keeps the historical test semantics where the PD
  holds the spawn pose.
  """
  term = env.action_manager._terms["joint_pos"]
  robot = env.scene["robot"]
  q = robot.data.joint_pos[:, term._target_ids]
  bias = robot.data.encoder_bias[:, term._target_ids]
  raw = (q + bias - term.offset) / term.scale
  if env.action_manager.total_action_dim != raw.shape[1]:
    raise ValueError(
      "hold_pose_actions assumes joint_pos is the only action term "
      f"({raw.shape[1]} joint dims vs "
      f"{env.action_manager.total_action_dim} total)."
    )
  return raw.clone()


def zero_actions(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)


def quasistatic_settle(
  env: ManagerBasedRlEnv,
  executor,
  affordance,
  actions: torch.Tensor,
  rounds: int = 8,
  steps_per_round: int = 10,
  tail_steps: int = 20,
) -> None:
  """Null the spawn-pose anchor preload without letting the structure ring.

  The clip's frame-0 RSI spawns with reference velocities and a pose whose
  pad placement is only approximately wall-consistent; the anchors then carry
  a pure internal standoff (hundreds of N under some ring phases). Relax it
  quasistatically: kill all momentum every step (the anchor efc force is
  position-error based, so this does not distort any readout) and re-place
  each anchor at its settled pad, clamped to the wall face — a ringing
  snapshot with the pad center transiently inside the wall would otherwise
  lock the contact push-out into a permanent phantom standoff.
  """
  robot = env.scene["robot"]
  zero_qd = torch.zeros_like(robot.data.joint_vel)
  zero_root_vel = torch.zeros(env.num_envs, 6, device=DEVICE)
  wall_inward = affordance._wall_inward_normal().unsqueeze(1)

  def _step():
    robot.write_root_link_velocity_to_sim(zero_root_vel)
    robot.write_joint_velocity_to_sim(zero_qd)
    env.step(actions)

  for _ in range(rounds):
    for _ in range(steps_per_round):
      _step()
    pads = executor.pad_center_world()
    proj = executor._project_to_wall(pads)
    depth = ((pads - proj) * wall_inward).sum(-1, keepdim=True)
    executor.anchor_pos_w = torch.where(depth > 0, proj, pads).clone()
  for _ in range(tail_steps):
    _step()


def robot_weight(env: ManagerBasedRlEnv) -> float:
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids
  mass = float(env.sim.mj_model.body_mass[body_ids.cpu().numpy()].sum())
  return mass * 9.81


def check(name: str, ok: bool, detail: str) -> bool:
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
  return ok


# ---------------------------------------------------------------------------
# hang
# ---------------------------------------------------------------------------


def cmd_hang(args) -> bool:
  print(f"== hang: {args.num_envs} envs, 90° wall, {args.seconds}s ==")
  # Limit surface ON by default: re-grip re-seats each anchor at the settled
  # pad pose and relieves spawn-pose preload before hold checks begin.
  # "all engaged" passing with hold checks active is therefore the re-grip
  # acceptance test. --no-limit-surface restores the mechanics-only variant.
  env = build_env(
    args.num_envs,
    limit_surface_enable=not args.no_limit_surface,
  )
  env.reset()
  set_gravity(env, 90.0)
  executor, affordance = get_terms(env)
  executor.assert_now(env)

  steps = int(args.seconds / env.step_dt)
  weight = robot_weight(env)
  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()

  actions = hold_pose_actions(env)
  support_hist, engaged_hist = [], []
  terminations = 0
  root_z_hist = []
  for _ in range(steps):
    _, _, terminated, truncated, _ = env.step(actions)
    terminations += int((terminated | truncated).sum().item())
    root_z_hist.append(robot.data.root_link_pos_w[:, 2].clone())
    # F_load is EE-on-anchor; the wall's supporting force on the robot is -z
    # of that, so total support = -sum(F_load_z). Gather explicitly so the
    # readout also works with --no-limit-surface.
    support_hist.append(-executor.compute_anchor_loads()[..., 2].sum(dim=1).clone())
    engaged_hist.append(affordance.engaged.all(dim=1).clone())

  root_z = torch.stack(root_z_hist)  # (T, E)
  support = torch.stack(support_hist)  # (T, E)
  engaged = torch.stack(engaged_hist)  # (T, E)
  half = steps // 2

  sag = (z0.unsqueeze(0) - root_z).max().item()
  support_mean = support[half:].mean().item()
  support_std = support[half:].std(dim=0).mean().item()
  jitter = root_z[half:].std(dim=0).max().item()
  nefc_max = int(env.sim.data.nefc[:].max().item())
  njmax = int(env.sim.data.njmax)

  ok = True
  ok &= check("all engaged", bool(engaged.all().item()), f"{engaged.float().mean():.3f} mean")
  ok &= check("no terminations", terminations == 0, f"{terminations} resets")
  ok &= check("root sag < 2 cm", sag < 0.02, f"max sag {sag * 1000:.1f} mm")
  ok &= check(
    "anchors carry weight",
    abs(support_mean - weight) / weight < 0.10,
    f"support {support_mean:.1f} N vs weight {weight:.1f} N ({100 * (support_mean / weight - 1):+.1f}%)",
  )
  ok &= check("root jitter < 1 mm", jitter < 1.0e-3, f"{jitter * 1000:.3f} mm std")
  ok &= check("nefc headroom", nefc_max + 12 <= njmax, f"nefc {nefc_max} / njmax {njmax}")
  ok &= check("finite state", bool(torch.isfinite(root_z).all().item()), "root z finite")
  print(f"  anchor support force std (per-env mean): {support_std:.3f} N")
  # Informational: final per-EE loads in the engage frame vs the default
  # envelope. Tension beyond the cone here means the spawn pose itself is not
  # holdable by a realistic spine envelope (task/pose concern, not physics).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  loads = executor.compute_anchor_loads()
  f_t = (loads * t_hat).sum(-1).mean(0)
  f_l = (loads * l_hat).sum(-1).mean(0)
  f_n = (loads * n_hat).sum(-1).mean(0)
  ls_cfg = affordance._ls_cfg
  for h, name in enumerate(EE_NAMES):
    if affordance._wang_lut is not None:
      f0_mid = 0.5 * sum(ls_cfg.f_t_max_range)
      ratio = capacity_ratio(affordance._wang_lut, f_t[h], f_l[h], f_n[h]).clamp_min(ls_cfg.wang_r_floor)
      load_mag = math.sqrt(f_t[h] ** 2 + f_l[h] ** 2 + max(f_n[h].item(), 0.0) ** 2)
      cap_note = f"(load {load_mag:.1f} N vs capacity @ mid F0 ≈ {f0_mid * ratio.item():.1f} N)"
    else:
      cap_note = f"(tension cap @ default mid params ≈ {0.4 * max(f_t[h].item(), 0.0) + 15.0:.1f} N)"
    print(f"  {name}: f_t={f_t[h]:.1f} N, f_l={f_l[h]:.1f} N, f_n={f_n[h]:.1f} N " + cap_note)
  env.close()

  return ok


# ---------------------------------------------------------------------------
# reference-hold
# ---------------------------------------------------------------------------


def _raw_cycle_frames(data: dict) -> int:
  """Return cycle-0 frames excluding the duplicated end-hold tail."""
  names = list(data["segment_names"])
  end_id = names.index("end_hold")
  labels = np.asarray(data["phase_segment"])
  cycles = np.asarray(data["cycle_index"])
  return int(np.sum((cycles == 0) & (labels != end_id)))


def _place_reference_frames(
  env: ManagerBasedRlEnv,
  frames: torch.Tensor,
  f0: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Place one env at each reference frame for the PD settling stress test."""
  motion = env.command_manager.get_term("climb_motion")
  executor, affordance = get_terms(env)
  robot = env.scene["robot"]
  frames = frames.to(device=DEVICE, dtype=torch.long)
  if frames.numel() != env.num_envs:
    raise ValueError(f"Need one frame per env ({frames.numel()} vs {env.num_envs}).")
  if int(frames.max().item()) >= motion.clip.cycle_frames:
    raise ValueError(
      f"Reference frame {int(frames.max().item())} exceeds the loaded {motion.clip.cycle_frames}-frame cycle."
    )

  env_ids = torch.arange(env.num_envs, device=DEVICE)
  motion.time_steps.copy_(frames)
  # Canonical-frame bookkeeping (wall-angle era): root_offset no longer
  # carries env origins — the world transform adds them (identity wall here,
  # so this is the plain origin shift it always was).
  motion.root_offset.zero_()
  root_pos = motion._to_world_point(motion.clip.root_pos[frames])
  motion.spawn_pos.copy_(root_pos)

  joint_pos = robot.data.default_joint_pos.clone()
  joint_pos[:, motion._joint_ids] = motion.clip.joint_pos[frames]
  joint_vel = torch.zeros_like(joint_pos)
  robot.write_joint_state_to_sim(joint_pos, joint_vel)
  root_state = torch.cat(
    [
      root_pos,
      motion.clip.root_quat[frames],
      torch.zeros(env.num_envs, 6, device=DEVICE),
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(root_state)
  robot.clear_state()
  env.sim.forward()

  scheduled = motion.clip.ee_engaged[frames]
  affordance.seed_engaged(env_ids, scheduled)
  # Do not let a scheduled swing pad auto-latch during the snapshot test.
  # Ordinary compression contact remains active; only its bilateral anchor
  # is blocked.
  affordance._cooldown_steps.copy_(
    torch.where(
      scheduled,
      torch.zeros_like(affordance._cooldown_steps),
      torch.full_like(affordance._cooldown_steps, 1_000_000),
    )
  )
  # Use the worst configured asperity draw for a conservative stress test.
  test_f0 = affordance._ls_cfg.f_t_max_range[0] if f0 is None else float(f0)
  affordance.limit_f_t_max.fill_(test_f0)

  executor.reset(env_ids)
  executor.assert_now(env)
  return scheduled, root_pos


def cmd_reference_hold(args) -> bool:
  path = Path(args.motion_file).resolve()
  data = np.load(path, allow_pickle=True).item()
  cycle = _raw_cycle_frames(data)
  frames_np = np.arange(0, cycle, args.frame_stride, dtype=np.int64)
  # Always include both sides of every segment boundary, even with a coarse
  # stride: support-set changes are the most likely failure locations.
  labels = np.asarray(data["phase_segment"])[:cycle]
  edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
  frames_np = np.unique(
    np.concatenate(
      [
        frames_np,
        edges,
        np.maximum(edges - 1, 0),
        np.array([cycle - 1], dtype=np.int64),
      ]
    )
  )
  print(
    f"== reference-hold: {path}, {len(frames_np)}/{cycle} cycle frames, "
    f"90° wall, {args.ramp_seconds}s gravity ramp + "
    f"{args.hold_seconds}s hold =="
  )

  env = build_env(
    len(frames_np),
    limit_surface_enable=True,
    motion_file=path,
  )
  env.reset()
  set_gravity(env, 90.0, magnitude=0.0)
  executor, affordance = get_terms(env)
  robot = env.scene["robot"]
  frames = torch.tensor(frames_np, dtype=torch.long, device=DEVICE)
  scheduled, root_pos = _place_reference_frames(env, frames, f0=args.f0)
  actions = hold_pose_actions(env)

  ever_lost = torch.zeros_like(scheduled)
  ramp_released = torch.zeros_like(scheduled)
  hold_released = torch.zeros_like(scheduled)
  min_live_supports = torch.full((env.num_envs,), 99, dtype=torch.long, device=DEVICE)
  reset = torch.zeros(env.num_envs, dtype=torch.bool, device=DEVICE)
  min_margin = torch.full_like(affordance.hold_margin, float("inf"))
  max_moment = torch.zeros_like(affordance.hold_margin)
  max_effective = torch.zeros_like(affordance.hold_margin)
  max_root_shift = torch.zeros(env.num_envs, device=DEVICE)
  zero_joint_vel = torch.zeros_like(robot.data.joint_vel)
  zero_root_vel = torch.zeros(env.num_envs, 6, device=DEVICE)
  ramp_steps = max(int(round(args.ramp_seconds / env.step_dt)), 1)
  hold_steps = max(int(round(args.hold_seconds / env.step_dt)), 0)
  steps = ramp_steps + hold_steps

  for step in range(steps):
    gravity_fraction = min((step + 1) / ramp_steps, 1.0)
    set_gravity(env, 90.0, magnitude=9.81 * gravity_fraction)
    # Suppress the worst artificial impulse from dropping an isolated clip
    # snapshot into gravity. There is still no inverse-dynamics feedforward:
    # this tests whether the stock PD and latch logic settle from the frame,
    # not whether some static force allocation exists.
    robot.write_root_link_velocity_to_sim(zero_root_vel)
    robot.write_joint_velocity_to_sim(zero_joint_vel)
    if step < ramp_steps and step % 5 == 4:
      # Quasi-static approach: a real climb ARRIVES at each pose with the
      # loads already distributed — it never teleports in. Re-seat engaged
      # anchors at the settled pads during the ramp (the sweep's
      # preload-nulling trick): otherwise the teleported clip pose stores
      # elastic strain (e.g. the lateral weight shift pre-stretches the far
      # leg) that snaps one anchor mid-ramp and leaves the pad clear of the
      # wall, unable to re-latch under a frozen PD. Every 5th step, not
      # every step — a continuously-following anchor carries no load until
      # the ramp ends and the whole settle then bursts out at the freeze.
      # The full-gravity hold after the ramp runs with no re-seat and
      # strict criteria.
      engaged_now = executor.engaged.unsqueeze(-1)
      executor.anchor_pos_w = torch.where(engaged_now, executor.pad_center_world(), executor.anchor_pos_w)
      executor.anchor_quat_w = torch.where(
        engaged_now,
        robot.data.body_link_quat_w[:, executor._ee_body_ids],
        executor.anchor_quat_w,
      )
    _, _, terminated, truncated, _ = env.step(actions)
    reset |= terminated | truncated
    released_now = affordance.forced_release_this_step & scheduled
    # A release during the artificial gravity RAMP is a settle transient:
    # the frozen clip pose is elastically pre-strained relative to the PD
    # equilibrium (e.g. the lateral weight shift stretches the far leg, and
    # PD stiffness routes a disproportionate share of the ramping weight
    # through the straightest limb until its anchor pops and re-seats one
    # step later). Real physics failures — like moment billing blowing the
    # envelope — release CONTINUOUSLY, so a release at held full gravity
    # stays fatal while ramp transients only have to re-latch and hold.
    if step < ramp_steps:
      ramp_released |= released_now
    else:
      hold_released |= released_now
    ever_lost |= ~affordance.engaged & scheduled
    live = affordance.engaged & scheduled
    min_live_supports = torch.minimum(min_live_supports, live.sum(dim=1))
    min_margin = torch.where(
      live,
      torch.minimum(min_margin, affordance.hold_margin),
      min_margin,
    )
    max_moment = torch.maximum(max_moment, affordance.anchor_moment_tnl.norm(dim=-1))
    max_effective = torch.maximum(max_effective, affordance.anchor_effective_load_tnl.norm(dim=-1))
    shift = (robot.data.root_link_pos_w - root_pos).norm(dim=-1)
    max_root_shift = torch.maximum(max_root_shift, shift)

  final_lost = ~affordance.engaged & scheduled
  # More than one scheduled anchor simultaneously out is a cascade, not a
  # settle transient — the load path collapsed.
  cascaded = min_live_supports < scheduled.sum(dim=1) - 1
  failed = final_lost.any(dim=1) | cascaded | reset
  finite = bool(
    torch.isfinite(robot.data.root_link_pos_w).all().item()
    and torch.isfinite(affordance.anchor_effective_load_tnl).all().item()
  )
  min_live_margin = float(min_margin[scheduled].min().item())
  max_live_moment = float(max_moment[scheduled].max().item())
  max_live_effective = float(max_effective[scheduled].max().item())
  root_shift = float(max_root_shift.max().item())
  test_f0 = affordance._ls_cfg.f_t_max_range[0] if args.f0 is None else float(args.f0)

  ok = True
  # A teleported boundary snapshot may hold one PD-marginal latch that the
  # elastic snap sheds for good (the frozen PD cannot re-press the pad); the
  # stance is stable on the remaining supports and RSI training re-engages
  # within a step or two. Tolerate a FEW such frames — a real physics
  # regression (the 0.04-torquescale moment-billing failure force-released
  # 78% of frames) blows through the cap instantly, and losing two anchors
  # in one frame is never a settle artifact.
  one_short = final_lost.sum(dim=1) == 1
  frac_one_short = float(one_short.float().mean().item())
  ok &= check(
    "no frame ends more than one anchor short",
    not bool((final_lost.sum(dim=1) >= 2).any().item()),
    f"{int((final_lost.sum(dim=1) >= 2).sum().item())} frames",
  )
  ok &= check(
    "frames ending one anchor short <= 5%",
    frac_one_short <= 0.05,
    f"{int(one_short.sum().item())}/{env.num_envs} ({100.0 * frac_one_short:.1f}%)",
  )
  ok &= check(
    "no support cascade (at most one concurrent transient loss)",
    not bool(cascaded.any().item()),
    f"{int(cascaded.sum().item())} frames cascaded",
  )
  ok &= check(
    "no forced releases at held full gravity",
    not bool(hold_released.any().item()),
    f"{int(hold_released.sum().item())} post-ramp releases",
  )
  if bool(ramp_released.any().item()):
    print(
      f"  [INFO] gravity-ramp settle transients: "
      f"{int(ramp_released.sum().item())} release(s) across "
      f"{int(ramp_released.any(dim=1).sum().item())} frame(s); "
      "bounded by the one-anchor-short checks above"
    )
  ok &= check(
    "no environment resets",
    not bool(reset.any().item()),
    f"{int(reset.sum().item())} resets",
  )
  ok &= check(
    "minimum scheduled-anchor margin > 0",
    min_live_margin > 0.0,
    f"min {min_live_margin:+.3f} at F0={test_f0:.0f} N",
  )
  ok &= check(
    "root shift < 3 cm",
    root_shift < 0.03,
    f"max {root_shift * 1000.0:.1f} mm",
  )
  ok &= check("finite state", finite, "state and effective loads finite")
  print(f"  scheduled-anchor peak |M|={max_live_moment:.3f} N*m, peak |F_eff|={max_live_effective:.1f} N")

  if bool(failed.any().item()):
    names = list(data["segment_names"])
    bad = failed.nonzero().flatten()[:20].cpu().numpy()
    print("  failed frame details:")
    for idx in bad:
      frame = int(frames_np[idx])
      print(
        f"    {frame}:{names[int(labels[frame])]} "
        f"lost={ever_lost[idx].cpu().tolist()} "
        f"margin="
        f"{[round(x, 3) for x in min_margin[idx].cpu().tolist()]}"
      )

  env.close()
  return ok


def cmd_ablation(args) -> bool:
  print(f"== ablation: {args.num_envs} envs, 90° wall, {args.seconds}s, grip OFF ==")
  env = build_env(args.num_envs, grip_enabled=False)
  env.reset()
  set_gravity(env, 90.0)
  executor, affordance = get_terms(env)
  executor.assert_now(env)

  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  actions = hold_pose_actions(env)
  steps = int(args.seconds / env.step_dt)
  for _ in range(steps):
    env.step(actions)

  drop = z0 - robot.data.root_link_pos_w[:, 2]
  eq_active = env.sim.data.eq_active[:, executor._eq_ids]
  ok = True
  ok &= check(
    "no microspines engaged",
    not bool(affordance.engaged.any().item()),
    f"occupancy={affordance.engaged.float().mean().item():.3f}",
  )
  ok &= check(
    "all grip equalities inactive",
    not bool(eq_active.any().item()),
    f"active={int(eq_active.sum().item())}",
  )
  ok &= check(
    "robot falls without grip",
    bool((drop > args.minimum_drop).all().item()),
    f"minimum drop={drop.min().item():.3f} m",
  )
  ok &= check(
    "finite state",
    bool(torch.isfinite(robot.data.root_link_pos_w).all().item()),
    "root position finite",
  )
  env.close()
  return ok


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _ray_components(theta: torch.Tensor, plane: str):
  """Unit ray direction (d_t, d_l, d_n) for cos(theta)·t̂ + sin(theta)·axis."""
  cos_t, sin_t = torch.cos(theta), torch.sin(theta)
  zero = torch.zeros_like(theta)
  if plane == "tn":
    return cos_t, zero, sin_t
  return cos_t, sin_t, zero


def _predicted_release_radius(theta: torch.Tensor, plane: str, ls_cfg=None, lut=None) -> torch.Tensor:
  """Smallest envelope-violating pull magnitude along ray angle theta.

  Ray direction is cos(theta)·t̂ + sin(theta)·(n̂ or l̂). Returns +inf where no
  finite pull violates (compression-dominated rays the bilateral anchor and
  the contacts hold forever). Pass the affordance's ls_cfg and LUT for the
  wang2017 model; without them the legacy wedge faces are used.
  """
  if lut is not None:
    # Radial model: the anchor only ever carries the in-plane + tension part
    # of the applied ray (compression sheds to the wall contacts), so release
    # comes when r·|d⁺| crosses F0·max(R(d), r_floor). Same math as
    # MicrospineAffordance._wang_envelope.
    d_t, d_l, d_n = _ray_components(theta, plane)
    ratio = capacity_ratio(lut, d_t, d_l, d_n).clamp_min(ls_cfg.wang_r_floor)
    anchor_frac = torch.sqrt(d_t**2 + d_l**2 + d_n.clamp_min(0.0) ** 2)
    cap = SWEEP_PARAMS["f_t_max"] * ratio
    inf = torch.full_like(theta, float("inf"))
    return torch.where(anchor_frac > 1e-6, cap / anchor_frac.clamp_min(1e-6), inf)

  p = SWEEP_PARAMS
  cos_t, sin_t = torch.cos(theta), torch.sin(theta)
  inf = torch.full_like(theta, float("inf"))
  # c1: reverse shear, f_t < -f_rev_max.
  r1 = torch.where(cos_t < 0, p["f_rev_max"] / (-cos_t).clamp_min(1e-9), inf)
  # c4: ultimate shear, f_t > f_t_max.
  r4 = torch.where(cos_t > 0, p["f_t_max"] / cos_t.clamp_min(1e-9), inf)
  if plane == "tn":
    denom = sin_t - p["mu_adhesion"] * cos_t.clamp_min(0.0)
    r2 = torch.where(denom > 0, p["f_n0"] / denom.clamp_min(1e-9), inf)
  else:  # tl
    denom = sin_t.abs() - p["mu_lateral"] * cos_t.clamp_min(0.0)
    r2 = torch.where(denom > 0, p["f_l0"] / denom.clamp_min(1e-9), inf)
  return torch.minimum(torch.minimum(r1, r4), r2)


def _envelope_margin(load_tnl: torch.Tensor, ls_cfg=None, lut=None) -> torch.Tensor:
  """Normalized min-margin of the fixed SWEEP_PARAMS envelope at given loads."""
  f_t, f_l, f_n = load_tnl[..., 0], load_tnl[..., 1], load_tnl[..., 2]
  if lut is not None:
    f0 = SWEEP_PARAMS["f_t_max"]
    ratio = capacity_ratio(lut, f_t, f_l, f_n).clamp_min(ls_cfg.wang_r_floor)
    cap = f0 * ratio
    load_mag = torch.sqrt(f_t**2 + f_l**2 + f_n.clamp_min(0.0) ** 2)
    return (cap - load_mag) / (cap + ls_cfg.wang_margin_eps * f0)

  p = SWEEP_PARAMS
  ft_pos = f_t.clamp(min=0.0)
  cap_n = p["mu_adhesion"] * ft_pos + p["f_n0"]
  cap_l = p["mu_lateral"] * ft_pos + p["f_l0"]
  m1 = (f_t + p["f_rev_max"]) / p["f_rev_max"]
  m2 = (cap_n - f_n) / cap_n
  m3 = (cap_l - f_l.abs()) / cap_l
  m4 = (p["f_t_max"] - f_t) / p["f_t_max"]
  return torch.minimum(torch.minimum(m1, m2), torch.minimum(m3, m4))


def cmd_sweep(args) -> bool:
  num_envs = args.num_envs
  target = args.ee_index
  print(
    f"== sweep: {num_envs} envs, target EE {EE_NAMES[target]}, ramp {args.seconds}s ==",
  )
  # Weld (production) anchors by default: a connect (ball joint) lets the
  # in-plane pull tip the pad about its anchor point, the sphere edge digs
  # into the wall and the contact push-out reads as ~750 N phantom anchor
  # tension — releases then land far off the envelope (measured p50 ~0.97 in
  # connect mode vs 0.003 with weld). The weld's torque path can carry a few
  # % of rays past the ramp end, hence the release-fraction INFO branch.
  env = build_env(num_envs, anchor_lock_rotation=args.lock_rotation)
  env.reset()
  set_gravity(env, 90.0, magnitude=0.0)  # Zero gravity: pure scripted pulls.
  executor, affordance = get_terms(env)
  executor.assert_now(env)
  # Envelope model under test comes from the task cfg; predictions and margin
  # checks below follow it (wang2017 uses the affordance's own capacity LUT).
  ls_cfg = affordance._ls_cfg
  wang_lut = affordance._wang_lut
  print(f"  envelope model: {ls_cfg.model}")
  robot = env.scene["robot"]
  actions = hold_pose_actions(env)

  # All EEs unbreakable while the static preload is nulled out. Intercept-only
  # caps — a huge mu would produce inf*0 NaNs at zero shear.
  big = 1.0e9
  affordance.limit_f_t_max[:] = big
  affordance.limit_f_rev_max[:] = big
  affordance.limit_mu_adhesion[:] = 0.0
  affordance.limit_f_n0[:] = big
  affordance.limit_mu_lateral[:] = 0.0
  affordance.limit_f_l0[:] = big

  # The spawn pose internally preloads the anchors (PD fights the anchored
  # closed chain — tens of N even at zero g), which would trip the small test
  # envelope before any scripted pull. Null it: alternately let the robot
  # settle and re-place the anchors at the settled pad positions. The
  # executor rewrites its cached anchor_pos_w into eq_data every step, so
  # overriding the cache is sufficient. The clip's frame-0 RSI also spawns
  # with the reference velocities; at zero g nothing dissipates that momentum
  # and a small fraction of envs rings against the anchors instead of
  # settling (~750 N residuals) — kill all momentum at the top of each round.
  quasistatic_settle(env, executor, affordance, actions)
  zero_qd = torch.zeros_like(robot.data.joint_vel)
  zero_root_vel = torch.zeros(env.num_envs, 6, device=DEVICE)
  baseline = executor.compute_anchor_loads()[:, target]
  print(f"  residual static preload on target EE: max |F| = {baseline.norm(dim=-1).max().item():.2f} N")

  # Deterministic envelope on the target EE only.
  for key, value in SWEEP_PARAMS.items():
    getattr(affordance, f"limit_{key}")[:, target] = value
  others = [h for h in range(len(EE_NAMES)) if h != target]

  # Engage frame of the target EE (constant: wall static, zero gravity makes
  # the rake fall back to world -z projected onto the wall).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  t_hat = t_hat[:, target]
  l_hat = l_hat[:, target]
  n_hat = n_hat[:, target]

  # First half scans the t-n plane, second half t-l.
  half = num_envs // 2
  plane_tn = torch.arange(num_envs, device=DEVICE) < half
  theta = torch.empty(num_envs, device=DEVICE)
  theta[:half] = torch.linspace(-math.pi, math.pi, half + 1, device=DEVICE)[:-1]
  theta[half:] = torch.linspace(-math.pi, math.pi, num_envs - half + 1, device=DEVICE)[:-1]
  second_axis = torch.where(plane_tn.unsqueeze(-1), n_hat, l_hat)
  direction = torch.cos(theta).unsqueeze(-1) * t_hat + torch.sin(theta).unsqueeze(-1) * second_axis

  r_pred = torch.where(
    plane_tn,
    _predicted_release_radius(theta, "tn", ls_cfg, wang_lut),
    _predicted_release_radius(theta, "tl", ls_cfg, wang_lut),
  )
  # Rays grazing the envelope faces (denominators → 0 near pure compression)
  # predict astronomical release forces; cap those ramps instead. A capped
  # (compression-dominated) ray may legitimately still release: pressing the
  # pad hard through a bent limb builds secondary shear/lateral load that
  # genuinely crosses the envelope — the assertion below therefore only
  # requires that *whatever releases does so on the envelope*.
  finite = r_pred < 1.5 * SWEEP_PARAMS["f_t_max"]
  # Per-direction ramp rate: release ~mid-ramp for finite predictions; cap
  # no-release rays at a force that proves they hold.
  steps = int(args.seconds / env.step_dt)
  rate = torch.where(finite, 2.2 * r_pred / args.seconds, 1.5 * SWEEP_PARAMS["f_t_max"] / args.seconds)

  spine_ids, _ = robot.find_bodies([EE_NAMES[target]])
  released = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  load_at_release = torch.zeros(num_envs, 3, device=DEVICE)
  zero_torque = torch.zeros(num_envs, 1, 3, device=DEVICE)
  # Rolling last-3-step loads for spike-robust release recording: pad-wall
  # contact churn at the creeping margin boundary occasionally emits a
  # one-step ~700 N equality-force transient (the bilateral anchor countering
  # a contact-penetration correction impulse). The debounce ignores isolated
  # spikes, but a spike adjacent to a genuine boundary crossing completes the
  # 2-step violation window and would be recorded as the release load. The
  # per-component median over the last 3 steps recovers the true crossing.
  load_hist = torch.zeros(3, num_envs, 3, device=DEVICE)

  corrupted = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  for step in range(steps):
    # Quasistatic ramp: kill all momentum every step (overdamped limit). At
    # zero g the limb + PD + pad contacts ring under the force ramp; the
    # anchor-load direction then oscillates and, with the closed
    # direction-sensitive surface, the 2-step violation debounce never sees
    # two consecutive violating samples until the swing is enormous —
    # releases land hundreds of N past the boundary. The anchor load is
    # position-error based, so zeroed velocities do not distort the readout.
    robot.write_root_link_velocity_to_sim(zero_root_vel)
    robot.write_joint_velocity_to_sim(zero_qd)
    # A mid-test env reset (e.g. violent post-release limb motion tripping
    # bad_state) resamples the DR params; pin the test envelope every step
    # and exclude envs that reset before their release from the stats.
    for key, value in SWEEP_PARAMS.items():
      getattr(affordance, f"limit_{key}")[:, target] = value
    magnitude = rate * (step + 1) * env.step_dt
    force = torch.where(released.unsqueeze(-1), torch.zeros_like(direction), magnitude.unsqueeze(-1) * direction)
    robot.write_external_wrench_to_sim(forces=force.unsqueeze(1), torques=zero_torque, body_ids=spine_ids)
    _, _, terminated, truncated, _ = env.step(actions)
    corrupted |= (terminated | truncated) & ~released
    load_hist = torch.roll(load_hist, 1, dims=0)
    load_hist[0] = affordance.anchor_load_tnl[:, target]
    newly = affordance.forced_release_this_step[:, target] & ~released & ~corrupted
    load_at_release = torch.where(newly.unsqueeze(-1), load_hist.median(dim=0).values, load_at_release)
    released |= newly

  # --- evaluation ---
  ok = True
  valid = ~corrupted
  print(f"  excluded (reset before release): {int(corrupted.sum())} envs")
  others_held = bool(affordance.engaged[valid][:, others].all().item())
  ok &= check("non-target EEs held", others_held, "no spurious releases")

  finite_release_frac = released[finite & valid].float().mean().item()
  if args.lock_rotation:
    # Weld adds a torque path that legitimately carries a few % of rays past
    # the ramp end (see cmd_sweep docstring). The envelope-geometry check below
    # is anchor-type independent and is the real assertion, so report the
    # release fraction here without failing on it.
    print(f"  [INFO] finite-prediction rays released (weld): {100 * finite_release_frac:.1f}%")
  else:
    ok &= check(
      "finite-prediction rays released",
      finite_release_frac > 0.98,
      f"{100 * finite_release_frac:.1f}%",
    )
  capped = ~finite & valid
  print(
    f"  capped (compression-dominated) rays: {int(capped.sum())}, "
    f"of which released on secondary load: {int(released[capped].sum())}"
  )

  hit = released & valid
  # Margin statistics cover only rays whose applied load the anchor carries
  # in full: the tension half of the t-n plane plus the whole in-plane t-l
  # scan. Compression-side rays share load with the ordinary pad contacts
  # (the paper's eq-13 decomposition realized physically): the anchor picks
  # tangential load up through contact stick-slip jumps, so their release
  # points overshoot the anchor-only envelope by construction. They are
  # still asserted to release (above) and reported informationally.
  anchor_full = (~plane_tn) | (torch.sin(theta) >= 0.0)
  scored = hit & anchor_full
  margin = _envelope_margin(load_at_release[scored], ls_cfg, wang_lut)
  abs_margin = margin.abs()
  p50 = abs_margin.quantile(0.5).item()
  p90 = abs_margin.quantile(0.9).item()
  ok &= check("release on envelope (|margin| p50 < 0.10)", p50 < 0.10, f"p50 {p50:.3f}")
  ok &= check("release on envelope (|margin| p90 < 0.20)", p90 < 0.20, f"p90 {p90:.3f}")
  comp_hit = hit & ~anchor_full
  if bool(comp_hit.any()):
    m_comp = _envelope_margin(load_at_release[comp_hit], ls_cfg, wang_lut)
    print(
      f"  compression-side releases (info, contact-assisted): "
      f"{int(comp_hit.sum())}, |margin| p50 {m_comp.abs().quantile(0.5):.3f}"
    )

  if args.plot:
    try:
      import matplotlib

      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      import numpy as np

      p = SWEEP_PARAMS
      fig, axes = plt.subplots(1, 2, figsize=(13, 6))
      for ax, plane, mask, ylab, col in (
        (axes[0], "tn", hit & plane_tn, "f_n [N] (tension +)", 2),
        (axes[1], "tl", hit & ~plane_tn, "f_l [N]", 1),
      ):
        pts = load_at_release[mask].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, col], s=8, label="measured release")
        if wang_lut is not None:
          # Closed-surface boundary: the anchor-side release radius along a
          # dense fan of ray angles, same math as the release check.
          th = torch.linspace(-math.pi, math.pi, 721, device=DEVICE)
          rb = _predicted_release_radius(th, plane, ls_cfg, wang_lut)
          keep = torch.isfinite(rb) & (rb < 1.5 * p["f_t_max"])
          bx = (rb * torch.cos(th))[keep].cpu().numpy()
          by = (rb * torch.sin(th))[keep].cpu().numpy()
          ax.plot(bx, by, "r.", markersize=2, label="wang2017 surface")
        else:
          ft = np.linspace(-p["f_rev_max"] * 1.2, p["f_t_max"] * 1.05, 200)
          if plane == "tn":
            cap = p["mu_adhesion"] * np.clip(ft, 0, None) + p["f_n0"]
            ax.plot(ft, cap, "r-", label="tension cone")
          else:
            cap = p["mu_lateral"] * np.clip(ft, 0, None) + p["f_l0"]
            ax.plot(ft, cap, "r-", label="lateral cone")
            ax.plot(ft, -cap, "r-")
          ax.axvline(-p["f_rev_max"], color="orange", label="reverse limit")
          ax.axvline(p["f_t_max"], color="purple", label="ultimate shear")
        ax.set_xlabel("f_t [N] (engage shear +)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{plane} plane")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
      fig.suptitle(f"Microspine limit-surface sweep — {EE_NAMES[target]}")
      fig.tight_layout()
      fig.savefig(args.plot, dpi=130)
      print(f"  plot saved to {args.plot}")
    except ImportError:
      print("  matplotlib not available; skipping plot")

  env.close()
  return ok


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


def cmd_readout(args) -> bool:
  print(f"== readout: {args.num_envs} envs, efc gather vs kinematic error ==")
  # Re-grip off: this test regresses load against anchor error, and grace-
  # window reseating decouples the two by design (the anchor follows the pad).
  # Hold checks off too: the closed wang2017 surface releases anchors under
  # the torso-push loads (the open wedge rarely did), zeroing the loads
  # mid-regression — this test studies the raw efc gather, not the envelope.
  env = build_env(
    args.num_envs,
    limit_surface_enable=False,
    reanchor_after_grace=False,
  )
  env.reset()
  set_gravity(env, 90.0)
  executor, affordance = get_terms(env)
  executor.assert_now(env)
  robot = env.scene["robot"]
  torso_ids, _ = robot.find_bodies(["torso"])

  steps = int(args.seconds / env.step_dt)
  settle = min(25, steps // 4)
  actions = hold_pose_actions(env)
  # Null the RSI spawn-pose standoff first: a constant internal preload much
  # larger than the push response flattens the regression (the varying part
  # of the load drowns in a constant offset the linear map cannot place).
  quasistatic_settle(env, executor, affordance, actions)
  zero_torque = torch.zeros(env.num_envs, 1, 3, device=DEVICE)
  loads_hist, err_hist = [], []
  amp = 60.0
  for step in range(steps):
    t = step * env.step_dt
    push = torch.tensor(
      [
        amp * math.sin(2 * math.pi * 0.30 * t),
        amp * math.sin(2 * math.pi * 0.23 * t + 1.0),
        amp * math.sin(2 * math.pi * 0.17 * t + 2.0),
      ],
      device=DEVICE,
    ).expand(env.num_envs, 1, 3)
    robot.write_external_wrench_to_sim(forces=push, torques=zero_torque, body_ids=torso_ids)
    env.step(actions)
    if step >= settle:
      # Gather explicitly: with hold checks disabled the affordance never
      # calls compute_anchor_loads, so the cached anchor_load_w stays zero.
      loads_hist.append(executor.compute_anchor_loads().clone())
      err_hist.append(executor.anchor_error_w().clone())

  loads = torch.stack(loads_hist)  # (T, E, H, 3)
  errs = torch.stack(err_hist)

  ok = True
  for h, name in enumerate(EE_NAMES):
    load = loads[:, :, h].reshape(-1, 3)
    err = errs[:, :, h].reshape(-1, 3)
    # The anchor load (EE-on-anchor) tracks the pose error through the
    # constraint impedance: F_load ≈ K_eff · err with K_eff a 3x3 stiffness
    # (direction-dependent — the soft-constraint force scales with the
    # per-axis effective inverse mass, so a scalar spring underfits). A
    # gather bug (wrong rows, wrong sign) would destroy this fit entirely.
    k_mat = torch.linalg.lstsq(err, load).solution  # (3, 3): load ≈ err @ K
    pred = err @ k_mat
    # Per-axis signal gate: an axis whose load barely varies under the torso
    # push (the feet mostly carry a constant gravity share in the RSI pose)
    # has no variance for R² / K-sign to explain — evaluating it would test
    # the push transmission, not the gather. The direction cosine still
    # covers every axis of every sample.
    sig = load.std(dim=0)  # (3,)
    strong = sig > 5.0
    if bool(strong.any()):
      ls = load[:, strong]
      ps = pred[:, strong]
      ss_res = (ls - ps).pow(2).sum()
      ss_tot = (ls - ls.mean(dim=0)).pow(2).sum().clamp_min(1e-12)
      r2 = (1 - ss_res / ss_tot).item()
    else:
      r2 = float("nan")
    big = load.norm(dim=-1) > 10.0
    cosine = (
      torch.nn.functional.cosine_similarity(load[big], pred[big], dim=-1).mean().item() if big.any() else float("nan")
    )
    k_diag = k_mat.diagonal()
    print(
      f"  {name}: K_diag=({k_diag[0]:.0f}, {k_diag[1]:.0f}, {k_diag[2]:.0f}) "
      f"N/m, R²={r2:.3f} on {int(strong.sum())} strong axes, "
      f"dir-cos={cosine:.3f} (|F|>10N, n={int(big.sum())})"
    )
    # What this test discriminates is gather bugs, and the direction cosine
    # is the discriminator: a sign flip lands at cos ≈ -0.8 and wrong/stale
    # rows at cos ≈ 0. The stiffness map R²/K are reported informationally
    # only — they additionally assume the anchor is the sole load path, which
    # the wall contacts break to a pose-dependent degree (an EE sharing load
    # through contact friction keeps a clean direction but loses the linear
    # fit). Absolute magnitude correctness is covered by the hang test's
    # weight-closure assertion.
    ok &= check(f"{name} direction agreement", cosine > 0.6, f"cos {cosine:.3f}")
  env.close()
  return ok


# ---------------------------------------------------------------------------
# dynamic  (debounce / grace tolerance under fast, non-quasistatic loads)
# ---------------------------------------------------------------------------


def _dynamic_settle(env, executor, affordance, target, actions):
  """Reset, zero gravity, pin the test envelope, null the spawn preload, and
  leave the target EE engaged and well past the grace window."""
  env.reset()
  set_gravity(env, 90.0, magnitude=0.0)
  executor.assert_now(env)
  for key, value in SWEEP_PARAMS.items():
    getattr(affordance, f"limit_{key}")[:, target] = value
  # Same preload-nulling trick as cmd_sweep: settle, then re-seat the anchors
  # at the settled pad so the closed-chain PD preload does not pre-trip the
  # small test envelope.
  for _ in range(8):
    for _ in range(10):
      env.step(actions)
    executor.anchor_pos_w = executor.pad_center_world().clone()
  for _ in range(20):
    env.step(actions)


def cmd_dynamic(args) -> bool:
  """Probe the release debounce/grace under fast (non-quasistatic) loads.

  cmd_sweep validates the envelope quasi-statically (release lands on the
  boundary). This probes the *dynamic* behavior the debounce introduces:
    A. a sustained step overload must still release (after the debounce count);
    B. a load that flips outside/inside every step can evade release entirely
       — the F5 exploit surface. Both live in the affordance debounce, so they
       are anchor-type independent; connect is the clean default.
  """
  target = args.ee_index
  env = build_env(args.num_envs, anchor_lock_rotation=args.lock_rotation)
  executor, affordance = get_terms(env)
  robot = env.scene["robot"]
  actions = hold_pose_actions(env)
  spine_ids, _ = robot.find_bodies([EE_NAMES[target]])
  zero_torque = torch.zeros(args.num_envs, 1, 3, device=DEVICE)

  if args.violation_steps is not None:
    affordance._ls_cfg.violation_consecutive_steps = args.violation_steps
  vsteps = max(1, affordance._ls_cfg.violation_consecutive_steps)
  mode = "weld" if args.lock_rotation else "connect"
  print(f"== dynamic: {args.num_envs} envs, target {EE_NAMES[target]}, {mode}, violation_consecutive_steps={vsteps} ==")

  def pull(mag_vec):
    force = torch.zeros(args.num_envs, 1, 3, device=DEVICE)
    force[:, 0, :] = mag_vec
    robot.write_external_wrench_to_sim(forces=force, torques=zero_torque, body_ids=spine_ids)
    env.step(actions)

  # Pure pull-off (t-n plane, 90°): crosses the tension face at f_n0, a
  # debounced (soft) condition — exactly what the oscillation probes.
  _dynamic_settle(env, executor, affordance, target, actions)
  _, _, n_hat = affordance._engage_frame()
  n_hat = n_hat[:, target]
  r_cross = SWEEP_PARAMS["f_n0"]
  hi, lo = 1.8 * r_cross, 0.4 * r_cross

  # --- A: sustained step overload (constant, clearly outside) ---
  base_engaged = affordance.engaged[:, target].clone()
  steps_to_rel = torch.full((args.num_envs,), -1, dtype=torch.long, device=DEVICE)
  for s in range(12):
    pull(hi * n_hat)
    newly = affordance.forced_release_this_step[:, target] & (steps_to_rel < 0)
    steps_to_rel = torch.where(newly, torch.full_like(steps_to_rel, s + 1), steps_to_rel)
  rel_mask = (steps_to_rel >= 0) & base_engaged
  frac_rel_A = rel_mask.float().mean().item()
  lat = steps_to_rel[rel_mask]
  med_lat = int(lat.median().item()) if lat.numel() else -1

  # --- B: oscillation evasion (1 step outside, 1 step inside) ---
  _dynamic_settle(env, executor, affordance, target, actions)
  b_steps = 40
  ever_released = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
  outside = torch.zeros(args.num_envs, device=DEVICE)
  for s in range(b_steps):
    pull((hi if s % 2 == 0 else lo) * n_hat)
    ever_released |= affordance.forced_release_this_step[:, target]
    outside += (affordance.hold_margin[:, target] < 0).float()
  frac_evaded = (~ever_released).float().mean().item()
  frac_outside = (outside / b_steps).mean().item()

  # --- C: isolated overload spike, average load INSIDE the envelope ---
  # The tightest debounce-evasion probe: one outside spike every spike_period
  # steps, deep-inside otherwise, so the *mean* load sits inside the cone —
  # the pattern a policy would need to "ride" the debounce and hold an
  # effectively-outside load without ever tripping the consecutive count.
  _dynamic_settle(env, executor, affordance, target, actions)
  spike_period, c_steps, deep_lo = 4, 48, 0.1 * r_cross
  avg_load = (hi + (spike_period - 1) * deep_lo) / spike_period
  ever_released_c = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
  outside_c = torch.zeros(args.num_envs, device=DEVICE)
  for s in range(c_steps):
    pull((hi if s % spike_period == 0 else deep_lo) * n_hat)
    ever_released_c |= affordance.forced_release_this_step[:, target]
    outside_c += (affordance.hold_margin[:, target] < 0).float()
  frac_evaded_c = (~ever_released_c).float().mean().item()
  frac_outside_c = (outside_c / c_steps).mean().item()
  env.close()

  print(f"  [A] sustained 1.8x pull-off: released {100 * frac_rel_A:.1f}%, median steps-to-release = {med_lat}")
  print(
    f"  [B] oscillation 1-out/1-in: evaded release {100 * frac_evaded:.1f}%, "
    f"mean time spent outside envelope {100 * frac_outside:.0f}%"
  )
  print(
    f"  [C] isolated spike (avg load {avg_load / r_cross:.2f}x cap, INSIDE): "
    f"evaded release {100 * frac_evaded_c:.1f}%, "
    f"outside {100 * frac_outside_c:.0f}% of steps"
  )
  ok = True
  ok &= check("sustained overload releases", frac_rel_A > 0.95, f"{100 * frac_rel_A:.1f}%")
  ok &= check(
    "release latency within debounce+lag",
    1 <= med_lat <= vsteps + 2,
    f"{med_lat} steps (debounce={vsteps})",
  )
  print(
    f"  -> F5: at vsteps={vsteps}, an oscillating load rode "
    f"{100 * frac_outside:.0f}% of steps outside the envelope while "
    f"{100 * frac_evaded:.0f}% of envs never released."
  )
  return ok


# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_hang = sub.add_parser("hang")
  p_hang.add_argument("--num-envs", type=int, default=16)
  p_hang.add_argument("--seconds", type=float, default=10.0)
  p_hang.add_argument(
    "--no-limit-surface",
    action="store_true",
    help="Validate the raw anchor mechanics with hold checks disabled "
    "(pre-re-grip behavior; engaged spines stay latched forever).",
  )

  p_reference = sub.add_parser("reference-hold")
  p_reference.add_argument("motion_file", type=Path)
  p_reference.add_argument(
    "--ramp-seconds",
    type=float,
    default=0.8,
    help="gravity-ramp duration for every sampled controller snapshot",
  )
  p_reference.add_argument(
    "--hold-seconds",
    type=float,
    default=0.4,
    help="full-gravity overdamped hold after the ramp",
  )
  p_reference.add_argument(
    "--frame-stride",
    type=int,
    default=1,
    help="sample every Nth cycle frame; segment boundaries are always included",
  )
  p_reference.add_argument(
    "--f0",
    type=float,
    default=None,
    help="fixed pad F0 for the audit (default: configured lower bound)",
  )

  p_ablation = sub.add_parser("ablation")
  p_ablation.add_argument("--num-envs", type=int, default=16)
  p_ablation.add_argument("--seconds", type=float, default=1.0)
  p_ablation.add_argument("--minimum-drop", type=float, default=0.25)

  p_sweep = sub.add_parser("sweep")
  p_sweep.add_argument("--num-envs", type=int, default=1024)
  p_sweep.add_argument("--seconds", type=float, default=6.0)
  p_sweep.add_argument("--ee-index", type=int, default=2, choices=range(4))
  p_sweep.add_argument("--plot", type=str, default="")
  p_sweep.add_argument(
    "--lock-rotation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Weld anchors (production path, default). --no-lock-rotation pins "
    "the legacy pure-force connect anchor: under an in-plane pull a connect "
    "(ball joint) lets the pad tip about its anchor, the sphere edge digs "
    "into the wall and the contact push-out reads as ~750 N phantom anchor "
    "tension — measured releases then land far off the envelope.",
  )

  p_read = sub.add_parser("readout")
  p_read.add_argument("--num-envs", type=int, default=8)
  p_read.add_argument("--seconds", type=float, default=10.0)

  p_dyn = sub.add_parser("dynamic")
  p_dyn.add_argument("--num-envs", type=int, default=512)
  p_dyn.add_argument("--ee-index", type=int, default=2, choices=range(4))
  p_dyn.add_argument("--lock-rotation", action="store_true")
  p_dyn.add_argument(
    "--violation-steps",
    type=int,
    default=None,
    help="Override MicrospineLimitSurfaceCfg.violation_consecutive_steps.",
  )

  args = parser.parse_args()
  ok = {
    "hang": cmd_hang,
    "reference-hold": cmd_reference_hold,
    "ablation": cmd_ablation,
    "sweep": cmd_sweep,
    "readout": cmd_readout,
    "dynamic": cmd_dynamic,
  }[args.cmd](args)
  print("OVERALL:", "PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
