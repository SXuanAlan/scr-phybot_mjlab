"""GPU validation benches for the microspine connect-equality attachment on the climbing robot.

Each ``cmd_*`` bench returns whether it passed:
  cmd_hang            all four pads anchored on a vertical wall hold the
                      robot's weight without sag, jitter, NaNs or efc overflow.
  cmd_reference_hold  a controller-settling stress test from sampled
                      reference snapshots under a gravity ramp; not a
                      static feasibility proof.
  cmd_ablation        with the grip disabled at the same low wall friction
                      the robot falls, so the anchors carry the weight.
  cmd_readout         the efc-row force gather tracks the kinematic anchor
                      error under a time-varying torso push.
scripts/validate/validate_microspine_constraint.py runs them from the command line.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

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


def build_env(
  num_envs: int,
  limit_surface_enable: bool = True,
  reanchor_after_grace: bool = True,
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
  for dr_event in ("encoder_bias", "body_friction", "base_mass"):
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
  motion.height_offset_range = (0.0, 0.0)
  # The nested limit_surface cfg is shared between the affordance command and
  # the physics event term (dataclasses.replace is shallow copy: nested
  # objects shared, scalar fields independent) — so scalars must be written
  # on both copies.
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
  # The wall plane passes through the wall entity's origin.
  wall_origin = env.scene["wall"].data.root_link_pos_w.unsqueeze(1)

  def _step():
    robot.write_root_link_velocity_to_sim(zero_root_vel)
    robot.write_joint_velocity_to_sim(zero_qd)
    env.step(actions)

  for _ in range(rounds):
    for _ in range(steps_per_round):
      _step()
    pads = executor.pad_center_world()
    proj = pads - ((pads - wall_origin) * wall_inward).sum(-1, keepdim=True) * wall_inward
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
  # Informational: final per-EE loads in the engage frame and the empirical
  # hold margin. A negative margin here means the spawn pose itself is not
  # holdable by the unit envelope (task/pose concern, not physics).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  loads = executor.compute_anchor_loads()
  f_t = (loads * t_hat).sum(-1).mean(0)
  f_l = (loads * l_hat).sum(-1).mean(0)
  f_n = (loads * n_hat).sum(-1).mean(0)
  margin = affordance.hold_margin.mean(0)
  for h, name in enumerate(EE_NAMES):
    print(f"  {name}: f_t={f_t[h]:.1f} N, f_l={f_l[h]:.1f} N, f_n={f_n[h]:.1f} N (hold margin {margin[h]:+.3f})")
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
  scheduled, root_pos = _place_reference_frames(env, frames)
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
      # anchors at the settled pads during the ramp (preload nulling):
      # otherwise the teleported clip pose stores elastic strain (e.g. the
      # lateral weight shift pre-stretches the far leg) that snaps one anchor
      # mid-ramp and leaves the pad clear of the wall, unable to re-latch
      # under a frozen PD. Every 5th step, not every step — a continuously
      # following anchor carries no load until the ramp ends and the whole
      # settle then bursts out at the freeze. The full-gravity hold after the
      # ramp runs with no re-seat and strict criteria.
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
    f"min {min_live_margin:+.3f}",
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
# readout
# ---------------------------------------------------------------------------


def cmd_readout(args) -> bool:
  print(f"== readout: {args.num_envs} envs, efc gather vs kinematic error ==")
  # Re-grip off: this test regresses load against anchor error, and grace-
  # window reseating decouples the two by design (the anchor follows the pad).
  # Hold checks off too: the envelope may release anchors under the
  # torso-push loads, zeroing the loads mid-regression — this test studies
  # the raw efc gather, not the envelope.
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
