"""Transition-only reward variants layered over the climb battery.

Two mount-specific pressures the cycle never needed:

* ``motion_swing_approach_masked`` — a WIDE-basin tow toward the upcoming
  anchor for a subset of EEs. The cycle's swings are short, so its approach
  funnel (std 0.05) never had to pay beyond ~15 cm; the mount's raise must
  bridge the ~0.5 m horizontal->overhead gap where that funnel pays ~0 and
  v4 plateaued (no local gradient — the spawn-bias window trains AT the
  latch, this pays the path TO it). Hands only by default: a wide kernel on
  the feet would pay a standing robot ~0.5/foot for merely existing 0.35 m
  from its wall anchors and reward premature foot lifts.
* masked joint/EE tracking — the blind-delivery scatter makes the clip's ARM
  path wrong by the spawn offset while the wall anchors stay environment-
  true, so the arm tracking terms fight the correction the contact terms ask
  for. The split variants let the arms track loose (weight/2, std*2) while
  legs+root stay strict — LadderMan Eq (1) asymmetric tracking, mapped onto
  our per-term kernels. Wired behind ``asym_arm_tracking`` (default OFF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
  _engage_match_gate,
  _motion,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "engaged_arm_stillness_penalty",
  "standing_foot_skate_penalty",
  "first_anchor_press_bonus",
  "first_charge_bonus",
  "first_hold_bonus",
  "first_weld_bonus",
  "force_target_scale",
  "motion_toe_brace_target",
  "motion_approach_progress",
  "motion_anchor_dwell_wage",
  "motion_predrag_wage",
  "motion_scheduled_press_gentle",
  "motion_scheduled_support_loss",
  "motion_swing_approach_masked",
  "motion_joint_pos_error_exp_masked",
  "motion_ee_pos_error_exp_masked",
  "motion_unscheduled_press_penalty",
  "press_overdrive_penalty",
  "pad_grind_penalty",
  "seeded_anchor_rip_penalty",
  "standing_root_xy_tracking",
  "standing_wall_clearance_penalty",
)


_support_grace_state: dict = {}


def seeded_anchor_rip_penalty(
  env: ManagerBasedRlEnv,
  affordance_command_name: str = "microspine_affordance",
  command_name: str = "climb_motion",
  reverse_grace_s: float = 0.3,
) -> torch.Tensor:
  """Full-price bill for demolishing an RSI-SEEDED anchor (seed economy #1).

  The rip-bill holiday exists so the policy's OWN experimental welds are
  cheap to break; it must not also price the two gifted anchors of a
  latch-window birth at 25 cents — the 08-15 autopsy showed the policy
  blowing both seeds within 5 steps of birth (all `ultimate`), and the
  economics never opposed it. Bills at a flat, un-annealed event price:

  * FAILURE-reason releases (tension, lateral, ultimate, tilt-peel), and
  * v52: OFF-SCHEDULE reverse-shear unhooks. The polite-dump mole (v51
    @600-774: first_hold pinned at 0, LH occupancy flat, seeds gone in
    <10 steps with only 1.3% failure-reason rips) — the legal unhook
    path cost ~0.075 under the holiday, so the policy tipped the gifts
    off the wall and went back to standing. A reverse unhook stays FREE
    only when the schedule has released that EE or its falling edge is
    within ``reverse_grace_s`` (the choreography's own hand-up releases).

  DR-forced releases (reason >= 6) stay exempt — drills are not the
  policy's crime. Pair with a negative weight; divided by step_dt so the
  configured weight IS the per-event bill.
  """
  aff = env.command_manager.get_term(affordance_command_name)
  reason = aff.release_reason
  hits = aff.released_was_seeded & (reason >= 2) & (reason <= 5)
  if reverse_grace_s >= 0.0:
    motion = _motion(env, command_name)
    lead = reverse_grace_s / motion.cycle_time_s
    scheduled_off = (~motion.ref_engaged) | (
      motion.ref_edge_countdowns[..., 0] <= lead
    )
    hits = hits | (
      aff.released_was_seeded & (reason == 1) & ~scheduled_off
    )
  return hits.sum(dim=-1).float() / env.step_dt


def standing_foot_skate_penalty():
  """Factory: bill sole horizontal speed while the schedule GROUNDS a foot.

  v54 (user's video verdict: the stance shuffles and the feet skate on
  the floor). The floor plant checked out — soles-only contact, mu 0.7,
  DR-exempt, identical to the rock-solid v42 era — so the skating is a
  BEHAVIOR the deadline economy never priced: root-xy pays position, leg
  tracking pays pose, and nothing bills a foot that slides or steps. This
  does: per foot, |sole xy velocity| while that foot's wall-mount rising
  edge is not yet imminent (the swing itself stays free through the same
  edge-lead gate the tows use). Zero once the foot is scheduled on the
  wall.
  """
  state: dict[str, object] = {"ids": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    body_names: tuple[str, ...] = ("left_ankle_roll", "right_ankle_roll"),
    foot_ee_indices: tuple[int, ...] = (2, 3),
    swing_lead_time_s: float = 2.5,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    robot = env.scene["robot"]
    if state["ids"] is None:
      state["ids"] = [robot.find_bodies([n])[0][0] for n in body_names]
    speed = robot.data.body_link_lin_vel_w[:, state["ids"], :2].norm(dim=-1)
    idx = torch.as_tensor(
      foot_ee_indices, dtype=torch.long, device=speed.device
    )
    grounded = ~motion.ref_engaged[:, idx]
    lead = swing_lead_time_s / motion.cycle_time_s
    grounded = grounded & (motion.ref_edge_countdowns[:, idx, 1] > lead)
    return (speed * grounded.float()).sum(dim=-1)

  return call


def engaged_arm_stillness_penalty():
  """Factory: bill arm-joint speed WHILE that side's hand is welded.

  "焊上就别动" written into the ledger (seed economy #3): tracking terms
  price position, not calmness — a jittering arm near the reference still
  collects, and every jerk loads the anchor toward the force/moment
  envelope. Per side: the left hand's engagement gates the left
  shoulder/elbow joints, so a staggered choreography (one hand holding,
  the other mid-swing) bills only the holding side. Hands only: the foot
  mounts' shin dynamics are owned by the leg tracking terms.
  """
  state: dict[str, object] = {"ids": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    affordance_command_name: str = "microspine_affordance",
    side_joint_regex: tuple[tuple[int, tuple[str, ...]], ...] = (
      (0, (".*left_shoulder.*", ".*left_elbow.*")),
      (1, (".*right_shoulder.*", ".*right_elbow.*")),
    ),
  ) -> torch.Tensor:
    robot = env.scene["robot"]
    if state["ids"] is None:
      state["ids"] = [
        (ee, robot.find_joints(list(regex))[0])
        for ee, regex in side_joint_regex
      ]
    aff = env.command_manager.get_term(affordance_command_name)
    out = torch.zeros(env.num_envs, device=env.device)
    for ee, ids in state["ids"]:
      speed = robot.data.joint_vel[:, ids].abs().mean(dim=-1)
      out += speed * aff.engaged[:, ee].float()
    return out

  return call


def motion_scheduled_support_loss(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  event_term_name: str = "microspine_grip",
  minimum_supports: int = 2,
  grace_s: float = 1.5,
) -> torch.Tensor:
  """Support-shortfall bill whose requirement follows the SCHEDULE.

  The cycle's ``microspine_support_loss`` bills ``2 - engaged`` flat every
  step — sound on the wall, where fewer than two anchors means falling. The
  mount's standing prefix stands on the FLOOR with zero anchors, so the
  inherited flat bill became a -8/s rent on merely existing that no income
  could cover (tracking ceiling ~4.5/s) while a termination costs -10 once
  (dt-scaled). The v15-v18 ledger under that economy: every episode ended
  by base_height, return strictly worsened with lifespan, and the observed
  "behaviors" — head-butt forward, hand-shove-topple backward — were the
  two fastest exits from a stance 0.2 m from the wall. The same rent also
  funded every crutch (a two-pad prop was the one net-positive state) and
  the v11 slam rush (each 0.1 s of earlier latch saved 0.8 of rent).

  Requirement here = ``min(minimum_supports, scheduled engaged count)``:
  zero while the clip stands, ramping with the choreography, and exact
  climb parity from any wall-borne stance (>= 2 scheduled) including the
  terminal hold.
  """
  motion = _motion(env, command_name)
  engaged = env.event_manager.get_term_cfg(event_term_name).func.engaged
  required = motion.ref_engaged.sum(dim=-1).clamp(max=minimum_supports)
  shortfall = (required - engaged.sum(dim=-1)).clamp_min(0).float()
  if grace_s > 0.0:
    # v51 grace ramp (the lunge's last funding line): the bill used to
    # land at FULL price the instant the schedule opened, so at the
    # deadline a 150 N slam-weld that stops the -8/s bleed NOW rationally
    # beat a careful gentle attempt that bleeds while it works. The bill
    # now ramps linearly over ``grace_s`` after a shortfall APPEARS
    # (schedule rise or rip alike); the long-run pressure is unchanged.
    state = _support_grace_state
    since = state.get("since")
    prev = state.get("prev")
    if since is None or since.shape != shortfall.shape:
      since = torch.zeros_like(shortfall)
      prev = torch.zeros_like(shortfall)
    fresh = env.episode_length_buf <= 1
    since = torch.where(fresh, torch.zeros_like(since), since)
    prev = torch.where(fresh, torch.zeros_like(prev), prev)
    increased = shortfall > prev
    since = torch.where(increased, torch.zeros_like(since), since + 1.0)
    state["since"] = since
    state["prev"] = shortfall
    ramp = (since * env.step_dt / grace_s).clamp(max=1.0)
    shortfall = shortfall * ramp
  return shortfall


def standing_wall_clearance_penalty():
  """Factory: bill head/upper-torso proximity to the wall while standing.

  The crutch whack-a-mole's systematic ending. v15 leaned the BODY (fixed
  by root-xy tracking + the pad tax), v16 bent at the HIPS and rested the
  HEAD — root at the reference spot, pads clean, hip deviation diluted by
  the 21-joint mean, and no sensor or term anywhere watching the skull.
  This fences the wall for the body parts that NEVER legitimately
  approach it before the feet mount: linear bill from ``clearance_m``
  out, full price at contact, standing prefix only (the hanging stances
  later bring the chest close by choreography, and the arms must reach
  the wall — neither is fenced).
  """
  state: dict[str, object] = {"ids": None, "half": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    geom_names: tuple[str, ...] = (
      "neck_pitch_collision",
      "torso_collision",
    ),
    clearance_m: float = 0.10,
  ) -> torch.Tensor:
    from mjlab.utils.lab_api.math import quat_apply

    motion = _motion(env, command_name)
    first_engage = getattr(motion, "_first_engage_frame", None)
    zeros = torch.zeros(
      motion.ref_engaged.shape[0], device=motion.ref_engaged.device
    )
    if first_engage is None:
      return zeros
    if state["ids"] is None:
      # GEOM SURFACES, not link origins — the v17 bug: the head's
      # collision sphere (r=0.07, centered 0.065 outboard of the
      # neck_pitch link origin) let the skull press the wall while the
      # ORIGIN still read 8 cm clear, so the fence collected pennies for
      # a full head-butt (caught by the user's eyes vs the telltale).
      # geom_size[:, 0] is the wall-ward half-extent for both the head
      # sphere and the chest box while the trunk is near upright.
      mj = env.sim.mj_model
      ids = [mj.geom(f"robot/{name}").id for name in geom_names]
      state["ids"] = ids
      state["half"] = torch.tensor(
        [float(mj.geom_size[g, 0]) for g in ids],
        device=motion.ref_engaged.device,
      )
    standing = motion.time_steps < first_engage
    pos = env.sim.data.geom_xpos[:, state["ids"], :]  # (E, G, 3)
    n_local = torch.zeros_like(pos[:, 0])
    n_local[:, 0] = -1.0
    n_w = quat_apply(motion.wall_quat, n_local)  # (E, 3)
    rel = pos - motion.wall_pivot_w.unsqueeze(1)
    dist = (rel * n_w.unsqueeze(1)).sum(dim=-1) - state["half"]  # surface
    intrusion = ((clearance_m - dist) / clearance_m).clamp(min=0.0, max=1.5)
    return intrusion.sum(dim=-1) * standing.float()

  return call


def motion_unscheduled_press_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  press_norm_n: float = 4.0,
  regrip_lead_time_s: float = 0.3,
  bill_all_press: bool = False,
) -> torch.Tensor:
  """Bill pad press OUTSIDE the scheduled seating windows.

  The v15 graduation video's verdict (user's eyes, not the lanes): the
  stand phase converged to LEANING — body against the wall, pads propped
  as a crutch below the slam-tax threshold, balance outsourced to the
  wall. Wall contact is a scheduled resource in this choreography: inside
  a seating window pressing is paid, outside it ANY pad press is a
  crutch and bills at the same saturating scale the pay uses. The
  complement of ``motion_scheduled_press_shaping`` — same window, sign
  flipped.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  press = (affordance.pad_press_force / press_norm_n).clamp(max=1.0)
  if bill_all_press:
    # Stand-first: NO free press window at all. The clamped stand clock
    # parks one frame short of the first rising edge, so the regrip
    # "imminent" lead below is permanently true there — every hand press
    # anywhere on the wall would ride the pre-latch exemption forever
    # (the v15 prop, relicensed). With press pay at zero the window has
    # no legitimate business, so any un-latched press bills.
    return (press * (~motion.robot_engaged).float()).sum(dim=-1)
  lead = regrip_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  window = (motion.ref_engaged | regrip_soon) & ~motion.robot_engaged
  return (press * (~window & ~motion.robot_engaged).float()).sum(dim=-1)


def standing_root_xy_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.1,
) -> torch.Tensor:
  """Pay root-xy proximity to the reference DURING THE STANDING PREFIX.

  The cycle deliberately never tracks global root position (the base is
  free on the wall); the standing prefix is the one stretch where global
  position IS the task — the robot must stand where the clip stands to
  reach the anchors, and a body slumped against the wall drifts exactly
  the coordinate nobody was watching. Active only before the clip's first
  scheduled engagement; zero (and harmless) elsewhere and for commands
  without the acyclic prologue.
  """
  motion = _motion(env, command_name)
  first_engage = getattr(motion, "_first_engage_frame", None)
  if first_engage is None:
    return torch.zeros(
      motion.ref_engaged.shape[0], device=motion.ref_engaged.device
    )
  standing = motion.time_steps < first_engage
  robot_xy = motion.robot.data.root_link_pos_w[:, :2]
  err = (motion.ref_root_pos_w[:, :2] - robot_xy).square().sum(dim=-1)
  return torch.exp(-err / std**2) * standing.float()


def motion_approach_progress():
  """Factory: PROGRESS-paid approach toward the upcoming anchor.

  The v13 endgame of the wide tow: position-paid proximity is a rent, and
  every patch just moved the tenant — slam (v10 tax) -> jam (v12 free-space
  cutoff) -> hover at the anchor minus one centimeter (v13 lanes: press
  income and overdrive both ~0 while the tow still paid). This variant
  pays the DERIVATIVE instead: each step earns the distance CLOSED since
  the previous active step (and refunds retreat symmetrically), so any
  closed loop nets zero — approach earns exactly once per meter, hovering
  earns nothing, and the only standing income near the anchor is the
  schedule's own press -> latch channel. Per-EE state resets whenever the
  pay mask was inactive last step (spawn, post-latch, post-rip), so no
  windfalls on teleports; a per-step clamp guards the rest.
  """
  state: dict[str, torch.Tensor | None] = {"prev": None, "active": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    ee_indices: tuple[int, ...] = (0, 1),
    gain_per_m: float = 4.0,
    step_clamp_m: float = 0.05,
    edge_lead_time_s: float | None = None,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    executor = affordance._get_executor()
    ref_on = motion.ref_engaged
    idx = torch.as_tensor(ee_indices, dtype=torch.long, device=ref_on.device)
    active = (~ref_on[:, idx]) & (~motion.robot_engaged[:, idx])
    if edge_lead_time_s is not None:
      # v28 schedule-proximity gate: approach pay exists for SCHEDULED
      # swings. The cycle satisfied that implicitly (ref goes false only
      # inside swing windows); the mount's 5.6 s all-false standing prefix
      # turned both tows into an unconditional wall-ward magnet that
      # taught the straight-line punch over the choreographed lateral
      # raise. Pay only when this EE's rising edge is imminent.
      lead = edge_lead_time_s / motion.cycle_time_s
      active = active & (motion.ref_edge_countdowns[:, idx, 1] <= lead)
    dist = (
      executor.pad_center_world()[:, idx] - motion.ref_anchor_pos_w[:, idx]
    ).norm(dim=-1)
    prev = state["prev"]
    prev_active = state["active"]
    if prev is None or prev.shape != dist.shape:
      prev = dist.clone()
      prev_active = torch.zeros_like(active)
    # Fresh episodes re-baseline (RSI teleports must not pay).
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    consecutive = active & prev_active & ~fresh
    progress = (prev - dist).clamp(min=-step_clamp_m, max=step_clamp_m)
    pay = gain_per_m * progress * consecutive.float()
    state["prev"] = dist
    state["active"] = active
    return pay.sum(dim=-1)

  return call


def press_overdrive_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  limit_n: float = 8.0,
  scale_n: float = 10.0,
  cap: float = 3.0,
) -> torch.Tensor:
  """Bill pad press force beyond ``limit_n`` — slamming must LOSE money.

  Pair with a negative weight. The v11 press probe measured hand presses
  of median 94 N / p95 379 N / max 443 N against a ~3 N reference press, a
  ~30 N static tipping budget and the ~400 N ultimate envelope — the pad
  ram simultaneously topples the robot and explains the 3-step ultimate
  rips of every seeded anchor. Saturating press PAY (12 N) makes slamming
  free, not costly; this term prices it: zero through ``limit_n``, then
  linear at 1/``scale_n`` per N, capped so outliers don't blow up the
  update. All four pads (the press sensors are pad-vs-WALL pairs; floor
  contact never reaches them).
  """
  motion = _motion(env, command_name)
  press = motion._affordance().pad_press_force
  over = ((press - limit_n) / scale_n).clamp(min=0.0, max=cap)
  return over.sum(dim=-1)


def motion_swing_approach_masked(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.4,
  ee_indices: tuple[int, ...] = (0, 1),
  press_cutoff_n: float | None = 6.0,
  edge_lead_time_s: float | None = None,
) -> torch.Tensor:
  """Wide-basin swing-approach shaping restricted to ``ee_indices``.

  Same activation and target as the cycle's ``motion_swing_approach``
  (scheduled OFF and actually released, distance to ``ref_anchor_pos_w``),
  but sliced to a subset of EEs so the wide kernel can serve the hands
  without funding the feet's stand-still income. Meant to be stacked ON TOP
  of the narrow funnel as a two-scale kernel: this term hands exploration a
  gradient from ~1 m out (exp(-(d/0.4)^2): 0.006 at the standing hang,
  0.21 at horizontal arms, 0.94 at the place pose), the untouched 0.05
  funnel takes over for the precision landing.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  executor = affordance._get_executor()
  ref_on = motion.ref_engaged
  idx = torch.as_tensor(ee_indices, dtype=torch.long, device=ref_on.device)
  active = (~ref_on[:, idx]) & (~motion.robot_engaged[:, idx])
  if edge_lead_time_s is not None:
    # Same schedule-proximity gate as motion_approach_progress (v28): pay
    # approach only when this EE's scheduled engagement is imminent.
    lead = edge_lead_time_s / motion.cycle_time_s
    active = active & (motion.ref_edge_countdowns[:, idx, 1] <= lead)
  if press_cutoff_n is not None:
    # The tow guides the approach through FREE SPACE only. Without this,
    # a pad rammed against the wall AT the anchor collects full tow pay
    # forever while never welding (distance ~0, no latch, no rip bill) —
    # the exact 94-443 N jamming equilibrium the v11 probe measured. Once
    # touching beyond a light brush, the press/latch machinery owns the
    # reward channel.
    active = active & (affordance.pad_press_force[:, idx] < press_cutoff_n)
  error = (
    executor.pad_center_world()[:, idx] - motion.ref_anchor_pos_w[:, idx]
  ).norm(dim=-1)
  pay = torch.exp(-((error / std) ** 2))
  return (pay * active.float()).sum(dim=-1)


def pad_grind_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  ee_indices: tuple[int, ...] = (0, 1),
  min_press_n: float = 2.0,
  affordance_command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Bill UNWELDED pressing that SLIDES along the wall — the visible grind.

  v56 (the user's recurring video verdict, finally priced directly): the
  event metrics kept improving while the hands audibly dragged across the
  surface between welds. Rubbing = press without weld + tangential motion;
  neither the slide bill (engaged pads only) nor the skate tax (floor)
  ever touched it. Per selected EE: |pad velocity projected into the wall
  plane| while pressed above ``min_press_n`` and not engaged. Pair with a
  negative weight.
  """
  from mjlab.utils.lab_api.math import quat_apply

  motion = _motion(env, command_name)
  aff = env.command_manager.get_term(affordance_command_name)
  robot = env.scene["robot"]
  ex = aff._get_executor()
  idx = torch.as_tensor(
    ee_indices, dtype=torch.long, device=motion.ref_engaged.device
  )
  vel = robot.data.body_link_lin_vel_w[:, ex._ee_body_ids, :][:, idx]
  n_local = torch.zeros(vel.shape[0], 3, device=vel.device)
  n_local[:, 0] = -1.0
  n_w = quat_apply(motion.wall_quat, n_local).unsqueeze(1)
  v_tan = vel - (vel * n_w).sum(dim=-1, keepdim=True) * n_w
  grinding = (
    (aff.pad_press_force[:, idx] > min_press_n) & ~aff.engaged[:, idx]
  )
  return (v_tan.norm(dim=-1) * grinding.float()).sum(dim=-1)


def motion_predrag_wage(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  shear_peak_n: float = 8.0,
  shear_band_hi_n: float = 16.0,
  overdrive_cutoff_n: float = 30.0,
  regrip_lead_time_s: float = 0.8,
  anchor_radius_m: float = 0.15,
  wage_window_s: float | None = 2.0,
) -> torch.Tensor:
  """Pre-weld DRAG wage (v61d ignition aid): pay down-shear while touching.

  The springfield weld rule is TOUCH + DRAG, and the iter-1467 diagnosis
  showed the drag half has no income anywhere: touching contacts carry
  p90 = 0.62 N of down-shear (max ever seen 3.83 N — not one step over
  the 4 N ignition gate). Pressing is paid, approaching is paid, seating
  a WELD is paid (first_charge) — dragging an unwelded touch pays
  nothing, so the policy presses statically forever. Dense from the first
  newton (the gate pays nothing below threshold; the wage is the gradient
  that reaches it). Same gates as the press wage: schedule window +
  regrip lead + the v56 anti-career time-box + at-anchor only. Slow
  heavy drags net positive against the pad_grind SPEED tax — together
  they shape the real seating stroke. Ignition scaffolding: anneals to
  zero before graduation (see the wage ladder in the cfg).
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  shear = getattr(affordance, "pad_shear_down", None)
  if shear is None:
    return torch.zeros(env.num_envs, device=env.device)
  lead = regrip_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  scheduled = motion.ref_engaged
  if wage_window_s is not None:
    elapsed = (
      motion.ref_run_time - motion.ref_edge_countdowns[..., 0]
    ) * motion.cycle_time_s
    scheduled = scheduled & (elapsed <= wage_window_s)
  window = (scheduled | regrip_soon) & ~motion.robot_engaged
  window = window & (affordance.contact_count > 0)
  executor = affordance._get_executor()
  dist = (
    executor.pad_center_world() - motion.ref_anchor_pos_w
  ).norm(dim=-1)
  window = window & (dist <= anchor_radius_m)
  pay = force_target_scale(
    shear.clamp_min(0.0),
    target_n=shear_peak_n,
    zero_at_n=overdrive_cutoff_n,
    floor=0.0,
    band_hi_n=shear_band_hi_n,
  )
  return (pay * window.float()).sum(dim=-1)


def motion_anchor_dwell_wage():
  """Factory: pre-weld DWELL wage — pay STAYING on the wall (v62e6, A of
  the user's A+B package, 08-19 evening).

  The missing rung UNDER the drag wage: the weld gate demands contact
  sustained through the debounce, but no term ever paid contact
  persistence itself — a hop-and-slap costs nothing, so the policy never
  learned to stay (video verdict: bounce, tremble, then an ~80-105 N
  slam). Pays per step while (scheduled window | regrip lead) x touching
  x unwelded x at-anchor — the predrag wage's exact gates — scaled by a
  consecutive-contact ramp (full pay at ``ramp_steps``); a broken touch
  zeroes the streak, so CONTINUITY itself is the earner and slap-cycling
  earns the floor forever. Streak state is self-healing across resets (a
  fresh episode starts un-touching, which zeroes the row). Ignition
  scaffolding: anneals out on the graduation timeline (dwell_wage_anneal).
  """
  state: dict[str, torch.Tensor | None] = {"streak": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    regrip_lead_time_s: float = 0.8,
    anchor_radius_m: float = 0.15,
    wage_window_s: float | None = 2.0,
    ramp_steps: int = 10,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    lead = regrip_lead_time_s / motion.cycle_time_s
    regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
    scheduled = motion.ref_engaged
    if wage_window_s is not None:
      elapsed = (
        motion.ref_run_time - motion.ref_edge_countdowns[..., 0]
      ) * motion.cycle_time_s
      scheduled = scheduled & (elapsed <= wage_window_s)
    window = (scheduled | regrip_soon) & ~motion.robot_engaged
    window = window & (affordance.contact_count > 0)
    executor = affordance._get_executor()
    dist = (
      executor.pad_center_world() - motion.ref_anchor_pos_w
    ).norm(dim=-1)
    window = window & (dist <= anchor_radius_m)
    streak = state["streak"]
    if streak is None or streak.shape != window.shape:
      streak = torch.zeros(
        window.shape, dtype=torch.float32, device=window.device
      )
    streak = torch.where(
      window, streak + 1.0, torch.zeros_like(streak)
    )
    state["streak"] = streak
    pay = (streak / float(max(ramp_steps, 1))).clamp(max=1.0)
    return (pay * window.float()).sum(dim=-1)

  return call


def motion_scheduled_press_gentle(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  press_peak_n: float = 8.0,
  press_band_hi_n: float = 14.0,
  regrip_lead_time_s: float = 0.3,
  overdrive_cutoff_n: float = 20.0,
  anchor_radius_m: float | None = None,
  wage_window_s: float | None = None,
) -> torch.Tensor:
  """Scheduled-press pay shaped as a TRIANGLE peaking at the reference press.

  Two generations of this wage taught the wrong lesson at the top end: the
  cycle's saturating shape paid a 250 N slam like a 4 N seat (v27 probe:
  median 70 N collecting full pay), and the v28 flat-top [3, 8] N gentle
  band still paid 8 N like 3 N — no gradient toward the reference press,
  and every newton above 3 spends the ~20-35 N standing tip-over budget
  for nothing (the physics reason the choreography presses exactly 3 N).
  The triangle makes the reference press the unique income peak: linear up
  from 0 to ``press_peak_n``, linear down to zero at
  ``overdrive_cutoff_n`` (where the slam tax takes over).
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  lead = regrip_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  scheduled = motion.ref_engaged
  if wage_window_s is not None:
    # v56 time-box (user question, answered honestly: "why does rubbing
    # pay at all?"): the wage is IGNITION scaffolding, and unlimited it
    # became a career — a weak limb parks pressed-at-the-anchor collecting
    # per-step pay while it waits (the visible wall-grind). Pay only
    # [lead before, wage_window_s after] each rising edge — aligned with
    # the contact law's grace: past it you are either welded (no wage
    # needed) or executed. New limbs (the feet) keep their full bridge.
    elapsed = (
      motion.ref_run_time - motion.ref_edge_countdowns[..., 0]
    ) * motion.cycle_time_s
    scheduled = scheduled & (elapsed <= wage_window_s)
  window = (scheduled | regrip_soon) & ~motion.robot_engaged
  if anchor_radius_m is not None:
    # v35 spatial anchor (the wage's cycle-invariant made explicit): in
    # the cycle a windowed press IS at the anchor — the hand is mid-regrip
    # centimeters away. The standing mount broke that: a hip-height wall
    # slap collected the full triangle wage with zero balance risk (the
    # v34 fullrun's "weird hand strike", caught by the user), outbidding
    # the risky overhead press it was meant to fund. Pay only presses
    # delivered AT that EE's own anchor.
    executor = affordance._get_executor()
    dist = (
      executor.pad_center_world() - motion.ref_anchor_pos_w
    ).norm(dim=-1)
    window = window & (dist <= anchor_radius_m)
  press = affordance.pad_press_force
  # v48 band wage: full pay across [press_peak_n, press_band_hi_n] (the
  # whole estimated weldable band), ramping to zero at the cutoff — the
  # point-target triangle bet the wage on a single derived number.
  pay = force_target_scale(
    press,
    target_n=press_peak_n,
    zero_at_n=overdrive_cutoff_n,
    floor=0.0,
    band_hi_n=press_band_hi_n,
  )
  return (pay * window.float()).sum(dim=-1)


def force_target_scale(
  force_n: torch.Tensor,
  target_n: float,
  zero_at_n: float,
  floor: float = 0.25,
  band_hi_n: float | None = None,
  hard_zero_n: float | None = None,
) -> torch.Tensor:
  """Target-BAND scale for force-conditioned pay (user doctrine,
  2026-08-15: force rewards are never "more is better" — they pay by
  closeness to the target; and the day's second lesson: the target is a
  BAND, not a point — every derivation of the weldable force carries
  model uncertainty, so the whole estimated band pays in full).

  Trapezoid: full pay across [``target_n``, ``band_hi_n``], linear rise
  from 0, linear fall to ``zero_at_n``, clamped to ``floor`` (the floor
  keeps a far-side learning signal during skill transitions — a hard
  zero would un-ignite the hot welds a transitioning policy still makes).
  ``band_hi_n=None`` degenerates to the triangle (point target).
  """
  hi = target_n if band_hi_n is None else band_hi_n
  up = force_n / target_n
  down = (zero_at_n - force_n) / max(zero_at_n - hi, 1e-6)
  out = torch.minimum(up, down).clamp(min=floor, max=1.0)
  if hard_zero_n is not None:
    # v50 tail guard (user catch: the falls are decided by the TAIL, not
    # the median — the floor kept paying 140 N slam-welds 0.6x while the
    # capped tax stopped hurting past ~50 N). The learning-material floor
    # survives only up to hard_zero_n; physically absurd forces pay zero.
    out = torch.where(force_n > hard_zero_n, torch.zeros_like(out), out)
  return out


def motion_toe_brace_target(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  sensor_name: str = "climb_toe_contact",
  toe_geom_names: tuple[str, ...] = (
    "left_ankle_roll_toe",
    "right_ankle_roll_toe",
  ),
  target_n: float = 12.0,
  band_hi_n: float = 20.0,
  zero_at_n: float = 48.0,
  off_schedule_cost: float = 2.5,
) -> torch.Tensor:
  """Toe-brace wage reshaped to TARGET-tracking (audit brace ~16 N).

  The climb original saturates at 8 N — "more is better" up to the cap,
  which paid the 79-345 N toe jams the same as the choreographed brace.
  On-schedule pay now peaks at the audit's reference force and decays
  either side; off-schedule pressing bills exactly as before.
  """
  from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
    _toe_sensor_row_perm,
  )

  motion = _motion(env, command_name)
  scheduled = motion.ref_toe_braced
  sensor = env.scene[sensor_name]
  data = sensor.data
  perm = _toe_sensor_row_perm(sensor, toe_geom_names)
  found = data.found[:, perm] > 0
  normal = data.force[:, perm, 0].clamp(min=0.0)
  pay = force_target_scale(
    normal, target_n, zero_at_n, floor=0.0, band_hi_n=band_hi_n
  )
  pay = pay * found.float()
  on = scheduled.float()
  bill = (normal / 8.0).clamp(max=1.0) * found.float()
  return (pay * on - off_schedule_cost * bill * (1.0 - on)).sum(dim=-1)


def first_anchor_press_bonus():
  """Factory: one-shot bonus for the FIRST gentle in-window anchor press.

  The v36 ledger: rotation learned, presses in the gentle band (median
  4.6 N), yet the press->debounce->weld conversion never fires on its own
  — the chain's very first successful attempt earns only ~0.03 of process
  wage before anything downstream exists to value it. This pays that
  first attempt like an event (divided by step_dt so the configured
  weight IS the bonus, the unsafe_release convention): once per episode,
  only inside the scheduled window, only at the EE's own anchor, only in
  the gentle band — none of which can be farmed (one payment, re-armed
  only by a fresh episode).
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    anchor_radius_m: float = 0.15,
    min_press_n: float = 1.5,
    max_press_n: float = 20.0,
    regrip_lead_time_s: float = 0.8,
    far_floor: float = 0.25,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    executor = affordance._get_executor()
    lead = regrip_lead_time_s / motion.cycle_time_s
    regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
    window = (motion.ref_engaged | regrip_soon) & ~motion.robot_engaged
    dist = (
      executor.pad_center_world() - motion.ref_anchor_pos_w
    ).norm(dim=-1)
    press = affordance.pad_press_force
    qual = (
      window
      & (dist <= anchor_radius_m)
      & (press >= min_press_n)
      & (press <= max_press_n)
    )
    # Seed economy #2: an EE born WITH a welded anchor collects no press
    # bonus this episode — otherwise demolishing the gift and redoing it
    # would be the paid route (the 08-15 autopsy's incentive audit).
    fresh = env.episode_length_buf <= 1
    birth = state.get("birth")
    if birth is None or birth.shape != qual.shape:
      birth = torch.zeros_like(qual)
    birth = torch.where(fresh.unsqueeze(-1), affordance.engaged, birth)
    state["birth"] = birth
    qual = qual & ~birth
    hit = qual.any(dim=-1)
    # v45: pay by closeness to the target press (user doctrine).
    scale = (
      force_target_scale(
        press, target_n=20.0, zero_at_n=55.0, band_hi_n=28.0, floor=far_floor, hard_zero_n=70.0
      )
      * qual.float()
    ).amax(dim=-1)
    paid = state["paid"]
    if paid is None or paid.shape != hit.shape:
      paid = torch.zeros_like(hit)
    fresh = env.episode_length_buf <= 1
    paid = paid & ~fresh
    pay = hit & ~paid
    state["paid"] = paid | hit
    return pay.float() * scale / env.step_dt

  return call


def first_weld_bonus():
  """Factory: one-shot bonus for the FIRST in-window hand WELD.

  The v37 lesson in one line: the first-press bonus (+3 for a 1.5 N
  instant) outpaid the actual weld prize (anchor_landing's 40 is dt-scaled
  to ~0.8), so 57% of episodes collected the touch and none converted —
  the margin between "touch" and "weld" was NEGATIVE. This term fixes the
  ladder's ordering: the weld itself (an engage rising edge of a hand
  inside its scheduled window) pays the biggest one-shot on the chain,
  once per episode, unfarmable by construction.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    ee_indices: tuple[int, ...] = (0, 1),
    regrip_lead_time_s: float = 0.8,
    far_floor: float = 0.25,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    lead = regrip_lead_time_s / motion.cycle_time_s
    regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
    window = motion.ref_engaged | regrip_soon
    idx = torch.as_tensor(
      ee_indices, dtype=torch.long, device=window.device
    )
    # Seed economy #2 (same as the press bonus): a hand born welded earns
    # no "first weld" this episode — the seed's rising edge happened before
    # the episode existed, and a rip-and-redo must not manufacture one.
    fresh = env.episode_length_buf <= 1
    birth = state.get("birth")
    if birth is None or birth.shape != affordance.engaged.shape:
      birth = torch.zeros_like(affordance.engaged)
    birth = torch.where(fresh.unsqueeze(-1), affordance.engaged, birth)
    state["birth"] = birth
    edge = (
      affordance.engaged_this_step[:, idx]
      & window[:, idx]
      & ~birth[:, idx]
    )
    hit = edge.any(dim=-1)
    # v45 force-target scaling (user doctrine: force rewards pay by
    # closeness to the target, never "more is better"). The edge-step
    # press often reads 0 (history-mean welds fire after a pulse), so a
    # short decaying peak tracks the true entry force.
    press = affordance.pad_press_force[:, idx]
    peak = state.get("peak")
    if peak is None or peak.shape != press.shape:
      peak = torch.zeros_like(press)
    peak = torch.maximum(peak * 0.8, press)
    entry = torch.where(edge, peak, torch.zeros_like(peak)).amax(dim=-1)
    scale = force_target_scale(
      entry, target_n=20.0, zero_at_n=55.0, band_hi_n=28.0, floor=far_floor, hard_zero_n=70.0
    )
    state["peak"] = peak
    paid = state["paid"]
    if paid is None or paid.shape != hit.shape:
      paid = torch.zeros_like(hit)
    fresh = env.episode_length_buf <= 1
    paid = paid & ~fresh
    pay = hit & ~paid
    state["paid"] = paid | hit
    return pay.float() * scale / env.step_dt

  return call


def first_charge_bonus():
  """Factory: one-shot bonus for the first FULL-CHARGE of a self-made weld.

  The v60 rung between weld and hold: the capacity ramp charges only under
  rake shear >= ``capacity_ramp_shear_n``, and on the wall gravity pays
  that for free (the cycle never needed the lesson — every welded limb
  hangs loaded from birth). The mount's hands weld from a STANDING body,
  feet carrying the weight, so a fresh weld can idle below the charge
  threshold forever and still sit at floor capacity when the schedule
  transfers weight — the fullrun's 3-8 step RH rips all die inside the
  12.5-step ramp. This rung prices the seating move itself.

  Per EE per episode — the weld/hold rungs' shared pot lets whichever limb
  fires first scoop the bonus (a seeded LH holds from birth and claims the
  hold rung at ``hold_steps`` every time); a per-limb ledger keeps this
  rung reachable for RH/RF. Seeds are excluded by the edge test (born
  engaged means no rising edge this episode — their full charge is a
  gift, not a skill); a rip-and-reweld stays eligible (its ramp restarts
  from zero, and churn already nets negative under the rip bills) but
  pays at most once per limb.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    ee_indices: tuple[int, ...] = (0, 1, 2, 3),
    anchor_radius_m: float = 0.15,
    regrip_lead_time_s: float = 0.8,
    full_fraction: float = 1.0,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    frac = affordance.seat_charge_fraction
    paid = state.get("paid")
    edged = state.get("edged")
    prev = state.get("prev")
    if paid is None or paid.shape != frac.shape:
      paid = torch.zeros(frac.shape, dtype=torch.bool, device=frac.device)
      edged = torch.zeros_like(paid)
      prev = torch.ones_like(frac)
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    paid = paid & ~fresh
    edged = (edged & ~fresh) | affordance.engaged_this_step
    prev = torch.where(fresh.expand_as(prev), frac, prev)
    # Rule-8 rung: the bar starts where the current distribution can
    # reach (the 08-18 probe: RH self-welds die at 3 steps / 0.16 charge,
    # so a full-charge-only rung would never fire) and ratchets to 1.0
    # via ``reward_param_schedule`` as lifetimes grow.
    crossed = (
      (prev < full_fraction)
      & (frac >= full_fraction)
      & affordance.engaged
    )
    # Same acceptance window as the weld rung: welds made inside the 0.8 s
    # regrip lead can finish charging before ref_engaged flips — judged at
    # the instantaneous window they would never collect.
    lead = regrip_lead_time_s / motion.cycle_time_s
    window = motion.ref_engaged | (
      motion.ref_edge_countdowns[..., 1] <= lead
    )
    executor = affordance._get_executor()
    dist = (
      executor.pad_center_world() - motion.ref_anchor_pos_w
    ).norm(dim=-1)
    hit = (
      crossed
      & edged
      & window
      & (dist <= anchor_radius_m)
      & ~paid
    )
    state["paid"] = paid | hit
    state["edged"] = edged
    state["prev"] = frac.clone()
    idx = torch.as_tensor(
      ee_indices, dtype=torch.long, device=frac.device
    )
    return hit[:, idx].sum(dim=-1).float() / env.step_dt

  return call


def first_hold_bonus():
  """Factory: one-shot bonus for the first SUSTAINED hand hold.

  The ladder's top step (rule 8, applied all the way up): touch < weld <
  HOLD. Pays once per episode when any selected EE has been continuously
  engaged for ``hold_steps`` — the skill the 0.06 s throwaway welds skip,
  and the one the engage-gate discount needs a bridge to (the gate's
  sustained income only compounds for welds that live).
  """
  state: dict[str, torch.Tensor | None] = {"run": None, "paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    ee_indices: tuple[int, ...] = (0, 1),
    hold_steps: int = 25,
    anchor_radius_m: float | None = None,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    idx = torch.as_tensor(
      ee_indices, dtype=torch.long, device=motion.ref_engaged.device
    )
    engaged = affordance.engaged[:, idx]
    if anchor_radius_m is not None:
      # v54: hold only counts AT the scheduled anchor — the position-blind
      # rung paid 7.5 cm-off holds and taught offset placement.
      ex = affordance._get_executor()
      dist = (
        ex.pad_center_world()[:, idx] - motion.ref_anchor_pos_w[:, idx]
      ).norm(dim=-1)
      engaged = engaged & (dist <= anchor_radius_m)
    run = state["run"]
    paid = state["paid"]
    if run is None or run.shape != engaged.shape:
      run = torch.zeros_like(engaged, dtype=torch.long)
      paid = torch.zeros(
        engaged.shape[0], dtype=torch.bool, device=engaged.device
      )
    fresh = env.episode_length_buf <= 1
    run = torch.where(
      fresh.unsqueeze(-1), torch.zeros_like(run), run
    )
    paid = paid & ~fresh
    run = (run + 1) * engaged.long()
    hit = (run == hold_steps).any(dim=-1)
    pay = hit & ~paid
    state["run"] = run
    state["paid"] = paid | hit
    return pay.float() / env.step_dt

  return call


def motion_joint_pos_error_exp_masked(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.2,
  joint_keywords: tuple[str, ...] = ("shoulder", "elbow"),
  exclude_matches: bool = False,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """Clip-joint tracking kernel over a keyword-selected joint subset.

  ``joint_keywords`` picks clip joints by substring; ``exclude_matches``
  flips the selection (True = track everything EXCEPT the matches). Kernel
  and engage-gate semantics mirror ``motion_joint_pos_error_exp`` so a
  strict/loose split preserves the gate economics of the term it replaces.
  """
  motion = _motion(env, command_name)
  names = motion.clip.joint_names
  sel = [
    i
    for i, name in enumerate(names)
    if any(key in name for key in joint_keywords) != exclude_matches
  ]
  if not sel:
    raise ValueError(
      f"joint_keywords {joint_keywords!r} (exclude_matches={exclude_matches})"
      f" selected no joints out of {names}."
    )
  ref = motion.ref_joint_pos
  idx = torch.as_tensor(sel, dtype=torch.long, device=ref.device)
  error = (
    (ref[:, idx] - motion.robot_joint_pos[:, idx])
    .square()
    .mean(dim=-1)
  )
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(
      motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction
    )
  return reward


def motion_ee_pos_error_exp_masked(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.1,
  ee_indices: tuple[int, ...] = (0, 1),
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """Root-frame EE tracking kernel over an index-selected EE subset.

  Mirrors ``motion_ee_pos_error_exp`` (root-relative on both sides, same
  engage gate) sliced to ``ee_indices`` — EE order is the microspine order
  (left_hand, right_hand, left_foot, right_foot).
  """
  motion = _motion(env, command_name)
  ref = motion.ref_ee_pos_r
  idx = torch.as_tensor(ee_indices, dtype=torch.long, device=ref.device)
  error = (
    (ref[:, idx] - motion.robot_ee_pos_r[:, idx])
    .square()
    .sum(dim=-1)
    .mean(dim=-1)
  )
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(
      motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction
    )
  return reward
