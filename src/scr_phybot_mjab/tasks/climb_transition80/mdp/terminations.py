"""Transition80 terminations: the legality of the state space lives here.

Doctrine (mentor call, 2026-08-21): rewards say how well, terminations say
whether. Three families:

  * tracking deviation (BeyondMimic ports with LadderMan-style persistence
    — a pose error must hold for ``persist_steps`` before it kills, so
    contact transients do not);
  * the schedule (contact loss = the LadderMan executioner, this package's
    own copy of the 2026-08-16 law; a physical release of a scheduled
    anchor is death, not a fine);
  * the engage constraint cone — Constraints-as-Terminations (Chane-Sane
    et al. 2024): four per-pad constraints, each a normalised violation in
    [0, 1]; the episode ends with probability max_i p_i·v_i (soft mode),
    or whenever any v_i > 0 (hard mode). Soft by default: the violation
    magnitude becomes a gradient instead of a wall (the v10 lesson: a
    strict 4-degree gate produced zero flow).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "bad_anchor_ori_termination",
  "bad_anchor_pos_termination",
  "bad_motion_body_pos_termination",
  "bad_pad_center_termination",
  "engage_constraint_termination",
  "illegal_wall_contact_termination",
  "scheduled_anchor_rip",
  "scheduled_contact_loss_termination",
)


def _persist(
  state: dict[str, torch.Tensor | None],
  env: ManagerBasedRlEnv,
  condition: torch.Tensor,
  persist_steps: int,
) -> torch.Tensor:
  """Consecutive-step counter with the reset-safe ``fresh`` idiom."""
  count = state.get("count")
  if count is None or count.shape != condition.shape:
    count = torch.zeros_like(condition, dtype=torch.long)
  fresh = env.episode_length_buf <= 1
  count = torch.where(fresh, torch.zeros_like(count), count)
  count = (count + 1) * condition.long()
  state["count"] = count
  return count >= max(1, int(persist_steps))


def bad_anchor_pos_termination():
  """Factory: root position off the reference by > threshold (3-D, world),
  persisting ``persist_steps`` policy steps."""
  state: dict[str, torch.Tensor | None] = {"count": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    threshold: float = 0.35,
    persist_steps: int = 10,
  ) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    error = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w).norm(dim=-1)
    return _persist(state, env, error > threshold, persist_steps)

  return call


def bad_anchor_ori_termination():
  """Factory: mjlab ``bad_anchor_ori`` (projected-gravity z mismatch between
  the reference and robot root frames) with persistence."""
  state: dict[str, torch.Tensor | None] = {"count": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    threshold: float = 0.8,
    persist_steps: int = 10,
  ) -> torch.Tensor:
    # Imported here, not at module level: torchrunx ships these closures to
    # the worker processes with cloudpickle, which serialises a nested
    # function BY VALUE together with the globals it references — and a
    # torch.jit ScriptFunction cannot be pickled (same idiom as the
    # transition's standing_wall_contact_termination).
    from mjlab.utils.lab_api.math import quat_apply_inverse

    cmd = env.command_manager.get_term(command_name)
    gravity = cmd.robot.data.gravity_vec_w
    ref_g = quat_apply_inverse(cmd.anchor_quat_w, gravity)
    rob_g = quat_apply_inverse(cmd.robot_anchor_quat_w, gravity)
    error = (ref_g[:, 2] - rob_g[:, 2]).abs()
    return _persist(state, env, error > threshold, persist_steps)

  return call


def bad_motion_body_pos_termination():
  """Factory: any tracked body (default: the four pads) off its reference
  target by > threshold, persisting ``persist_steps``."""
  state: dict[str, torch.Tensor | None] = {"count": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    threshold: float = 0.30,
    persist_steps: int = 10,
    body_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    if body_names is None:
      idx = cmd.pad_slots
    else:
      idx = cmd.tracking_body_indices(body_names)
    error = (cmd.body_pos_target_w[:, idx] - cmd.robot_body_pos_w[:, idx]).norm(dim=-1)
    return _persist(state, env, (error > threshold).any(dim=-1), persist_steps)

  return call


def bad_pad_center_termination():
  """Factory: any pad FACE center off its reference face center by >
  threshold, persisting ``persist_steps`` (v1g: replaces the body-origin
  test for the pads — the origin sat inside the 0.30 m line while the face
  parked 46 cm from the anchor)."""
  state: dict[str, torch.Tensor | None] = {"count": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    threshold: float = 0.30,
    persist_steps: int = 10,
  ) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    error = (cmd.ref_pad_center_w - cmd.robot_pad_center_w).norm(dim=-1)
    return _persist(state, env, (error > threshold).any(dim=-1), persist_steps)

  return call


def scheduled_anchor_rip(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
  release_lead_time_s: float = 0.5,
  seed_settle_exempt_s: float = 0.5,
) -> torch.Tensor:
  """A pad the schedule has ENGAGED is physically released -> terminate.

  Every limit-surface release counts (tension, lateral, ultimate, tilt and
  the reverse-shear "free" release alike): while the chart says hold, a
  release is a failure of the hold, whatever the physics label. Exempt:
  R4 reseat pending; releases within ``release_lead_time_s`` of the
  scheduled falling edge (letting go slightly early is the choreography);
  and RSI-SEEDED anchors lost within ``seed_settle_exempt_s`` of birth —
  the kinematic birth drops the body weight onto welds the policy has not
  yet loaded (a80 hold-still probe, 2026-08-21: 150-270 N foot-pad shear
  spikes at step 2, compliance-driven lateral/tension/reverse-shear rips
  at steps 5-19), so an immediate death there would be a gradient desert
  at guard expiry. A lost seed still has to be re-welded inside the
  contact law's grace, so the consequence is delayed into a state with
  gradient, not removed. No grace otherwise, no fine — the v57 seed-rip
  bill taught price-sensitivity, never holding.
  """
  motion = env.command_manager.get_term(command_name)
  aff = env.command_manager.get_term(affordance_command_name)
  released = aff.forced_release_this_step
  lead = release_lead_time_s / motion.cycle_time_s
  release_soon = motion.ref_edge_countdowns[..., 0] <= lead
  rip = released & motion.ref_engaged & ~aff.dr_release_pending & ~release_soon
  if seed_settle_exempt_s > 0.0:
    settle_steps = max(1, int(round(seed_settle_exempt_s / env.step_dt)))
    settling = (env.episode_length_buf <= settle_steps).unsqueeze(-1)
    was_seeded = getattr(aff, "released_was_seeded", None)
    if was_seeded is not None:
      rip = rip & ~(was_seeded & settling)
  return rip.any(dim=-1)


def illegal_wall_contact_termination():
  """Factory: a listed body geom's SURFACE crosses the wall plane.

  Own copy of the 2026-08-13 wall executioner with the standing-prefix gate
  made optional (``gate="none"`` = whole episode).
  These links carry no wall collision at all, so a surface crossing is a
  state hardware cannot reach; no value may flow through it. The a80
  reference keeps every listed link >= +10.8 mm clear on every frame.
  """
  state: dict[str, object] = {"ids": None, "half": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    geom_names: tuple[str, ...] = ("neck_pitch_collision", "torso_collision"),
    threshold_m: float = 0.0,
    gate: str = "none",
    entity_name: str = "robot",
  ) -> torch.Tensor:
    from mjlab.utils.lab_api.math import quat_apply  # cloudpickle-safe

    motion = env.command_manager.get_term(command_name)
    if state["ids"] is None:
      mj = env.sim.mj_model
      ids = [mj.geom(f"{entity_name}/{name}").id for name in geom_names]
      state["ids"] = ids
      state["half"] = torch.tensor([float(mj.geom_size[g, 0]) for g in ids], device=env.device)
    pos = env.sim.data.geom_xpos[:, state["ids"], :]  # (E, G, 3)
    n_local = torch.zeros_like(pos[:, 0])
    n_local[:, 0] = -1.0
    n_w = quat_apply(motion.wall_quat, n_local)  # outward wall normal
    rel = pos - motion.wall_pivot_w.unsqueeze(1)
    dist = (rel * n_w.unsqueeze(1)).sum(dim=-1) - state["half"]
    crossed = dist.min(dim=-1).values < threshold_m
    if gate == "standing":
      first_engage = getattr(motion, "_first_engage_frame", None)
      if first_engage is None:
        return torch.zeros_like(crossed)
      crossed = crossed & (motion.time_steps < first_engage)
    return crossed

  return call


def engage_constraint_termination():
  """Factory: the microspine engage constraint cone (CaT).

  Per pad, on pads touching the wall (``contact_count > 0``) but not
  engaged (C1-C3) or at any time (C4), normalised violations
  ``v = clip((x - limit) / limit, 0, 1)``:

    C1 tilt        angle between the pad face axis and the wall inward
                   normal (``face_cos``) > ``tilt_limit_deg``
    C2 grind/slam  tangential pad speed while touching > ``grind_limit_mps``;
                   normal impact speed on the contact rising edge >
                   ``slam_limit_mps``
    C3 force       unengaged pad press force > ``force_limit_n``
    C4 unscheduled wall touch (or press > ``touch_force_n``) while the
                   schedule says OFF and no rising edge is due within
                   ``edge_lead_time_s`` (binary)

  ``mode="cat"``: terminate with probability δ = max_i p_i·v_i (Bernoulli
  inside the term — rsl_rl sees an ordinary ``terminated``); ``"hard"``:
  any v_i > 0; ``"off"``: never (play). Exempt: ``dr_release_pending``
  pads. The violations and δ are cached on the callable
  (``last_violations``, ``last_delta``) for the metric lanes.

  Persistence (v1b, 2026-08-21): C1, C2-grind, C3 and C4 count only once
  the violating condition has held ``persist_steps`` consecutive policy
  steps per pad (5 = 0.1 s, the weld engine's own debounce). The v1 launch
  showed why: with σ = 0.6 exploration a 0.1 rad/step target jitter moves a
  touching pad at ~1 m/s and slams it with tens of N, so a third of all
  envs violated "grind" on EVERY step — the cone was killing the noise
  floor, lifetimes stayed at 6 steps while σ collapsed 0.60 → 0.40 in 44
  iterations (the v14 desert by another door). A 1-2 step spike is noise;
  a 0.1 s sustained grind/press/crooked touch is behaviour. The slam test
  (impact speed on the contact rising edge) is inherently one-step and
  stays instantaneous.
  """
  state: dict[str, torch.Tensor | None] = {
    "touch": None,
    "vn": None,
    "n_tilt": None,
    "n_grind": None,
    "n_force": None,
    "n_unsched": None,
    "seed_lost": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    affordance_command_name: str = "microspine_affordance",
    mode: str = "cat",
    tilt_limit_deg: float = 15.0,
    grind_limit_mps: float = 0.30,
    slam_limit_mps: float = 0.6,
    force_limit_n: float = 80.0,
    touch_force_n: float = 3.0,
    edge_lead_time_s: float = 2.0,
    p_tilt: float = 0.5,
    p_vel: float = 0.5,
    p_force: float = 1.0,
    p_unscheduled: float = 1.0,
    ee_indices: tuple[int, ...] = (0, 1, 2, 3),
    persist_steps: int = 5,
    seed_settle_exempt_s: float = 0.5,
  ) -> torch.Tensor:
    from mjlab.utils.lab_api.math import quat_apply  # cloudpickle-safe

    motion = env.command_manager.get_term(command_name)
    aff = env.command_manager.get_term(affordance_command_name)
    executor = aff._get_executor()
    engaged = aff.engaged
    touch = aff.contact_count > 0
    press = aff.pad_press_force
    face_cos = aff.face_cos.clamp(-1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(face_cos))

    vel = motion.robot.data.body_link_lin_vel_w[:, executor._ee_body_ids, :]
    n_local = torch.zeros(vel.shape[0], 3, device=vel.device)
    n_local[:, 0] = -1.0
    n_w = quat_apply(motion.wall_quat, n_local).unsqueeze(1)  # outward
    v_normal_in = -(vel * n_w).sum(dim=-1)  # > 0 = moving INTO the wall
    v_tan = (vel - (vel * n_w).sum(dim=-1, keepdim=True) * n_w).norm(dim=-1)

    prev_touch = state["touch"]
    prev_vn = state["vn"]
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    if prev_touch is None or prev_touch.shape != touch.shape:
      prev_touch = torch.zeros_like(touch)
      prev_vn = torch.zeros_like(v_normal_in)
    prev_touch = torch.where(fresh, torch.zeros_like(prev_touch), prev_touch)
    prev_vn = torch.where(fresh, torch.zeros_like(prev_vn), prev_vn)
    rising = touch & ~prev_touch
    impact = torch.maximum(v_normal_in, prev_vn)

    def _over(value: torch.Tensor, limit: float) -> torch.Tensor:
      return ((value - limit) / max(limit, 1.0e-6)).clamp(0.0, 1.0)

    lead = edge_lead_time_s / motion.cycle_time_s
    regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
    free = ~engaged & ~aff.dr_release_pending
    # Seed-settle exemption, the rip law's mirror (monitor finding, v1b):
    # a seeded weld torn off by the birth transient leaves a free pad
    # pressed on the wall at settle-transient speeds/forces — the cone
    # killed it in the same 0.5 s the rip law had just spared. A pad whose
    # seed was lost inside the settle window is exempt from C1-C4 until the
    # window closes (counters keep running, nothing bills).
    seed_lost = state["seed_lost"]
    if seed_lost is None or seed_lost.shape != touch.shape:
      seed_lost = torch.zeros_like(touch)
    seed_lost = torch.where(fresh.expand_as(touch), torch.zeros_like(seed_lost), seed_lost)
    was_seeded = getattr(aff, "released_was_seeded", None)
    if was_seeded is not None:
      seed_lost = seed_lost | was_seeded
    seed_lost = seed_lost & ~engaged  # a re-weld ends the exemption
    state["seed_lost"] = seed_lost
    if seed_settle_exempt_s > 0.0:
      settle_steps = max(1, int(round(seed_settle_exempt_s / env.step_dt)))
      settling = (env.episode_length_buf <= settle_steps).unsqueeze(-1)
      free = free & ~(seed_lost & settling)
    touching = touch & free
    v_tilt = _over(tilt_deg, tilt_limit_deg) * touching.float()
    v_grind = _over(v_tan, grind_limit_mps) * touching.float()
    v_slam = _over(impact, slam_limit_mps) * (rising & free).float()
    v_force = _over(press, force_limit_n) * free.float()
    v_unsched = ((touch | (press > touch_force_n)) & ~motion.ref_engaged & ~regrip_soon & free).float()

    # Persistence: a violation bills only once it has held persist_steps
    # consecutive steps on that pad (counters reset on a clean step and at
    # episode start); the billed magnitude is the current step's.
    k = max(1, int(persist_steps))
    fresh_pad = fresh.expand_as(touch)
    for key, v in (
      ("n_tilt", v_tilt),
      ("n_grind", v_grind),
      ("n_force", v_force),
      ("n_unsched", v_unsched),
    ):
      count = state[key]
      if count is None or count.shape != v.shape:
        count = torch.zeros_like(v, dtype=torch.long)
      count = torch.where(fresh_pad, torch.zeros_like(count), count)
      count = (count + 1) * (v > 0.0).long()
      state[key] = count
    v_tilt = v_tilt * (state["n_tilt"] >= k).float()
    v_grind = v_grind * (state["n_grind"] >= k).float()
    v_force = v_force * (state["n_force"] >= k).float()
    v_unsched = v_unsched * (state["n_unsched"] >= k).float()

    idx = list(ee_indices)
    violations = {
      "tilt": v_tilt[:, idx],
      "grind": v_grind[:, idx],
      "slam": v_slam[:, idx],
      "force": v_force[:, idx],
      "unscheduled": v_unsched[:, idx],
    }
    weighted = torch.stack(
      [
        p_tilt * violations["tilt"],
        p_vel * violations["grind"],
        p_vel * violations["slam"],
        p_force * violations["force"],
        p_unscheduled * violations["unscheduled"],
      ],
      dim=0,
    )
    delta = weighted.amax(dim=(0, 2)).clamp(0.0, 1.0)  # (E,)
    call.last_violations = violations
    call.last_delta = delta

    state["touch"] = touch.clone()
    state["vn"] = v_normal_in.clone()

    if mode == "off":
      return torch.zeros_like(delta, dtype=torch.bool)
    if mode == "hard":
      return torch.stack(list(violations.values()), dim=0).amax(dim=(0, 2)) > 0.0
    if mode != "cat":
      raise ValueError(f"Unknown constraint mode {mode!r}.")
    return torch.rand_like(delta) < delta

  call.last_violations = None
  call.last_delta = None
  return call


def scheduled_contact_loss_termination():
  """Factory: terminate when a SCHEDULED contact stays missing past grace.

  The LadderMan enforcement (arXiv 2606.05873, Table 5: "required ladder
  contacts lost for >30 steps -> terminate"), this package's own copy of the
  2026-08-16 law: one counter per EE over ``ref_engaged & ~satisfied``
  (satisfied = welded within ``anchor_radius_m`` of the EE's OWN scheduled
  anchor — on a featureless wall every spot welds); ANY reset of the
  condition clears it. Covers late first latch, ripped-and-not-re-seated
  and skipped re-engagement alike. R4 reseat holds are exempt.
  """
  state: dict[str, torch.Tensor | None] = {"count": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    affordance_command_name: str = "microspine_affordance",
    grace_s: float = 2.0,
    anchor_radius_m: float | None = None,
  ) -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)
    aff = env.command_manager.get_term(affordance_command_name)
    satisfied = aff.engaged
    if anchor_radius_m is not None:
      executor = aff._get_executor()
      dist = (executor.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
      satisfied = satisfied & (dist <= anchor_radius_m)
    missing = motion.ref_engaged & ~satisfied & ~aff.dr_release_pending
    count = state["count"]
    if count is None or count.shape != missing.shape:
      count = torch.zeros_like(missing, dtype=torch.long)
    fresh = env.episode_length_buf <= 1
    count = torch.where(fresh.unsqueeze(-1), torch.zeros_like(count), count)
    count = (count + 1) * missing.long()
    state["count"] = count
    grace_steps = max(1, int(round(grace_s / env.step_dt)))
    return (count > grace_steps).any(dim=-1)

  return call
