"""Transition-only termination variants.

``standing_wall_contact_termination`` — the mole family's structural
ending (user doctrine, 2026-08-13): during the standing prefix, only the
microspine pads and the toes may ever touch the wall; for every other
link, wall contact IS a fall and ends the episode exactly like
``base_height``. The v16-v21 fence iterations priced proximity and the
policy folded around whichever geoms the list named (head -> shoulder ->
knee + dangling arm); a termination has no such null space.

Mechanically these links carry NO wall collision at all (the clean
pad/toe contact interface), so "contact" is geometric: the geom SURFACE
crossing the wall plane. That state cannot exist on hardware — the real
wall pushes back — so no value may flow through it. Standing prefix only:
the hanging stances later legally bring the chest close, and there the
sim-gap state (chest pressed on the wall) maps to a legal hardware state,
not a fall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import _motion

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "scheduled_contact_loss_termination",
  "standing_wall_contact_termination",
)


def scheduled_contact_loss_termination():
  """Factory: terminate when a SCHEDULED contact stays missing past grace.

  The LadderMan enforcement (arXiv 2606.05873, Table 5: "required ladder
  contacts lost for >30 steps -> terminate"), sized for the mount: the
  open-loop clock is only honest if desyncing from its contact schedule is
  fatal — the mismatch DISCOUNT provably was not (v52 @1706: LH mastered,
  RH occupancy pinned at 0.002, the chain parked at the door it never had
  to open). One counter per EE over ``ref_engaged & ~engaged``; ANY reset
  of the condition (weld made, schedule released) clears it. Covers all
  three crimes with no special cases: late first latch, ripped-and-not-
  re-seated (a rip starts the same clock), and skipped re-engagement in
  the hand walk-up. R4 reseat holds are exempt; the stand-first clamp
  never reaches an engaged frame, so the term is inert there. Safe only
  in the post-v19 economy (living net-positive, +27 measured) — under an
  inverted ledger an executioner is a free exit door.

  ``grace_s`` 2.0 vs LadderMan's ~0.6: our latch is approach + press +
  debounce under blind-delivery scatter, not a rung grab.
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
      # v54 position-aware law: on a featureless wall EVERY spot welds, so
      # "engaged" alone let 7.5 cm-off welds stop the death clock (the
      # v53 probe's verdict — LadderMan never had this hole: you cannot
      # grab a rung that is not there). Only a weld within radius of the
      # EE's OWN scheduled anchor satisfies the schedule.
      ex = aff._get_executor()
      dist = (ex.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
      satisfied = satisfied & (dist <= anchor_radius_m)
    missing = (
      motion.ref_engaged & ~satisfied & ~aff.dr_release_pending
    )
    count = state["count"]
    if count is None or count.shape != missing.shape:
      count = torch.zeros_like(missing, dtype=torch.long)
    fresh = env.episode_length_buf <= 1
    count = torch.where(
      fresh.unsqueeze(-1), torch.zeros_like(count), count
    )
    count = (count + 1) * missing.long()
    state["count"] = count
    grace_steps = max(1, int(round(grace_s / env.step_dt)))
    return (count > grace_steps).any(dim=-1)

  return call


def standing_wall_contact_termination():
  """Factory: terminate on illegal wall contact during the standing prefix.

  ``geom_names`` should list every link that never legitimately reaches
  the wall before first engage (head, torso, pelvis, hips, knees,
  shoulders, upper arms). Hands, forearms and elbows stay out — the place
  pose lays them on the wall (reference elbow bottoms out at +2.9 cm and
  the delivery scatter can erase that margin) — as do feet and toes,
  whose pad/toe contacts are real physics under the press economy.
  Geom SURFACES, not link origins (the v17 telltale lesson: origins
  under-read by up to ~13 cm).
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
    threshold_m: float = 0.0,
  ) -> torch.Tensor:
    from mjlab.utils.lab_api.math import quat_apply

    motion = _motion(env, command_name)
    first_engage = getattr(motion, "_first_engage_frame", None)
    if first_engage is None:
      return torch.zeros(
        motion.ref_engaged.shape[0],
        dtype=torch.bool,
        device=motion.ref_engaged.device,
      )
    if state["ids"] is None:
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
    return standing & (dist.min(dim=-1).values < threshold_m)

  return call
