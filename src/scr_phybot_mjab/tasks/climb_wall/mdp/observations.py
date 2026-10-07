"""Custom observation functions for the wall-climbing task."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "ee_disengage_time_obs",
  "height_above_spawn",
  "microspine_anchor_load",
  "microspine_contact_fraction",
  "microspine_cooldown_obs",
  "microspine_engaged_state",
  "microspine_forced_release",
  "microspine_hold_margin",
  "microspine_pad_normal_force",
  "prev_action",
  "prev_prev_action",
  "privileged_critic_group",
  "velocity_command",
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from the previous step (one step before current `last_action`)."""
  return env.action_manager.prev_action


def prev_prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from two steps ago."""
  return env.action_manager.prev_prev_action


def height_above_spawn(
  env: ManagerBasedRlEnv,
  baseline_z: float = 2.0,
  motion_command_name: str = "climb_motion",
) -> torch.Tensor:
  """Along-wall base altitude minus a fixed baseline. Shape (E, 1).

  The critic's altitude channel: without it the height income reads as pure
  noise (hidden-state variance in V). Measured ALONG the climbing surface —
  ``(root - wall_base_line) . wall_up`` — which is world z minus baseline on
  a vertical wall (bit-identical to the retired world-z form) and, under
  wall tilt, the same scalar the height rewards pay on. Using raw world z
  here would mix wall-normal standoff into "altitude" at every tilted rung.
  """
  motion = env.command_manager.get_term(motion_command_name)
  robot_pos = env.scene["robot"].data.root_link_pos_w
  altitude = ((robot_pos - motion.wall_pivot_w) * motion.wall_up_w).sum(dim=-1)
  return (altitude - baseline_z).unsqueeze(-1)  # (E, 1)


def velocity_command(env: ManagerBasedRlEnv, command_name: str = "climb_velocity") -> torch.Tensor:
  """Commanded linear velocity (world frame), shape (E, 3).

  Read from the live command term, so the observation and the progress
  terms share one per-env target.
  """
  return env.command_manager.get_command(command_name)


def microspine_pad_normal_force(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE pad-normal reaction force in Newtons, (E, 4) — load-cell model.

  Models the pad-mounted force channel the gimbal sensing co-design will
  provide: the wall-normal reaction the pad transmits, continuous across the
  latch event. Two sim sources are blended because a latched pad's contact
  proxies are retracted, so ``pad_press_force`` reads structurally zero exactly when the anchor
  carries its biggest load. Released pads report the sphere-grid compression
  sum; latched pads the anchor equality-constraint normal reaction
  (``anchor_load_tnl[..., 2]`` is f_n, positive = pull-off tension, so
  compression is its negation clamped at zero). Privileged tensors serve
  only as the signal SOURCE; the output is the physically continuous
  quantity a real load cell measures. Noise/latency/quantization live on
  the ObservationTermCfg, to be recalibrated against the real sensor.
  """
  affordance = env.command_manager.get_term(command_name)
  anchor_compression = (-affordance.anchor_load_tnl[..., 2]).clamp(min=0.0)
  return torch.where(affordance.engaged, anchor_compression, affordance.pad_press_force)


def microspine_engaged_state(env: ManagerBasedRlEnv, command_name: str = "microspine_affordance") -> torch.Tensor:
  """Per-EE microspine latch state, shape ``(E, num_ee)`` in ``{0, 1}``.

  Read from the affordance command, the source of truth, which is already
  set on the reset observation; the physics event mirrors it into the
  equalities one event pass later.
  """
  return env.command_manager.get_term(command_name).engaged.float()


def microspine_contact_fraction(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  spheres_per_ee: int = 4,
) -> torch.Tensor:
  """Per-EE engaged fraction of the virtual spine field (seating quality).

  A rigid pad on a flat wall generically touches with a single proxy, so a
  touching-proxy count would sit at 0.25 whenever the pad presses. The spine
  fraction is what the engage gate consumes ('how flat is the pad'), so the
  policy sees the quantity it is judged on.
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.spine_fraction.clamp(0.0, 1.0)


def microspine_cooldown_obs(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  cooldown_steps: int = 30,
) -> torch.Tensor:
  """Normalized per-EE re-engagement cooldown known to the grip controller."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.cooldown_steps.float() / max(1, cooldown_steps)


def ee_disengage_time_obs(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  clip: float = 0.5,
) -> torch.Tensor:
  """Normalized per-EE continuous disengage time, shape ``(E, H)``.

  The timer is maintained by ``MicrospineConstraint`` and reset to 0 while an
  EE is engaged. Clipping keeps the observation bounded and matches the
  max-time gate used by the capped disengage-time gait reward.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  physics = term_cfg.func
  if not hasattr(physics, "current_disengage_time"):
    return torch.zeros_like(physics.engaged, dtype=torch.float32)

  denom = max(float(clip), 1.0e-6)
  return physics.current_disengage_time.clamp(max=denom) / denom


# ---------------------------------------------------------------------------
# Privileged microspine state (critic-only).
#
# These read the limit-surface diagnostics that MicrospineAffordance maintains
# They are sim-only quantities (a real robot cannot sense its hold margin), so
# they belong in the critic group, never the actor.
# ---------------------------------------------------------------------------


def microspine_hold_margin(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE normalized distance to the hold envelope, shape (E, H).

  1 = unloaded, 0 = on the envelope boundary, < 0 = violating (about to
  release once debounced).
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.hold_margin


def microspine_anchor_load(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE anchor load in the engage frame (f_t, f_l, f_n), shape (E, 3H).

  Raw Newtons — pair with ``scale=0.01`` in the term cfg to keep the
  normalized magnitude O(1).
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.anchor_load_tnl.flatten(1)


def microspine_forced_release(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE limit-surface release flag this step, shape (E, H) in {0, 1}."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float()


def privileged_critic_group(
  actor_group: ObservationGroupCfg,
  command_name: str = "microspine_affordance",
) -> ObservationGroupCfg:
  """Asymmetric actor-critic group: actor terms + privileged microspine state.

  The extras (hold margin, anchor loads, forced releases) are what a real
  robot cannot sense. Term cfgs are re-instantiated (``dataclasses.replace``)
  so the two groups never share mutable cfg objects; the shallow copies
  deliberately SHARE the ``params`` dicts with the actor terms (preexisting
  behavior — do not deep-copy). Call AFTER every actor-term edit so the
  copies pick them up.
  """
  critic_terms = {name: dataclasses.replace(term) for name, term in actor_group.terms.items()}
  critic_terms["microspine_hold_margin"] = ObservationTermCfg(
    func=microspine_hold_margin,
    params={"command_name": command_name},
  )
  critic_terms["microspine_anchor_load"] = ObservationTermCfg(
    func=microspine_anchor_load,
    params={"command_name": command_name},
    scale=0.01,  # Newtons → O(1).
  )
  critic_terms["microspine_forced_release"] = ObservationTermCfg(
    func=microspine_forced_release,
    params={"command_name": command_name},
  )
  return ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
