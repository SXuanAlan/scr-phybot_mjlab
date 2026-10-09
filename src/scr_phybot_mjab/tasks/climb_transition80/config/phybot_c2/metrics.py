"""Transition metrics: climb's, plus handoff readiness, rips, hold margins and the engage-cone lanes."""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg

from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.commands import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.metrics import climb_metrics

_COMMANDS = {"command_name": MOTION_COMMAND, "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND}


def transition80_metrics() -> dict[str, MetricsTermCfg]:
  return {
    **climb_metrics(),
    "mount_handoff_ready": MetricsTermCfg(func=mdp.mount_handoff_ready, params=dict(_COMMANDS)),
    "anchor_rips": MetricsTermCfg(func=mdp.scheduled_anchor_rips, params={"reward_name": "anchor_rip_penalty"}),
    "anchor_ripped_pads": MetricsTermCfg(
      func=mdp.scheduled_anchor_ripped_pads, params={"reward_name": "anchor_rip_penalty"}
    ),
    "scheduled_hold_margin": MetricsTermCfg(func=mdp.scheduled_hold_margin, params=dict(_COMMANDS)),
    "rh_hold_margin": MetricsTermCfg(func=mdp.scheduled_hold_margin, params={**_COMMANDS, "ee_index": 1}),
    **{
      f"constraint_{key}_violation": MetricsTermCfg(
        func=mdp.engage_constraint_violation, params={"termination_name": "engage_constraint", "key": key}
      )
      for key in ("tilt", "grind", "slam", "force", "unscheduled")
    },
    "constraint_delta": MetricsTermCfg(
      func=mdp.engage_constraint_delta, params={"termination_name": "engage_constraint"}
    ),
  }
