"""Climbing metrics: microspine occupancy and health, actuation saturation and climb progress."""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg

from scr_phybot_mjab.tasks.climb_wall import mdp

from .commands import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND


def climb_metrics() -> dict[str, MetricsTermCfg]:
  """Microspine occupancy and health, actuation saturation and climb progress."""
  occupancy = {
    f"{name}_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": index},
    )
    for index, name in enumerate(("left_hand", "right_hand", "left_foot", "right_foot"))
  }
  return {
    "microspine_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    **occupancy,
    "microspine_forced_release_rate_hz": MetricsTermCfg(
      func=mdp.microspine_forced_release_rate,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "microspine_engage_event_rate_hz": MetricsTermCfg(
      func=mdp.microspine_engage_events,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "microspine_hold_margin_engaged": MetricsTermCfg(
      func=mdp.microspine_hold_margin_engaged,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "action_saturation_fraction": MetricsTermCfg(
      func=mdp.action_saturation_fraction,
      params={"clip": 1.0},
    ),
    "joint_target_clamp_fraction": MetricsTermCfg(
      func=mdp.joint_target_clamp_fraction,
      params={"action_name": "joint_pos"},
    ),
    "joint_soft_limit_violation": MetricsTermCfg(
      func=mdp.joint_soft_limit_violation,
      params={"asset_name": "robot"},
    ),
    "motion_height_gained_m": MetricsTermCfg(
      func=mdp.motion_height_gained,
      params={"command_name": MOTION_COMMAND},
    ),
  }
