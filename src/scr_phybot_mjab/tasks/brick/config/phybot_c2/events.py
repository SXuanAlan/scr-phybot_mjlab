"""Brick events: the scene reset, the grip constraint and the startup randomization."""

from __future__ import annotations

import math

from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.constraint import make_constraint_event_cfg


def brick_events(mc: MicrospineCfg, randomize: bool) -> dict[str, EventTermCfg]:
  events = {
    "reset_scene": EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset"),
    "microspine_grip": make_constraint_event_cfg(mc, affordance_name="microspine_affordance"),
  }
  if randomize:
    events["brick_density"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("brick", body_names=("brick",)),
        "alpha_range": (0.5 * math.log(0.5), 0.5 * math.log(1.5)),
      },
    )
    events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_ankle_roll_collision$",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    )
    events["servo_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    )
  return events
