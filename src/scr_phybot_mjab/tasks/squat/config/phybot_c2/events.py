"""Startup domain randomization of the squat task."""

from __future__ import annotations

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_FOOT_COLLISION_GEOMS

from .scene import TORSO_BODY


def squat_events() -> dict[str, EventTermCfg]:
  """Friction, torso mass and encoder bias, sampled once per environment.

  No pushes: the feet stay planted, and strong periodic velocity pushes
  teach the policy to stomp while it keeps the joint angles.
  """
  return {
    # Friction is split as in walk: non-foot geoms 0.6-1.0 here, feet below.
    "body_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(?!(left|right)_ankle_roll_collision$).*_collision$",)),
        "operation": "abs",
        "ranges": (0.6, 1.0),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,)),
        "operation": "add",
        "ranges": (-5.0, 5.0),
      },
    ),
    # Each foot is sampled on its own, so the policy cannot rely on perfectly
    # symmetric contact, which the real floor rarely gives.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(PHYBOT_C2_FOOT_COLLISION_GEOMS,)),
        "operation": "abs",
        "ranges": (0.4, 1.0),
        "shared_random": False,
      },
    ),
    # The policy observes joint angles offset by a fixed per-episode bias (the
    # sim-to-real encoder error). The tracking rewards read the true joint
    # positions, so only what the policy sees changes.
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }
