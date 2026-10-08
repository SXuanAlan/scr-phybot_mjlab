"""Reset perturbations, pushes and startup randomization of friction and torso mass."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_FOOT_COLLISION_GEOMS,
  PHYBOT_C2_POLICY_JOINT_NAMES,
)
from scr_phybot_mjab.tasks.common.phybot_c2 import TORSO_BODY
from scr_phybot_mjab.tasks.walk import mdp

from .variants import WalkVariant

_RESET_POSE_RANGE = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)}


def walk_events(variant: WalkVariant) -> dict[str, EventTermCfg]:
  """The amp variant resets through its amp_reset command, so it has no reset events; only walk is pushed."""
  if variant == "walk":
    events = _walk_resets()
  elif variant == "stand":
    events = _stand_resets()
  else:
    events = {}
  events |= _domain_rand()
  if variant == "stand":
    events["reset_robot_upper_joints"] = _stand_upper_body_reset()
  return events


def _walk_resets() -> dict[str, EventTermCfg]:
  return {
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": dict(_RESET_POSE_RANGE),
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.1, 0.1),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={
        "position_range": (0.8, 1.2),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Frequent pushes with vertical and angular components (MJLab G1 style)
    # force the policy to learn firm footing instead of a compliant crouch.
    "push_robot": EventTermCfg(
      func=base_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.8, 0.8),
          "y": (-0.8, 0.8),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        }
      },
    ),
  }


def _stand_resets() -> dict[str, EventTermCfg]:
  """One reset perturbation makes the recovery problem; repeated pushes would
  make a final stationary phase impossible.

  The leg offsets stay small enough that the reset itself cannot cross the
  feet, while the upper body gets the full recovery range. Unlike walk's
  scaled reset, the offsets are in radians, so zero-valued joints are
  perturbed as well.
  """
  return {
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": dict(_RESET_POSE_RANGE),
        "velocity_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          "z": (0.0, 0.0),
          "roll": (-0.10, 0.10),
          "pitch": (-0.10, 0.10),
          "yaw": (-0.10, 0.10),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (-0.05, 0.05),
        "asset_cfg": SceneEntityCfg("robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES[:12], preserve_order=True),
      },
    ),
  }


def _stand_upper_body_reset() -> EventTermCfg:
  return EventTermCfg(
    func=base_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.15, 0.15),
      "velocity_range": (-0.10, 0.10),
      "asset_cfg": SceneEntityCfg("robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES[12:], preserve_order=True),
    },
  )


def _domain_rand() -> dict[str, EventTermCfg]:
  return {
    # MuJoCo geom friction; the official C2 also randomizes dynamic friction
    # and restitution buckets, which have no direct hook here. The feet get
    # their own, wider range below.
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
    # Foot-floor sliding friction. Listed after body_friction, so it wins
    # even if the negative lookahead above ever fails to exclude the feet.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(PHYBOT_C2_FOOT_COLLISION_GEOMS,)),
        "operation": "abs",
        "ranges": (0.4, 1.0),
        "shared_random": True,
      },
    ),
  }
