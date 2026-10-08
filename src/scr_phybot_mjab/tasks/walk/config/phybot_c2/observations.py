"""Walking observations. Term order is the checkpoint and export layout contract."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.common.phybot_c2 import (
  BASE_LIN_VEL_SENSOR,
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_SITE_NAMES,
  policy_joint_cfg,
  standing_actor_terms,
)
from scr_phybot_mjab.tasks.walk import mdp

from .commands import TWIST_COMMAND


def walk_observations(amp: bool) -> dict[str, ObservationGroupCfg]:
  """Ten stacked frames for actor and critic, as in the official PhyBot Lab.

  The actor sees the standing tasks' actor terms with the twist command.
  The critic adds the base velocity and the feet's height, air time and
  contact; its other terms are the actor's term cfgs themselves. The amp
  variant adds the discriminator's group.
  """
  actor = standing_actor_terms(TWIST_COMMAND)
  critic = {
    "base_lin_vel": ObservationTermCfg(func=base_mdp.builtin_sensor, params={"sensor_name": BASE_LIN_VEL_SENSOR}),
    **actor,
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height, params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)}
    ),
    "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params={"sensor_name": FEET_GROUND_CONTACT_SENSOR}),
    "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": FEET_GROUND_CONTACT_SENSOR}),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces, params={"sensor_name": FEET_GROUND_CONTACT_SENSOR}
    ),
  }
  groups = {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=True, history_length=10),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False, history_length=10),
  }
  if amp:
    groups["amp"] = _amp_group()
  return groups


def _amp_group() -> ObservationGroupCfg:
  """The discriminator's input, one frame: policy-joint positions and velocities and the base twist."""
  joints = policy_joint_cfg()
  return ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": joints}),
      "joint_vel": ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": joints}),
      "base_lin_vel": ObservationTermCfg(func=base_mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
  )
