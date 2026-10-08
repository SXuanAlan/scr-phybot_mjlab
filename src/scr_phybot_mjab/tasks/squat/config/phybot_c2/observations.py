"""Squat observations. Term order is the checkpoint and export layout contract."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from scr_phybot_mjab.tasks.common.observations import projected_gravity_from_framequat_sensor
from scr_phybot_mjab.tasks.common.phybot_c2 import (
  BASE_ANG_VEL_SENSOR,
  BASE_LIN_VEL_SENSOR,
  BASE_QUAT_SENSOR,
  policy_joint_cfg,
)

from .commands import MOTION_COMMAND


def squat_observations() -> dict[str, ObservationGroupCfg]:
  """A 10-step history of proprioception and the phase clock, as in walk.

  The real robot has no linear-velocity estimate, so only the critic sees
  the base velocity; its other terms are the actor's term cfgs themselves.
  """
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": BASE_ANG_VEL_SENSOR},
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor,
      params={"sensor_name": BASE_QUAT_SENSOR},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": policy_joint_cfg()},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": policy_joint_cfg()},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": MOTION_COMMAND}),
  }
  critic = {
    "base_lin_vel": ObservationTermCfg(func=base_mdp.builtin_sensor, params={"sensor_name": BASE_LIN_VEL_SENSOR}),
    **actor,
  }
  return {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=True, history_length=10),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False, history_length=10),
  }
