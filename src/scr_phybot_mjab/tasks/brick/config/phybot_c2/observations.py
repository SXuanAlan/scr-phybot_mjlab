"""Brick observations. Term order is the checkpoint and export layout contract."""

from __future__ import annotations

import copy

from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg

from scr_phybot_mjab.tasks.brick import mdp
from scr_phybot_mjab.tasks.common.observations import projected_gravity_from_framequat_sensor
from scr_phybot_mjab.tasks.common.phybot_c2 import BASE_ANG_VEL_SENSOR, BASE_QUAT_SENSOR, policy_joint_cfg


def brick_observations() -> dict[str, ObservationGroupCfg]:
  """A 10-step actor history of proprioception and the task command; the critic adds the brick's physical state."""
  asset = policy_joint_cfg()
  actor = {
    "base_ang_vel": ObservationTermCfg(func=base_mdp.builtin_sensor, params={"sensor_name": BASE_ANG_VEL_SENSOR}),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor, params={"sensor_name": BASE_QUAT_SENSOR}
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel, params={"asset_cfg": asset}, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel, params={"asset_cfg": asset}, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
    "task_command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": "brick_task"}),
  }
  return {
    "actor": ObservationGroupCfg(terms=actor, history_length=10, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(
      terms={**copy.deepcopy(actor), "physical_state": ObservationTermCfg(func=mdp.privileged_state)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
