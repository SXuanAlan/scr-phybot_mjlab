"""Squat observations. Term order is the checkpoint and export layout contract."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from scr_phybot_mjab.tasks.common.phybot_c2 import BASE_LIN_VEL_SENSOR, standing_actor_terms

from .commands import MOTION_COMMAND


def squat_observations() -> dict[str, ObservationGroupCfg]:
  """A 10-step history of the standing tasks' actor terms, whose command is the phase clock.

  The critic adds the base velocity; its other terms are the actor's term
  cfgs themselves.
  """
  actor = standing_actor_terms(MOTION_COMMAND)
  critic = {
    "base_lin_vel": ObservationTermCfg(func=base_mdp.builtin_sensor, params={"sensor_name": BASE_LIN_VEL_SENSOR}),
    **actor,
  }
  return {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=True, history_length=10),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False, history_length=10),
  }
