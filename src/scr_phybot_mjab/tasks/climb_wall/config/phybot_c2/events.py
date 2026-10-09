"""Climbing events: the scene-default reset.

env_cfgs.py builds the microspine grip event from the finished pad config,
under MICROSPINE_GRIP_EVENT.
"""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg

# A manager key, public like the command names in commands.py.
MICROSPINE_GRIP_EVENT = "microspine_grip"


def climb_reset_events() -> dict[str, EventTermCfg]:
  """Only the scene-default reset: the motion command performs reference-state
  initialization after the reset events, so joint or base perturbation events
  would be overwritten."""
  return {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
  }
