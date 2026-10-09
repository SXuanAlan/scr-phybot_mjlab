"""Climbing terminations: timeout, a fall along the wall, divergence, lost tracking and lost support."""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from scr_phybot_mjab.tasks.climb_wall import mdp

from .commands import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND


def climb_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # The robot lost grip and fell toward the wall base. Along-wall altitude,
    # not world z: a world-z bar would fire at t=0 on tilted walls.
    "base_height": TerminationTermCfg(
      func=mdp.root_below_wall_altitude,
      params={"minimum_altitude": 0.3},
    ),
    # NaN and divergence guard; mjlab has no global NaN sink.
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 20.0},
    ),
    # BeyondMimic early termination: end rollouts whose tracked bodies left
    # the reference (fallen or tangled).
    "bad_tracking": TerminationTermCfg(
      func=mdp.motion_body_tracking_divergence,
      params={
        "command_name": MOTION_COMMAND,
        "position_threshold": 0.3,
        "orientation_threshold": 1.0,
        "height_threshold": 0.25,
      },
    ),
    "lost_support": TerminationTermCfg(
      func=mdp.prolonged_support_loss(), params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "timeout_s": 0.4}
    ),
  }
