"""PHYBot C2 wall-climbing environment configurations.

The climb imitates a CAD reference cycle with BeyondMimic-style tracking
rewards and failure-adaptive RSI, on four microspine pads (hands and feet)
judged by the empirical unit model. The main task uses twelve units per pad
and a curriculum to 90 degrees; the Corners variants use the four corner
units, with a curriculum to 60 degrees or a fixed 80-degree wall.

Each section lives in its own module and is built directly in its final
form; this module composes them and applies the play overrides.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from scr_phybot_mjab.microspine.constraint import make_constraint_event_cfg
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.motions import (
  CAD_CLIMB_CORNERS60_MOTION,
  CAD_CLIMB_CORNERS80_MOTION,
  CAD_CLIMB_MOTION,
)

from .commands import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND, climb_commands
from .curricula import climb_curriculum
from .domain_rand import DOMAIN_RAND_EVENTS, actor_obs_noise, domain_rand_events
from .events import MICROSPINE_GRIP_EVENT, climb_reset_events
from .metrics import climb_metrics
from .microspine import climb_microspine_cfg
from .observations import climb_observations
from .rewards import climb_rewards
from .robot import climb_actions, climb_robot_cfg
from .scene import DECIMATION, climb_scene, climb_sim, climb_viewer
from .stance import climb_toe_contacts
from .terminations import climb_terminations


def phybot_c2_climb_wall_env_cfg(
  play: bool = False,
  domain_rand: bool = True,
  unit_layout: str | None = None,
  wall_angle_deg: float | None = None,
  *,
  motion_file: Path = CAD_CLIMB_MOTION,
  angle_stages=mdp.WALL_ANGLE_STAGES,
) -> ManagerBasedRlEnvCfg:
  """The constraint-based PHYBot C2 wall-climbing configuration.

  Args:
    play: evaluation mode (single env, pinned curriculum, no randomization).
    domain_rand: observation noise and startup friction, mass and encoder
      randomization.
    unit_layout: active CAD unit sites, "all" (default) or the four
      "corners" of the hardware variants.
    wall_angle_deg: pin training and play to this angle from horizontal,
      without jitter. None keeps the curriculum's angle ladder.
    motion_file: the reference cycle to imitate.
    angle_stages: the wall-angle curriculum ladder.
  """
  if wall_angle_deg is not None and not 0.0 < wall_angle_deg <= 90.0:
    raise ValueError("wall_angle_deg must be in (0, 90]")
  microspine_cfg = climb_microspine_cfg(unit_layout or "all")
  toe_contacts = climb_toe_contacts()
  events = {
    **climb_reset_events(),
    # Takes a shallow copy of the finished microspine cfg (sharing only its
    # limit_surface), so microspine_cfg must not change after this.
    MICROSPINE_GRIP_EVENT: make_constraint_event_cfg(microspine_cfg, affordance_name=MICROSPINE_AFFORDANCE_COMMAND),
  }
  if domain_rand:
    events |= domain_rand_events()
  cfg = ManagerBasedRlEnvCfg(
    decimation=DECIMATION,
    episode_length_s=30.0,
    scene=climb_scene(climb_robot_cfg(toe_contacts), microspine_cfg, toe_contacts),
    sim=climb_sim(),
    viewer=climb_viewer(),
    observations=climb_observations(microspine_cfg, actor_obs_noise() if domain_rand else None),
    actions=climb_actions(),
    commands=climb_commands(microspine_cfg, motion_file),
    rewards=climb_rewards(),
    curriculum=climb_curriculum(angle_stages),
    events=events,
    terminations=climb_terminations(),
    metrics=climb_metrics(),
  )
  if play:
    apply_play_overrides(cfg)
  if wall_angle_deg is not None:
    cfg.curriculum["wall_angle"].params["angle_stages"] = [{"iter": 0, "angle_deg": float(wall_angle_deg)}]
    cfg.curriculum["wall_angle"].params["angle_jitter_deg"] = 0.0
  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Deterministic single-env evaluation on the final task."""
  # One world: the training batch would simulate thousands of invisible robots.
  cfg.scene.num_envs = 1
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  # Remove the randomization events entirely: neutral ranges would still
  # overwrite the XML frictions under operation="abs".
  for name in DOMAIN_RAND_EVENTS:
    cfg.events.pop(name, None)
  # play.py can still replace the pinned stage from the CLI or the checkpoint
  # iteration. Without jitter, rollouts are deterministic at the pinned angle.
  mdp.pin_curricula_to_final_stage(cfg)
  cfg.curriculum["wall_angle"].params["angle_jitter_deg"] = 0.0
  # Start from the cycle's canonical stance, so a rollout reads as one
  # continuous climb from frame 0.
  motion = cfg.commands[MOTION_COMMAND]
  motion.sampling_mode = "start"
  motion.height_offset_range = (0.0, 0.0)


def phybot_c2_climb_wall_corners60_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The four-corner task with a 45 -> 55 -> 60 degree curriculum.

  Shares the main task's control, rewards, RSI and domain randomization.
  Evaluation uses the terminal 60-degree angle and the canonical start.
  """
  return phybot_c2_climb_wall_env_cfg(
    play=play,
    unit_layout="corners",
    motion_file=CAD_CLIMB_CORNERS60_MOTION,
    angle_stages=mdp.CORNERS60_WALL_ANGLE_STAGES,
  )


def phybot_c2_climb_wall_corners80_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Four active CAD corner units on a fixed 80-degree wall."""
  return phybot_c2_climb_wall_env_cfg(
    play=play, unit_layout="corners", wall_angle_deg=80.0, motion_file=CAD_CLIMB_CORNERS80_MOTION
  )
