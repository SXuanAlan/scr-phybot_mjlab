"""PHYBot C2 80-degree walk-to-climb transition: stand at the wall, latch all four pads, hold.

The robot tracks the a80 mount clip on a wall pinned at 80 degrees, with
the real robot's PD table. Rewards are BeyondMimic's tracking kernels plus
one engage-progress potential; legality comes from terminations (tracking
deviation, the contact law, a wall fence and the engage cone as
Constraints-as-Terminations), and births are sampled adaptively. Climb's
scene, simulation, actions, grip physics, reset event and domain
randomization are reused as they are.

Each section lives in its own module and is built directly in its final
form; this module composes them and applies the play overrides.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from scr_phybot_mjab.microspine.constraint import make_constraint_event_cfg
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import (
  DECIMATION,
  MICROSPINE_AFFORDANCE_COMMAND,
  MICROSPINE_GRIP_EVENT,
  MOTION_COMMAND,
  climb_reset_events,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.domain_rand import (
  DOMAIN_RAND_EVENTS,
  actor_obs_noise,
  domain_rand_events,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.robot import climb_actions, climb_robot_cfg
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.scene import climb_scene, climb_sim, climb_viewer
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.stance import climb_toe_contacts

from .commands import transition80_commands
from .curricula import GIMBAL_TRAVEL_EVENTS, gimbal_travel_events, transition80_curriculum
from .metrics import transition80_metrics
from .microspine import transition80_microspine_cfg
from .observations import transition80_observations
from .rewards import transition80_rewards
from .robot import standing_init_state, transition80_actuators
from .schedules import ANTIRAKE_LADDER, GIMBAL_TRAVEL_STAGES, RIP_PENALTY_STAGES, stage_final
from .terminations import transition80_terminations


def phybot_c2_climb_transition80_env_cfg(play: bool = False, domain_rand: bool = True) -> ManagerBasedRlEnvCfg:
  """The 80-degree transition configuration.

  Args:
    play: evaluation mode (single env, start at frame 0, no scatter, engage
      cone off).
    domain_rand: climb's observation noise and startup friction, mass and
      encoder randomization.
  """
  microspine_cfg = transition80_microspine_cfg()
  toe_contacts = climb_toe_contacts()
  robot_cfg = climb_robot_cfg(toe_contacts, actuators=transition80_actuators(), init_state=standing_init_state())
  events = {
    **climb_reset_events(),
    MICROSPINE_GRIP_EVENT: make_constraint_event_cfg(microspine_cfg, affordance_name=MICROSPINE_AFFORDANCE_COMMAND),
  }
  if domain_rand:
    events |= domain_rand_events()
  events |= gimbal_travel_events()
  cfg = ManagerBasedRlEnvCfg(
    decimation=DECIMATION,
    # The 22.6 s clip plus the birth settle, with margin.
    episode_length_s=35.0,
    scene=climb_scene(robot_cfg, microspine_cfg, toe_contacts),
    sim=climb_sim(),
    viewer=climb_viewer(),
    observations=transition80_observations(microspine_cfg, actor_obs_noise() if domain_rand else None),
    actions=climb_actions(),
    commands=transition80_commands(microspine_cfg),
    rewards=transition80_rewards(),
    curriculum=transition80_curriculum(),
    events=events,
    terminations=transition80_terminations(),
    metrics=transition80_metrics(),
  )
  if play:
    apply_play_overrides(cfg)
  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Deterministic single-env evaluation of the whole mount from its first frame."""
  cfg.scene.num_envs = 1
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  # Remove the randomization events entirely: neutral ranges would still
  # overwrite the XML frictions under operation="abs".
  for name in DOMAIN_RAND_EVENTS:
    cfg.events.pop(name, None)
  motion = cfg.commands[MOTION_COMMAND]
  motion.sampling_mode = "start"
  motion.spawn_xy_noise = (0.0, 0.0)
  motion.spawn_yaw_noise = 0.0
  motion.spawn_early_fraction = 0.0
  motion.spawn_bias_frame_windows = ()
  cfg.terminations["engage_constraint"].params["mode"] = "off"
  # The rip bill, the anti-rake gate and the gimbal travel take their final
  # values; the tracking and contact laws keep their stage-0 thresholds.
  cfg.rewards["anchor_rip_penalty"].params["weight_per_pad"] = stage_final(RIP_PENALTY_STAGES)
  cfg.curriculum = {name: cfg.curriculum[name] for name in ("wall_angle", "engage_antirake_ladder")}
  cfg.curriculum["engage_antirake_ladder"].params["stages"] = [{"iter": 0, "value": stage_final(ANTIRAKE_LADDER)}]
  gimbal_stages = [{"iter": 0, "value": stage_final(GIMBAL_TRAVEL_STAGES)}]
  for name, _ in GIMBAL_TRAVEL_EVENTS:
    cfg.events[name].params["stages"] = gimbal_stages
