"""PHYBot C2 full-body squat tracking.

One episode plays the standalone squat clip once from its first frame, one
clip frame per policy step, with plain PPO. The policy tracks every policy
joint, the reference base and the planted feet. The robot, sensors and
deployed PD table are walk's.

Each section lives in its own module and is built directly in its final
form; this module composes them and applies the play overrides.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_ACTION_SCALE, PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.squat.motions import SQUAT_MOTION

from .commands import MOTION_COMMAND, squat_commands
from .events import squat_events
from .motion import motion_duration_s
from .observations import squat_observations
from .rewards import squat_rewards
from .scene import squat_robot_cfg, squat_scene, squat_sim, squat_viewer
from .terminations import squat_terminations


def phybot_c2_squat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The squat task; ``play`` replays the exact clip without noise or randomization."""
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=motion_duration_s(SQUAT_MOTION),
    is_finite_horizon=True,
    scene=squat_scene(squat_robot_cfg()),
    sim=squat_sim(),
    viewer=squat_viewer(),
    observations=squat_observations(),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES,
        preserve_order=True,
        scale=PHYBOT_C2_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands=squat_commands(),
    events=squat_events(),
    rewards=squat_rewards(),
    terminations=squat_terminations(),
  )
  if play:
    apply_play_overrides(cfg)
  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """The exact standing start and fixed physical parameters.

  The randomization events stay, pinned to single values, so a validation
  run can set the foot friction it tests (scripts/eval/accept_squat.py).
  """
  cfg.observations["actor"].enable_corruption = False
  motion = cfg.commands[MOTION_COMMAND]
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (0.0, 0.0)
  motion.joint_velocity_range = (0.0, 0.0)
  cfg.events["body_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["foot_friction"].params["ranges"] = (0.7, 0.7)
  cfg.events["base_mass"].params["ranges"] = (0.0, 0.0)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
