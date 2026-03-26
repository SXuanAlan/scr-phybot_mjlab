"""PHYBot Big V2 wall-climbing environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from spider_rl_mjlab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg


def phybot_big_v2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot Big V2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_big_v2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override initial state at task level.
  robot_cfg.init_state.pos = (0, 0.0, 1.1)
  robot_cfg.init_state.joint_pos = {
    "left_hip_pitch": -0.2,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.4,
    "left_ankle_pitch": -0.2,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.2,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.4,
    "right_ankle_pitch": -0.2,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "left_shoulder_pitch": -2.79,
    "left_shoulder_roll": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": -0.27,
    "left_elbow_yaw": 0.0,
    "right_shoulder_pitch": -2.79,
    "right_shoulder_roll": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": -0.27,
    "right_elbow_yaw": 0.0,
  }

  # Wall: center position in world frame.
  wall_cfg.init_state.pos = (0.8, 0.0, 2.5)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_BIG_V2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
