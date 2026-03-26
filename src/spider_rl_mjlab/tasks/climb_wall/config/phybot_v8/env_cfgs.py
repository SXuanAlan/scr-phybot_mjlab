"""PHYBot V8 wall-climbing environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from spider_rl_mjlab.assets import PHYBOT_V8_ACTION_SCALE, get_phybot_v8_robot_cfg, get_wall_cfg
from spider_rl_mjlab.assets.microspine.config import MicrospineCfg
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg

# PHYBot V8 arm end-effectors are elbow_yaw (not elbow_pitch like C2).
PHYBOT_V8_MICROSPINE_CFG = MicrospineCfg(
  ee_body_names=(
    "left_ankle_roll",
    "right_ankle_roll",
    "left_hand",
    "right_hand",
  ),
)


def phybot_v8_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot V8 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_v8_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override initial state at task level.
  robot_cfg.init_state.pos = (0, 0.0, 1.1)  # (x, y, z) in world frame
  robot_cfg.init_state.joint_pos = {
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.0,
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.0,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "left_shoulder_pitch": -2.79,
    "left_shoulder_roll": -0.34,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": -0.22,
    "left_elbow_yaw": 0.0,
    "right_shoulder_pitch": -2.79,
    "right_shoulder_roll": 0.34,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": -0.22,
    "right_elbow_yaw": 0.0,
  }

  # Wall: center position in world frame.
  # Wall half-size is [0.5, 1.5, 2.5] (1m thick, 3m wide, 5m tall).
  # Wall surface facing the robot is at pos.x - 0.5.
  wall_cfg.init_state.pos = (0.8, 0.0, 2.5)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_V8_ACTION_SCALE

  # Override microspine config with V8-specific EE body names.
  cfg.events["microspine_adhesion"].params["microspine_cfg"] = PHYBOT_V8_MICROSPINE_CFG

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
