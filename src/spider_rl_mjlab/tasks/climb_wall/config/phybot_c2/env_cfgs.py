"""PHYBot C2 wall-climbing environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from spider_rl_mjlab.assets import PHYBOT_C2_ACTION_SCALE, get_phybot_c2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg


def phybot_c2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  cfg.scene.entities = {
    "robot": get_phybot_c2_robot_cfg(),
    "wall": get_wall_cfg(),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False

  return cfg
