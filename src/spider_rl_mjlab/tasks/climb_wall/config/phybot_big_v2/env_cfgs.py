"""PHYBot Big V2 wall-climbing environment configurations."""

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.assets.microspine import MicrospineCfg, MicrospineController
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg

# End-effector collision geom names for microspine contact pairs.
# Hands: 2x4 sub-geom grid per hand (same as forearm_hand).
_HAND_SUB_GEOMS = [
  f"{side}_hand_c_{i}_{j}"
  for side in ("left", "right")
  for i in range(2)
  for j in range(4)
]
_ALL_EE_GEOMS = _HAND_SUB_GEOMS


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

  # Explicit contact pairs between EE geoms and wall for microspine adhesion.
  def _add_ee_wall_pairs(spec: mujoco.MjSpec) -> None:
    for gname in _ALL_EE_GEOMS:
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.condim = 3
      pair.friction = [20.0, 20.0, 0.005, 0.0001, 0.0001]
      pair.solref = [0.02, 1.5]
      pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]
      pair.solreffriction = [0.02, 1.5]
      pair.margin = 0.005
      pair.gap = 0.0

  cfg.scene.spec_fn = _add_ee_wall_pairs

  # Initial push: 20N +x on each hand for 2s to engage microspines, then release.
  from spider_rl_mjlab.tasks.microspine_test.mdp import PhaseForce
  cfg.events["hand_push_left"] = EventTermCfg(
    func=PhaseForce,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "body_name": "left_hand",
      "phase1_force": (30.0, 0.0, 0.0),
      "phase1_duration": 2.0,
      "phase2_force": (0.0, 0.0, 0.0),
    },
  )
  cfg.events["hand_push_right"] = EventTermCfg(
    func=PhaseForce,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "body_name": "right_hand",
      "phase1_force": (30.0, 0.0, 0.0),
      "phase1_duration": 2.0,
      "phase2_force": (0.0, 0.0, 0.0),
    },
  )

  # Microspine adhesion: force model for hands and feet.
  # Hand collision geoms: sphere at local (-0.02, 0, 0), radius 0.035m.
  #   Contact surface offset from body origin: -0.02 + 0.035 = 0.015m in -x (local).
  #   After scoping to world: depends on arm pose, but offset magnitude ~0.015m.
  # Foot collision geoms: cylinders at local (0.05, ±0.03, -0.042), radius 0.015m.
  #   Contact surface: the cylinder side closest to wall.
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=(
          "left_hand", "right_hand",
        ),
        disengage_distance=0.3,
        k_normal=500.0,
        c_normal=200.0,
        f_normal_max=100.0,
        k_tangential=2000.0,
        c_tangential=500.0,
        f_tangential_max=300.0,
        aniso_ratio=0.3,
        f_lateral_max=100.0,
      ),
      "wall_cfg": SceneEntityCfg("wall"),
    },
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_BIG_V2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
