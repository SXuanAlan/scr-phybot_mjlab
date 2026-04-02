"""PHYBot Big V2 wall-climbing environment configurations."""

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.assets.microspine import MicrospineCfg, MicrospineController, add_adhesion_actuators
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg


# Hand sub-geom 2x4 grid (matching forearm_hand layout) + foot collision geoms.
_HAND_GEOMS = [
  f"{side}_hand_c_{i}_{j}"
  for side in ("left", "right")
  for i in range(2)
  for j in range(4)
]
_FOOT_GEOMS = [
  f"{side}_ankle_roll_collision_{i}"
  for side in ("left", "right")
  for i in (1, 2)
]
_EE_GEOMS = _HAND_GEOMS + _FOOT_GEOMS


def phybot_big_v2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot Big V2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_big_v2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override initial state at task level.
  # Position robot so hands are touching wall inner face (x=0.3).
  robot_cfg.init_state.pos = (-0.10, 0.0, 1.1)
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
    "left_shoulder_pitch": -2.0,
    "left_shoulder_roll": -0.17,
    "left_shoulder_yaw": -0.5,
    "left_elbow_pitch": -0.26,
    "left_elbow_yaw": 1.5,
    "right_shoulder_pitch": -2.0,
    "right_shoulder_roll": 0.16,
    "right_shoulder_yaw": 0.5,
    "right_elbow_pitch": -0.26,
    "right_elbow_yaw": -1.5,
  }

  # Wall: center position in world frame.
  wall_cfg.init_state.pos = (0.8, 0.0, 2.5)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  # EE body names for adhesion actuators.
  _EE_BODY_NAMES = ("left_hand", "right_hand")
  _ADHESION_FORCE = 1200.0

  # Contact pairs: hand/foot geoms vs wall with condim=6 + high sliding friction.
  def _add_ee_wall_pairs(spec: mujoco.MjSpec) -> None:
    for gname in _EE_GEOMS:
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.condim = 6
      pair.friction = [30.0, 30.0, 0.01, 0.01, 0.01]
      pair.solref = [0.02, 5.0]
      pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]
      pair.solreffriction = [0.02, 5.0]
      pair.margin = 0.005
      pair.gap = 0.0

  cfg.scene.spec_fn = _add_ee_wall_pairs

  # Per-sphere microspine adhesion via external wrench.
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=("left_hand", "right_hand"),
        adhesion_force=_ADHESION_FORCE,
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
