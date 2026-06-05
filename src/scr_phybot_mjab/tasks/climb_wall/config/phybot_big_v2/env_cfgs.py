"""PHYBot Big V2 wall-climbing environment configurations."""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from scr_phybot_mjab.microspine import MicrospineCfg
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import gravity_angle, reward_weight


# Foot collision geoms get wall contact pairs, but no adhesion actuators.
_FOOT_GEOMS = [
  f"{side}_ankle_roll_collision_{i}"
  for side in ("left", "right")
  for i in (1, 2)
]
_VELOCITY_COMMAND_NAME = "climb_velocity"
_LEG_ALL = [".*_hip_.*", ".*_knee", ".*_ankle_.*"]
_LEG_HIP_KNEE = [".*_hip_.*", ".*_knee"]
_ANKLE_ONLY = [".*_ankle_.*"]
_HIP_YAW_ROLL = [".*_hip_yaw", ".*_hip_roll"]
_TORSO = ["waist_yaw", "waist_roll"]


def _make_big_v2_rewards() -> dict[str, RewardTermCfg]:
  robot_cfg = SceneEntityCfg("robot")
  return {
    "hand_orientation": RewardTermCfg(
      func=mdp.hand_orientation_toward_wall,
      weight=15.0,
      params={"asset_cfg": robot_cfg},
    ),
    "hand_grip_time": RewardTermCfg(
      func=mdp.hand_grip_time_min_climb(),
      weight=10.0,
      params={"tau": 0.5, "vz_cmd": 0.08, "cmd_eps": 0.0},
    ),
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.climb_linear_velocity_xy_gauss,
      weight=4.0,
      params={
        "asset_cfg": robot_cfg,
        "command_name": _VELOCITY_COMMAND_NAME,
        "sigma": 0.05,
      },
    ),
    "track_lin_vel_z": RewardTermCfg(
      func=mdp.climb_linear_velocity_z_gauss,
      weight=15.0,
      params={
        "asset_cfg": robot_cfg,
        "command_name": _VELOCITY_COMMAND_NAME,
        "sigma": 0.05,
      },
    ),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_gauss,
      weight=10.0,
      params={
        "asset_cfg": robot_cfg,
        "sigma": 0.3,
        "pitch_scale": 0.0,
        "roll_scale": 1.0,
      },
    ),
    "torso_orientation": RewardTermCfg(
      func=mdp.torso_orientation_gauss,
      weight=10.0,
      params={
        "asset_cfg": robot_cfg,
        "body_name": "waist_roll",
        "sigma": 0.3,
        "pitch_scale": 0.0,
        "roll_scale": 1.0,
      },
    ),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "hand_slide": RewardTermCfg(
      func=mdp.hand_slide_penalty,
      weight=-0.1,
      params={"asset_cfg": robot_cfg},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_ONLY)},
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_YAW_ROLL)},
    ),
    "joint_deviation_torso": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_TORSO)},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.25e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_HIP_KNEE)},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.5e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ALL)},
    ),
    "dof_vel_legs_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ALL)},
    ),
  }


def _make_big_v2_curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "hand_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_orientation",
        "weight_stages": [{"iter": 0, "weight": 15.0}],
      },
    ),
    "hand_grip_time_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_grip_time",
        "weight_stages": [{"iter": 0, "weight": 10.0}],
      },
    ),
    "track_lin_vel_xy_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "track_lin_vel_xy",
        "weight_stages": [{"iter": 0, "weight": 4.0}],
      },
    ),
    "track_lin_vel_z_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "track_lin_vel_z",
        "weight_stages": [{"iter": 0, "weight": 15.0}],
      },
    ),
    "hand_slide_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_slide",
        "weight_stages": [{"iter": 0, "weight": -0.1}],
      },
    ),
    "flat_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "flat_orientation",
        "weight_stages": [{"iter": 0, "weight": 10.0}],
      },
    ),
    "torso_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "torso_orientation",
        "weight_stages": [{"iter": 0, "weight": 10.0}],
      },
    ),
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0, "angle_deg": 45.0},
          {"iter": 1000, "angle_deg": 45.0},
          {"iter": 1500, "angle_deg": 51.0},
          {"iter": 2000, "angle_deg": 59.0},
          {"iter": 2500, "angle_deg": 68.0},
          {"iter": 3000, "angle_deg": 90.0},
        ],
      },
    ),
  }


def phybot_big_v2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot Big V2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_big_v2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override initial state at task level. IMPORTANT: build a fresh
  # InitialStateCfg rather than mutating robot_cfg.init_state in place — the
  # robot factory returns a shared keyframe singleton that is also used by
  # other tasks (e.g. microspine_test). Mutating it would cause the last task
  # registered to overwrite init poses for earlier ones.
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(-0.67, 0.0, 2.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.435,
      "left_hip_roll": 0.64,
      "left_hip_yaw": 0.955,
      "left_knee": 0.4,
      "left_ankle_pitch": -0.2,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.435,
      "right_hip_roll": -0.64,
      "right_hip_yaw": -0.955,
      "right_knee": 0.4,
      "right_ankle_pitch": -0.2,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "waist_roll": 0.0,
      "left_shoulder_pitch": -2.79,
      "left_shoulder_roll": -0.17,
      "left_shoulder_yaw": -0.5,
      "left_elbow_pitch": -0.4,
      "left_elbow_yaw": 0.45,
      "right_shoulder_pitch": -2.79,
      "right_shoulder_roll": 0.17,
      "right_shoulder_yaw": 0.5,
      "right_elbow_pitch": -0.4,
      "right_elbow_yaw": -0.45,
    },
  )

  # Wall: center position in world frame (10m tall wall centered at z=5).
  wall_cfg.init_state.pos = (0, 0.0, 5.0)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  microspine_cfg = MicrospineCfg(
    ee_body_names=("left_hand", "right_hand"),
    adhesion_force=1200.0,
    wall_geom_name="wall/wall_geom",
  )

  # Contact pairs + adhesion actuators via spec_fn.
  # Hands: microspine-scale friction (30) to anchor against the wall.
  # Feet:  normal shoe-scale friction (1.5) so the policy cannot use them
  # as a second pair of "grippers" to kick up along the wall.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    microspine_cfg.add_contact_pairs_and_actuators(spec)
    for gname in _FOOT_GEOMS:
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.condim = 6
      pair.friction = [1.5, 1.5, 0.5, 0.5, 0.5]
      pair.solref = [0.02, 5.0]
      pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]
      pair.solreffriction = [0.02, 5.0]
      pair.margin = 0.01
      pair.gap = 0.01
    # Disable the terrain ground plane — the gravity-tilt curriculum replaces
    # the role of a floor, and removing the plane keeps the robot honest
    # about needing to grip the wall.
    try:
      terrain_geom = spec.geom("terrain/terrain")
      terrain_geom.contype = 0
      terrain_geom.conaffinity = 0
    except Exception:
      pass

  cfg.scene.spec_fn = _spec_fn

  # Per-sphere microspine adhesion via MuJoCo adhesion actuators.
  cfg.events["microspine_adhesion"] = microspine_cfg.make_physics_event_cfg()

  cfg.rewards = _make_big_v2_rewards()
  cfg.curriculum = _make_big_v2_curriculum()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_BIG_V2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Play-only debug: print engage/disengage transitions for env 0 and
    # draw the current gravity vector in the viewer.
    cfg.events["microspine_adhesion"].params["microspine_cfg"].debug = True

  return cfg
