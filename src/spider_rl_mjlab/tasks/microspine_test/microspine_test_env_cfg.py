"""Microspine adhesion test task configuration.

Factory function to create a base microspine test config.
Uses a single forearm+hand against a wall. Gravity disabled; 10N in +x for 1s.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from spider_rl_mjlab.tasks.climb_wall.mdp.curriculums import reward_weight
from spider_rl_mjlab.tasks.microspine_test import mdp


def make_microspine_test_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base microspine adhesion test configuration."""

  num_envs = 128
  env_spacing = 5.0

  ##
  # Observations
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "joint_pos_rel": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "is_engaged": ObservationTermCfg(func=mdp.is_engaged_obs),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(r"^(?!adh_).*",),
      scale=0.25,
      use_default_offset=True,
    ),
  }

  ##
  # Rewards
  ##

  robot_cfg = SceneEntityCfg("robot")

  rewards = {
    # Dense guide before any contact — policy feels the wall from ~15cm out.
    "hand_to_wall_dist": RewardTermCfg(
      func=mdp.hand_to_wall_dist,
      weight=10.0,
      params={"asset_cfg": robot_cfg, "sigma": 0.05},
    ),
    # Palm normal should face +x (wall normal).
    "palm_orientation": RewardTermCfg(
      func=mdp.palm_orientation,
      weight=10.0,
      params={"asset_cfg": robot_cfg, "sigma": 0.15},
    ),
    # Dense signal once contact exists: press every sphere toward the engage threshold.
    "engage_progress": RewardTermCfg(
      func=mdp.engage_progress,
      weight=5.0,
    ),
    # Sparse bonus once the hand latches.
    "engaged_bonus": RewardTermCfg(
      func=mdp.engaged_bonus,
      weight=10.0,
    ),
    # Dense reward for maintaining engagement — ramps with tanh(t/0.33),
    # saturates at ~1 s of continuous engagement.
    "engaged_time": RewardTermCfg(
      func=mdp.engaged_time,
      weight=10.0,
      params={"tau": 0.33},
    ),
    # tanh-saturating reward on the top-k longest per-sphere contact streaks.
    "contact_continuous_time": RewardTermCfg(
      func=mdp.contact_continuous_time,
      weight=10.0,
      params={"top_k": 3},
    ),
    # # Engagement reaction force — disabled for now to isolate other signals.
    # "engagement_drag": RewardTermCfg(
    #   func=mdp.engagement_drag,
    #   weight=1.0,
    #   params={"asset_cfg": robot_cfg},
    # ),
    # --- Standing stability ---
    # Upright orientation: exp(-||g_xy||^2), 1.0 when perfectly upright.
    # Penalize base leaning forward toward wall — force arms to reach instead.
    # Deadzone=5cm allows normal balance oscillation without penalty.
    "base_forward_penalty": RewardTermCfg(
      func=mdp.base_forward_penalty,
      weight=-20.0,
      params={"asset_cfg": robot_cfg, "deadzone": 0.05},
    ),
    "torso_upright": RewardTermCfg(
      func=mdp.torso_upright,
      weight=2.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Penalize non-flat orientation (upper body tilt).
    "flat_orientation": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-2.0,
    ),
    # Penalize base height dropping below standing height.
    "base_height_penalty": RewardTermCfg(
      func=envs_mdp.root_height_below_minimum,
      weight=-5.0,
      params={"minimum_height": 0.9},
    ),
    # Keep joints near default standing pose.
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {".*": 0.5},
      },
    ),
    # --- Smoothness & energy penalties ---
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.02,
    ),
    "joint_torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1e-6,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
    ),
  }

  ##
  # Curriculum — phased reward weight scheduling
  #
  #   Phase 0 (0–500 iter):    Stand stable, no wall shaping
  #   Phase 1 (500–3000 iter): Guide hands toward wall + contact
  #   Phase 2 (3000+ iter):    Engage + maintain adhesion
  ##

  P0_END = 500    # stand stable
  P1_END = 1500   # approach & contact wall

  curriculum = {
    # Wall shaping: off in Phase 0, on in Phase 1
    "hand_to_wall_dist_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_to_wall_dist",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": 10.0},
        ],
      },
    ),
    "palm_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "palm_orientation",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": 10.0},
        ],
      },
    ),
    # Contact & engage: off until Phase 2, strong rewards to drive hands to wall.
    "engage_progress_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "engage_progress",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P1_END, "weight": 8.0},
        ],
      },
    ),
    "contact_continuous_time_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "contact_continuous_time",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P1_END, "weight": 10.0},
        ],
      },
    ),
    "engaged_bonus_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "engaged_bonus",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P1_END, "weight": 10.0},
        ],
      },
    ),
    "engaged_time_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "engaged_time",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P1_END, "weight": 10.0},
        ],
      },
    ),
    # Standing stability: strong in Phase 0, keep HIGH throughout — Phase 2
    # engage rewards tempt the policy to over-press, so stability must hold.
    "torso_upright_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "torso_upright",
        "weight_stages": [
          {"iter": 0,      "weight": 15.0},
          {"iter": P0_END, "weight": 10.0},
          {"iter": P1_END, "weight": 10.0},
        ],
      },
    ),
    "flat_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "flat_orientation",
        "weight_stages": [
          {"iter": 0,      "weight": -10.0},
          {"iter": P0_END, "weight": -8.0},
          {"iter": P1_END, "weight": -8.0},
        ],
      },
    ),
    "base_height_penalty_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "base_height_penalty",
        "weight_stages": [
          {"iter": 0,      "weight": -15.0},
          {"iter": P0_END, "weight": -10.0},
          {"iter": P1_END, "weight": -10.0},
        ],
      },
    ),
    "posture_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "posture",
        "weight_stages": [
          {"iter": 0,      "weight": 10.0},
          {"iter": P0_END, "weight": 3.0},
          {"iter": P1_END, "weight": 1.0},
        ],
      },
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.1, 0.1),
      },
    ),
    "reset_robot_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
        },
        "velocity_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
        },
      },
    ),
    # External force is handled by MicrospineController via xfrc_applied.
    # See env_cfgs.py for force params (total_force, total_force_after).
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.3},
    ),
    # Terminate when torso tilts too far (fallen over).
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 1.5},
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  ##
  # Build config
  ##

  return ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=10.0,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=env_spacing,
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002, cone="elliptic", integrator="implicitfast", impratio=10.0, ccd_iterations=500),
      njmax=1000,
      nconmax=500,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 1.5),
      distance=8.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    curriculum=curriculum,
    events=events,
    terminations=terminations,
  )
