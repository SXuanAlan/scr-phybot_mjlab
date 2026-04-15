"""Wall-climbing task configuration.

Factory function to create a base climb-wall task config.
Robot-specific configurations call the factory and customize as needed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from spider_rl_mjlab.tasks.climb_wall import mdp
from spider_rl_mjlab.tasks.climb_wall.mdp.curriculums import gravity_angle, reward_weight


def make_climb_wall_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base wall-climbing task configuration."""

  num_envs = 128
  env_spacing = 8.0

  ##
  # Observations
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "prev_action": ObservationTermCfg(func=mdp.prev_action),       # a_{t-1}
        "prev_prev_action": ObservationTermCfg(func=mdp.prev_prev_action),  # a_{t-2}
        "gait_clock": ObservationTermCfg(func=mdp.gait_clock),
      },
      concatenate_terms=True,  # Concatenate all terms into a single vector.
      enable_corruption=False, # Enable observation corruption (e.g., adding noise).
    ),
  }

  ##
  # Actions
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(r"^(?!adh_).*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    ),
  }

  ##
  # Rewards
  ##

  robot_cfg = SceneEntityCfg("robot")

  rewards = {
    # Shaping: guide hands toward wall (continuous, works at any distance)
    "hands_toward_wall": RewardTermCfg(
      func=mdp.hands_toward_wall,
      weight=15.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Hand contact face orientation: local -x should point toward wall (+x)
    "hand_orientation": RewardTermCfg(
      func=mdp.hand_orientation_toward_wall,
      weight=15.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Reward contact (per-sphere average)
    "hands_on_wall": RewardTermCfg(
      func=mdp.hands_on_wall,
      weight=20.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Bonus: both hands simultaneously touching (product of per-hand ratios)
    "both_hands_bonus": RewardTermCfg(
      func=mdp.both_hands_bonus,
      weight=30.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Keep COM close to wall
    "com_near_wall": RewardTermCfg(
      func=mdp.com_near_wall,
      weight=5.0,
      params={"asset_cfg": robot_cfg, "wall_x": 0.3},
    ),
    # Reward forward velocity toward wall
    "forward_velocity": RewardTermCfg(
      func=mdp.forward_velocity,
      weight=10.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Stay upright — prevent kneeling
    "torso_orientation": RewardTermCfg(
      func=mdp.torso_orientation,
      weight=5.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Penalize non-flat orientation (upper body tilt)
    "flat_orientation": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-5.0,
    ),
    # Maintain standing height
    "base_height_penalty": RewardTermCfg(
      func=envs_mdp.root_height_below_minimum,
      weight=-10.0,
      params={"minimum_height": 0.9},
    ),
    # Keep joints near default standing pose
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=5.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {".*": 0.5},
      },
    ),
    # Penalize body angular velocity (anti-shaking)
    "torso_ang_vel": RewardTermCfg(
      func=mdp.torso_ang_vel_xy,
      weight=-1.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Penalize lateral linear velocity (stay in place)
    "lin_vel_xy": RewardTermCfg(
      func=mdp.lin_vel_xy,
      weight=-1.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Smoothness: penalize jerky actions
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.1,
    ),
    # Energy: penalize large torques
    "joint_torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1e-6,
    ),
    # Joint limit penalties
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
    ),
    # --- Gait rewards (adapted from Um et al. 2025, Table I) -----------------
    # Two-hand trot: left swings in φ∈[0,0.5), right offset by 0.5.
    # Start with weight=0 here; curriculum turns them on from P2_END onward.
    "gait_match": RewardTermCfg(
      func=mdp.gait_match,
      weight=0.0,
      params={"asset_cfg": robot_cfg},
    ),
    "swing_hand_height": RewardTermCfg(
      func=mdp.swing_hand_height,
      weight=0.0,
      params={"asset_cfg": robot_cfg, "lift_target": 0.5},
    ),
    "swing_hand_clearance": RewardTermCfg(
      func=mdp.swing_hand_clearance,
      weight=0.0,   # penalty — curriculum sets negative weight
      params={"asset_cfg": robot_cfg, "lift_target": 0.5},
    ),
    "stance_hand_anchor": RewardTermCfg(
      func=mdp.stance_hand_anchor,
      weight=0.0,   # penalty — curriculum sets negative weight
      params={"asset_cfg": robot_cfg},
    ),
    "climb_vz": RewardTermCfg(
      func=mdp.climb_vz,
      weight=0.0,
      params={"asset_cfg": robot_cfg},
    ),
  }

  ##
  # Curriculum — SINGLE unified timeline.
  #
  # Robot spawns already near the wall (-0.7, 0, 2) with arms reaching
  # toward it — no need to learn standing/walking. Phase 0 goes directly
  # into "grip the wall".
  #
  #   Phase 0 [0      → P0_END ]  grip the wall          gravity 30°
  #   Phase 1 [P0_END → P1_END ]  gait rewards kick in   gravity 30°
  #   Phase 2 [P1_END → P2_END ]  gravity ramps 30°→90° while gait continues
  #   Phase 3 [P2_END → ...    ]  fully vertical wall + alternating gait
  ##

  P0_END =  5000   # hands reliably on wall
  P1_END = 15000   # gait learned at 30° gravity
  P2_END = 45000   # gravity reached full vertical

  curriculum = {
    # --- Phase 0 → 1 : wall-gripping rewards (on from the start) ----------
    # These shape "move hands to wall → orient correctly → contact →
    # both hands engaged". At iter 0 gravity is already 30° (pushes robot
    # into wall), so from the very first frame the robot is pressed up
    # against the wall and just needs to learn to engage.
    "hands_toward_wall_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hands_toward_wall",
        "weight_stages": [
          {"iter": 0,      "weight": 15.0},
          {"iter": P1_END, "weight": 5.0},   # fade once gait is established
        ],
      },
    ),
    "hand_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_orientation",
        "weight_stages": [
          {"iter": 0,      "weight": 15.0},
          {"iter": P1_END, "weight": 5.0},
        ],
      },
    ),
    "com_near_wall_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "com_near_wall",
        "weight_stages": [
          {"iter": 0, "weight": 5.0},
        ],
      },
    ),
    "hands_on_wall_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hands_on_wall",
        "weight_stages": [
          {"iter": 0, "weight": 20.0},
        ],
      },
    ),
    # Both-hands bonus: strong early to teach "grip with both", then fades
    # when gait starts (alternating gait ≠ always both-hands).
    "both_hands_bonus_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "both_hands_bonus",
        "weight_stages": [
          {"iter": 0,      "weight": 30.0},
          {"iter": P0_END, "weight": 15.0},
          {"iter": P1_END, "weight": 5.0},
        ],
      },
    ),
    # Forward velocity / lin_vel_xy: no approach phase, kept at 0.
    "forward_velocity_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "forward_velocity",
        "weight_stages": [{"iter": 0, "weight": 0.0}],
      },
    ),
    "lin_vel_xy_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "lin_vel_xy",
        "weight_stages": [{"iter": 0, "weight": 0.0}],
      },
    ),
    # --- Phase 1 onwards : gait rewards turn on at P0_END ------------------
    "gait_match_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "gait_match",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": 8.0},  # soft sine version — bumped to
                                              # compensate for weaker signal
                                              # at swing↔stance boundaries
        ],
      },
    ),
    "swing_hand_height_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "swing_hand_height",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": 10.0},
        ],
      },
    ),
    "swing_hand_clearance_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "swing_hand_clearance",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": -50.0},
        ],
      },
    ),
    "stance_hand_anchor_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "stance_hand_anchor",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": -2.0},
        ],
      },
    ),
    "climb_vz_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "climb_vz",
        "weight_stages": [
          {"iter": 0,      "weight": 0.0},
          {"iter": P0_END, "weight": 20.0},
        ],
      },
    ),
    # --- Stability rewards : constant throughout (moderate weights) -------
    "torso_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "torso_orientation",
        "weight_stages": [{"iter": 0, "weight": 5.0}],
      },
    ),
    "flat_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "flat_orientation",
        "weight_stages": [{"iter": 0, "weight": -2.0}],
      },
    ),
    "base_height_penalty_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "base_height_penalty",
        "weight_stages": [{"iter": 0, "weight": -10.0}],
      },
    ),
    "posture_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "posture",
        "weight_stages": [{"iter": 0, "weight": 1.0}],
      },
    ),
    "torso_ang_vel_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "torso_ang_vel",
        "weight_stages": [{"iter": 0, "weight": -1.0}],
      },
    ),
    # --- Gravity : 30° held through Phase 1, ramps to 90° in Phase 2 ------
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0,      "angle_deg": 30.0},
          {"iter": P1_END, "angle_deg": 30.0},   # gait learned at 30°
          {"iter": P2_END, "angle_deg": 90.0},   # fully vertical
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
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": robot_cfg, "max_velocity": 20.0},
    ),
  }

  ##
  # Build config
  ##

  return ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=30.0,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=env_spacing,
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002, cone="elliptic", integrator="implicitfast", impratio=10.0, ccd_iterations=1000),
      njmax=1000,
      nconmax=500,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(1.5, 0.0, 4.0),
      distance=12.0,
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
