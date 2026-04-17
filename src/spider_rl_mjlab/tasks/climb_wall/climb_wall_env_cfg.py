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
      weight=40.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Posture regularisation: penalise deviation from the init keyframe.
    # Tight std on torso/legs keeps the body compact and stable; looser std
    # on shoulders/elbows allows the reach-engage motion the policy must
    # learn. Uses mjlab's built-in Gaussian posture reward.
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=4.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "std": {
          ".*waist_.*": 0.3,
          ".*hip_.*": 0.5,
          ".*knee": 0.6,
          ".*ankle_.*": 0.6,
          ".*shoulder_.*": 1.2,
          ".*elbow_.*": 1.2,
        },
      },
    ),
    # Reward contact (per-sphere average)
    "hands_on_wall": RewardTermCfg(
      func=mdp.hands_on_wall,
      weight=20.0,
      params={"asset_cfg": robot_cfg},
    ),
    # --- Gait rewards (Walk-These-Ways + custom-sine-waves style) -----------
    # Two-hand trot: left stance plateau p ∈ [0.5, 1); right offset by 0.5.
    # Sine-arc hand-to-wall distance tracking (arXiv:2204.04157 Wu 2022):
    "swing_hand_height": RewardTermCfg(
      func=mdp.swing_hand_height,
      weight=0.0,
      params={"asset_cfg": robot_cfg, "lift_amplitude": 0.4, "sigma": 0.18},
    ),
    # Primary climbing objective: current height above spawn.
    # Position-based — "be high" is the reward, not "move up fast".
    # spawn z = 2.0 (robot_cfg.init_state.pos), wall top at z ≈ 10.
    "climb_height": RewardTermCfg(
      func=mdp.climb_height,
      weight=30.0,
      params={"asset_cfg": robot_cfg, "baseline_z": 2.0},
    ),
    # Secondary: small velocity bonus kept as a dense gradient during early
    # training, before the policy has yet climbed high enough for
    # ``climb_height`` to differentiate behaviours.
    "climb_vz": RewardTermCfg(
      func=mdp.climb_vz,
      weight=20.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Heavy penalty when BOTH hands are unlatched (robot in free-fly).
    # Removes the burst-hop exploit of climb_vz: a second of flight loses
    # more reward than the height you could gain by hopping.
    "airborne_penalty": RewardTermCfg(
      func=mdp.airborne_penalty,
      weight=-30.0,
      params={},
    ),
    # Stance anti-slide: penalise tangential velocity when hand should be
    # anchored (σ=0.15 m/s — clear sliding threshold for spider scale).
    "stance_phase_velocity": RewardTermCfg(
      func=mdp.stance_phase_velocity,
      weight=0.0,
      params={"asset_cfg": robot_cfg, "sigma": 0.15},
    ),
  }

  ##
  # Curriculum — SINGLE unified timeline.
  #
  # Robot spawns already near the wall (-0.7, 0, 2) with arms reaching
  # toward it — no need to learn standing/walking. Gripping and gait
  # rewards are ALL active from iter 0; only the wall angle is scheduled.
  #
  #   Phase 1 [0      → P1_END ]  grip + gait together   gravity 30°
  #   Phase 2 [P1_END → P2_END ]  gravity ramps 30° → 90°
  #   Phase 3 [P2_END → ...    ]  fully vertical wall
  ##

  P1_END = 15000   # gait learned at 30° gravity
  P2_END = 45000   # gravity reached full vertical

  curriculum = {
    # --- Wall-gripping rewards : shape "move hands to wall → orient →
    # contact → both hands engaged". Fade the strongest shapers out once
    # gait is established so they don't fight the alternating pattern.
    "hands_toward_wall_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hands_toward_wall",
        "weight_stages": [
          {"iter": 0,      "weight": 15.0},
          {"iter": P1_END, "weight": 5.0},
        ],
      },
    ),
    "hand_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_orientation",
        "weight_stages": [
          {"iter": 0, "weight": 40.0},
        ],
      },
    ),
    "posture_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "posture",
        "weight_stages": [
          {"iter": 0, "weight": 4.0},
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
    # --- Gait rewards : active from iter 0 alongside gripping shapers ------
    "swing_hand_height_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "swing_hand_height",
        "weight_stages": [
          {"iter": 0, "weight": 0.0},
        ],
      },
    ),
    "climb_height_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "climb_height",
        "weight_stages": [
          {"iter": 0, "weight": 30.0},
        ],
      },
    ),
    "climb_vz_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "climb_vz",
        "weight_stages": [
          {"iter": 0, "weight": 20.0},
        ],
      },
    ),
    "airborne_penalty_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "airborne_penalty",
        "weight_stages": [
          {"iter": 0, "weight": -30.0},
        ],
      },
    ),
    # --- Stance anti-slide ---------------------------------------------------
    "stance_phase_velocity_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "stance_phase_velocity",
        "weight_stages": [
          {"iter": 0, "weight": 10.0},
        ],
      },
    ),
    # --- Gravity : 30° held through Phase 1, ramps to 90° in Phase 2 ------
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0,      "angle_deg": 30.0},
          {"iter": P1_END, "angle_deg": 30.0},
          {"iter": P2_END, "angle_deg": 90.0},
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
