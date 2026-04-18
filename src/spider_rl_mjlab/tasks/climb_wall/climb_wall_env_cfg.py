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
        # Per-hand latch state (left, right). Lets the policy coordinate
        # motion with grip transitions.
        "hand_engaged_state": ObservationTermCfg(func=mdp.hand_engaged_state),
        # Robot z above spawn — critic needs this to value climbing.
        "height_above_spawn": ObservationTermCfg(func=mdp.height_above_spawn),
        # Commanded world-frame velocity (vx, vy, vz). Currently fixed;
        # ``track_lin_vel`` rewards try to match the commanded values.
        "velocity_command": ObservationTermCfg(
          func=mdp.velocity_command,
          params={"vx_cmd": 0.0, "vy_cmd": 0.0, "vz_cmd": 0.3},
        ),
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
      weight=8.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Hand contact face orientation: local -x should point toward wall (+x)
    "hand_orientation": RewardTermCfg(
      func=mdp.hand_orientation_toward_wall,
      weight=15.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Posture regularisation: penalise deviation from the init keyframe.
    # Tight std on torso/legs keeps the body compact and stable; looser std
    # on shoulders/elbows allows the reach-engage motion the policy must
    # learn. Uses mjlab's built-in Gaussian posture reward.
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=3.0,
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
      weight=8.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Paper (arXiv:2204.04157) sine-arc swing reference — horizontal
    # component: swing hand traces an arc AWAY from wall with peak
    # distance ``lift_amplitude`` (mid-swing) and returns to 0 at stance.
    # Amplitude 0.1 m and sigma 0.18 m are deliberately small/wide to give
    # early learning a broad reward basin: tiny swing attempts (a few cm
    # off the wall) still get meaningful reward. Scale up later if needed.
    "swing_hand_height": RewardTermCfg(
      func=mdp.swing_hand_height,
      weight=10.0,
      params={"asset_cfg": robot_cfg, "lift_amplitude": 0.1, "sigma": 0.18},
    ),
    # Paper sine-arc swing reference — vertical component: swing hand
    # traces an arc UPWARD (hand_z above root_z) with peak reach at
    # mid-swing, falling to 0 at stance. Together with ``swing_hand_height``
    # this fully specifies the task-space swing trajectory (pull away from
    # wall + reach up). Reach peak 0.15 m matches the paper's clearance
    # height order of magnitude (Cassie uses ~15 cm). sigma 0.20 widens
    # the tracking basin for easier exploration.
    "swing_hand_reach_up": RewardTermCfg(
      func=mdp.swing_hand_reach_up,
      weight=15.0,
      params={"asset_cfg": robot_cfg, "reach_height": 0.15, "sigma": 0.20},
    ),
    # World-frame velocity tracking, per axis. Gaussian kernel
    # exp(-(v_cmd - v)^2 / sigma^2) — symmetric around the command, so
    # under- and over-shoot both cost reward. Commands are fixed here
    # (xy=0, vz=+0.3 m/s) and exposed via ``velocity_command`` obs.
    # Weighting: vz > vx ≈ vy > regularization (posture, stance). vz
    # is the primary climbing signal; vx/vy keep the robot from drifting
    # along the wall.
    "track_vx": RewardTermCfg(
      func=mdp.track_lin_vel,
      weight=6.0,
      params={"asset_cfg": robot_cfg, "axis": 0, "target": 0.0, "sigma": 0.5},
    ),
    "track_vy": RewardTermCfg(
      func=mdp.track_lin_vel,
      weight=6.0,
      params={"asset_cfg": robot_cfg, "axis": 1, "target": 0.0, "sigma": 0.5},
    ),
    "track_vz": RewardTermCfg(
      func=mdp.track_lin_vel,
      weight=40.0,
      params={"asset_cfg": robot_cfg, "axis": 2, "target": 0.3, "sigma": 0.25},
    ),
    # Stance anti-slide: penalise tangential velocity when hand should be
    # anchored (σ=0.15 m/s — clear sliding threshold for spider scale).
    "stance_phase_velocity": RewardTermCfg(
      func=mdp.stance_phase_velocity,
      weight=3.0,
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
          {"iter": 0,      "weight": 8.0},
          {"iter": P1_END, "weight": 3.0},
        ],
      },
    ),
    "hand_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_orientation",
        "weight_stages": [
          {"iter": 0, "weight": 15.0},
        ],
      },
    ),
    "posture_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "posture",
        "weight_stages": [
          {"iter": 0, "weight": 3.0},
        ],
      },
    ),
    "hands_on_wall_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hands_on_wall",
        "weight_stages": [
          {"iter": 0, "weight": 8.0},
        ],
      },
    ),
    # --- Gait rewards : active from iter 0 alongside gripping shapers ------
    "swing_hand_height_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "swing_hand_height",
        "weight_stages": [
          {"iter": 0, "weight": 10.0},
        ],
      },
    ),
    "swing_hand_reach_up_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "swing_hand_reach_up",
        "weight_stages": [
          {"iter": 0, "weight": 15.0},
        ],
      },
    ),
    "track_vx_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "track_vx",
        "weight_stages": [
          {"iter": 0, "weight": 6.0},
        ],
      },
    ),
    "track_vy_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "track_vy",
        "weight_stages": [
          {"iter": 0, "weight": 6.0},
        ],
      },
    ),
    "track_vz_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "track_vz",
        "weight_stages": [
          {"iter": 0, "weight": 40.0},
        ],
      },
    ),
    # --- Stance anti-slide ---------------------------------------------------
    "stance_phase_velocity_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "stance_phase_velocity",
        "weight_stages": [
          {"iter": 0, "weight": 3.0},
        ],
      },
    ),
    # --- Gravity : 30° held through Phase 1, ramps to 90° in Phase 2 ------
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          # Start very easy: 10° means gravity is mostly +x (pressing into
          # wall) and only 17% along wall (-z). Robot can discover reach
          # motion without body weight fighting the arm. Ramp to 30° by
          # P1_END once a climbing gait has emerged, then to 90° by P2_END.
          {"iter": 0,      "angle_deg": 10.0},
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
    # ``both_hands_unlatched`` termination DISABLED — letting policy
    # explore freely without hand-grip hard constraints. The function is
    # kept in terminations.py as a fallback to re-enable later.
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
