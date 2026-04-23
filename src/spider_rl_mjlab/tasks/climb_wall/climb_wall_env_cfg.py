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

  num_envs = 4096
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
          params={"vx_cmd": 0.0, "vy_cmd": 0.0, "vz_cmd": 0.08},
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

  # ---- Joint name patterns (used by multiple regulation terms) -----------
  _LEG_ALL = [".*_hip_.*", ".*_knee", ".*_ankle_.*"]
  _LEG_HIP_KNEE = [".*_hip_.*", ".*_knee"]
  _ANKLE_ONLY = [".*_ankle_.*"]
  _HIP_YAW_ROLL = [".*_hip_yaw", ".*_hip_roll"]
  _TORSO = ["waist_yaw", "waist_roll"]

  rewards = {
    # ====================================================================
    # Task-specific rewards (drive climbing behaviour)
    # ====================================================================
    # Hand contact face orientation: local -x should point toward wall (+x).
    # Gait-phase-agnostic: both hands rewarded for staying oriented.
    "hand_orientation": RewardTermCfg(
      func=mdp.hand_orientation_toward_wall,
      weight=15.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Gait: climbing port of IsaacLab ``feet_air_time_positive_biped``.
    # Strict single-stance (∑e == 1). See mdp/rewards.py for details.
    "hand_grip_time": RewardTermCfg(
      func=mdp.hand_grip_time_min_climb(),
      weight=30.0,
      params={"tau": 0.5, "vz_cmd": 0.08, "cmd_eps": 0.0},
    ),
    # Velocity command tracking (climbing analog of ``track_lin_vel_xy_exp``).
    # Only vz — xy tracking not meaningful for wall climbing.
    "track_vz": RewardTermCfg(
      func=mdp.climb_vz_gauss,
      weight=40.0,
      params={"asset_cfg": robot_cfg, "target": 0.08, "sigma": 0.05},
    ),
    # Pelvis upright — roll-only Gaussian (σ=0.3). Pitch is loose (climbing
    # needs some pelvis pitch toward the wall).
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_gauss,
      weight=10.0,
      params={"asset_cfg": robot_cfg, "sigma": 0.3, "pitch_scale": 0.0, "roll_scale": 1.0},
    ),
    # Torso upright — closes the waist-bend exploit where pelvis stays
    # level but ``waist_roll`` body folds sideways.
    "torso_orientation": RewardTermCfg(
      func=mdp.torso_orientation_gauss,
      weight=10.0,
      params={"asset_cfg": robot_cfg, "body_name": "waist_roll",
              "sigma": 0.3, "pitch_scale": 0.0, "roll_scale": 1.0},
    ),

    # ====================================================================
    # G1-walking-style regulation (ported from
    # isaaclab_tasks.manager_based.locomotion.velocity.config.g1).
    # Weights copied verbatim where they translate directly; joint-set
    # scopes adapted for climbing (arms are NOT regulated because they're
    # active climbing executors).
    # ====================================================================
    # is_terminated — fires only on failure termination (not time_out).
    # Weight −200 matches G1 exactly.
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # Climbing analog of G1 ``feet_slide``: penalise tangential motion of
    # an engaged hand. Weight −0.1 matches G1.
    "hand_slide": RewardTermCfg(
      func=mdp.hand_slide_penalty,
      weight=-0.1,
      params={"asset_cfg": robot_cfg},
    ),
    # G1 ``dof_pos_limits`` on ankles (weight −1.0). Ankles are the joints
    # most at risk of hitting soft limits for both walking and climbing
    # posture; other joints have plenty of travel room.
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_ONLY)},
    ),
    # G1 ``joint_deviation_hip`` (hip_yaw + hip_roll, weight −0.1). Keeps
    # legs from splaying outward.
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_YAW_ROLL)},
    ),
    # G1 ``joint_deviation_torso`` (weight −0.1). ``waist_yaw`` + ``waist_roll``
    # should not twist — torso_orientation handles roll; this also pins yaw
    # (which the orientation rewards do not cover).
    "joint_deviation_torso": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_TORSO)},
    ),
    # NOTE: G1 also penalises ``joint_deviation_arms`` (shoulders, elbows)
    # and ``joint_deviation_fingers``. Deliberately OMITTED for climbing —
    # arms are the primary climbing actuators and deviating far from the
    # keyframe is exactly what the task requires. PHYBot Big V2 has no
    # fingers either.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    # G1 ``dof_acc_l2`` on hip + knee. In G1 these are the big-torque
    # leg joints; for climbing, same scope keeps the legs from
    # dynamically whipping.
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.25e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_HIP_KNEE)},
    ),
    # G1 ``dof_torques_l2`` on hip + knee + ankle. Standard locomotion
    # effort cost applied to the full leg chain.
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.5e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ALL)},
    ),
    # Quiet legs: penalize large leg joint velocities so the lower body
    # doesn't flail. L2 quadratic — tiny for small motions, grows fast
    # for big ones, discouraging the "kick wall upward" style.
    "dof_vel_legs_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ALL)},
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

  P1_END = 1000   # gait learned at 45°→60° gravity (scaled for 4096 envs)
  P2_END = 3000   # gravity reached full vertical

  curriculum = {
    # --- Wall-gripping rewards : shape "move hands to wall → orient →
    # contact → both hands engaged". Fade the strongest shapers out once
    # gait is established so they don't fight the alternating pattern.
    "hand_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_orientation",
        "weight_stages": [
          {"iter": 0, "weight": 15.0},
        ],
      },
    ),
    # --- Gait reward (min-grip-time) : active from iter 0 -------------------
    "hand_grip_time_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_grip_time",
        "weight_stages": [
          {"iter": 0, "weight": 30.0},
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
    "hand_slide_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "hand_slide",
        "weight_stages": [
          {"iter": 0, "weight": -0.1},
        ],
      },
    ),
    "flat_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "flat_orientation",
        "weight_stages": [
          {"iter": 0, "weight": 10.0},
        ],
      },
    ),
    "torso_orientation_weight": CurriculumTermCfg(
      func=reward_weight,
      params={
        "reward_name": "torso_orientation",
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
          # Hold at 45° through iter P1_END so policy can first learn basic
          # grip + alternation at a fixed load. Then ramp 45° → 90° over
          # iter 1000-3000, with waypoints evenly spaced in sin(angle) =
          # along-wall gravity load (so load increases at a uniform rate,
          # not bursty early / stale late like linear-in-angle would be).
          # Interp function _interp_angle does sin-space interpolation.
          #   iter    angle     sin(angle) = wall-load
          #   ----    -----     ----------
          #      0    45°       0.707      start
          #   1000    45°       0.707      ← P1_END (hold, grip + gait stable)
          #   1500    51°       0.777
          #   2000    59°       0.857
          #   2500    68°       0.927
          #   3000    90°       1.000      ← P2_END (full vertical)
          {"iter": 0,         "angle_deg": 45.0},
          {"iter": P1_END,    "angle_deg": 45.0},
          {"iter": 1500,      "angle_deg": 51.0},
          {"iter": 2000,      "angle_deg": 59.0},
          {"iter": 2500,      "angle_deg": 68.0},
          {"iter": P2_END,    "angle_deg": 90.0},
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

  # ====================================================================
  # Terminations — G1-walking minimal set adapted for climbing.
  # G1 has: ``time_out`` + ``base_contact`` (torso touches ground).
  # Climb analog: ``time_out`` + ``base_height`` (robot fell off the wall
  # below the spawn floor). ``bad_state`` stays purely as NaN / physics
  # divergence safety.
  # ====================================================================
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # Climb analog of G1's ``base_contact``: the robot has lost grip and
    # fallen below the spawn floor. No terrain collision is active, so
    # without this termination a fallen robot floats into void.
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.3},
    ),
    # NaN / magnitude divergence guard. Not in G1 (IsaacLab has a global
    # NaN sink elsewhere); mjlab does not, so we keep this defensively.
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": robot_cfg, "max_velocity": 20.0},
    ),
    # Both hands off the wall: the robot has lost all grip. Triggers
    # ``termination_penalty`` (-200) via ``is_terminated`` and kills the
    # "single-hand cheat" strategy — if the held hand ever releases
    # without the other engaging, the episode ends with big penalty.
    "both_hands_unlatched": TerminationTermCfg(
      func=mdp.both_hands_unlatched,
      params={"event_term_name": "microspine_adhesion"},
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
      mujoco=MujocoCfg(timestep=0.002, cone="elliptic", integrator="implicitfast", impratio=10.0, ccd_iterations=100),
      njmax=250,
      nconmax=50,
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
