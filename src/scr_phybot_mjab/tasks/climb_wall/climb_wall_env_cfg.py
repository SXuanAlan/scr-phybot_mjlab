"""Wall-climbing task configuration.

Factory function to create a base climb-wall task config.
Robot-specific configurations call the factory and customize as needed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.tasks.climb_wall import mdp

##
# Manager-key names — single source of truth, shared by this base factory,
# the robot configs, and climb_transition. FROZEN: the keys are public API
# (tyro CLI paths, params/env.yaml, wandb names, phi_meta.json, and scripts
# hard-code them — engage_raster / priv_ablate / export_policy / play).
##

VELOCITY_COMMAND = "climb_velocity"
MOTION_COMMAND = "climb_motion"
MICROSPINE_AFFORDANCE_COMMAND = "microspine_affordance"
MICROSPINE_GRIP_EVENT = "microspine_grip"
MICROSPINE_CONTACT_SENSOR = "microspine_wall_contact"
TOE_CONTACT_SENSOR = "climb_toe_contact"

# One PPO rollout must hold a full release -> cooldown -> re-engage -> hold
# sequence (~0.7 s). Must equal the phybot_c2 rl_cfg's num_steps_per_env
# (test_climb_task_config cross-checks).
_STEPS_PER_ITER = 96


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
        # Robot z above spawn — critic needs this to value climbing.
        "height_above_spawn": ObservationTermCfg(func=mdp.height_above_spawn),
        # Commanded world-frame velocity (vx, vy, vz). The command manager
        # owns the target so obs and reward stay synchronized.
        "velocity_command": ObservationTermCfg(
          func=mdp.velocity_command,
          params={"command_name": VELOCITY_COMMAND},
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
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    ),
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    VELOCITY_COMMAND: mdp.ClimbVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      ranges=mdp.ClimbVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        lin_vel_z=(0.08, 0.08),  # Currently fixed to 0.08
      ),
    )
  }

  ##
  # Robot-specific configs own concrete rewards.
  ##

  rewards = {}

  ##
  # Curriculum — wall-angle ladder x tilt-peel anneal (tables + doctrine live
  # in mdp.curriculums; play/distill/polish pin them to the terminal stage via
  # mdp.pin_curricula_to_final_stage). The angle is realized by TILTING THE
  # WALL about its base line under real gravity (mdp.wall_angle) — at the
  # terminal 90° rung the world is exactly climb_transition's.
  ##

  curriculum = {
    "wall_angle": CurriculumTermCfg(
      func=mdp.wall_angle,
      params={
        "angle_stages": [dict(stage) for stage in mdp.WALL_ANGLE_STAGES],
        "steps_per_iter": _STEPS_PER_ITER,
        "angle_jitter_deg": 5.0,
      },
    ),
    "tilt_peel": CurriculumTermCfg(
      func=mdp.tilt_peel_schedule,
      params={
        "stages": [dict(stage) for stage in mdp.TILT_PEEL_STAGES],
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "steps_per_iter": _STEPS_PER_ITER,
      },
    ),
  }

  ##
  # Events
  ##

  # Only the scene-default reset: the motion command performs reference-state
  # initialization (it writes the robot's pose/velocities AFTER reset events
  # run), so generic joint/base perturbation events would only be overwritten.
  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
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
  robot_cfg = SceneEntityCfg("robot")
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # Climb analog of G1's ``base_contact``: the robot has lost grip and
    # fallen toward the wall base. Along-wall altitude, NOT world z — the
    # mjlab builtin's absolute z bar would fire at t=0 on tilted rungs
    # (the RSI spawn's world z shrinks with the wall angle).
    "base_height": TerminationTermCfg(
      func=mdp.root_below_wall_altitude,
      params={"minimum_altitude": 0.3},
    ),
    # NaN / magnitude divergence guard. Not in G1 (IsaacLab has a global
    # NaN sink elsewhere); mjlab does not, so we keep this defensively.
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": robot_cfg, "max_velocity": 20.0},
    ),
    # no_progress executioner REMOVED 2026-07-30 (it truncated every learning
    # episode); if a dance-forever basin returns, reintroduce a curriculum-
    # gated stall bar (mdp.no_progress_stall), never a flat one. (notes:
    # no_progress obituary)
    # DeepMimic early termination: end rollouts whose pose left the reference
    # entirely (fallen/tangled). Threshold is generous — a clean hang at the
    # wrong phase scores ~0.1-0.2 rad mean error and survives.
    "bad_tracking": TerminationTermCfg(
      func=mdp.motion_tracking_divergence,
      params={"command_name": MOTION_COMMAND, "threshold": 0.7},
    ),
  }

  ##
  # Metrics — microspine occupancy/health telemetry + actuation saturation +
  # climb progress. All params are manager-key names; robot-agnostic.
  ##

  metrics = {
    "microspine_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    # Reseat-hold telemetry: clock-freeze duty cycle and pending-EE count
    # (hold_fraction >> interval-implied duty = slow/timed-out reseats).
    # (notes: metrics)
    "microspine_reseat_hold_fraction": MetricsTermCfg(
      func=mdp.microspine_reseat_hold_fraction,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "microspine_reseat_pending_count": MetricsTermCfg(
      func=mdp.microspine_reseat_pending_count,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "left_hand_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": 0},
    ),
    "right_hand_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": 1},
    ),
    "left_foot_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": 2},
    ),
    "right_foot_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": 3},
    ),
    "microspine_forced_release_rate_hz": MetricsTermCfg(
      func=mdp.microspine_forced_release_rate,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "microspine_engage_event_rate_hz": MetricsTermCfg(
      func=mdp.microspine_engage_events,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "microspine_hold_margin_engaged": MetricsTermCfg(
      func=mdp.microspine_hold_margin_engaged,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    "action_saturation_fraction": MetricsTermCfg(
      func=mdp.action_saturation_fraction,
      params={"clip": 1.0},
    ),
    "joint_target_clamp_fraction": MetricsTermCfg(
      func=mdp.joint_target_clamp_fraction,
      params={"action_name": "joint_pos"},
    ),
    "joint_soft_limit_violation": MetricsTermCfg(
      func=mdp.joint_soft_limit_violation,
      params={"asset_name": "robot"},
    ),
    "motion_height_gained_m": MetricsTermCfg(
      func=mdp.motion_height_gained,
      params={"command_name": MOTION_COMMAND},
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
      lookat=(1.5, 0.0, 1.2),
      distance=5.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    curriculum=curriculum,
    events=events,
    terminations=terminations,
    metrics=metrics,
  )
