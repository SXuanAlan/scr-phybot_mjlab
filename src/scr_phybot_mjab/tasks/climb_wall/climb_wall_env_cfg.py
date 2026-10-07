"""Robot-independent sections of the wall-climbing task.

The manager-key names, the wall-angle curriculum, terminations, metrics and
the reset event live here; ``config/<robot>/`` builds the robot-bound
sections (scene, commands, observations, rewards, domain randomization) and
composes the env config.
"""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from scr_phybot_mjab.tasks.climb_wall import mdp

# Manager-key names, shared by every climb config and climb_transition80.
# They are public: CLI override paths, saved run configs, wandb names and
# scripts such as export_policy and play use them.
VELOCITY_COMMAND = "climb_velocity"
MOTION_COMMAND = "climb_motion"
MICROSPINE_AFFORDANCE_COMMAND = "microspine_affordance"
MICROSPINE_GRIP_EVENT = "microspine_grip"
MICROSPINE_CONTACT_SENSOR = "microspine_wall_contact"
TOE_CONTACT_SENSOR = "climb_toe_contact"
UNDESIRED_CONTACT_SENSOR = "climb_undesired_contact"

STEPS_PER_ITER = 96
"""Policy steps per PPO iteration; curricula convert iterations with it.

One rollout must hold a full release -> cooldown -> re-engage -> hold
sequence (~0.7 s). Equals the phybot_c2 runner's num_steps_per_env
(test_climb_task_config cross-checks).
"""


DECIMATION = 10
"""Physics steps per policy step (0.002 s timestep, 50 Hz policy)."""


def climb_curriculum(angle_stages=mdp.WALL_ANGLE_STAGES) -> dict[str, CurriculumTermCfg]:
  """The wall-angle ladder, realized by tilting the wall about its base line
  under real gravity (``mdp.wall_angle``). Play pins it to the final stage."""
  return {
    "wall_angle": CurriculumTermCfg(
      func=mdp.wall_angle,
      params={
        "angle_stages": [dict(stage) for stage in angle_stages],
        "steps_per_iter": STEPS_PER_ITER,
        "angle_jitter_deg": 5.0,
      },
    ),
  }


def climb_reset_events() -> dict[str, EventTermCfg]:
  """Only the scene-default reset: the motion command performs reference-state
  initialization after the reset events, so joint or base perturbation events
  would be overwritten."""
  return {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
  }


def climb_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # The robot lost grip and fell toward the wall base. Along-wall altitude,
    # not world z: a world-z bar would fire at t=0 on tilted walls.
    "base_height": TerminationTermCfg(
      func=mdp.root_below_wall_altitude,
      params={"minimum_altitude": 0.3},
    ),
    # NaN and divergence guard; mjlab has no global NaN sink.
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 20.0},
    ),
    # BeyondMimic early termination: end rollouts whose tracked bodies left
    # the reference (fallen or tangled).
    "bad_tracking": TerminationTermCfg(
      func=mdp.motion_body_tracking_divergence,
      params={
        "command_name": MOTION_COMMAND,
        "position_threshold": 0.3,
        "orientation_threshold": 1.0,
        "height_threshold": 0.25,
      },
    ),
    "lost_support": TerminationTermCfg(
      func=mdp.prolonged_support_loss(), params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "timeout_s": 0.4}
    ),
  }


def climb_metrics() -> dict[str, MetricsTermCfg]:
  """Microspine occupancy and health, actuation saturation and climb progress."""
  occupancy = {
    f"{name}_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "ee_index": index},
    )
    for index, name in enumerate(("left_hand", "right_hand", "left_foot", "right_foot"))
  }
  return {
    "microspine_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    ),
    **occupancy,
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
