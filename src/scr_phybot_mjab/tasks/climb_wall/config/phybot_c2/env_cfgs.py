"""PHYBot C2 wall-climbing environment configuration.

The reward is outcome-based: it values upward body progress and confirmed
new anchors, while static pose/contact quality is expressed only as error or
safety cost. This prevents a motionless hanging policy from accumulating a
positive return merely by keeping its current grips.

C2-specific notes:
  * 4 grip end-effectors: left/right hand_spine + left/right foot_spine.
    Each spine has a 2x2 sphere grid (16 spheres total).
  * No ``waist_roll`` joint/body — Spot/Anymal-style torso-deviation terms
    are not migrated here (intentionally minimal).
"""

from __future__ import annotations

import dataclasses

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.assets.wall import get_wall_cfg
from scr_phybot_mjab.microspine.config import (
  MicrospineAffordanceCfg,
  MicrospineCfg,
)
from scr_phybot_mjab.microspine.contact import (
  add_contact_pairs,
  add_pad_wall_contact_excludes,
  get_sphere_geom_names,
)
from scr_phybot_mjab.microspine.constraint import (
  add_attachment_equalities,
  make_constraint_event_cfg,
)
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import (
  PHYBOT_C2_WALK_COLLISION,
)


def _policy_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
  )


def phybot_c2_climb_wall_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the constraint-based PHYBot C2 wall-climbing configuration."""
  velocity_command_name = "climb_velocity"
  gait_command_name = "climb_gait"
  microspine_affordance_name = "microspine_affordance"
  microspine_contact_sensor_name = "microspine_wall_contact"
  microspine_event_name = "microspine_grip"

  microspine_cfg = MicrospineCfg(
    ee_body_names=(
      "left_hand_spine",
      "right_hand_spine",
      "left_foot_spine",
      "right_foot_spine",
    ),
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    # The first curriculum stage starts from a stable four-anchor stance and
    # teaches one physical release/re-grip. Later stages switch reset support
    # to hands-only through the performance-gated curriculum.
    initial_engaged_ee_names=(
      "left_hand_spine",
      "right_hand_spine",
      "left_foot_spine",
      "right_foot_spine",
    ),
    # Ordinary contact is only for compression and seating. Tangential support
    # must come primarily from the microspine equality/limit surface.
    contact_friction=(0.2, 0.2, 0.005, 1.0e-4, 1.0e-4),
    # A released limb must be able to re-seat before a partially supported
    # robot falls. Keep the physical gate (two spheres above 3 N) unchanged,
    # but use task-rate timing: 0.20 s cooldown + 0.06 s debounce. The former
    # 0.60 s/0.10 s pair left otherwise-valid contacts one step short of every
    # re-engagement in a stochastic rollout.
    reengage_cooldown_steps=10,
    engage_consecutive_steps=3,
  )
  # Constraint grip engages from compression count + face alignment; the
  # limit surface owns release.
  # A real spine pad locks position and orientation across the contact patch.
  # Capacity follows the pad-local rake direction.
  microspine_cfg.anchor_lock_rotation = True
  microspine_cfg.limit_surface.rake_mode = "ee_local"

  cfg = make_climb_wall_env_cfg()
  cfg.scene.num_envs = 2048
  cfg.commands[velocity_command_name].ranges.lin_vel_z = (0.02, 0.02)
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  terms = cfg.observations["actor"].terms
  terms["base_lin_vel"].func = mdp.builtin_sensor
  terms["base_lin_vel"].params = {"sensor_name": "robot/baselink-velocity"}
  terms["base_ang_vel"].func = mdp.builtin_sensor
  terms["base_ang_vel"].params = {"sensor_name": "robot/baselink-gyro"}
  terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  terms["projected_gravity"].params = {"sensor_name": "robot/baselink-quat"}
  # Match MJLab/IsaacLab velocity tracking style: no explicit gait phase
  # observation. Gait should emerge from velocity tracking + grip shaping.
  cfg.observations["actor"].terms.pop("gait_clock", None)
  cfg.observations["actor"].terms.pop("hand_engaged_state", None)

  cfg.commands[microspine_affordance_name] = MicrospineAffordanceCfg(
    microspine_cfg=microspine_cfg,
    contact_sensor_name=microspine_contact_sensor_name,
    physics_event_name=microspine_event_name,
  )
  cfg.commands[gait_command_name] = mdp.ClimbGaitCommandCfg(
    entity_name="robot",
    affordance_name=microspine_affordance_name,
    physics_event_name=microspine_event_name,
    ee_body_names=microspine_cfg.ee_body_names,
    sequence=(2, 1, 3, 0),  # left foot -> right hand -> right foot -> left hand
    step_height=0.06,
    confirmation_steps=(
      microspine_cfg.limit_surface.hold_grace_steps
      + microspine_cfg.limit_surface.violation_consecutive_steps
    ),
  )

  robot_cfg = get_phybot_c2_robot_cfg()
  # One shared CollisionCfg is the source of truth for both tasks. Climb adds
  # sphere-wall pairs and grip equalities, but does not alter robot geoms.
  robot_cfg.collisions = (PHYBOT_C2_WALK_COLLISION,)
  # Keep the robot collision model byte-for-byte equivalent to walk. The
  # climb wall uses an analytic plane for contact while retaining its finite
  # box as a non-colliding visual; this avoids convex CCD substitutions on the
  # robot itself.
  wall_cfg = get_wall_cfg(analytic_collision_plane=True)

  # Override init state at task level (do NOT mutate the shared keyframe
  # singleton on robot_cfg.init_state — other c2 tasks reuse it).
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.245, 0.0, 1.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.815,
      "left_hip_roll": 0.09,
      "left_hip_yaw": -0.265,
      "left_knee": 0.765,
      "left_ankle_pitch": 1.34,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.815,
      "right_hip_roll": -0.09,
      "right_hip_yaw": 0.265,
      "right_knee": 0.765,
      "right_ankle_pitch": 1.34,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "neck_yaw": 0.0,
      "neck_pitch": 0.0,
      "left_shoulder_pitch": -2.0,
      "left_shoulder_roll": -0.17,
      "left_shoulder_yaw": 0.0,
      "left_elbow_pitch": -1.14,
      "right_shoulder_pitch": -2.0,
      "right_shoulder_roll": 0.17,
      "right_shoulder_yaw": 0.0,
      "right_elbow_pitch": -1.14,
    },
    joint_vel={".*": 0.0},
  )

  # Climb-specific upper-body PD. The deploy walk gains (arms k=50, waist
  # k=20/d=10) cannot hold the hang posture: the torso sags ~2 cm, pads peel
  # off the wall and no episode survives past base_height termination.
  # Effort limits stay at the deploy hardware values (static hang peaks at
  # ~15% of the 58 Nm cap); shoulder_yaw keeps the deploy gain. Rebuild via
  # dataclasses.replace — the actuator cfgs are shared module singletons.
  climb_upper_body_gains = {
    "waist_yaw": (100.0, 5.0),
    ".*_shoulder_pitch": (100.0, 5.0),
    ".*_shoulder_roll": (100.0, 5.0),
    ".*_elbow_pitch": (100.0, 5.0),
  }
  robot_cfg.articulation = dataclasses.replace(
    robot_cfg.articulation,
    # Reserve 5% of the hard range at each end. The action term below clamps
    # PD targets to these limits; the physical hard stop remains a final guard.
    soft_joint_pos_limit_factor=0.9,
    actuators=tuple(
      dataclasses.replace(act, stiffness=gains[0], damping=gains[1])
      if (gains := climb_upper_body_gains.get(act.target_names_expr[0]))
      is not None
      else act
      for act in robot_cfg.articulation.actuators
    ),
  )

  # Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5) — same as tune_pose,
  # so the spawned base position lines up with what was visualized there.

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 500)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=microspine_contact_sensor_name,
      primary=ContactMatch(
        mode="geom",
        pattern=get_sphere_geom_names(microspine_cfg),
        entity="robot",
      ),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      global_frame=False,
      history_length=10,
    ),
  )

  # Ordinary per-sphere contacts handle compression and pre-engage sliding;
  # per-EE equalities provide grip.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_pad_wall_contact_excludes(spec, microspine_cfg)
    add_attachment_equalities(spec, microspine_cfg)
    # Disable the aligned terrain ground plane — gravity-tilt curriculum
    # replaces the floor; this forces the robot to actually grip the wall.
    configure_phybot_c2_floor_geom(spec, enable_contact=False)

  cfg.scene.spec_fn = _spec_fn

  cfg.events[microspine_event_name] = make_constraint_event_cfg(
    microspine_cfg,
    affordance_name=microspine_affordance_name,
  )

  # ------------------------------------------------------------------------
  # Rewards — progress is the only sustained positive signal. Safe posture,
  # contact, and alignment are zero-cost constraints rather than positive
  # income, so a motionless hang cannot become a local reward optimum.
  # ------------------------------------------------------------------------
  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- task progress. The first 0.5 s follows the physically settled pose;
    # reset sag and reset-time passive engagement earn nothing.
    "climb_height_progress": RewardTermCfg(
      func=mdp.climb_height_progress(),
      weight=2.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "command_threshold": 0.01,
        "max_normalized_progress": 1.5,
        "settle_time_s": 0.5,
      },
    ),
    "signed_height_progress": RewardTermCfg(
      func=mdp.signed_climb_height_progress(),
      weight=0.5,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "command_threshold": 0.01,
        "max_normalized_progress": 1.0,
        "settle_time_s": 0.5,
      },
    ),
    "confirmed_anchor_advance": RewardTermCfg(
      func=mdp.confirmed_anchor_advance(),
      weight=0.5,
      params={
        "event_term_name": microspine_event_name,
        "settle_time_s": 0.5,
        "confirmation_steps": (
          microspine_cfg.limit_surface.hold_grace_steps
          + microspine_cfg.limit_surface.violation_consecutive_steps
        ),
        "minimum_swing_time": 0.1,
        "step_height": 0.06,
        "frontier_bonus_scale": 0.5,
      },
    ),
    "commanded_anchor_advance": RewardTermCfg(
      func=mdp.commanded_anchor_advance,
      weight=4.0,
      params={"command_name": gait_command_name},
    ),
    "productive_gait_cycle": RewardTermCfg(
      func=mdp.productive_gait_cycle,
      weight=2.0,
      params={"command_name": gait_command_name},
    ),
    # -- tracking and workspace costs. These are zero at the desired state;
    # unlike Gaussian rewards they never pay the robot merely to stand still.
    "climb_velocity_z_error": RewardTermCfg(
      func=mdp.climb_velocity_z_error_l2,
      weight=-0.5,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "velocity_scale": 0.05,
      },
    ),
    "lateral_velocity_y_l2": RewardTermCfg(
      func=mdp.climb_lateral_velocity_y_l2,
      weight=-0.2,
      params={"asset_cfg": robot_cfg_e, "velocity_scale": 0.1},
    ),
    "wall_distance_band_l2": RewardTermCfg(
      func=mdp.wall_distance_band_l2,
      weight=-0.2,
      params={
        "asset_cfg": robot_cfg_e,
        "target_distance": 0.255,
        "tolerance": 0.08,
        "distance_scale": 0.1,
      },
    ),
    "ee_face_parallel_error": RewardTermCfg(
      func=mdp.ee_face_parallel_error,
      weight=-0.5,
      params={
        "asset_cfg": robot_cfg_e,
        "ee_body_names": microspine_cfg.ee_body_names,
        "event_term_name": microspine_event_name,
        "command_name": microspine_affordance_name,
        "face_local_axis": microspine_cfg.face_local_axis,
      },
    ),
    # -- hybrid grip safety. Two, three, and four anchors are all acceptable;
    # only insufficient support is penalized. A reverse-shear unhook is free
    # while safe, whereas structural failure modes carry one-shot event costs.
    "microspine_support_loss": RewardTermCfg(
      func=mdp.microspine_support_loss,
      weight=-3.0,
      params={
        "event_term_name": microspine_event_name,
        "minimum_supports": 2,
      },
    ),
    "climb_gait_support_shortfall": RewardTermCfg(
      func=mdp.climb_gait_support_shortfall,
      weight=-2.0,
      params={"command_name": gait_command_name},
    ),
    "unsafe_microspine_release": RewardTermCfg(
      func=mdp.unsafe_microspine_release_penalty,
      weight=-1.0,
      params={
        "command_name": microspine_affordance_name,
        "minimum_supports": 2,
        "settle_time_s": 0.5,
      },
    ),
    "unscheduled_microspine_release": RewardTermCfg(
      func=mdp.unscheduled_microspine_release,
      weight=-2.0,
      params={"command_name": gait_command_name},
    ),
    "climb_gait_phase_timeout": RewardTermCfg(
      func=mdp.climb_gait_phase_timeout,
      # Kept as an event marker; the real deterrent is the gait_phase_timeout
      # TERMINATION (fines lost the economics twice against stretch-farming).
      weight=-5.0,
      params={"command_name": gait_command_name},
    ),
    "climb_swing_progress_shaping": RewardTermCfg(
      func=mdp.climb_swing_progress_shaping,
      # Pays the risky middle of a regrip (weight shift + reach) that the
      # 2026-07-17 frames showed exploration never crosses. A full approach
      # earns ~1.0 step-unit; potential-based so retreat/oscillation nets 0+.
      weight=2.0,
      params={"command_name": gait_command_name, "progress_cap": 0.5},
    ),
    "climb_seat_press_shaping": RewardTermCfg(
      func=mdp.climb_seat_press_shaping,
      # Partial credit for the hardest regrip step (pressing the commanded
      # pad into the wall). Saturates at an engage-grade press
      # (4 spheres x 3 N threshold = force_scale 12 N), so per-step income
      # is capped and grinding into the wall earns nothing extra.
      weight=1.0,
      params={
        "command_name": gait_command_name,
        "affordance_command_name": microspine_affordance_name,
        "force_scale": 12.0,
      },
    ),
    "commanded_ee_target_error": RewardTermCfg(
      func=mdp.commanded_ee_target_error_l2,
      weight=-0.2,
      params={"command_name": gait_command_name},
    ),
    "ee_slide_penalty": RewardTermCfg(
      func=mdp.ee_slide_penalty,
      weight=-0.1,
      params={
        "asset_cfg": robot_cfg_e,
        "hand_body_names": microspine_cfg.ee_body_names,
        "event_term_name": microspine_event_name,
      },
    ),
    # -- regularization.
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.ang_vel_xy_l2,
      weight=-0.05,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={"asset_cfg": robot_cfg_e},
    ),
    "bounded_dof_acc_l2": RewardTermCfg(
      func=mdp.bounded_joint_acc_l2,
      weight=-0.02,
      params={
        "asset_cfg": robot_cfg_e,
        "acceleration_scale": 100.0,
        "max_cost": 10.0,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    # The target clamp is the safety guard; this term gives the policy a
    # learning signal to stay away from the guard and from physical joint stops.
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": _policy_joint_cfg()},
    ),
    # RewardManager scales by dt, so -500 is a -10 one-shot failure cost.
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-500.0),
  }

  cfg.terminations["no_progress"] = TerminationTermCfg(
    func=mdp.no_progress_stall(),
    params={
      "asset_cfg": robot_cfg_e,
      "command_name": velocity_command_name,
      "settle_time_s": 0.5,
      "stall_time_s": 5.0,
      "minimum_progress": 0.01,
      "command_threshold": 0.01,
    },
  )
  cfg.terminations["gait_curriculum_success"] = TerminationTermCfg(
    func=mdp.gait_curriculum_success,
    time_out=True,
    params={"command_name": gait_command_name},
  )
  # Disqualification, not a fine: reward-level timeout penalties lost the
  # economics twice (fine -0.08/ep vs stretch-farmed income +0.12/ep).
  # Failure termination also collects the -500 termination_penalty.
  cfg.terminations["gait_phase_timeout"] = TerminationTermCfg(
    func=mdp.gait_phase_timeout,
    params={"command_name": gait_command_name},
  )

  cfg.metrics = {
    "microspine_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": microspine_affordance_name},
    ),
    "left_hand_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": microspine_affordance_name, "ee_index": 0},
    ),
    "right_hand_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": microspine_affordance_name, "ee_index": 1},
    ),
    "left_foot_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": microspine_affordance_name, "ee_index": 2},
    ),
    "right_foot_engagement_occupancy": MetricsTermCfg(
      func=mdp.microspine_engagement_occupancy,
      params={"command_name": microspine_affordance_name, "ee_index": 3},
    ),
    "microspine_forced_release_rate_hz": MetricsTermCfg(
      func=mdp.microspine_forced_release_rate,
      params={"command_name": microspine_affordance_name},
    ),
    "microspine_engage_event_rate_hz": MetricsTermCfg(
      func=mdp.microspine_engage_events,
      params={"command_name": microspine_affordance_name},
    ),
    "microspine_hold_margin_engaged": MetricsTermCfg(
      func=mdp.microspine_hold_margin_engaged,
      params={"command_name": microspine_affordance_name},
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
    "gait_productive_regrip_rate_hz": MetricsTermCfg(
      func=mdp.gait_productive_regrips,
      params={"command_name": gait_command_name},
    ),
    "gait_productive_cycle_rate_hz": MetricsTermCfg(
      func=mdp.gait_productive_cycles,
      params={"command_name": gait_command_name},
    ),
    "gait_release_phase_occupancy": MetricsTermCfg(
      func=mdp.gait_phase_occupancy,
      params={"command_name": gait_command_name, "phase": 0},
    ),
    "gait_swing_phase_occupancy": MetricsTermCfg(
      func=mdp.gait_phase_occupancy,
      params={"command_name": gait_command_name, "phase": 1},
    ),
    "gait_seat_phase_occupancy": MetricsTermCfg(
      func=mdp.gait_phase_occupancy,
      params={"command_name": gait_command_name, "phase": 2},
    ),
    "gait_transfer_phase_occupancy": MetricsTermCfg(
      func=mdp.gait_phase_occupancy,
      params={"command_name": gait_command_name, "phase": 3},
    ),
    "gait_target_release_rate_hz": MetricsTermCfg(
      func=mdp.gait_target_release_rate,
      params={"command_name": gait_command_name},
    ),
    "gait_target_contact_rate_hz": MetricsTermCfg(
      func=mdp.gait_target_contact_rate,
      params={"command_name": gait_command_name},
    ),
    "gait_regrip_rate_hz": MetricsTermCfg(
      func=mdp.gait_regrip_rate,
      params={"command_name": gait_command_name},
    ),
  }

  all_anchors = microspine_cfg.ee_body_names
  hand_anchors = ("left_hand_spine", "right_hand_spine")
  foot_anchors = ("left_foot_spine", "right_foot_spine")
  cfg.curriculum = {
    "climb_difficulty": CurriculumTermCfg(
      func=mdp.performance_gated_climb,
      params={
        "gait_command_name": gait_command_name,
        "affordance_command_name": microspine_affordance_name,
        "min_evaluated_episodes": 512,
        "advance_success_rate": 0.7,
        # Skill ladder v2 (2026-07-17). Cross-round data: upward FOOT regrips
        # emerge quickly under the plain full-cycle stages (productive rate
        # 0.40 Hz by iter 150 on the previous run), so those stages stay as
        # proven. The one skill that never emerged is HAND regrips — so after
        # basic competence the ladder forces them in the safest possible
        # setting: all four anchors engaged at reset, only hands cycle, and a
        # tiny 2 cm commanded step. A feet-only stage was tried and removed:
        # hands pinned forever let the body sag cycle over cycle (negative
        # anchor advance, engagement collapse).
        "stages": [
          # The regrip cycle alternates foot -> hand -> foot -> hand, so ANY
          # two-consecutive-regrip stage implicitly demands a hand regrip.
          # That is why every historical run stalled exactly at the 2-regrip
          # stage: it examined hands at full difficulty. The dedicated hand
          # rung therefore comes BEFORE any multi-regrip stage.
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": all_anchors,
            "required_productive_regrips": 1,
            "randomize_start_limb": True,
          },
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": all_anchors,
            "gait_ee_names": hand_anchors,
            "step_height": 0.02,
            "required_productive_regrips": 1,
            "randomize_start_limb": True,
          },
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": all_anchors,
            "gait_ee_names": hand_anchors,
            "step_height": 0.02,
            "required_productive_regrips": 2,
            "randomize_start_limb": True,
          },
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": all_anchors,
            "required_productive_regrips": 2,
            "randomize_start_limb": True,
          },
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": all_anchors,
            "required_productive_regrips": 4,
            "randomize_start_limb": True,
          },
          {
            "angle_deg": 45.0,
            "initial_engaged_ee_names": hand_anchors,
            "required_productive_regrips": 4,
            "randomize_start_limb": False,
          },
          {
            "angle_deg": 60.0,
            "initial_engaged_ee_names": hand_anchors,
            "required_productive_regrips": 4,
            "randomize_start_limb": False,
          },
          {
            "angle_deg": 75.0,
            "initial_engaged_ee_names": hand_anchors,
            "required_productive_regrips": 4,
            "randomize_start_limb": False,
          },
          {
            "angle_deg": 90.0,
            "initial_engaged_ee_names": hand_anchors,
            "required_productive_regrips": 4,
            "randomize_start_limb": False,
          },
        ],
      },
    )
  }

  cfg.actions["joint_pos"] = mdp.SoftJointPositionActionCfg(
    entity_name="robot",
    actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
    scale=PHYBOT_C2_ACTION_SCALE,
    use_default_offset=True,
  )

  actor_terms = cfg.observations["actor"].terms
  actor_terms["joint_pos_rel"].params = {"asset_cfg": _policy_joint_cfg()}
  actor_terms["joint_vel_rel"].params = {"asset_cfg": _policy_joint_cfg()}
  actor_terms["microspine_engaged_state"] = ObservationTermCfg(
    func=mdp.microspine_engaged_state,
    params={"command_name": microspine_affordance_name},
  )
  actor_terms["microspine_contact_fraction"] = ObservationTermCfg(
    func=mdp.microspine_contact_fraction,
    params={
      "command_name": microspine_affordance_name,
      "spheres_per_ee": (
        microspine_cfg.spheres_per_ee_rows * microspine_cfg.spheres_per_ee_cols
      ),
    },
  )
  actor_terms["microspine_disengage_time"] = ObservationTermCfg(
    func=mdp.ee_disengage_time_obs,
    params={"event_term_name": microspine_event_name, "clip": 1.5},
  )
  actor_terms["microspine_cooldown"] = ObservationTermCfg(
    func=mdp.microspine_cooldown_obs,
    params={
      "command_name": microspine_affordance_name,
      "cooldown_steps": microspine_cfg.reengage_cooldown_steps,
    },
  )
  actor_terms["climb_gait_command"] = ObservationTermCfg(
    func=mdp.climb_gait_command,
    params={"command_name": gait_command_name},
  )

  # Asymmetric actor-critic: the critic group is the actor group plus
  # privileged microspine state (hold margin, anchor loads, forced releases)
  # that a real robot cannot sense. Term cfgs are re-instantiated so the two
  # groups never share mutable cfg objects. Built after all actor-term edits
  # above so the copies pick them up.
  critic_terms = {
    name: dataclasses.replace(term) for name, term in actor_terms.items()
  }
  critic_terms["microspine_hold_margin"] = ObservationTermCfg(
    func=mdp.microspine_hold_margin,
    params={"command_name": microspine_affordance_name},
  )
  critic_terms["microspine_anchor_load"] = ObservationTermCfg(
    func=mdp.microspine_anchor_load,
    params={"command_name": microspine_affordance_name},
    scale=0.01,  # Newtons → O(1).
  )
  critic_terms["microspine_forced_release"] = ObservationTermCfg(
    func=mdp.microspine_forced_release,
    params={"command_name": microspine_affordance_name},
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  if play:
    # The viewer shows one world. Keeping the training batch of 4096 here
    # simulates thousands of invisible robots and is the main cause of
    # sub-real-time playback unless callers remember ``--num-envs 1``.
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Debug transition logging forces CPU-GPU synchronization in the physics
    # step. Keep normal play fast; it can still be enabled explicitly on a
    # locally constructed config when diagnosing the state machine.
    microspine_cfg.debug = False
    cfg.events[microspine_event_name].params["microspine_cfg"].debug = False
    # Evaluation uses the final task directly and must not truncate after a
    # curriculum goal. CLI gravity overrides can still edit this single stage.
    final_stage = cfg.curriculum["climb_difficulty"].params["stages"][-1]
    cfg.curriculum["climb_difficulty"].params["stages"] = [final_stage]
    cfg.terminations.pop("gait_curriculum_success", None)
    # Evaluation rollouts should show the full behavior, not end on the
    # training-time disobedience disqualification.
    cfg.terminations.pop("gait_phase_timeout", None)

  return cfg
