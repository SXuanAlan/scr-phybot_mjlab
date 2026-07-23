"""PHYBot C2 wall-climbing environment configuration.

DeepMimic-style layout (2026-07-21 redesign, replacing the commanded-gait
state machine):

  * A motion command (``climb_motion``) indexes the IK regrip cycle
    ``assets/motions/climb_cycle.npy``; episode resets are reference-state
    initialization (random cycle phase, latches seeded from the clip).
  * ``motion_*`` tracking rewards teach the cycle SHAPE (joint space,
    root-frame EE positions, root orientation, engage schedule). No global
    position is tracked — the clip's world trajectory is used only for reset
    placement and the debug ghost.
  * Height/velocity progress rewards remain the climbing DRIVE, and the
    microspine engage/safety battery remains the physics contract.

C2-specific notes:
  * 4 grip end-effectors: left/right hand_spine + left/right foot_spine.
    Each spine has a 2x2 sphere grid (16 spheres total).
  * No ``waist_roll`` joint/body — Spot/Anymal-style torso-deviation terms
    are not migrated here (intentionally minimal).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

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


_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"
_CLIMB_MOTION_FILE = _MOTIONS_DIR / "climb_cycle.npy"


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
  motion_command_name = "climb_motion"
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
    # Fallback reset latches only: the motion command's RSI overwrites them
    # per env with the sampled clip frame's engage schedule right after the
    # affordance's own reset (see seed_engaged). All-four matches frame 0.
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
  # Command exactly the average climb rate one reference cycle delivers
  # (step_height / cycle duration), so the height-progress normalization and
  # the commanded-velocity observation agree with the motion being imitated.
  clip_vz = mdp.clip_average_climb_rate(_CLIMB_MOTION_FILE)
  cfg.commands[velocity_command_name].ranges.lin_vel_z = (clip_vz, clip_vz)
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  terms = cfg.observations["actor"].terms
  terms["base_lin_vel"].func = mdp.builtin_sensor
  terms["base_lin_vel"].params = {"sensor_name": "robot/baselink-velocity"}
  terms["base_ang_vel"].func = mdp.builtin_sensor
  terms["base_ang_vel"].params = {"sensor_name": "robot/baselink-gyro"}
  terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  terms["projected_gravity"].params = {"sensor_name": "robot/baselink-quat"}
  # Drop the base factory's fixed-period gait clock and legacy hand-latch
  # obs: the motion command below provides the phase clock (cos/sin, tied to
  # the reference cycle) and the reference engage schedule, and the
  # microspine_engaged_state term added later covers all four latches.
  cfg.observations["actor"].terms.pop("gait_clock", None)
  cfg.observations["actor"].terms.pop("hand_engaged_state", None)

  cfg.commands[microspine_affordance_name] = MicrospineAffordanceCfg(
    microspine_cfg=microspine_cfg,
    contact_sensor_name=microspine_contact_sensor_name,
    physics_event_name=microspine_event_name,
  )
  # Registered AFTER the affordance on purpose: its reset (RSI) seeds the
  # per-env latch state and must run after the affordance's own reset (the
  # command asserts this ordering).
  cfg.commands[motion_command_name] = mdp.ClimbMotionCommandCfg(
    entity_name="robot",
    motion_file=str(_CLIMB_MOTION_FILE),
    affordance_command_name=microspine_affordance_name,
    ee_body_names=microspine_cfg.ee_body_names,
    sampling_mode="uniform",  # DeepMimic RSI: random cycle phase per episode.
    joint_pos_noise=0.02,
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
  # Climb-zero stance v3 (2026-07-22): v2 with the hand pads raised 1.5 cm
  # (pad z 1.738 -> 1.753) per the higher-reach request. On the wristless
  # 4-DoF arm the flat-pad wall point is a function of base standoff alone,
  # so the raise is bought by standing closer: base 0.173 m from the wall
  # (v2 was 0.196); the legs re-solve with the foot pads 1.0 cm lower
  # (z=0.566). Pads stay flat on the wall (|gap| <= 0.1 mm, tilt <= 0.7
  # deg). Elbow straightens to -0.54 — the regime of v2's rejected 0.176 m
  # variant — but the swing-singularity blowup that killed that variant
  # predates the current gen_climb_clip.py safeguards (mid-swing pad-rot
  # floor 20, posture anchor 0.6, frozen other-arm reference): the v3 cycle
  # regenerates with swing pad-rot residual <= 0.2 deg and no yaw whip.
  # Measured cycle cost vs regenerating on v2 (left-hand landing 3.1 mm,
  # roll margin 1.8%, swing-corner wall graze +1.5 mm — the left-swing miss
  # is inherent: the still-low right anchor drags the base): v3 lands the
  # left hand 4.4 mm off schedule, shoulder-roll peaks 1.3% from the stop,
  # graze +2.7 mm. Knee-to-wall clearance holds at 1.73 cm (less knee flex
  # cancels the closer stance). A +2.0 cm / 0.166 m variant was generated
  # and rejected: it sits exactly on the reach frontier (4.9 mm miss, roll
  # 1.1%).
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.327, 0.0, 1.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.4592,
      "left_hip_roll": 0.1607,
      "left_hip_yaw": -0.1584,
      "left_knee": 0.4198,
      "left_ankle_pitch": 1.256,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.4592,
      "right_hip_roll": -0.1607,
      "right_hip_yaw": 0.1584,
      "right_knee": 0.4198,
      "right_ankle_pitch": 1.256,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "neck_yaw": 0.0,
      "neck_pitch": 0.0,
      "left_shoulder_pitch": -2.5999,
      "left_shoulder_roll": -0.166,
      "left_shoulder_yaw": 0.0,
      "left_elbow_pitch": -0.5386,
      "right_shoulder_pitch": -2.5999,
      "right_shoulder_roll": 0.166,
      "right_shoulder_yaw": 0.0,
      "right_elbow_pitch": -0.5386,
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
    # Reserve 1% of the hard range at each end. The action term below clamps
    # PD targets to these limits; the physical hard stop remains a final
    # guard. Widened from 0.9 (2026-07-21): the reference cycle works close
    # to some stops — shoulder_roll kinematically peaks 0.06 rad from its
    # hard limit at state 4 (one hand high, other anchored low) — and
    # tracking must never fight the target clamp. The clip keeps its own
    # clearance elsewhere via the gen script's limit-avoidance springs.
    soft_joint_pos_limit_factor=0.98,
    actuators=tuple(
      dataclasses.replace(act, stiffness=gains[0], damping=gains[1])
      if (gains := climb_upper_body_gains.get(act.target_names_expr[0]))
      is not None
      else act
      for act in robot_cfg.articulation.actuators
    ),
  )

  # Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5). NOTE the stance-v2
  # base x (1.304) no longer matches what scripts/tune_pose.py visualized for
  # the v1 stance; the stance is now designed by the IK stance search
  # (2026-07-21) with pads pinned to the wall plane.

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
  # Reference-state initialization owns the robot's reset pose and velocities
  # (the motion command writes them AFTER reset events run); the generic
  # perturbation events would only be overwritten.
  cfg.events.pop("reset_robot_joints", None)
  cfg.events.pop("reset_robot_base", None)

  # ------------------------------------------------------------------------
  # Rewards — DeepMimic split. ``motion_*`` exp kernels teach the regrip
  # cycle (no global position tracking); height/velocity progress terms
  # drive actual climbing; the microspine battery enforces the physics.
  # ------------------------------------------------------------------------
  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- motion imitation (positive tracking income, BeyondMimic exp kernel).
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_pos_error_exp,
      weight=2.0,
      params={"command_name": motion_command_name, "std": 0.2},
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_vel_error_exp,
      weight=0.5,
      params={"command_name": motion_command_name, "std": 1.5},
    ),
    # Root-frame EE positions: shapes reach and swing arcs without pinning
    # the base to the clip's world trajectory. 2.5 (was 1.5, motionref5):
    # the reach toward the next hold is the first link of the regrip chain
    # (reach -> press -> latch); it must out-earn holding a parked limb
    # still, or the weld attractor keeps winning the swing phase.
    "motion_ee_pos": RewardTermCfg(
      func=mdp.motion_ee_pos_error_exp,
      weight=2.5,
      params={"command_name": motion_command_name, "std": 0.1},
    ),
    "motion_root_ori": RewardTermCfg(
      func=mdp.motion_root_ori_error_exp,
      weight=0.5,
      params={"command_name": motion_command_name, "std": 0.3},
    ),
    # The clip's engage schedule replaces the old commanded-gait machinery:
    # a latch state disagreeing with the reference phase costs per step.
    # -1.0 (was -0.25, raised 2026-07-21): the motionref1 audit showed the
    # policy ignoring the schedule wholesale (match 0.44, LF parked at 0.03
    # occupancy) because a wrong latch state cost pocket change next to the
    # height income. ~2 mismatched pads must read as -2/s against a budgeted
    # ~3.1/s task income, not -0.5/s against 22/s.
    # grace 0.25 s (2026-07-22): the bill ramps linearly over the last 0.25 s
    # to a scheduled flip (debounce 0.06 s + cooldown 0.20 s make instant
    # compliance impossible) — the same countdown signal the policy sees in
    # motion_command, so chart -> latch timing is a learnable mapping. Deep
    # inside a segment (>88% of the cycle) the price is unchanged.
    "motion_engage_mismatch": RewardTermCfg(
      func=mdp.motion_engage_mismatch,
      weight=-1.0,
      params={"command_name": motion_command_name, "grace_time_s": 0.25},
    ),
    # Seat-phase step credit (2026-07-22 evening, motionref5): every run so
    # far welded one limb pair because a regrip's first motions earn nothing
    # until the latch closes (motionref4 audit: feet occ 1.00 / hands 0.04 at
    # iter 1450 — the 7th occurrence). Pays pad press inside the scheduled
    # regrip window only (ref_engaged, or regrip countdown <= 0.3 s — the
    # observed channel), stops at latch (engaged anchors carry tension), and
    # the +0.4 ceiling stays below the -1.0 mismatch bill so pressing
    # without latching is strictly net-negative.
    "motion_scheduled_press": RewardTermCfg(
      func=mdp.motion_scheduled_press_shaping,
      weight=0.4,
      params={"command_name": motion_command_name, "regrip_lead_time_s": 0.3},
    ),
    # -- task drive: pay for actual height, normalized by the commanded rate.
    # The reference climbs in one 0.8 s burst per 4.3 s cycle (peak ~10x the
    # average rate), so the per-step cap must clear the burst: 12 x average.
    # The first 0.5 s follows the physically settled pose; reset sag and
    # reset-time passive engagement earn nothing.
    # rate_budget_scale 1.25 (added 2026-07-21): the per-step cap alone let a
    # scramble gait bank 12x continuously — motionref1 climbed at 20x the
    # commanded rate and height income (21.9/s) buried the whole imitation
    # battery (1.4/s). The cumulative budget pays honest bursts from headroom
    # banked in the cycle's non-climbing phases but rate-limits sustained
    # overspeed to 1.25x the commanded climb distance.
    "climb_height_progress": RewardTermCfg(
      func=mdp.climb_height_progress(),
      weight=2.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "command_threshold": 0.01,
        "max_normalized_progress": 12.0,
        "settle_time_s": 0.5,
        "rate_budget_scale": 1.25,
      },
    ),
    # The signed term's upper cap is the reference itself (2026-07-22): net
    # upward income may never exceed the height the clip has gained at this
    # phase, so income tracks the schedule exactly — no flat-rate slack.
    "signed_height_progress": RewardTermCfg(
      func=mdp.signed_climb_height_progress(),
      weight=0.5,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "command_threshold": 0.01,
        "max_normalized_progress": 12.0,
        "settle_time_s": 0.5,
        "reference_budget_command": motion_command_name,
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
        # Base standoff of the climb-zero v3 stance (1.5 - 1.327).
        "target_distance": 0.173,
        "tolerance": 0.08,
        "distance_scale": 0.1,
      },
    ),
    # -- microspine physics contract. Two, three, and four anchors are all
    # acceptable; only insufficient support is penalized. A reverse-shear
    # unhook is free while safe; structural failures carry one-shot costs.
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
    "microspine_support_loss": RewardTermCfg(
      func=mdp.microspine_support_loss,
      # -4.0 (was -3.0 pre-DeepMimic): the tracking terms raised the ceiling
      # of positive income to ~4.5/s, and a one-anchor dangle still earning
      # ~2.4-3.5/s of partial tracking must stay strictly net-negative.
      weight=-4.0,
      params={
        "event_term_name": microspine_event_name,
        "minimum_supports": 2,
      },
    ),
    # -2.5 (was -1.0, raised 2026-07-21): motionref1's scramble gait ripped
    # anchors at 3.5 Hz, 40% of them genuine tension pull-offs, and the
    # per-event bill (~-0.4/s) never registered. Reverse-shear unhooks stay
    # free by design — this prices only the failure modes.
    "unsafe_microspine_release": RewardTermCfg(
      func=mdp.unsafe_microspine_release_penalty,
      weight=-2.5,
      params={
        "command_name": microspine_affordance_name,
        "minimum_supports": 2,
        "settle_time_s": 0.5,
      },
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
      # 0.03 (was 0.01): the anti-hack executioner against farming partial
      # tracking income while creeping. 1 cm / 5 s let a creeper survive on
      # 14% of the commanded rate (and graze past curriculum stage 0); the
      # reference's worst honest 5 s window rises 5.9 cm, so 3 cm keeps a
      # 2x margin for sloppy-but-real cycling.
      "minimum_progress": 0.03,
      "command_threshold": 0.01,
    },
  )
  # DeepMimic early termination: end rollouts whose pose left the reference
  # entirely (fallen/tangled). Threshold is generous — a clean hang at the
  # wrong phase scores ~0.1-0.2 rad mean error and survives.
  cfg.terminations["bad_tracking"] = TerminationTermCfg(
    func=mdp.motion_tracking_divergence,
    params={"command_name": motion_command_name, "threshold": 0.7},
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
    "motion_height_gained_m": MetricsTermCfg(
      func=mdp.motion_height_gained,
      params={"command_name": motion_command_name},
    ),
  }

  # Fixed-iteration gravity schedule (2026-07-21, replacing the performance
  # gate): graduation-by-height let a 20x-overspeed scramble gait promote
  # 45->75 deg inside 400 iterations while ignoring the engage schedule
  # entirely. Angle is now a function of training time alone — each rung gets
  # a guaranteed 3000-iteration residence for the imitation terms to shape
  # HOW the robot climbs before gravity gets harder. steps_per_iter must
  # match rl_cfg num_steps_per_env.
  # Three rungs 45/60/90 with +-5 deg per-episode randomization (2026-07-22):
  # each env draws its own angle at reset (45 rung spans 40-50, 60 spans
  # 55-65), which blurs the rung boundaries enough to retire the dedicated
  # 75 rung; the final vertical stage runs exactly 90 — jitter there would
  # mix in >90 overhangs, a different task. 90 gets the back half (6000 it).
  cfg.curriculum = {
    "gravity_angle": CurriculumTermCfg(
      func=mdp.gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0, "angle_deg": 45.0},
          {"iter": 3000, "angle_deg": 60.0},
          {"iter": 6000, "angle_deg": 90.0},
        ],
        "steps_per_iter": 96,
        "angle_jitter_deg": 5.0,
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
  # Cycle phase clock (cos/sin) + the clip's engage schedule chart: current
  # per-EE engage bits plus each EE's countdown to its next release and next
  # regrip (2026-07-22). The countdowns are the raster rows' upcoming edges
  # fused with the phase, so "prepare to let go / to re-latch" is visible in
  # advance instead of memorized from the clock. 14 dims total.
  actor_terms["motion_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": motion_command_name},
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
    # Evaluation uses the final task directly. play.py's gravity resolver
    # (CLI override or checkpoint-iteration lookup) can still replace this
    # single stage. Jitter off: evaluation and diagnostic rollouts (engage
    # raster) must be deterministic at the pinned angle.
    final_angle = cfg.curriculum["gravity_angle"].params["angle_stages"][-1]
    cfg.curriculum["gravity_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": final_angle["angle_deg"]}
    ]
    cfg.curriculum["gravity_angle"].params["angle_jitter_deg"] = 0.0
    # Evaluation starts at the cycle's canonical stance instead of a random
    # RSI phase, so rollouts read as one continuous climb from frame 0.
    cfg.commands[motion_command_name].sampling_mode = "start"
    cfg.commands[motion_command_name].joint_pos_noise = 0.0

  return cfg
