"""PHYBot C2 wall-climbing environment configuration.

Empirical climbing uses a CAD reference, 16 BeyondMimic-adapted rewards and
failure-adaptive RSI. The main task has 12 units per pad and the original
curriculum to 90 degrees; Corners-A60 has four units and a curriculum to 60.
Legacy Wang transition tasks retain their previous tracking/RSI contract.

Layer split: task structure (obs skeleton, velocity command, terminations,
metrics, curriculum) lives in ``climb_wall_env_cfg.py``; reusable machinery
(stage tables, curriculum pin, obs-group builders) in ``mdp``; this file binds
the PHYBot C2 robot — microspine pad physics, stance, PD table, sensors, DR —
and owns the reward table. Each ``(notes: <key>)`` pointer references the full
campaign narrative in docs/climb_env_cfg_notes.md.

C2-specific notes:
  * 4 grip end-effectors: left/right hand_spine + left/right foot_spine.
    Each pad has 2x2 contact proxies, separate from its 12 or 4 active units.
  * No ``waist_roll`` joint/body — Spot/Anymal-style torso-deviation terms
    are not migrated here (intentionally minimal).
"""

from __future__ import annotations

import dataclasses

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from scr_phybot_mjab.assets.motions import (
  CAD_CLIMB_INITIAL_STATE as _CLIMB_INITIAL_STATE_FILE,
  CAD_CLIMB_MOTION as _CLIMB_MOTION_FILE,
  LEGACY_CLIMB_MOTION as _LEGACY_CLIMB_MOTION_FILE,
)
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry
from scr_phybot_mjab.microspine.contact_sensor import WrenchContactSensorCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_ACTUATOR_ANKLE_PITCH,
  PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
  PHYBOT_C2_ACTUATOR_ELBOW_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_ROLL,
  PHYBOT_C2_ACTUATOR_HIP_YAW,
  PHYBOT_C2_ACTUATOR_KNEE,
  PHYBOT_C2_ACTUATOR_NECK,
  PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
  PHYBOT_C2_ACTUATOR_SHOULDER_ROLL,
  PHYBOT_C2_ACTUATOR_SHOULDER_YAW,
  PHYBOT_C2_ACTUATOR_WAIST_YAW,
  PHYBOT_C2_FOOT_TOES,
  PHYBOT_C2_PAD_GIMBAL,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
  get_spec,
)
from scr_phybot_mjab.assets.wall import get_wall_cfg
from scr_phybot_mjab.microspine.config import (
  MicrospineAffordanceCfg,
  MicrospineCfg,
)
from scr_phybot_mjab.microspine.contact import (
  ToeContact,
  add_contact_pairs,
  add_pad_wall_contact_excludes,
  add_arm_wall_contact_pairs,
  add_toe_contact_pairs,
  add_toe_geoms,
  get_sphere_geom_names,
  toe_geom_name,
)
from scr_phybot_mjab.microspine.constraint import (
  add_attachment_equalities,
  make_constraint_event_cfg,
)
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import (
  MICROSPINE_AFFORDANCE_COMMAND,
  MICROSPINE_CONTACT_SENSOR,
  MICROSPINE_GRIP_EVENT,
  MOTION_COMMAND,
  TOE_CONTACT_SENSOR,
  VELOCITY_COMMAND,
  make_climb_wall_env_cfg,
)
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
  domain_rand: bool = True,
  student_obs: bool = False,
  robustness_release: bool = False,
  contact_model: str = "empirical_unit",
  unit_layout: str | None = None,
  wall_angle_deg: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the constraint-based PHYBot C2 wall-climbing configuration.

  The PD gains and the s2r DR package are the shipping recipe; DR stays
  switchable for ablations.

  Args:
    play: evaluation mode (single env, pinned curriculum, no corruption).
    domain_rand: s2r hardening — obs noise, encoder bias, friction/mass DR,
      random forced-release events (walk/squat parity minus push_robot).
    student_obs: add the deployable "student" observation group (distillation
      target). Off for plain PPO variants: the extra 850-dim group costs
      ~1.3 GB of rollout storage at 4096 envs.
    robustness_release: inject random unscheduled latch releases. A FINAL
      stage on a policy that already masters exact-90 — see the block that
      installs it for why it is not part of the base DR package.
    unit_layout: active CAD unit sites. The main task uses all twelve units;
      explicit "corners" selects the four-unit hardware variants.
    wall_angle_deg: pin both training and play to this angle from horizontal,
      without curriculum jitter. None retains the configured angle ladder.
  """
  ##
  # Microspine physics. The cfg built here is FINAL at the end of this
  # section: the grip event (Events below) snapshots its top-level scalars
  # via a shallow copy (only ``.limit_surface`` stays the shared object), so
  # no scalar may be edited past that point — the sole sanctioned late write
  # is the play block's dual-owner ``.debug`` sync.
  ##

  if contact_model not in ("empirical_unit", "wang2017"):
    raise ValueError(f"Unknown climbing contact model: {contact_model}")
  unit_layout = unit_layout or "all"
  if contact_model != "empirical_unit" and unit_layout != "all":
    raise ValueError("Active unit selection requires the empirical unit model")
  if wall_angle_deg is not None and not 0.0 < wall_angle_deg <= 90.0:
    raise ValueError("wall_angle_deg must be in (0, 90]")
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
    # Task-rate timing: 0.20 s cooldown + 0.06 s debounce — the former
    # 0.60/0.10 pair left otherwise-valid contacts one step short of every
    # re-engagement. Empirical mode adds the per-active-unit preload gate below.
    # (notes: reengage_cooldown_steps)
    reengage_cooldown_steps=10,
    engage_consecutive_steps=3,
  )
  # Contact seating + face alignment engage the grip; the limit surface
  # owns release. A real pad locks position AND orientation across the patch,
  # with capacity along the pad-local rake direction.
  microspine_cfg.anchor_lock_rotation = True
  microspine_cfg.limit_surface.rake_mode = "ee_local"
  microspine_cfg.limit_surface.unit_layout = unit_layout
  # Billing spans come from the CAD unit-center sites at cfg-build time:
  # 63 mm along the rake and 84 mm laterally, independent of proxy dimensions.
  pad_geometry = matching_pad_geometry(get_spec(), microspine_cfg.ee_body_names,
    rake_local_axis=microspine_cfg.limit_surface.ee_rake_local_axis,
    face_local_axis=microspine_cfg.face_local_axis, unit_layout=unit_layout)
  span_rake, span_lateral = pad_geometry.spans
  microspine_cfg.limit_surface.unit_rows_rake = pad_geometry.rows
  microspine_cfg.limit_surface.unit_cols_lateral = pad_geometry.cols
  microspine_cfg.limit_surface.pad_span_rake_m = span_rake
  microspine_cfg.limit_surface.pad_span_lateral_m = span_lateral
  # Distribute the wrench about C over the selected CAD sites. Empirical
  # mode judges each active unit's draw; legacy Wang retains F0/12.
  # Equal stiffness and first-unit pad release require full-pad validation.
  microspine_cfg.limit_surface.moment_billing_mode = "per_unit"
  if PHYBOT_C2_PAD_GIMBAL:
    # Gimbal-era weld (2026-08-15): the pad self-aligns through the passive
    # wrist ball joint, so the weld is honestly RIGID — a latched pad does
    # not move relative to the wall. The 07-31 objection to rigid rotation
    # rows (closed-chain PD conflicts billed as ~6 N*m parasitic moments,
    # docs/microspine_tilt_calibration.md par.2) is resolved mechanically:
    # within the +-3 deg free travel only spring torque is transmitted.
    # At the stops, structural reaction also enters the weld moment. The
    # new reference uses the lower x stop; its PLA strength is unmeasured.
    microspine_cfg.anchor_torquescale = 1.0
  else:
    # 0.01: rotational rows compliant enough that PD-parasitic couples shed
    # into force redistribution instead of billing local pad moment, while
    # the honest-arm billing still guards sustained prying. (notes:
    # anchor_torquescale)
    microspine_cfg.anchor_torquescale = 0.01
  # The climbing candidate uses the measured single-unit distribution.
  # Explicit Wang opt-in preserves existing transition/spring experiments.
  microspine_cfg.limit_surface.model = contact_model
  microspine_cfg.limit_surface.f_t_max_range = (400.0, 650.0)
  if contact_model == "empirical_unit":
    # Capacity randomness is entirely in persistent per-unit empirical draws.
    # The dataset does not identify additional recruitment/seat multipliers.
    microspine_cfg.seat_quality_scales_capacity = False
    microspine_cfg.engage_min_spine_fraction = 1.0
    microspine_cfg.engage_unit_preload_n = 10.0
    microspine_cfg.limit_surface.capacity_ramp_floor = 1.0
    microspine_cfg.limit_surface.capacity_ramp_time_s = 0.0
    microspine_cfg.limit_surface.redraw_f_t_max_per_latch = False
    microspine_cfg.limit_surface.unit_catch_empty_prob = 0.0
    # Legacy diagnostic only: tested per-unit S-domain endpoint times the
    # active count, not a measured pad capacity or a DR draw.
    shear_domain_max = 30.0 * pad_geometry.count
    microspine_cfg.limit_surface.f_t_max_range = (shear_domain_max, shear_domain_max)
  if PHYBOT_C2_PAD_GIMBAL:
    # Tilt-peel retired: the suspension is now a real +-3 deg joint.
    # At the gimbal stops the arm's
    # torque flows into the weld and the per-unit moment billing owns the
    # failure. (0, 0) disables the angle check; the rigid weld keeps
    # capture-relative pad rotation at ~0 regardless.
    microspine_cfg.limit_surface.tilt_peel_max_deg_range = (0.0, 0.0)
  else:
    # "4-deg" calibration: 5 mm suspension travel / 74.7 mm row spacing =
    # ~3.8 deg before the tension-side row loads structurally and pulls out
    # (forced release, reason 5); the reference stays <= 1.1 deg. (notes:
    # tilt_peel_max_deg_range)
    microspine_cfg.limit_surface.tilt_peel_max_deg_range = (3.5, 4.5)
  # Latched pads hand the whole load path to the bilateral anchor; the
  # redundant sphere contacts (motionref6's margin 'barcode') switch off per
  # env while latched. (notes: disable_contacts_while_latched)
  microspine_cfg.disable_contacts_while_latched = True

  ##
  # Base cfg + sensor-backed base observations.
  ##

  cfg = make_climb_wall_env_cfg()
  cfg.scene.num_envs = 2048
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  if PHYBOT_C2_PAD_GIMBAL:
    # No angle-based release means no tilt-peel anneal ladder either — the
    # schedule would just rewrite live bands onto a retired channel.
    cfg.curriculum.pop("tilt_peel", None)
  terms = cfg.observations["actor"].terms
  terms["base_lin_vel"].func = mdp.builtin_sensor
  terms["base_lin_vel"].params = {"sensor_name": "robot/baselink-velocity"}
  terms["base_ang_vel"].func = mdp.builtin_sensor
  terms["base_ang_vel"].params = {"sensor_name": "robot/baselink-gyro"}
  terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  terms["projected_gravity"].params = {"sensor_name": "robot/baselink-quat"}

  ##
  # Commands.
  ##

  # Command exactly the average climb rate one reference cycle delivers
  # (step_height / cycle duration), so the height-progress normalization and
  # the commanded-velocity observation agree with the motion being imitated.
  motion_file = _CLIMB_MOTION_FILE if contact_model == "empirical_unit" else _LEGACY_CLIMB_MOTION_FILE
  clip_vz = mdp.clip_average_climb_rate(motion_file)
  cfg.commands[VELOCITY_COMMAND].ranges.lin_vel_z = (clip_vz, clip_vz)
  cfg.commands[MICROSPINE_AFFORDANCE_COMMAND] = MicrospineAffordanceCfg(
    microspine_cfg=microspine_cfg,
    contact_sensor_name=MICROSPINE_CONTACT_SENSOR,
    physics_event_name=MICROSPINE_GRIP_EVENT,
    skip_zero_time_update=contact_model == "empirical_unit",
  )
  # Registered AFTER the affordance on purpose: its reset (RSI) seeds the
  # per-env latch state and must run after the affordance's own reset (the
  # command asserts this ordering).
  cfg.commands[MOTION_COMMAND] = mdp.ClimbMotionCommandCfg(
    entity_name="robot",
    motion_file=str(motion_file),
    affordance_command_name=MICROSPINE_AFFORDANCE_COMMAND,
    ee_body_names=microspine_cfg.ee_body_names,
    sampling_mode="uniform",  # DeepMimic RSI: random cycle phase per episode.
    # Unprojected joint jitter moves already-seeded contacts off the wall.
    joint_pos_noise=0.0 if contact_model == "empirical_unit" else 0.02,
    # Along-wall spawn-height randomization (on top of the vertical-height
    # compensation), sized to the 2 m wall: at -0.30 the toes stay ~0.2 m
    # above the base line; at +0.15 the hands end an episode's climb
    # (~1.53 + 0.15 + 0.2) still under the 2 m top. Decouples absolute
    # altitude from cycle phase; play pins it to zero.
    height_offset_range=(-0.30, 0.0 if contact_model == "empirical_unit" else 0.15),
  )
  if contact_model == "empirical_unit":
    motion = cfg.commands[MOTION_COMMAND]
    motion.sampling_mode = "adaptive"
    motion.synchronize_reset_phase = True
    motion.initialize_seeded_anchors = True
    motion.tracking_body_names = (
      "torso", "torso_upper", "neck_pitch",
      "left_shoulder_roll", "left_elbow_pitch", "left_hand_spine",
      "right_shoulder_roll", "right_elbow_pitch", "right_hand_spine",
      "left_hip_yaw", "left_knee", "left_ankle_roll", "left_foot_spine",
      "right_hip_yaw", "right_knee", "right_ankle_roll", "right_foot_spine",
    )

  ##
  # Robot entity: toe geoms, shared walk collision, stance, PD table.
  ##

  robot_cfg = get_phybot_c2_robot_cfg()
  # Foot toes brace the wall during foot regrips (a released pad must never
  # slide on the surface). The geoms live on the ROBOT entity spec — sensors
  # resolve geom names against the entity; the wall PAIRS are scene-level
  # (_spec_fn below). (notes: toe geoms)
  toe_contacts = tuple(
    ToeContact(ee_body_name=ee, body_name=body, pos=pos, radius=radius)
    for ee, body, pos, radius in PHYBOT_C2_FOOT_TOES
  )

  def _robot_spec_with_toes() -> mujoco.MjSpec:
    spec = get_spec()
    add_toe_geoms(spec, toe_contacts, robot_prefix="")
    return spec

  robot_cfg.spec_fn = _robot_spec_with_toes
  # One shared CollisionCfg is the source of truth for both tasks. Climb adds
  # sphere-wall pairs and grip equalities, but does not alter robot geoms.
  robot_cfg.collisions = (PHYBOT_C2_WALK_COLLISION,)

  # Override init state at task level (do NOT mutate the shared keyframe
  # singleton on robot_cfg.init_state — other c2 tasks reuse it).
  # Climb-zero stance v3: hand pads at z=1.753 bought by standing closer
  # (base standoff 0.173 m from the wall plane), pads flat within 0.1 mm /
  # 0.7 deg; the 0.166 m higher-reach variant sits on the reach frontier and
  # was rejected. (notes: stance v3)
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
  if contact_model == "empirical_unit" and _CLIMB_INITIAL_STATE_FILE.exists():
    import json
    initial = json.loads(_CLIMB_INITIAL_STATE_FILE.read_text())
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
      pos=tuple(initial["root_position"]),
      rot=tuple(initial["root_quaternion_wxyz"]),
      joint_pos=initial["joint_positions"], joint_vel={".*": 0.0},
    )

  # pd100 table: v3's training gains — full upper chain INCLUDING
  # shoulder_yaw at 100/10 (kd=10 lifts the hanging-chain zeta 0.35 -> 0.5),
  # legs 100/10, ankles 50/5, neck default. (notes: pd100)
  # dataclasses.replace, not assignment: the actuator cfgs are shared
  # module singletons and walk/squat use the same objects.
  climb_actuators = (
    dataclasses.replace(PHYBOT_C2_ACTUATOR_HIP_PITCH, stiffness=100.0, damping=10.0),
    dataclasses.replace(PHYBOT_C2_ACTUATOR_HIP_ROLL, stiffness=100.0, damping=10.0),
    dataclasses.replace(PHYBOT_C2_ACTUATOR_HIP_YAW, stiffness=100.0, damping=10.0),
    dataclasses.replace(PHYBOT_C2_ACTUATOR_KNEE, stiffness=100.0, damping=10.0),
    PHYBOT_C2_ACTUATOR_ANKLE_PITCH,  # 50/5
    PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
    dataclasses.replace(PHYBOT_C2_ACTUATOR_WAIST_YAW, stiffness=100.0, damping=10.0),
    dataclasses.replace(
      PHYBOT_C2_ACTUATOR_SHOULDER_PITCH, stiffness=100.0, damping=10.0
    ),
    dataclasses.replace(
      PHYBOT_C2_ACTUATOR_SHOULDER_ROLL, stiffness=100.0, damping=10.0
    ),
    dataclasses.replace(
      PHYBOT_C2_ACTUATOR_SHOULDER_YAW, stiffness=100.0, damping=10.0
    ),
    dataclasses.replace(PHYBOT_C2_ACTUATOR_ELBOW_PITCH, stiffness=100.0, damping=10.0),
    PHYBOT_C2_ACTUATOR_NECK,
  )
  robot_cfg.articulation = dataclasses.replace(
    robot_cfg.articulation,
    # Reserve 1% of the hard range at each end: the reference cycle works
    # 0.06 rad from some stops and tracking must never fight the target
    # clamp; the physical hard stop remains the final guard. (notes: soft
    # limit 0.98)
    soft_joint_pos_limit_factor=0.98,
    actuators=climb_actuators,
  )

  ##
  # Scene: wall entity, contact sensors, composed-scene spec edits.
  ##

  # Analytic plane for wall contact (finite box stays as non-colliding
  # visual): keeps the robot collision model byte-for-byte equal to walk —
  # no convex CCD substitutions on the robot. (notes: wall plane)
  wall_cfg = get_wall_cfg(analytic_collision_plane=True)
  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }
  # Ordinary corner-block contacts handle compression and pre-engage sliding;
  # per-EE equalities provide grip.
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 500)
  pad_sensor_cfg = WrenchContactSensorCfg if contact_model == "empirical_unit" else ContactSensorCfg
  pad_sensor_kwargs = {}
  if contact_model == "empirical_unit":
    # Four flat boxes per pad can produce 64 simultaneous pad-wall contacts
    # before any floor/toe contacts. The legacy 50-contact buffer drops forces
    # needed by the preload gate. The four-world 60-degree reference-PD
    # probe also reached 513 total contacts, overflowing a 128/world pool.
    # Reserve headroom for ordinary body/floor contacts after loss of grip.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 512)
    pad_sensor_kwargs["reference_sites"] = {
      name: f"robot/{name.rsplit('_c_', 1)[0]}_contact_center"
      for name in get_sphere_geom_names(microspine_cfg)
    }
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    pad_sensor_cfg(
      name=MICROSPINE_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern=get_sphere_geom_names(microspine_cfg),
        entity="robot",
      ),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      # A flat box can generate several contacts: sum their forces instead
      # of keeping only the strongest point. Netforce is in world coordinates;
      # the affordance projects it onto the live wall normal.
      fields=("found", "force", "normal", "tangent") + (
        ("torque", "pos") if contact_model == "empirical_unit" else ()
      ),
      reduce="netforce",
      num_slots=1,
      global_frame=False,
      history_length=cfg.decimation,
      **pad_sensor_kwargs,
    ),
    # Foot-toe brace force: feeds the scheduled toe-brace shaping reward.
    ContactSensorCfg(
      name=TOE_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern=tuple(toe_geom_name(toe) for toe in toe_contacts),
        entity="robot",
      ),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      global_frame=False,
    ),
  )
  if contact_model == "empirical_unit":
    cfg.scene.sensors += (
      ContactSensorCfg(
        name="climb_undesired_contact",
        primary=ContactMatch(
          mode="body", pattern=".*", entity="robot",
          exclude=(".*_spine.*", ".*_ankle_roll")),
        fields=("found", "force"), reduce="maxforce", num_slots=1,
        history_length=cfg.decimation,
      ),
    )

  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_pad_wall_contact_excludes(spec, microspine_cfg)
    add_toe_contact_pairs(spec, microspine_cfg, toe_contacts)
    if microspine_cfg.arm_wall_contact_geoms:
      add_arm_wall_contact_pairs(spec, microspine_cfg)
    add_attachment_equalities(spec, microspine_cfg)
    # Real floor, real gravity: under the wall-angle representation the
    # ground plane is physically meaningful again (a fall LANDS instead of
    # dropping into void) and the terminal 90° world matches
    # climb_transition's, which re-enables the same floor. The wall's base
    # line sits ON this plane at every angle, so floor and tilted wall
    # coexist without the plane sweeping the robot's spawn region.
    configure_phybot_c2_floor_geom(spec, enable_contact=True)

  cfg.scene.spec_fn = _spec_fn

  ##
  # Events. make_constraint_event_cfg snapshots the FINALIZED microspine cfg
  # (shallow copy: scalars frozen into the event's copy, .limit_surface
  # shared) — see the Microspine section header.
  ##

  cfg.events[MICROSPINE_GRIP_EVENT] = make_constraint_event_cfg(
    microspine_cfg,
    affordance_name=MICROSPINE_AFFORDANCE_COMMAND,
  )

  if domain_rand:
    ##
    # Domain randomization — v3's full walk-parity s2r set, carried from
    # birth (from-scratch-with-noise beats resume-then-inject by ~0.015 on
    # the clean exam). (notes: v4 DR revival)
    ##
    obs_noise = {
      "base_lin_vel": Unoise(n_min=-0.5, n_max=0.5),
      "base_ang_vel": Unoise(n_min=-0.2, n_max=0.2),
      "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
      "joint_pos_rel": Unoise(n_min=-0.01, n_max=0.01),
      "joint_vel_rel": Unoise(n_min=-1.5, n_max=1.5),
    }
    for term_name, noise in obs_noise.items():
      cfg.observations["actor"].terms[term_name].noise = noise
    cfg.observations["actor"].enable_corruption = True
    cfg.events["body_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        # Walk's regex: every collision geom except the feet. Spine-sphere
        # and toe-wall contacts are pair-driven, so the microspine
        # calibration stays untouched. (notes: body_friction)
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(r"^(?!(left|right)_ankle_roll_collision$).*_collision$",),
        ),
        "operation": "abs",
        "ranges": (0.6, 1.0),
      },
    )
    cfg.events["base_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
        # +-2 kg, not walk's +-5: a +5 kg draw pushes the worst F0 lottery
        # out of the Wang envelope (thinnest hold margin 12.3%). (notes:
        # base_mass)
        "ranges": (-2.0, 2.0),
        "operation": "add",
      },
    )
    cfg.events["encoder_bias"] = EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        # v3-exact +-0.01 rad: a 4-8 mm unobservable EE offset against the
        # ~5 mm seating travel — the single suspect for real-pad reseat
        # misses. (notes: encoder_bias)
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    )
  if robustness_release:
    # Optional recovery drill, off by default for the empirical candidate.
    # It is separate from failure randomness inferred from bench data.
    cfg.events["random_release"] = EventTermCfg(
      mode="interval",
      interval_range_s=(6.0, 12.0),
      func=mdp.random_microspine_release,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
    )

  ##
  # Rewards — DeepMimic split. ``motion_*`` exp kernels teach the regrip
  # cycle (no global position tracking); height/velocity progress terms
  # drive actual climbing; the microspine battery enforces the physics.
  ##

  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- motion imitation (positive tracking income, BeyondMimic exp kernel).
    # Pose income is gated by schedule adherence: floor 0.25 keeps birth
    # learning fed (engagement-blind pose kernels funded the welded statue);
    # grace = 0.3 of the violated window so a weld-through forfeits ~70% of
    # the gate credit regardless of clip timing. (notes: motion_joint_pos)
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_pos_error_exp,
      weight=2.0,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.2,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_vel_error_exp,
      weight=0.5,
      params={"command_name": MOTION_COMMAND, "std": 1.5},
    ),
    # Root-frame EE tracking shapes reach and swing arcs without pinning the
    # base to the clip's world trajectory; 2.5 so the reach toward the next
    # hold out-earns holding a parked limb still. (notes: motion_ee_pos)
    "motion_ee_pos": RewardTermCfg(
      func=mdp.motion_ee_pos_error_exp,
      weight=2.5,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.1,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    "motion_root_ori": RewardTermCfg(
      func=mdp.motion_root_ori_error_exp,
      weight=0.5,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.3,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    # ~2 mismatched pads must bill -2/s against a budgeted ~3.1/s task
    # income; the bill ramps over the violated window's outer 30% — the same
    # countdown the policy sees, and it survives clip retiming. (notes:
    # motion_engage_mismatch)
    "motion_engage_mismatch": RewardTermCfg(
      func=mdp.motion_engage_mismatch,
      weight=-1.0,
      params={"command_name": MOTION_COMMAND, "grace_fraction": 0.3},
    ),
    # Pays pad press inside the scheduled regrip window only and stops at
    # latch; the +0.4 ceiling stays under the -1.0 mismatch bill so pressing
    # without latching is strictly net-negative. (notes:
    # motion_scheduled_press)
    "motion_scheduled_press": RewardTermCfg(
      func=mdp.motion_scheduled_press_shaping,
      weight=0.4,
      params={"command_name": MOTION_COMMAND, "regrip_lead_time_s": 0.3},
    ),
    # The released foot's regrip support is the TOE pressing the wall (a
    # sliding unloaded pad destroys spine tips). Pays saturated toe force in
    # scheduled brace windows; outside-window pressing bills 0.3x — that
    # reaction loads the fresh anchor toward its cheap release directions.
    # (notes: motion_toe_brace)
    "motion_toe_brace": RewardTermCfg(
      func=mdp.motion_toe_brace_match,
      weight=0.4,
      params={
        "command_name": MOTION_COMMAND,
        "sensor_name": TOE_CONTACT_SENSOR,
        "press_norm_n": 8.0,
      },
    ),
    # -- task drive AS an imitation term: climb income follows the ghost —
    # exp kernel on (robot climb - reference climb) since settling, decaying
    # when falling behind OR running ahead ("any rise pays" funded every
    # reference-ignoring gait). (notes: motion_ref_height)
    "motion_ref_height": RewardTermCfg(
      func=mdp.motion_ref_height_tracking(),
      weight=2.5,
      params={
        "asset_cfg": robot_cfg_e,
        # 0.83 step heights — the designed kernel ratio kept for the 30 mm
        # clip (0.05 would halve the per-step climb gradient). (notes:
        # motion_ref_height std)
        "motion_command_name": MOTION_COMMAND,
        "std": 0.025,
        "settle_time_s": 0.5,
      },
    ),
    # Pays each scheduled latch once by where it LANDS on the anchor ladder
    # (the shape terms are root-relative and the gate checks timing only, so
    # cycling in place otherwise collects nearly full pay). Ceiling ~0.75/s
    # vs ~4.5/s tracking; in-window relatch farming pays zero. (notes:
    # motion_anchor_landing)
    "motion_anchor_landing": RewardTermCfg(
      func=mdp.motion_anchor_landing(),
      weight=40.0,
      params={"command_name": MOTION_COMMAND, "std": 0.03},
    ),
    # Dense progress toward ``ref_anchor_pos_w`` while the EE is scheduled
    # OFF and released — the landing bonus alone is a chicken-and-egg that
    # exploration never samples. Ceiling 0.5/s per swinging EE, under the
    # mismatch bill. (notes: motion_swing_approach)
    "motion_swing_approach": RewardTermCfg(
      func=mdp.motion_swing_approach,
      weight=0.5,
      params={"command_name": MOTION_COMMAND, "std": 0.05},
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
        "event_term_name": MICROSPINE_GRIP_EVENT,
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "face_local_axis": microspine_cfg.face_local_axis,
      },
    ),
    # Capture-relative rotation priced from 3 deg so the 3.5-4.5 deg
    # tilt-peel cliff is felt as a gradient before it fires; the reference
    # needs <= 1.1 deg. (notes: engaged_tilt)
    "engaged_tilt": RewardTermCfg(
      func=mdp.engaged_tilt_penalty,
      weight=-1.0,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "threshold_deg": 3.0,
        "scale_deg": 5.0,
      },
    ),
    "microspine_support_loss": RewardTermCfg(
      func=mdp.microspine_support_loss,
      # A one-anchor dangle still earning partial tracking (~2.4-3.5/s of
      # the ~4.5/s ceiling) must stay strictly net-negative. (notes:
      # microspine_support_loss)
      weight=-4.0,
      params={
        "event_term_name": MICROSPINE_GRIP_EVENT,
        "minimum_supports": 2,
      },
    ),
    # Base failure-mode costs; empirical release-load shaping below also
    # prices loaded reverse-shear unhooks. (notes:
    # unsafe_microspine_release)
    "unsafe_microspine_release": RewardTermCfg(
      func=mdp.unsafe_microspine_release_penalty,
      weight=-2.5,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "minimum_supports": 2,
        "settle_time_s": 0.5,
        # Reverse-shear stays free only AT scheduled releases; 0.3x while
        # the clip says ENGAGED (motionref16b farmed the free direction as
        # a release valve).
        "motion_command_name": MOTION_COMMAND,
        "scheduled_reverse_cost": 0.3,
        "scheduled_reverse_grace_s": 0.25,
        # Tilt-peel prices between tension (0.25) and ultimate (1.0): always
        # avoidable — the reference needs <= 1.1 deg.
        "tilt_release_cost": 0.5,
      },
    ),
    "ee_slide_penalty": RewardTermCfg(
      func=mdp.ee_slide_penalty,
      weight=-0.1,
      params={
        "asset_cfg": robot_cfg_e,
        "hand_body_names": microspine_cfg.ee_body_names,
        "event_term_name": MICROSPINE_GRIP_EVENT,
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
    # Spike tax above 70% of the deploy torque caps: leaves support duty and
    # the gentle unhook untaxed; one saturated joint costs 8/s — comparable
    # to the timing income a near-saturation yank shortcuts. (notes:
    # torque_headroom)
    "torque_headroom": RewardTermCfg(
      func=mdp.torque_headroom_penalty,
      weight=-8.0,
      params={"asset_cfg": robot_cfg_e, "threshold": 0.7},
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
    # At -0.01 mean-chatter was economically free (3-4 Hz hand buzz straight
    # through the 60 ms engage debounce); walk prices -0.25/-0.3, climb
    # keeps headroom for legitimately fast regrip reaches. (notes:
    # action_rate_l2)
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
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

  if contact_model == "empirical_unit":
    # Use the actual active-unit loads captured by the hold judge. Near-zero
    # shear/lateral gets one scheduled event bonus; shear-loaded releases
    # cost more than equally loaded lateral releases, in either sign.
    cfg.rewards["microspine_release_load"] = RewardTermCfg(
      func=mdp.microspine_release_load_reward(),
      weight=1.0,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "motion_command_name": MOTION_COMMAND,
        "shear_scale_n": 2.0,
        "lateral_scale_n": 2.0,
        "shear_cost": 0.25,
        "lateral_cost": 0.1,
        "max_load_cost": 4.0,
        "minimum_supports": 3,
        "release_lead_time_s": 0.25,
        "settle_time_s": 0.5,
      },
    )
    # BeyondMimic body tracking + the agreed climbing/contact adaptation.
    # Legacy Wang transition tasks retain their existing reward contract.
    pose_params = {
      "command_name": MOTION_COMMAND, "engage_gate_floor": 0.25,
      "engage_gate_grace_fraction": 0.3,
    }
    body_rewards = {
      "motion_body_pos": RewardTermCfg(
        func=mdp.motion_body_position_error_exp, weight=1.0,
        params={**pose_params, "std": 0.1}),
      "motion_body_ori": RewardTermCfg(
        func=mdp.motion_body_orientation_error_exp, weight=1.0,
        params={**pose_params, "std": 0.3}),
      "motion_body_lin_vel": RewardTermCfg(
        func=mdp.motion_body_linear_velocity_error_exp, weight=1.0,
        params={"command_name": MOTION_COMMAND, "std": 1.0}),
      "motion_body_ang_vel": RewardTermCfg(
        func=mdp.motion_body_angular_velocity_error_exp, weight=1.0,
        params={"command_name": MOTION_COMMAND, "std": 3.14}),
    }
    retained = (
      "motion_root_ori", "motion_ref_height", "motion_anchor_landing", "motion_swing_approach",
      "motion_engage_mismatch", "microspine_support_loss", "unsafe_microspine_release",
      "microspine_release_load", "action_rate_l2", "dof_pos_limits", "termination_penalty",
    )
    cfg.rewards = {**body_rewards, **{name: cfg.rewards[name] for name in retained}}
    cfg.rewards["undesired_contacts"] = RewardTermCfg(
      func=mdp.undesired_body_contacts, weight=-0.1,
      params={"sensor_name": "climb_undesired_contact", "threshold": 1.0})
    from mjlab.managers.termination_manager import TerminationTermCfg
    cfg.terminations["bad_tracking"] = TerminationTermCfg(
      func=mdp.motion_body_tracking_divergence,
      params={"command_name": MOTION_COMMAND, "position_threshold": 0.3,
              "orientation_threshold": 1.0, "height_threshold": 0.25})
    cfg.terminations["lost_support"] = TerminationTermCfg(
      func=mdp.prolonged_support_loss(),
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "timeout_s": 0.4})

  ##
  # Actions.
  ##

  cfg.actions["joint_pos"] = mdp.SoftJointPositionActionCfg(
    entity_name="robot",
    actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
    scale=PHYBOT_C2_ACTION_SCALE,
    use_default_offset=True,
  )

  ##
  # Observations — climb additions to the actor group, then the derived
  # critic/student groups (order is the checkpoint/export layout contract).
  ##

  actor_terms = cfg.observations["actor"].terms
  actor_terms["joint_pos_rel"].params = {"asset_cfg": _policy_joint_cfg()}
  actor_terms["joint_vel_rel"].params = {"asset_cfg": _policy_joint_cfg()}
  actor_terms["microspine_engaged_state"] = ObservationTermCfg(
    func=mdp.microspine_engaged_state,
    params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
  )
  actor_terms["microspine_contact_fraction"] = ObservationTermCfg(
    func=mdp.microspine_contact_fraction,
    params={
      "command_name": MICROSPINE_AFFORDANCE_COMMAND,
      "spheres_per_ee": (
        microspine_cfg.spheres_per_ee_rows * microspine_cfg.spheres_per_ee_cols
      ),
    },
  )
  actor_terms["microspine_disengage_time"] = ObservationTermCfg(
    func=mdp.ee_disengage_time_obs,
    params={"event_term_name": MICROSPINE_GRIP_EVENT, "clip": 1.5},
  )
  actor_terms["microspine_cooldown"] = ObservationTermCfg(
    func=mdp.microspine_cooldown_obs,
    params={
      "command_name": MICROSPINE_AFFORDANCE_COMMAND,
      "cooldown_steps": microspine_cfg.reengage_cooldown_steps,
    },
  )
  # Cycle phase clock (cos/sin) + the clip's engage schedule chart: per-EE
  # engage bits, per-EE countdowns to next release/regrip, per-foot
  # toe-brace flags — 16 dims total. (notes: motion_command obs)
  actor_terms["motion_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": MOTION_COMMAND},
  )

  # Critic = actor terms + privileged microspine state; must be built AFTER
  # every actor-term edit above so the replace-copies pick them up (the
  # builder deliberately shares params dicts — see mdp.privileged_critic_group).
  cfg.observations["critic"] = mdp.privileged_critic_group(
    cfg.observations["actor"], command_name=MICROSPINE_AFFORDANCE_COMMAND
  )

  if student_obs:
    # Deployable student contract: hardware-measurable terms only, 89 dims x
    # 50-frame history = 4450, APPEND-ONLY layout (C++ packer offsets) — see
    # mdp.student_observation_group. (notes: student group design)
    cfg.observations["student"] = mdp.student_observation_group(
      policy_joint_cfg=_policy_joint_cfg(),
      motion_command_name=MOTION_COMMAND,
      affordance_command_name=MICROSPINE_AFFORDANCE_COMMAND,
      gyro_sensor_name="robot/baselink-gyro",
      quat_sensor_name="robot/baselink-quat",
    )

  if play:
    ##
    # Play overrides — deterministic single-env evaluation on the final task.
    ##

    # One-world viewer: keeping the training batch here simulates thousands
    # of invisible robots (remember ``--num-envs 1``). (notes: play)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    if "student" in cfg.observations:
      cfg.observations["student"].enable_corruption = False
    # Deterministic physics probes: drop the DR events entirely (neutral
    # ranges would still overwrite XML frictions under operation="abs").
    for dr_event in ("encoder_bias", "body_friction", "base_mass", "random_release"):
      cfg.events.pop(dr_event, None)
    # Debug transition logging forces CPU-GPU sync in the physics step; keep
    # play fast. Dual-owner write: the grip event holds its own shallow copy
    # of the microspine cfg (see the Events section).
    microspine_cfg.debug = False
    cfg.events[MICROSPINE_GRIP_EVENT].params["microspine_cfg"].debug = False
    # Evaluation uses the final task directly (calibrated tilt band; play.py's
    # gravity resolver — CLI override or checkpoint-iteration lookup — can
    # still replace the pinned stage). Jitter off: evaluation and diagnostic
    # rollouts (engage raster) must be deterministic at the pinned angle.
    mdp.pin_curricula_to_final_stage(cfg)
    cfg.curriculum["wall_angle"].params["angle_jitter_deg"] = 0.0
    # Evaluation starts at the cycle's canonical stance instead of a random
    # RSI phase, so rollouts read as one continuous climb from frame 0.
    cfg.commands[MOTION_COMMAND].sampling_mode = "start"
    cfg.commands[MOTION_COMMAND].joint_pos_noise = 0.0
    cfg.commands[MOTION_COMMAND].height_offset_range = (0.0, 0.0)

  if wall_angle_deg is not None:
    cfg.curriculum["wall_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": float(wall_angle_deg)}]
    cfg.curriculum["wall_angle"].params["angle_jitter_deg"] = 0.0

  return cfg


def phybot_c2_climb_wall_corners60_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Full four-corner climbing task with a 45 -> 55 -> 60 degree curriculum.

  Shares the main task's control, rewards, RSI and domain randomization.
  Evaluation uses the terminal 60-degree angle and canonical reference start.
  """
  from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS60_MOTION

  cfg = phybot_c2_climb_wall_env_cfg(play=play, unit_layout="corners")
  cfg.commands[MOTION_COMMAND].motion_file = str(CAD_CLIMB_CORNERS60_MOTION)
  clip_vz = mdp.clip_average_climb_rate(CAD_CLIMB_CORNERS60_MOTION)
  cfg.commands[VELOCITY_COMMAND].ranges.lin_vel_z = (clip_vz, clip_vz)
  cfg.curriculum["wall_angle"].params["angle_stages"] = [
    dict(stage) for stage in mdp.CORNERS60_WALL_ANGLE_STAGES
  ]
  if play:
    mdp.pin_curricula_to_final_stage(cfg)
  return cfg


def phybot_c2_climb_wall_corners80_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Four active CAD corner units on a fixed 80-degree wall."""
  from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS80_MOTION
  cfg = phybot_c2_climb_wall_env_cfg(play=play,unit_layout="corners",wall_angle_deg=80.)
  cfg.commands[MOTION_COMMAND].motion_file = str(CAD_CLIMB_CORNERS80_MOTION)
  return cfg


def phybot_c2_climb_wall_distill_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Distillation variant: the climb task plus the student observation group.

  The teacher (loaded from a climb PPO checkpoint) reads the actor group, so
  its corruption is switched off — labels come from clean privileged state;
  the student group keeps noise + history. The wall-angle curriculum is
  pinned to the ladder's final angle — a HARD angle: the pinned stage
  carries no jitter and the terminal 90 is sampler-exempt anyway
  (``jitter_below_deg``). For band-jittered distill rollouts (a band
  teacher's neighborhood), re-pin with ``mdp.pin_curricula_to_band``.
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    play=play, student_obs=True, robustness_release=False
  )
  # No release drills: DAgger/estimator rollouts want the undisturbed
  # schedule — drills add student label variance and, in RL settings,
  # subsidize welding (the polish-v1 post-mortem). (notes: distill no-drill)
  cfg.observations["actor"].enable_corruption = False
  if not play:
    mdp.pin_curricula_to_final_stage(cfg)
  return cfg


def phybot_c2_climb_wall_polish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Phase-3 RL polish: PPO with the STUDENT group as the actor input.

  BC-init + RL polish breaks the distillation ceiling: the distilled student
  only imitates states its all-knowing teacher visits, so active information
  gathering (probe-loading a pad before committing) can only emerge here,
  where the acting policy pays for its own engage-state uncertainty. Same
  pinned curricula as the distill variant; unlike it the actor group's
  corruption is NOT disabled — there is no teacher reading it, and the
  acting (student) group carries its own noise regardless. The runner cfg
  points obs_groups at {"actor": ("student",), "critic": ("critic",)}.
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    play=play, student_obs=True, robustness_release=False
  )
  if not play:
    mdp.pin_curricula_to_final_stage(cfg)
  return cfg
