"""PHYBot C2 wall-climbing environment configuration.

DeepMimic-style layout (2026-07-21 redesign, replacing the commanded-gait
state machine):

  * A motion command (``climb_motion``) indexes the IK regrip cycle
    ``assets/motions/climb_cycle_moment_safe.npy``; episode resets are
    reference-state initialization (random cycle phase, latches seeded from
    the clip).
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
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

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
  add_toe_contact_pairs,
  add_toe_geoms,
  get_sphere_geom_names,
  pad_grid_spans,
  toe_geom_name,
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
# Recruitment-paced toe-brace crawl (4.52 s / 30 mm cycle, 6.6 mm/s): the
# fast clip's poses with every dwell restored to >= 0.3 s and foot swings to
# 0.75 s — the fast timing put seat/regrip holds (0.2 s) UNDER the 0.25 s
# fresh-anchor recruitment ramp, so each transfer loaded an immature anchor
# (motionref16's abandoned right foot). Gates 2026-07-30: static audit
# 0.8773 worst-util PASS, schedule-timing audit PASS (worst maturity gap
# 0.28 s >= ramp), reference-hold PASS (1/68 transients, margin +0.058).
_CLIMB_MOTION_FILE = _MOTIONS_DIR / "climb_cycle_moment_safe_paced.npy"


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
  robustness_release: bool = True,
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
  """
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
  # Moment billing measures the pad from the robot XML itself: the billing
  # spans are the contact-sphere grid extents (74.7 mm rake x 96 mm lateral
  # on this model). Regenerate nothing when the XML pad changes; this
  # derives at cfg-build time.
  span_rake, span_lateral = pad_grid_spans(
    get_spec(), microspine_cfg, robot_prefix=""
  )
  microspine_cfg.limit_surface.pad_span_rake_m = span_rake
  microspine_cfg.limit_surface.pad_span_lateral_m = span_lateral
  # Per-unit billing (2026-08-04): distribute the weld's 6D wrench onto the
  # paper's 3x4 spine-unit grid (bolt-group statics) and judge every unit's
  # 3D load against its F0/12 share of the closed surface — weakest unit
  # releases the pad. Supersedes the worst_edge fold whose arm = span/2 is
  # exact only for the two-row point abstraction (the real spread array
  # concentrates bending on its edge units ~1.5-1.8x harder per axis:
  # span*(N+1)/(6*(N-1)) = s/3 rake, 0.278*s lateral). The arm-sweep bench
  # (logs/eval/arm_sweep) showed clean climbing is insensitive to billing
  # harshness — only worst-F0 forced-release recoveries feel it.
  # moment_arm_fraction stays at its 0.25 default, used only when a bench
  # (engage_raster arm=) explicitly flips back to worst_edge mode.
  microspine_cfg.limit_surface.moment_billing_mode = "per_unit"
  # Weld rotational-residual scale, calibrated with the reference-hold
  # stress test (validate_microspine_constraint.py reference-hold): the old
  # 1 m default routed whole-body pitch couples through ~6 N*m of local pad
  # moment ("four enormous flanges"); 0.04 still trapped 0.7-1.8 N*m of
  # PD-parasitic moment in the wristless hands and force-released 65/79
  # reference frames. At 0.01 the rotational rows are compliant enough that
  # parasitic couples shed into force redistribution (peak |M| <= 0.07 N*m,
  # all frames hold, margin +0.077 at the F0 = 400 N DR floor) while the
  # honest-arm billing above still guards genuine sustained prying.
  microspine_cfg.anchor_torquescale = 0.01
  # Closed Wang-2017 hold envelope (paper-faithful): tension capacity exists
  # only alongside engaged shear, the forward lobe closes at the origin, and
  # against-rake capacity collapses to the side-contact residual. F0 range
  # raised versus the wedge's f_t_max because tension now taxes the whole
  # surface: the 90 deg frame-0 hang loads hands at ~70 N shear + 14 N
  # tension (~12 deg elevation, capacity ~0.6 * F0) — validated by
  # scripts/validate_microspine_constraint.py hang. The paper's 12-tile palm
  # demonstrated ~700 N pure shear, so 400-650 N per pad is conservative.
  microspine_cfg.limit_surface.model = "wang2017"
  microspine_cfg.limit_surface.f_t_max_range = (400.0, 650.0)
  # Tilt-peel calibration (2026-07-31, "4-deg" pass): the spine suspensions
  # conform force-free over le_max = 5 mm (Wang 2017 Sec 2.3); rotating the
  # chassis about the loaded edge row exhausts that travel at 5/74.7 mm =
  # ~3.8 deg, after which the tension-side row is loaded structurally and
  # pulls out — forced release, reason 5. Per-grip lottery like F0. The
  # reference's engaged rotations measure <= 1.1 deg (3x margin at the DR
  # floor); the pre-calibration policy rode the unlimited softness to
  # 12-22 deg — exactly the sim2real region this closes.
  microspine_cfg.limit_surface.tilt_peel_max_deg_range = (3.5, 4.5)
  # Latched pads hand their whole load path to the bilateral anchor; the
  # sphere contacts (a redundant second path that fights the anchor and made
  # the measured loads oscillate — motionref6's 90-deg margin 'barcode')
  # switch off per env while latched and return on release (2026-07-24).
  microspine_cfg.disable_contacts_while_latched = True

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
  # Foot toes brace the wall during foot regrips so the released spine pad
  # never slides on the surface. The geoms live on the ROBOT entity spec
  # (sensors resolve geom names against the entity, not the composed scene);
  # their wall contact pairs are scene-level, added in _spec_fn below.
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

  # pd100 table (2026-08-02 user call: revive deploypd-v3 from model_3300).
  # These are the gains v3 was trained with — byte-matched against
  # logs/.../2026-08-01_20-45-59_deploypd-v3-pd100/params/env.yaml — and a
  # resumed checkpoint must get the plant it learned on: full upper chain
  # INCLUDING shoulder_yaw at 100/10 (kd=10 lifts the hanging-chain zeta
  # 0.35 -> 0.5), legs 100/10 over the 100/5 module default, ankles 50/5,
  # neck default. (The 18c-exact 100/5 table this replaces served the
  # robust-ladder campaign; it lives in the jv run dirs' param dumps.)
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
    # Reserve 1% of the hard range at each end. The action term below clamps
    # PD targets to these limits; the physical hard stop remains a final
    # guard. Widened from 0.9 (2026-07-21): the reference cycle works close
    # to some stops — shoulder_roll kinematically peaks 0.06 rad from its
    # hard limit at state 4 (one hand high, other anchored low) — and
    # tracking must never fight the target clamp. The clip keeps its own
    # clearance elsewhere via the gen script's limit-avoidance springs.
    soft_joint_pos_limit_factor=0.98,
    actuators=climb_actuators,
  )

  # Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5). NOTE the stance-v2
  # base x (1.304) no longer matches what scripts/tune_pose.py visualized for
  # the v1 stance; the stance is now designed by the IK stance search
  # (2026-07-21) with pads pinned to the wall plane.

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }
  # Ordinary per-sphere contacts handle compression and pre-engage sliding;
  # per-EE equalities provide grip.
  toe_contact_sensor_name = "climb_toe_contact"

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
    # Foot-toe brace force: feeds the scheduled toe-brace shaping reward.
    ContactSensorCfg(
      name=toe_contact_sensor_name,
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

  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_pad_wall_contact_excludes(spec, microspine_cfg)
    add_toe_contact_pairs(spec, microspine_cfg, toe_contacts)
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

  if domain_rand:
    # v4 revival (2026-08-02 user call): resume deploypd-v3/model_3300 with
    # v3's FULL walk-parity DR, byte-matched to that run's params/env.yaml.
    # From-scratch-with-noise beats resume-then-inject: the graded R1/R2
    # ladder (see the jv run dirs, 2026-08-02 midday) measured a permanent
    # ~0.015 clean-exam tax injecting joint_vel noise into the noise-naive
    # 18c line, while v3 carried this whole set from birth and examined
    # 0.98 at 75-85 deg. The one thing that killed v3 was the gravity
    # bridge, not the DR — see the angle_stages note below.
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
        # Walk's regex: every collision geom except the feet. The spine
        # plates match too but their contype is 0 (walk CollisionCfg), so
        # randomizing them is inert. Spine-sphere and toe-wall contacts are
        # pair-driven — geom friction does not reach them (pair friction
        # overrides), so the microspine calibration stays untouched.
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
        # +-2 kg, not walk's +-5: the static climb audit's thinnest hold
        # margin is 12.3%, and a +5 kg draw pushes the worst F0 lottery out
        # of the Wang envelope — no-fault forced releases would pollute the
        # training signal. Revisit after a hang-bench pass at +2 kg.
        "ranges": (-2.0, 2.0),
        "operation": "add",
      },
    )
    cfg.events["encoder_bias"] = EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        # v3-exact: +-0.01 rad on every joint, drawn once per episode-world.
        # At 0.4 m reach this is a 4-8 mm unobservable EE offset against the
        # ~5 mm seating travel — the single suspect for real-pad reseat
        # misses, which is why it ships in the base DR set.
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    )
  if robustness_release:
    # Unscheduled-release recovery drill, deliberately NOT part of the base
    # DR package (2026-08-01 night, user call). Rationale: an unplanned
    # detach every ~5.5 s teaches "losing a latch mid-cycle is survivable",
    # which is exactly the experience that makes weld-to-survive cheap once
    # the wall gets steep — the engage-mismatch bill drifted monotonically
    # through the 75-deg rung while it was on. The recovery skill it trains
    # is still wanted for the distilled student (which cannot observe latch
    # state), so it comes back as a FINAL stage once exact-90 is mastered,
    # on top of a policy whose schedule discipline is already fixed.
    # R4 stage (2026-08-03): default ON now that exact-90 is mastered — the
    # drill pairs with the reseat-hold clock freeze (affordance.reseat_hold)
    # so recovery is a trained skill, not schedule truancy. Interval starts
    # gentle (6-12 s); tighten to (3, 8) once the recovery exam passes. For
    # a FROM-SCRATCH retrain launch with robustness_release=False until
    # exact-90 is mastered (2026-08-01 rationale: unplanned-release-is-normal
    # economics makes welding cheap before schedule discipline forms).
    cfg.events["random_release"] = EventTermCfg(
      mode="interval",
      interval_range_s=(6.0, 12.0),
      func=mdp.random_microspine_release,
      params={"command_name": microspine_affordance_name},
    )

  # ------------------------------------------------------------------------
  # Rewards — DeepMimic split. ``motion_*`` exp kernels teach the regrip
  # cycle (no global position tracking); height/velocity progress terms
  # drive actual climbing; the microspine battery enforces the physics.
  # ------------------------------------------------------------------------
  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- motion imitation (positive tracking income, BeyondMimic exp kernel).
    # engage_gate_floor 0.25 (motionref10): pose income multiplied by the
    # schedule-adherence gate — "looking right" pays in full only while
    # "gripping right". The engagement-blind pose kernels were the welded
    # statue's last income pillar (ref8/ref9 both parked a diagonal at
    # match~0.5 and lived off pose tracking; the cycle's poses never leave
    # stance far enough for the statue to lose the kernel). Floor keeps a
    # quarter of the income ungated so birth pose learning never starves;
    # the gate uses the same edge-distance grace ramp as the mismatch bill.
    # engage_gate_grace_fraction 0.3 (2026-07-30, motionref17): the gate's
    # edge grace is a fraction of the violated window's own length, not a
    # fixed 0.25 s — the fixed grace silently discounted welding whenever
    # the clip was retimed (fast clip: only the middle 0.1 s of a 0.6 s foot
    # window billed full price). At 0.3 a weld-through always forfeits ~70%
    # of the window's gate credit, clip timing notwithstanding.
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_pos_error_exp,
      weight=2.0,
      params={
        "command_name": motion_command_name,
        "std": 0.2,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
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
      params={
        "command_name": motion_command_name,
        "std": 0.1,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    "motion_root_ori": RewardTermCfg(
      func=mdp.motion_root_ori_error_exp,
      weight=0.5,
      params={
        "command_name": motion_command_name,
        "std": 0.3,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    # The clip's engage schedule replaces the old commanded-gait machinery:
    # a latch state disagreeing with the reference phase costs per step.
    # -1.0 (was -0.25, raised 2026-07-21): the motionref1 audit showed the
    # policy ignoring the schedule wholesale (match 0.44, LF parked at 0.03
    # occupancy) because a wrong latch state cost pocket change next to the
    # height income. ~2 mismatched pads must read as -2/s against a budgeted
    # ~3.1/s task income, not -0.5/s against 22/s.
    # grace_fraction 0.3 (2026-07-30, was grace_time_s 0.25): the bill ramps
    # linearly over the outer 30% of the violated window to a scheduled flip
    # (debounce 0.06 s + cooldown 0.20 s make instant compliance impossible)
    # — the same countdown signal the policy sees in motion_command. As a
    # fraction it survives clip retiming: a weld-through always pays ~0.7 of
    # the window at full price (the fixed 0.25 s grace had cut the fast
    # clip's foot-window weld bill nearly in half).
    "motion_engage_mismatch": RewardTermCfg(
      func=mdp.motion_engage_mismatch,
      weight=-1.0,
      params={"command_name": motion_command_name, "grace_fraction": 0.3},
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
    # Foot-toe brace shaping (2026-07-29, toe-brace reference): the released
    # foot's support during its regrip is the TOE pressing the wall — the
    # spine pad now travels clear of the surface (sliding an unloaded pad
    # destroys spine tips). Pays saturated toe normal force inside the clip's
    # scheduled brace windows only; pressing outside the window (e.g. while
    # this foot's own pad is latched) bills 0.3x of the same scale — that
    # reaction loads the fresh anchor toward its cheap release directions.
    "motion_toe_brace": RewardTermCfg(
      func=mdp.motion_toe_brace_match,
      weight=0.4,
      params={
        "command_name": motion_command_name,
        "sensor_name": toe_contact_sensor_name,
        "press_norm_n": 8.0,
      },
    ),
    # -- task drive AS an imitation term (2026-07-24, motionref8): the climb
    # income follows the ghost. The free-standing height rewards ("any rise
    # pays") funded every reference-ignoring gait — scramble, arms-only,
    # legs-only — and each takeoff bought climb income with falling schedule
    # adherence; three rounds of caps (per-step, flat budget, ref-profile
    # ceiling) were patches on that split incentive. Now the payout is an exp
    # kernel on (robot climb - reference climb) since settling: full on the
    # profile (where RSI places every episode), decaying symmetrically when
    # falling behind OR running ahead. std 0.05 ~ one step height: a one-step
    # lag still earns, a runaway scramble earns nothing.
    "motion_ref_height": RewardTermCfg(
      func=mdp.motion_ref_height_tracking(),
      weight=2.5,
      params={
        "asset_cfg": robot_cfg_e,
        "motion_command_name": motion_command_name,
        # 0.83 step heights, the ratio the term was designed at (std 0.05
        # against the old 60 mm step). Left at 0.05 for the 30 mm clip, one
        # step moved the kernel only 0.6 sigma and the per-step climb
        # gradient halved — measured iter-1400 basin: schedule match 0.98
        # with net climb at ~5% of the ghost (ref12 hit 0.18 m/episode by
        # iter 300 on the original ratio).
        "std": 0.025,
        "settle_time_s": 0.5,
      },
    ),
    # Landing accuracy on the anchor ladder (2026-07-30): the shape terms
    # are root-relative and the engage gate checks timing only, so cycling
    # in place collected nearly full imitation pay. This pays each scheduled
    # latch once by where it LANDS (the reference anchors climb one step per
    # cycle). Ceiling ~0.75/s at perfect landings (4 per 4.24 s cycle x
    # 40 x dt) — meaningful against the ~4.5/s tracking ceiling without
    # dominating; in-window relatch farming pays zero (once per window).
    "motion_anchor_landing": RewardTermCfg(
      func=mdp.motion_anchor_landing(),
      weight=40.0,
      params={"command_name": motion_command_name, "std": 0.03},
    ),
    # Dense approach shaping toward the landing rung (2026-07-30,
    # motionref17): the landing bonus alone is a chicken-and-egg — reaching
    # a rung one step UP needs the whole swing aimed there, and exploration
    # never samples it (motionref16 landing income: 3-5% of ceiling). Pays
    # the swing's progress toward ``ref_anchor_pos_w`` while the EE is
    # scheduled OFF and released; ceiling 0.5/s per swinging EE, well under
    # the mismatch bill so hovering released outside swing windows never
    # profits.
    "motion_swing_approach": RewardTermCfg(
      func=mdp.motion_swing_approach,
      weight=0.5,
      params={"command_name": motion_command_name, "std": 0.05},
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
    # Engaged-tilt tax (recalibrated 2026-07-31, "4-deg" pass): capture-
    # relative rotation priced from 3 deg so the tilt-peel cliff (DR
    # 3.5-4.5 deg, reason 5) is felt as a gradient before it fires. The
    # reference uses <= 1.1 deg — schedule-true execution pays nothing.
    "engaged_tilt": RewardTermCfg(
      func=mdp.engaged_tilt_penalty,
      weight=-1.0,
      params={
        "command_name": microspine_affordance_name,
        "threshold_deg": 3.0,
        "scale_deg": 5.0,
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
        # Reverse-shear unhooks stay free AT scheduled releases (the edge
        # ramp zeroes the bill there) but cost 0.3x while the clip says
        # ENGAGED: motionref16b farmed the free direction as a release
        # valve — 43-45 latch-jiggle-unhook cycles per 30 s on the
        # abandoned right foot, none of them priced (2026-07-30).
        "motion_command_name": motion_command_name,
        "scheduled_reverse_cost": 0.3,
        "scheduled_reverse_grace_s": 0.25,
        # Tilt-peel (reason 5) prices between tension (0.25) and ultimate
        # (1.0): rotating a live patch off its grip is a structural failure
        # the policy can always avoid (the reference needs <= 1.1 deg).
        "tilt_release_cost": 0.5,
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
    # Spike tax above 70% of the deploy torque caps (2026-07-30,
    # motionref17b): unpriced spike torque let the policy trade the
    # weight-shift unhook for a near-saturation yank (0.45 -> 1.00 of the
    # 58 Nm shoulder cap between model_2175 and model_4300, +7% climb).
    # Threshold form leaves support duty and the gentle unhook untaxed; -8
    # makes one saturated joint cost 8/s — a 1-2 step yank bills ~0.3-0.6,
    # comparable to the timing income it shortcuts.
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
    # -0.1 (was -0.01): at -0.01 chatter was economically free — model_3000's
    # hovering hands buzzed the policy MEAN at 3-4 Hz, +-4-5 cm, ~40 mrad
    # target flips per step (deterministic rollout, torque duty far from
    # limits), which also blows through the 60 ms engage debounce window.
    # Walk prices the same term at -0.25/-0.3; climbing keeps more headroom
    # for the legitimately fast regrip reaches.
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

  # no_progress executioner REMOVED (2026-07-30, motionref17). Under the
  # motionref16 economics it truncated every learning episode at ~8-13 s and
  # priced failed regrip experiments at -10, which dismantled the very
  # learning ladder motionref15 had climbed (dance -> choreography -> climb).
  # The climb pressure is now carried by motion_ref_height +
  # motion_anchor_landing + motion_swing_approach; if a dance-forever basin
  # returns, reintroduce a curriculum-gated stall bar instead of a flat one
  # (see mdp.no_progress_stall, kept for that purpose).
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
    # R4 reseat-hold telemetry: clock-freeze duty cycle and how many EEs are
    # pending. hold_fraction >> interval-implied duty means reseats are slow
    # or timing out; ~0 with random_release on means holds resolve instantly.
    "microspine_reseat_hold_fraction": MetricsTermCfg(
      func=mdp.microspine_reseat_hold_fraction,
      params={"command_name": microspine_affordance_name},
    ),
    "microspine_reseat_pending_count": MetricsTermCfg(
      func=mdp.microspine_reseat_pending_count,
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

  # ------------------------------------------------------------------------
  # Training schedule — gravity ladder x tilt-peel anneal, one timeline
  # (from-scratch rewrite 2026-08-02 evening, user call: the config must
  # fully retrain from iter 0; the resume-era pinned variants live in the
  # deploypd-v4-* run dirs and git history).
  #
  #    iter | gravity pool      | tilt-peel band (tax thresh/scale)
  #   ------+-------------------+----------------------------------
  #       0 | 45 (global +-5)   | 12-15 deg (10/8)  wide birth band
  #    1200 |                   | 8-10 deg  (6.5/6)
  #    1500 | 75 +-12 (63-87)   |
  #    2400 |                   | 5-7 deg   (4/5)
  #    3000 |                   | 3.5-4.5   (3/5)   calibrated, final
  #    3550 | 85 +-3  (82-88)   |
  #    4200 | 88 +-2  (86-90)   |
  #    4800 | 90 exact          |
  #
  # The iters transcribe the ONE validated trajectory: v3 ran 0->3300 on
  # exactly this 45/75/tilt prefix (from birth, full DR + pd100), and the
  # v4 bridge campaign took the same lineage 3300->5050 through
  # 85/88/exact-90 to the ship gate (90-deg cal-band exams
  # 0.962/0.961/0.963 across three consecutive panels, zero falls).
  # Design rules baked into the numbers:
  #   - gravity is a function of training time alone (2026-07-21:
  #     graduation-by-height once promoted a 20x-overspeed scramble), and
  #     every hop overlaps its neighbor's pool (63-87 / 82-88 / 86-90 /
  #     90) so zero-compression arrives in-distribution — v3 died on the
  #     unbridged 75+-12 -> 88+-2 cliff (weld-hands basin: at 90 a hand
  #     swing leaves 3 anchors on pure shear+tension, 12.3% margin at the
  #     F0=400 worst draw). The final stage runs exactly 90; jitter there
  #     would mix in >90 overhangs, a different task.
  #   - the tilt anneal reaches the honest physics (3000) before the wall
  #     gets steep (3550): every steep rung trains AND examines on the
  #     calibrated band. Birth needs the wide band — cold starts on
  #     3.5-4.5 peel apart within ~1 s (motionref18: zero learning).
  #   - anneal steps and rung boundaries stay >=300 iters apart (fresh2b:
  #     never stack two shocks inside a consolidation window).
  #   - the iters are CEILINGS: drive by mastery gates (precise-angle exam
  #     >=0.95 x2 zero falls; per-EE occ_pol-occ_ref < +0.04 is the weld
  #     early warning) via kill/edit/resume, and when pulling a rung IN,
  #     push the later rungs OUT (the 88 rung auto-fired at 4200 mid-soak
  #     once; it survived only by luck of near-readiness).
  #   - examine ALL rungs every cycle (periodic_eval.sh probes
  #     75/85/88/90): steep-only pools erode shallow angles — accepted
  #     under the ckpt-portfolio doctrine (bank each rung's graduate),
  #     but the erosion must stay measured, never silent.
  # steps_per_iter must match rl_cfg num_steps_per_env.
  gravity_stages = [
    {"iter": 0, "angle_deg": 45.0},
    {"iter": 1500, "angle_deg": 75.0, "jitter_deg": 12.0},
    {"iter": 3550, "angle_deg": 85.0, "jitter_deg": 3.0},
    {"iter": 4200, "angle_deg": 88.0, "jitter_deg": 2.0},
    {"iter": 4800, "angle_deg": 90.0},
  ]
  tilt_peel_stages = [
    {"iter": 0, "deg_range": (12.0, 15.0),
     "tax_threshold_deg": 10.0, "tax_scale_deg": 8.0},
    {"iter": 1200, "deg_range": (8.0, 10.0),
     "tax_threshold_deg": 6.5, "tax_scale_deg": 6.0},
    {"iter": 2400, "deg_range": (5.0, 7.0),
     "tax_threshold_deg": 4.0, "tax_scale_deg": 5.0},
    {"iter": 3000, "deg_range": (3.5, 4.5),
     "tax_threshold_deg": 3.0, "tax_scale_deg": 5.0},
  ]
  cfg.curriculum = {
    "gravity_angle": CurriculumTermCfg(
      func=mdp.gravity_angle,
      params={
        "angle_stages": gravity_stages,
        "steps_per_iter": 96,
        "angle_jitter_deg": 5.0,
      },
    ),
    "tilt_peel": CurriculumTermCfg(
      func=mdp.tilt_peel_schedule,
      params={
        "stages": tilt_peel_stages,
        "command_name": microspine_affordance_name,
        "steps_per_iter": 96,
      },
    ),
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
  # regrip (2026-07-22), plus the per-foot toe-brace flags (2026-07-29).
  # 16 dims total.
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

  if student_obs:
    # Deployable observation contract for the distilled student. Every term
    # is hardware-measurable: IMU gyro + gravity, joint encoders, the
    # policy's own action, and the motion command — a pure 50 Hz clock
    # lookup exported alongside the policy. Deliberately absent: base_lin_vel
    # and height_above_spawn (no estimator on the robot), the microspine
    # latch state (no engagement sensing; the reference schedule inside
    # motion_command plus proprioceptive history carries it), the constant
    # velocity_command, and prev/prev_prev_action (the history covers them).
    # 89 dims per frame x history 10 = 890; layout is term-major with frames
    # oldest->newest (the export sidecar documents the C++ contract).
    # pad_normal_force (2026-08-03) is the one non-proprioceptive channel:
    # the gimbal-mounted load cell the sensing co-design provides — it gives
    # the student a physical engage/loss estimator instead of pure schedule
    # faith, and it is the deploy-side trigger for the R4 clock pause.
    student_terms = {
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/baselink-gyro"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity_from_framequat_sensor,
        params={"sensor_name": "robot/baselink-quat"},
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "joint_pos_rel": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": _policy_joint_cfg()},
        noise=Unoise(n_min=-0.01, n_max=0.01),
      ),
      "joint_vel_rel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": _policy_joint_cfg()},
        noise=Unoise(n_min=-1.5, n_max=1.5),
      ),
      "last_action": ObservationTermCfg(func=mdp.last_action),
      "motion_command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": motion_command_name},
      ),
      # APPEND-ONLY from here: inserting mid-dict shifts every downstream
      # block of the exported C++ offset table. Pipeline reminder: noise is
      # applied per-frame BEFORE the history append; the 1-2 step delay
      # models load-cell/CAN latency and deliberately survives play/export
      # (the real sensor is late too — delay is not corruption-gated).
      "pad_normal_force": ObservationTermCfg(
        func=mdp.microspine_pad_normal_force,
        params={"command_name": microspine_affordance_name},
        noise=Unoise(n_min=-5.0, n_max=5.0),
        clip=(0.0, 200.0),
        scale=0.01,
        delay_min_lag=1,
        delay_max_lag=2,
        delay_per_env=True,
      ),
    }
    cfg.observations["student"] = ObservationGroupCfg(
      terms=student_terms,
      concatenate_terms=True,
      enable_corruption=True,
      # 50 frames = 1.0 s (2026-08-03, Route-B step 0 follow-up): the 10-frame
      # window (0.2 s) cannot span the 0.6 s reengage cooldown, so latch
      # timers were uninferable; 1 s also covers debounce + typical reseat
      # and gives the height/slow-factor estimators a usable integration
      # window. 89 dims/frame x 50 = 4450 — the packer contract.
      history_length=50,
    )

  if play:
    # The viewer shows one world. Keeping the training batch of 4096 here
    # simulates thousands of invisible robots and is the main cause of
    # sub-real-time playback unless callers remember ``--num-envs 1``.
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    if "student" in cfg.observations:
      cfg.observations["student"].enable_corruption = False
    # Evaluation and diagnostic rollouts are deterministic physics probes:
    # drop the DR events entirely (neutral ranges would still overwrite XML
    # frictions under operation="abs").
    for dr_event in ("encoder_bias", "body_friction", "base_mass", "random_release"):
      cfg.events.pop(dr_event, None)
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
    # Evaluation always runs the CALIBRATED tilt-peel band (the annealing
    # ladder is a training aid, not physics).
    final_tilt = cfg.curriculum["tilt_peel"].params["stages"][-1]
    cfg.curriculum["tilt_peel"].params["stages"] = [
      {"iter": 0, "deg_range": final_tilt["deg_range"]}
    ]
    # Evaluation starts at the cycle's canonical stance instead of a random
    # RSI phase, so rollouts read as one continuous climb from frame 0.
    cfg.commands[motion_command_name].sampling_mode = "start"
    cfg.commands[motion_command_name].joint_pos_noise = 0.0

  return cfg


def phybot_c2_climb_wall_distill_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Distillation variant: the climb task plus the student observation group.

  The teacher (loaded from a climb PPO checkpoint) reads the actor group, so
  its corruption is switched off — labels come from clean privileged state;
  the student group keeps noise + history. The gravity curriculum is pinned
  to the ladder's final angle while KEEPING the training jitter, so rollouts
  cover the neighborhood the teacher actually graduated on (re-point this
  after the teacher's exam if it passes a wider angle set).
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    play=play, student_obs=True, robustness_release=False
  )
  # No release drills here (2026-08-03): DAgger/estimator rollouts want the
  # undisturbed schedule — drills + the reseat freeze add label variance for
  # the student and, in RL settings, subsidize welding (the polish-v1
  # post-mortem). The recovery SKILL still transfers through imitation of
  # the press-on-schedule behavior (model_250 already passed 2/2 drills).
  cfg.observations["actor"].enable_corruption = False
  if not play:
    final_angle = cfg.curriculum["gravity_angle"].params["angle_stages"][-1]
    cfg.curriculum["gravity_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": final_angle["angle_deg"]}
    ]
    final_tilt = cfg.curriculum["tilt_peel"].params["stages"][-1]
    cfg.curriculum["tilt_peel"].params["stages"] = [
      {"iter": 0, "deg_range": final_tilt["deg_range"]}
    ]
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
    final_angle = cfg.curriculum["gravity_angle"].params["angle_stages"][-1]
    cfg.curriculum["gravity_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": final_angle["angle_deg"]}
    ]
    final_tilt = cfg.curriculum["tilt_peel"].params["stages"][-1]
    cfg.curriculum["tilt_peel"].params["stages"] = [
      {"iter": 0, "deg_range": final_tilt["deg_range"]}
    ]
  return cfg
