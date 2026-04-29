"""PHYBot C2 wall-climbing environment configuration.

Reward set is a port of IsaacLab's quadruped velocity-tracking template
(``locomotion/velocity/velocity_env_cfg.py``) with two task-specific swaps:

  * Velocity tracking uses a climb-specific world-frame command fixed at
    ``vz_cmd = 0.08`` m/s.
  * Contact-sensor signals → microspine ``engaged`` flag for all 4 spines.

C2-specific notes:
  * 4 grip end-effectors: left/right hand_spine + left/right foot_spine.
    Each spine has a 2x2 sphere grid (16 spheres total).
  * No ``waist_roll`` joint/body — Spot/Anymal-style torso-deviation terms
    are not migrated here (intentionally minimal).
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets import (
  PHYBOT_C2_ACTION_SCALE,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
  get_wall_cfg,
)
from spider_rl_mjlab.assets.microspine import (
  MicrospineCfg,
  MicrospineController,
  add_adhesion_actuators,
)
from spider_rl_mjlab.tasks.climb_wall import mdp
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg


_EE_BODY_NAMES: tuple[str, ...] = (
  "left_hand_spine",
  "right_hand_spine",
  "left_foot_spine",
  "right_foot_spine",
)
_SPHERES_PER_EE_ROWS = 2
_SPHERES_PER_EE_COLS = 2
_ADHESION_FORCE = 1200.0  # per-EE max adhesion force [N]
_REENGAGE_COOLDOWN_STEPS = 15  # 0.3 s at climb_wall control dt = 0.002 * 10
_ENGAGE_CONSECUTIVE_STEPS = 15  # 0.3 s at climb_wall control dt = 0.002 * 10
_VZ_CMD = 0.08            # initial climb-velocity target [m/s]
_BASE_QUAT_SENSOR = "robot/baselink-quat"
_BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
_BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"

# All 4 spines × 4 spheres = 16 sphere geom names. These are the contact
# patches that pair with the wall.
_EE_GEOMS = [
  f"{ee}_c_{i}_{j}"
  for ee in _EE_BODY_NAMES
  for i in range(_SPHERES_PER_EE_ROWS)
  for j in range(_SPHERES_PER_EE_COLS)
]


def _use_xml_base_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  terms = cfg.observations["actor"].terms
  terms["base_lin_vel"].func = mdp.builtin_sensor
  terms["base_lin_vel"].params = {"sensor_name": _BASE_LIN_VEL_SENSOR}
  terms["base_ang_vel"].func = mdp.builtin_sensor
  terms["base_ang_vel"].params = {"sensor_name": _BASE_ANG_VEL_SENSOR}
  terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  terms["projected_gravity"].params = {"sensor_name": _BASE_QUAT_SENSOR}


def phybot_c2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  _use_xml_base_sensors(cfg)
  # Match MJLab/IsaacLab velocity tracking style: no explicit gait phase
  # observation. Gait should emerge from velocity tracking + adhesion shaping.
  cfg.observations["actor"].terms.pop("gait_clock", None)

  robot_cfg = get_phybot_c2_robot_cfg()
  wall_cfg = get_wall_cfg()

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

  # Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5) — same as tune_pose,
  # so the spawned base position lines up with what was visualized there.

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  # Per-sphere wall-contact pairs (high friction) + adhesion actuators.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    for gname in _EE_GEOMS:
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.condim = 6
      pair.friction = [30.0, 30.0, 1.0, 1.0, 1.0]
      pair.solref = [0.02, 5.0]
      pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]
      pair.solreffriction = [0.02, 5.0]
      pair.margin = 0.01
      pair.gap = 0.01
    add_adhesion_actuators(
      spec,
      _EE_BODY_NAMES,
      _ADHESION_FORCE,
      rows=_SPHERES_PER_EE_ROWS,
      cols=_SPHERES_PER_EE_COLS,
    )
    # Disable the aligned terrain ground plane — gravity-tilt curriculum
    # replaces the floor; this forces the robot to actually grip the wall.
    configure_phybot_c2_floor_geom(spec, enable_contact=False)

  cfg.scene.spec_fn = _spec_fn

  # Per-spine microspine adhesion via MuJoCo body-level wrench.
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=_EE_BODY_NAMES,
        adhesion_force=_ADHESION_FORCE,
        spheres_per_ee_rows=_SPHERES_PER_EE_ROWS,
        spheres_per_ee_cols=_SPHERES_PER_EE_COLS,
        engage_force_threshold=5.0,
        engage_min_spheres=1,
        engage_parallel_cos_threshold=0.0,
        # Re-engage only when friction is opposite to the release trigger:
        # fric_up < -30 N, filtered to suppress contact noise.
        engage_friction_threshold=30.0,
        engage_friction_filter_alpha=0.8,
        engage_consecutive_steps=_ENGAGE_CONSECUTIVE_STEPS,
        # Disengage criterion: friction force on the EE projected onto each
        # spine's local -z axis. Static stance has friction supporting the
        # body (fric on the spine in world +z, ≈ spine local +z) → projects
        # NEGATIVELY onto local -z, no trigger. Lifting the spine flips the
        # wall friction (opposes lift, points along world -z ≈ spine local
        # -z) → projects POSITIVELY, triggers when ≥ 30 N. Body-local axis
        # so the criterion follows the spine body if it rotates.
        friction_axis_world=None,
        hand_up_local=(0.0, 0.0, -1.0),
        disengage_friction_threshold=30.0,
        disengage_friction_filter_alpha=0.8,
        disengage_consecutive_steps=5,
        disengage_pull_threshold=0.0,
        disengage_lift_velocity_threshold=0.0,
        reengage_cooldown_steps=_REENGAGE_COOLDOWN_STEPS,
        wall_geom_name="wall/wall_geom",
        wall_entity_name="wall",
      ),
    },
  )

  # ------------------------------------------------------------------------
  # Rewards — Anymal D quadruped template (locomotion/velocity), with climb
  # velocity command tracking and engage signal instead of contact. Track and
  # gait weights are down-scaled so dense velocity tracking does not swamp the
  # gait signal.
  # ------------------------------------------------------------------------
  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- task: commanded world-frame climb velocity target. Split x/y from z
    # so lateral drift correction is softer than upward climb-rate tracking.
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.climb_linear_velocity_xy_gauss,
      weight=4.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": "climb_velocity",
        "sigma": 0.05,
      },
    ),
    "track_lin_vel_z": RewardTermCfg(
      func=mdp.climb_linear_velocity_z_gauss,
      weight=8.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": "climb_velocity",
        "sigma": 0.05,
      },
    ),
    "climb_height_progress": RewardTermCfg(
      func=mdp.climb_height_progress(),
      weight=10.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": "climb_velocity",
        "command_threshold": 0.01,
        "max_normalized_progress": 1.5,
      },
    ),
    # -- penalties (Anymal weights verbatim except as noted).
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
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.0e-5,
      params={"asset_cfg": robot_cfg_e},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    # -- gait shaping from the 2026-04-28_14-51-31 run: dense min-mode
    # timer gated on exactly two engaged EEs (two-anchor / two-swing).
    "ee_grip_time": RewardTermCfg(
      func=mdp.hand_grip_time_min_climb(),
      weight=10.0,
      params={
        "tau": 0.5,
        "vz_cmd": _VZ_CMD,
        "cmd_eps": 0.0,
        "event_term_name": "microspine_adhesion",
        "target_engaged_count": 2,
      },
    ),
    "ee_slide_penalty": RewardTermCfg(
      func=mdp.ee_slide_penalty,
      weight=-0.1,
      params={
        "asset_cfg": robot_cfg_e,
        "hand_body_names": _EE_BODY_NAMES,
        "event_term_name": "microspine_adhesion",
      },
    ),
    # Spine contact face (local -y) staying flush against the wall — MIN
    # across 4 EEs (strictest: any 1 spine rotated off-axis drops reward).
    "ee_face_parallel_to_wall": RewardTermCfg(
      func=mdp.ee_face_parallel_to_wall,
      weight=2.0,
      params={
        "asset_cfg": robot_cfg_e,
        "ee_body_names": _EE_BODY_NAMES,
        "face_local_axis": (0.0, -1.0, 0.0),
      },
    ),
    # -- optional Anymal terms, default disabled (weight 0) — left in the
    # cfg so they can be flipped on without restructuring.
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    # -- termination penalty (common addition not in Anymal generic).
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  # Curriculum: keep ONLY gravity-tilt schedule. Drop per-reward weight
  # schedules from the big_v2 base (they reference reward names that no
  # longer exist, and Anymal-style training uses constant weights). Keep the
  # climb velocity command fixed for now.
  cfg.curriculum = {"gravity_angle": cfg.curriculum["gravity_angle"]}
  # Override the schedule for C2: hold 30° longer, then ramp to vertical.
  # 30° is 0.5g along-wall load; it is still an active climb, but gives the
  # controller more time to stabilize grip/gait before the wall gets steep.
  cfg.curriculum["gravity_angle"].params["angle_stages"] = [
    {"iter":    0, "angle_deg": 30.0},
    {"iter": 1500, "angle_deg": 30.0},   # extended hold — learn stable gait
    {"iter": 2500, "angle_deg": 60.0},   # steep, 0.87g
    {"iter": 3500, "angle_deg": 90.0},   # full vertical
  ]

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events["microspine_adhesion"].params["microspine_cfg"].debug = True

  return cfg
