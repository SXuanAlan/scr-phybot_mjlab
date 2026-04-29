"""PHYBot Big V2 wall-climbing environment configurations."""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.assets.microspine import MicrospineCfg, MicrospineController, add_adhesion_actuators
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg


# Hand sub-geom 2x4 grid (matching forearm_hand layout) + foot collision geoms.
_HAND_GEOMS = [
  f"{side}_hand_c_{i}_{j}"
  for side in ("left", "right")
  for i in range(2)
  for j in range(4)
]
_FOOT_GEOMS = [
  f"{side}_ankle_roll_collision_{i}"
  for side in ("left", "right")
  for i in (1, 2)
]
_EE_GEOMS = _HAND_GEOMS + _FOOT_GEOMS


def phybot_big_v2_climb_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot Big V2 wall-climbing configuration."""
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_big_v2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override initial state at task level. IMPORTANT: build a fresh
  # InitialStateCfg rather than mutating robot_cfg.init_state in place — the
  # robot factory returns a shared keyframe singleton that is also used by
  # other tasks (e.g. microspine_test). Mutating it would cause the last task
  # registered to overwrite init poses for earlier ones.
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(-0.67, 0.0, 2.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.435,
      "left_hip_roll": 0.64,
      "left_hip_yaw": 0.955,
      "left_knee": 0.4,
      "left_ankle_pitch": -0.2,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.435,
      "right_hip_roll": -0.64,
      "right_hip_yaw": -0.955,
      "right_knee": 0.4,
      "right_ankle_pitch": -0.2,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "waist_roll": 0.0,
      "left_shoulder_pitch": -2.79,
      "left_shoulder_roll": -0.17,
      "left_shoulder_yaw": -0.5,
      "left_elbow_pitch": -0.4,
      "left_elbow_yaw": 0.45,
      "right_shoulder_pitch": -2.79,
      "right_shoulder_roll": 0.17,
      "right_shoulder_yaw": 0.5,
      "right_elbow_pitch": -0.4,
      "right_elbow_yaw": -0.45,
    },
  )

  # Wall: center position in world frame (10m tall wall centered at z=5).
  wall_cfg.init_state.pos = (0, 0.0, 5.0)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  # EE body names for adhesion actuators.
  _EE_BODY_NAMES = ("left_hand", "right_hand")
  _ADHESION_FORCE = 1200.0
  _REENGAGE_COOLDOWN_STEPS = 15  # 0.3 s at climb_wall control dt = 0.002 * 10
  _ENGAGE_CONSECUTIVE_STEPS = 15  # 0.3 s at climb_wall control dt = 0.002 * 10

  # Contact pairs + adhesion actuators via spec_fn.
  # Hands: microspine-scale friction (30) to anchor against the wall.
  # Feet:  normal shoe-scale friction (1.5) so the policy cannot use them
  # as a second pair of "grippers" to kick up along the wall.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    for gname in _EE_GEOMS:
      is_hand = gname in _HAND_GEOMS
      tan_friction = 30.0 if is_hand else 1.5
      other_friction = 1.0 if is_hand else 0.5
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.condim = 6
      pair.friction = [tan_friction, tan_friction, other_friction, other_friction, other_friction]
      pair.solref = [0.02, 5.0]
      pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]
      pair.solreffriction = [0.02, 5.0]
      pair.margin = 0.01
      pair.gap = 0.01
    add_adhesion_actuators(spec, _EE_BODY_NAMES, _ADHESION_FORCE)
    # Disable the terrain ground plane — the gravity-tilt curriculum replaces
    # the role of a floor, and removing the plane keeps the robot honest
    # about needing to grip the wall.
    try:
      terrain_geom = spec.geom("terrain/terrain")
      terrain_geom.contype = 0
      terrain_geom.conaffinity = 0
    except Exception:
      pass

  cfg.scene.spec_fn = _spec_fn

  # Per-sphere microspine adhesion via MuJoCo adhesion actuators.
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=("left_hand", "right_hand"),
        adhesion_force=_ADHESION_FORCE,
        engage_force_threshold=5.0,
        # Lenient engagement: any single sphere in contact is enough, as long
        # as the hand is roughly parallel to the (tilted) wall surface.
        engage_min_spheres=1,
        engage_parallel_cos_threshold=0.9397,  # cos(20°)
        # Re-engage only when friction is opposite to the release trigger:
        # fric_up < -30 N, filtered to suppress contact noise.
        engage_friction_threshold=30.0,
        engage_friction_filter_alpha=0.8,
        engage_consecutive_steps=_ENGAGE_CONSECUTIVE_STEPS,
        # Latched adhesion releases when friction along hand-up exceeds
        # 30 N.
        disengage_friction_threshold=30.0,
        disengage_friction_filter_alpha=0.8,
        # Suppress single-step CCD jitter oscillations: require sustained
        # overload for 5 frames before actually releasing, and block
        # re-engagement for 0.3 s (15 controller steps) after release so
        # a brief jitter spike can't cause engage→release→engage on
        # consecutive steps.
        disengage_consecutive_steps=5,
        reengage_cooldown_steps=_REENGAGE_COOLDOWN_STEPS,
        wall_geom_name="wall/wall_geom",
        wall_entity_name="wall",
      ),
    },
  )



  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_BIG_V2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Play-only debug: print engage/disengage transitions for env 0 and
    # draw the current gravity vector in the viewer.
    cfg.events["microspine_adhesion"].params["microspine_cfg"].debug = True

  return cfg
