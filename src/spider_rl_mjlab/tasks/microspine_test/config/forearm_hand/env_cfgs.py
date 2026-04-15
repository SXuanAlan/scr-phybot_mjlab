"""PHYBot Big V2 microspine test environment configurations."""

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets import PHYBOT_BIG_V2_ACTION_SCALE, get_phybot_big_v2_robot_cfg, get_wall_cfg
from spider_rl_mjlab.assets.microspine import MicrospineCfg, MicrospineController, add_adhesion_actuators
from spider_rl_mjlab.tasks.microspine_test.microspine_test_env_cfg import make_microspine_test_env_cfg


# Hand sub-geom 2x4 grid + foot collision geoms (same as climb_wall).
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


def forearm_hand_microspine_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot Big V2 microspine test configuration."""
  cfg = make_microspine_test_env_cfg()
  # Train: only Phase 1 (engage). Play: full Phase 1→2→3 to visualize holding.
  cfg.episode_length_s = 12.0 if play else 5.0

  robot_cfg = get_phybot_big_v2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Position robot close to wall — hands should be near the wall inner face.
  robot_cfg.init_state.pos = (-0.05, 0.0, 1.0)
  robot_cfg.init_state.joint_pos = {
    "left_hip_pitch": -0.2,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.4,
    "left_ankle_pitch": -0.2,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.2,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.4,
    "right_ankle_pitch": -0.2,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "left_shoulder_pitch": -2.79,
    "left_shoulder_roll": -0.17,
    "left_shoulder_yaw": -0.5,
    "left_elbow_pitch": -0.26,
    "left_elbow_yaw": 1.5,
    "right_shoulder_pitch": -2.79,
    "right_shoulder_roll": 0.34,
    "right_shoulder_yaw": 0.5,
    "right_elbow_pitch": -0.26,
    "right_elbow_yaw": -1.5,
  }

  # Wall position (same as climb_wall).
  wall_cfg.init_state.pos = (0.8, 0.0, 2.5)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  _ADHESION_FORCE = 1200.0
  _EE_BODY_NAMES = ("left_hand", "right_hand")

  # Contact pairs + adhesion actuators via spec_fn (same as climb_wall).
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
    add_adhesion_actuators(spec, _EE_BODY_NAMES, _ADHESION_FORCE)

  cfg.scene.spec_fn = _spec_fn

  # Two-phase test force: Phase 1 push (+x) then Phase 2 shear (-z).
  # This is the test harness — DO NOT add adhesion logic here.
  from spider_rl_mjlab.tasks.microspine_test.mdp import PhaseForce
  # Train: zero external force in Phase 1 — policy must press the wall using
  # joint actions alone. Play: full P1→P2→P3 stress test.
  phase_force_params: dict = {
    "asset_cfg": SceneEntityCfg("robot"),
    "body_name": "right_hand",
    "phase1_duration": 5.0,
    "phase1_mag_range": (0.0, 0.0),
    "phase1_local_torque": (0.0, 0.0, 0.0),
  }
  if play:
    phase_force_params.update({
      "phase2_duration": 4.0,
      "phase2_mag_range": (50.0, 300.0),
      "phase2_angle_range": (0.0, 45.0),
      "phase3_mag_range": (50.0, 200.0),
    })
  else:
    phase_force_params.update({
      "phase2_duration": 0.0,
      "phase2_mag_range": (0.0, 0.0),
      "phase2_angle_range": (0.0, 0.0),
    })
  cfg.events["phase_force"] = EventTermCfg(
    func=PhaseForce,
    mode="step",
    params=phase_force_params,
  )

  # Play only: freeze policy joint targets after Phase 1 ends, so the test
  # forces in Phase 2/3 hit a static pose maintained purely by adhesion.
  if play:
    from spider_rl_mjlab.tasks.microspine_test.mdp import FreezeAfterPhase1
    cfg.events["freeze_after_phase1"] = EventTermCfg(
      func=FreezeAfterPhase1,
      mode="step",
      params={"phase1_duration": 3.0},
    )

  # Per-sphere microspine adhesion via MuJoCo adhesion actuators (same as climb_wall).
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=("left_hand", "right_hand"),
        adhesion_force=_ADHESION_FORCE,
        engage_force_threshold=2.0,
        engage_min_fraction=0.75,  # 6 of 8 spheres — tolerates small palm tilt
        wall_geom_name="wall/wall_geom",
        debug=True,
      ),
    },
  )

  # Contact force debug visualization (independent of controller).
  from spider_rl_mjlab.tasks.microspine_test.mdp import ContactForceVis
  cfg.events["contact_force_vis"] = EventTermCfg(
    func=ContactForceVis,
    mode="step",
    params={
      "target_geom_patterns": ["left_hand_c_", "right_hand_c_"],
      "wall_geom_name": "wall/wall_geom",
      "force_scale": 0.003,
      "ext_force_body_names": ["left_hand", "right_hand"],
    },
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_BIG_V2_ACTION_SCALE

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
