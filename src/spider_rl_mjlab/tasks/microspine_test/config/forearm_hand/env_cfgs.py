"""Forearm+hand microspine test environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets.forearm_hand import FOREARM_HAND_ACTION_SCALE, get_forearm_hand_robot_cfg
from spider_rl_mjlab.assets.microspine import MicrospineCfg, MicrospineController
from spider_rl_mjlab.assets.wall import get_wall_cfg
from spider_rl_mjlab.tasks.microspine_test.microspine_test_env_cfg import make_microspine_test_env_cfg


def forearm_hand_microspine_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create forearm+hand microspine test configuration."""
  cfg = make_microspine_test_env_cfg()

  robot_cfg = get_forearm_hand_robot_cfg()
  wall_cfg = get_wall_cfg()

  # "Hand standing upright" posture: forearm hangs vertically, hand presses against wall.
  #
  # 180° rotation around z: flips -x body axis to +x (world), so the hand sphere
  # (local pos=(-0.02,0,0)) ends up at +x relative to the arm chain.
  #
  # Geometry (180° z rotation, chain still extends in -z):
  #   Base (elbow)   : (B_x,       0, 1.5)
  #   Hand body      : (B_x+0.01,  0, 1.327)
  #   Sphere center  : (B_x+0.03,  0, 1.327)  [sphere pos=(-0.02,0,0) → (0.02,0,0) after rotation]
  #   Sphere radius  : 0.035 m
  #   Sphere +x edge : B_x + 0.065
  #
  # Wall center at (0.55, 0, 2.5): inner face at x = env_origin_x + 0.05
  # With B_x = 0 (robot init at env origin): sphere edge = 0.065 > 0.05 → ~15mm initial penetration.
  # Elbow cylinders max +x edge: B_x + 0.04 < 0.05 → elbow does NOT hit wall. ✓
  #
  # The 300 N force in -z tests pure shear adhesion of the microspine.
  robot_cfg.init_state.pos = (-0.04, 0.0, 1.5)  # sphere edge ~1cm from wall; phase-1 force pushes in gently
  robot_cfg.init_state.rot = (0, 1, 0.0, 0.0)  # 180° around x: arm extends upward, hand presses wall in +x
  robot_cfg.init_state.joint_pos = {
    "right_elbow_yaw": 0.0,
  }

  # Wall center at (0.5, 0, 2.5): inner face at x=0, covers z=[0, 5].
  wall_cfg.init_state.pos = (0.55, 0.0, 2.5)

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  # Microspine adhesion (only right hand).
  cfg.events["microspine_adhesion"] = EventTermCfg(
    func=MicrospineController,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "microspine_cfg": MicrospineCfg(
        ee_body_names=("right_hand",),
        max_penetration=0.15,        # allow up to 15cm penetration before disengaging
        penetration_hysteresis=0.05,
      ),
      "wall_cfg": SceneEntityCfg("wall"),
    },
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = FOREARM_HAND_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
