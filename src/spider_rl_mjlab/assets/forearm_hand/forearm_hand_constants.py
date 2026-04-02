"""Forearm + hand robot constants and configuration.

Minimal model: right elbow (2 DOF) + hand for microspine adhesion testing.
Reuses meshes from phybot_big_v2.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets.
##

FOREARM_HAND_XML: Path = Path(__file__).resolve().parent / "xml" / "forearm_hand.xml"
assert FOREARM_HAND_XML.exists(), f"MJCF not found: {FOREARM_HAND_XML}"

# Meshes live in the phybot_big_v2 directory.
PHYBOT_MESHES: Path = Path(__file__).resolve().parent.parent / "phybot_big_v2" / "meshes"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PHYBOT_MESHES, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(FOREARM_HAND_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

FOREARM_HAND_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("right_elbow_yaw",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

##
# Keyframe config.
##

FOREARM_HAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.5),
  rot=(1.0, 0.0, 0.0, 0.0),  # no rotation — arm hangs in -z
  joint_pos={
    "right_elbow_yaw": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOREARM_HAND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*", "right_hand_c_.*"),
  condim=6,
  friction=(30.0, 0.01, 0.01),
  solref=(0.02, 1.5),
  solimp=(0.9, 0.99, 0.001, 0.5, 2),
  priority={"right_hand_c_.*": 1},
)

##
# Final config.
##

FOREARM_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(FOREARM_HAND_ACTUATOR_ELBOW_YAW,),
  soft_joint_pos_limit_factor=0.9,
)


def get_forearm_hand_robot_cfg() -> EntityCfg:
  """Get a fresh forearm+hand robot configuration instance."""
  return EntityCfg(
    init_state=FOREARM_HAND_KEYFRAME,
    collisions=(FOREARM_HAND_COLLISION,),
    spec_fn=get_spec,
    articulation=FOREARM_HAND_ARTICULATION,
  )


FOREARM_HAND_ACTION_SCALE = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_forearm_hand_robot_cfg())
  viewer.launch(robot.spec.compile())
