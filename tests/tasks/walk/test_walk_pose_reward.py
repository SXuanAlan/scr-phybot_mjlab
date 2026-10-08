"""The walk pose reward weighs body joints only; the pad gimbals are passive."""

import mujoco
import pytest
from mjlab.utils.lab_api.string import resolve_matching_names, resolve_matching_names_values

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import phybot_c2_walk_env_cfg


@pytest.mark.parametrize("variant", ["walk", "amp"])
def test_pose_reward_std_tables_cover_the_same_body_joints(variant):
  model = get_spec().compile()
  joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  params = phybot_c2_walk_env_cfg(variant=variant).rewards["pose"].params
  _, names = resolve_matching_names(params["asset_cfg"].joint_names, joints)
  assert names and not any("gimbal" in name for name in names)
  lengths = {
    key: len(resolve_matching_names_values(params[key], names)[2])
    for key in ("std_standing", "std_walking", "std_running")
  }
  assert set(lengths.values()) == {len(names)}, lengths
