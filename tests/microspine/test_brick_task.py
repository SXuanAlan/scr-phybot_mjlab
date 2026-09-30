"""Task contracts and observable phase gates for floor manipulation."""

import mjlab  # noqa: F401
import numpy as np
import torch

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.tasks.brick.command import interpolate_reference, measured_phase_gates
from scr_phybot_mjab.tasks.brick.env_cfg import brick_env_cfg
from scr_phybot_mjab.tasks.brick.reference import load_reference


def gates(**overrides):
  inputs = dict(
    hand_error=0.0,
    lift=0.12,
    speed=0.0,
    goal_distance=0.0,
    ground_gap=0.0,
    orientation_error=0.0,
    hand_clearance=0.03,
    stand_error=0.0,
    upright=0.0,
  )
  inputs.update(overrides)
  values = {key: torch.tensor([value]) for key, value in inputs.items()}
  return measured_phase_gates(**values, valid=torch.tensor([True]))[0]


def test_ground_pushing_cannot_complete_lift_or_carry():
  result = gates(lift=0.0)
  assert not result[2] and not result[3]
  assert result[4]


def test_release_requires_both_hands_clear_and_brick_resting():
  assert gates()[5]
  assert not gates(hand_clearance=0.005)[5]
  assert not gates(ground_gap=0.03)[5]
  assert not gates(speed=0.2)[5]
  assert not gates(orientation_error=0.4)[5]


def test_reference_interpolation_has_closed_endpoints_and_is_per_environment():
  v = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
  p = torch.tensor([0, 1, 0])
  out = interpolate_reference(v, p, torch.tensor([0.0, 1.0, 0.5]))
  torch.testing.assert_close(out[0], v[0, 0])
  torch.testing.assert_close(out[1], v[1, -1])
  torch.testing.assert_close(out[2], (v[0, 1] + v[0, 2]) / 2)


def test_task_uses_dynamic_brick_and_independent_opposing_faces():
  cfg = brick_env_cfg(play=True)
  spec = cfg.scene.entities["brick"].spec_fn()
  model = spec.compile()
  assert model.nv == 6
  np.testing.assert_allclose(model.body_mass.sum(), 1.0)
  spine = cfg.commands["microspine_affordance"].microspine_cfg
  assert len(spine.ee_body_names) == len(spine.surfaces) == 2
  assert not spine.initial_on and not spine.reanchor_after_grace
  assert spine.surfaces[0].scoped_body == spine.surfaces[1].scoped_body == "brick/brick"
  np.testing.assert_allclose(spine.surfaces[0].normal, -np.asarray(spine.surfaces[1].normal))
  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
  assert "physical_state" not in cfg.observations["actor"].terms
  assert "physical_state" in cfg.observations["critic"].terms
  assert len(cfg.actions["joint_pos"].actuator_names) == 21


def test_nominal_reference_passes_geometry_and_preserves_joint_order():
  reference = load_reference()
  assert reference["joint_pos"].shape[-1] == 21
  np.testing.assert_allclose(reference["joint_pos"][0, 0], reference["joint_pos"][-1, -1])
  for k in range(6):
    np.testing.assert_allclose(reference["joint_pos"][k, -1], reference["joint_pos"][k + 1, 0], atol=0.03)
  np.testing.assert_allclose(reference["brick_size"], BrickCfg().size)
  assert reference["brick_pos"][2, -1, 2] - reference["brick_pos"][0, 0, 2] >= 0.10
