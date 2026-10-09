"""CAD climbing reference geometry and pre-latch audit contracts."""

import mujoco
import numpy as np
import pytest

import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface
from scr_phybot_mjab.tasks.climb_wall.motions import CAD_CLIMB_MOTION
from scr_phybot_mjab.tools.climbing_reference import cad_cycle_audit as audit
from scr_phybot_mjab.tools.climbing_reference.model import WallGeometry, build_reference_model


@pytest.fixture(scope="module")
def reference():
  clip = np.load(CAD_CLIMB_MOTION, allow_pickle=True).item()
  model, sites, bodies, toes, _ = build_reference_model(tuple(clip["ee_list"]))
  return clip, model, sites, bodies, toes, audit


def test_climbing_full_clip_matches_current_cad_fk(reference):
  clip, model, sites, bodies, _, audit = reference
  data = mujoco.MjData(model)
  max_body = max_center = 0.0
  for frame in range(len(clip["joint_positions"])):
    audit.set_frame(model, data, clip, frame)
    for i, ee in enumerate(clip["ee_list"]):
      max_body = max(max_body, np.linalg.norm(data.xpos[bodies[ee]] - clip["ee_positions"][frame, i]))
      max_center = max(max_center, np.linalg.norm(data.site_xpos[sites[ee]] - clip["pad_positions"][frame, i]))
  assert max_body < 2e-6
  assert max_center < 2e-6


def test_climbing_loop_closes_complete_joint_state(reference):
  clip, *_ = reference
  cycle = int(np.flatnonzero(clip["phase_segment"] == list(clip["segment_names"]).index("end_hold"))[0])
  for key in ("joint_positions", "passive_joint_positions", "root_quaternion"):
    assert np.allclose(clip[key][cycle], clip[key][0], atol=1e-6, rtol=0)
  assert np.allclose(
    clip["root_position"][cycle] - clip["root_position"][0], [0, 0, clip["step_height"]], atol=1e-6, rtol=0
  )


def test_eighty_degree_audit_rotates_contacts_but_keeps_real_gravity(reference):
  clip, model, sites, bodies, _, audit = reference
  tilted = {**clip, "wall_angle": 80.0, "unit_layout": "corners"}
  data = mujoco.MjData(model)
  wall = WallGeometry(80.0, 1.5)
  audit.set_frame(model, data, tilted, 0)
  for i, ee in enumerate(clip["ee_list"]):
    np.testing.assert_allclose(
      data.site_xpos[sites[ee]], wall.rotate_point(clip["pad_positions"][0, i]), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(data.xpos[bodies[ee]], wall.rotate_point(clip["ee_positions"][0, i]), atol=2e-6, rtol=0)
    assert abs(wall.distance(data.site_xpos[sites[ee]])) < 2e-6
  np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
  np.testing.assert_allclose(data.qfrc_bias[:3], -model.body_mass.sum() * model.opt.gravity, atol=1e-8)


def test_corner80_asset_keeps_canonical_cad_poses_and_identifies_active_units(reference):
  from scr_phybot_mjab.tasks.climb_wall.motions import CAD_CLIMB_CORNERS80_MOTION

  clip, *_ = reference
  variant = np.load(CAD_CLIMB_CORNERS80_MOTION, allow_pickle=True).item()
  assert variant["wall_angle"] == 80.0
  assert variant["unit_layout"] == "corners"
  assert variant["active_units_per_pad"] == 4
  assert variant["engage_unit_preload_n"] == 10.0
  for key in (
    "root_position",
    "root_quaternion",
    "joint_positions",
    "passive_joint_positions",
    "ee_engaged",
    "ee_anchor_positions",
    "pad_positions",
    "toe_braced",
  ):
    np.testing.assert_array_equal(variant[key], clip[key])


def test_audit_toe_loads_obey_the_actual_task_friction_cone(reference):
  import inspect

  from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg

  clip, model, sites, bodies, toes, audit = reference
  cfg = phybot_c2_climb_wall_env_cfg()
  mu = cfg.commands["microspine_affordance"].microspine_cfg.contact_friction[0]
  parameters = inspect.signature(audit.static_frame).parameters
  assert parameters["preload_friction"].default == parameters["toe_friction"].default == mu
  data = mujoco.MjData(model)
  audit.set_frame(model, data, clip, 50)
  row, _ = audit.static_frame(model, data, clip, 50, sites, bodies, toes, EmpiricalLimitSurface())
  assert row["characteristic_feasible"]
  assert row["toe_load_wall_xyz_n"]
  for force in row["toe_load_wall_xyz_n"].values():
    assert np.linalg.norm(force[1:]) <= mu * force[0] + 1e-5


def test_climbing_leaves_three_touching_off_frames_before_each_latch(reference):
  clip, *_ = reference
  edges = np.argwhere(np.diff(clip["ee_engaged"].astype(int), axis=0) > 0)
  assert len(edges) == 4
  for previous, pad in edges:
    edge = int(previous + 1)
    assert not clip["ee_engaged"][edge - 3 : edge, pad].any()
    assert np.abs(clip["pad_positions"][edge - 3 : edge, pad, 0] - 1.5).max() < 0.00015
    # The reference retains the downward seating rake during this window.
    assert (np.diff(clip["pad_positions"][edge - 3 : edge + 1, pad, 2]) <= 1e-6).all()


def test_sixty_degree_hold_can_carry_more_than_ten_n_compression_per_unit(reference):
  clip, model, sites, bodies, toes, audit = reference
  tilted = {**clip, "wall_angle": 60.0, "unit_layout": "corners"}
  data = mujoco.MjData(model)
  audit.set_frame(model, data, tilted, 0)
  row, _ = audit.static_frame(model, data, tilted, 0, sites, bodies, toes, EmpiricalLimitSurface())
  assert row["characteristic_feasible"]
  assert row["maximum_unit_compression_n"] > 10.0


def test_preload_audit_cannot_credit_adhesion_to_an_approaching_pad(reference):
  clip, model, sites, bodies, toes, audit = reference
  data = mujoco.MjData(model)
  surface = EmpiricalLimitSurface()
  previous, index = np.argwhere(np.diff(clip["ee_engaged"].astype(int), axis=0) > 0)[0]
  frame, ee = int(previous), clip["ee_list"][index]
  audit.set_frame(model, data, clip, frame)
  row, _ = audit.static_frame(
    model, data, clip, frame, sites, bodies, toes, surface, preload_names=(ee,), unit_preload_n=10000.0
  )
  assert not row["linear_feasible"]
  assert row["preload_contacts"] == [ee]
  assert row["required_unit_preload_n"] == 10000.0
