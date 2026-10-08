"""Physical invariants and geometry changes across runtime/offline consumers."""

import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from scr_phybot_mjab.microspine.pad_geometry import pad_geometry_from_sites
from scr_phybot_mjab.microspine.wrench import (
  distribute_wrench_to_units,
  distribute_wrench_to_units_numpy,
  unit_wrench_matrix,
)
from scr_phybot_mjab.tools.microspine.stochastic_pad import StochasticPad


def test_reference_ik_preserves_unselected_passive_joint_coordinates():
  import scr_phybot_mjab.tasks  # noqa: F401  # Initialize task registry first.
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
  from scr_phybot_mjab.tools.climbing_reference.kinematics import ClimbIK, base_task

  ik = ClimbIK()
  ik.set_state1()
  passive = [adr for name, adr in ik.qpos_addresses.items() if "_gimbal_" in name]
  before = ik.data.qpos[passive].copy()
  target = ik.data.qpos[:3] + np.array([0.0, 0.0, 0.01])
  posture = {name: ik.data.qpos[ik.qpos_addresses[name]] for name in PHYBOT_C2_POLICY_JOINT_NAMES}
  ik.solve_frame(
    [base_task(ik, target, (100.0, 100.0, 100.0))],
    ik.dof_selection(PHYBOT_C2_POLICY_JOINT_NAMES),
    posture,
    posture_weight=0.2,
    iters=3,
  )
  # The CAD rest pose is on the -3 degree stop. A policy-only solve must
  # not silently apply the selected-joint interior margin to that state.
  assert len(passive) == 12
  assert np.array_equal(ik.data.qpos[passive], before)


def make_pad(rows, cols, rake_spacing, lateral_spacing):
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="pad")
  center = np.array([0.0, -0.04, 0.075])
  body.add_site(name="pad_contact_center", pos=center)
  for row in range(rows):
    for col in range(cols):
      body.add_site(
        name=f"pad_unit_{row}_{col}",
        pos=center + np.array([(col - (cols - 1) / 2) * lateral_spacing, 0.0, (row - (rows - 1) / 2) * rake_spacing]),
      )
  return spec


@pytest.mark.parametrize("rows,cols,rake,lateral", [(3, 4, 0.0315, 0.028), (4, 5, 0.027, 0.019)])
def test_changed_asset_grid_drives_torch_numpy_and_probability_model(rows, cols, rake, lateral):
  spec = make_pad(rows, cols, rake, lateral)
  geometry = pad_geometry_from_sites(spec, "pad")
  compiled = pad_geometry_from_sites(spec.compile(), "pad")
  np.testing.assert_allclose(compiled.unit_positions_tln, geometry.unit_positions_tln)
  assert (geometry.rows, geometry.cols) == (rows, cols)
  np.testing.assert_allclose(geometry.spans, [(rows - 1) * rake, (cols - 1) * lateral])
  rng = np.random.default_rng(607)
  wrench = rng.normal(size=(50, 6))
  loads = distribute_wrench_to_units_numpy(wrench, geometry)
  # Conservation is independent of the allocator's implementation.
  np.testing.assert_allclose(loads.sum(axis=-2), wrench[:, :3], atol=1e-12)
  moments = np.cross(geometry.unit_positions_tln, loads).sum(axis=-2)
  np.testing.assert_allclose(moments, wrench[:, 3:], atol=1e-12)
  matrix_loads = np.einsum("uac,ec->eua", unit_wrench_matrix(geometry), wrench)
  np.testing.assert_allclose(matrix_loads, loads, atol=1e-12)
  tensors = [torch.tensor(x) for x in np.moveaxis(wrench, -1, 0)]
  torch_loads = torch.stack(
    distribute_wrench_to_units(
      *tensors,
      torch.tensor(geometry.x.copy()),
      torch.tensor(geometry.y.copy()),
      geometry.sxx,
      geometry.syy,
      geometry.srr,
    ),
    dim=-1,
  )
  np.testing.assert_allclose(torch_loads.numpy(), loads, atol=1e-12)
  pad = StochasticPad(draws=16, geometry=geometry)
  assert pad.u.shape == (16, rows * cols)
  np.testing.assert_allclose(
    np.stack(pad.units(100.0, 10.0, 0.2), axis=-1),
    distribute_wrench_to_units_numpy([100.0, 10.0, 0.0, 0.0, 0.0, 0.2], geometry)[:, :2],
  )


def test_doubling_rake_lever_halves_unit_peeling_load():
  narrow = pad_geometry_from_sites(make_pad(3, 4, 0.02, 0.03), "pad")
  wide = pad_geometry_from_sites(make_pad(3, 4, 0.04, 0.03), "pad")
  wrench = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  narrow_load = distribute_wrench_to_units_numpy(wrench, narrow)
  wide_load = distribute_wrench_to_units_numpy(wrench, wide)
  np.testing.assert_allclose(wide_load, narrow_load * 0.5, atol=1e-12)


def test_compressed_pad_bending_boundary_is_set_by_pull_off_on_tensile_side():
  geometry = pad_geometry_from_sites(make_pad(3, 4, 0.0315, 0.028), "pad", unit_layout="corners")
  pad = StochasticPad(draws=16, geometry=geometry)
  pad.u.fill(0.5)
  pad.v.fill(1.0)
  angles = np.array([0.0, np.pi / 2])
  radii = pad.bending(40.0, angles, fn=-80.0)  # 20 N/unit before bending.
  assert np.isfinite(radii).all() and (radii > 0).all()
  for angle, radius in zip(angles, radii):
    loads = distribute_wrench_to_units_numpy(
      [40.0, 0.0, -80.0, radius * np.sin(angle), radius * np.cos(angle), 0.0], geometry
    )
    cap, _, valid, _ = pad.surface.query_numpy(loads[:, 0], loads[:, 1], 0.5)
    assert valid.all() and np.min(loads[:, 2]) < -20.0
    np.testing.assert_allclose(np.min(cap - loads[:, 2]), 0.0, atol=1e-8)


def test_four_corners_keep_cad_levers_and_conserve_the_complete_wrench():
  spec = make_pad(3, 4, 0.0315, 0.028)
  geometry = pad_geometry_from_sites(spec, "pad", unit_layout="corners")
  assert geometry.count == 4
  assert geometry.unit_names == ("pad_unit_0_0", "pad_unit_0_3", "pad_unit_2_0", "pad_unit_2_3")
  np.testing.assert_allclose(geometry.spans, [0.063, 0.084])
  wrench = np.random.default_rng(608).normal(size=(30, 6))
  loads = distribute_wrench_to_units_numpy(wrench, geometry)
  np.testing.assert_allclose(loads.sum(axis=-2), wrench[:, :3], atol=1e-12)
  np.testing.assert_allclose(np.cross(geometry.unit_positions_tln, loads).sum(axis=-2), wrench[:, 3:], atol=1e-12)
  np.testing.assert_allclose(distribute_wrench_to_units_numpy([0, 0, -40, 0, 0, 0], geometry)[:, 2], -10.0)


def test_unknown_unit_layout_is_rejected():
  with pytest.raises(ValueError, match="unit layout"):
    pad_geometry_from_sites(make_pad(3, 4, 0.02, 0.03), "pad", unit_layout="middle")


def test_missing_unit_and_inconsistent_center_are_rejected():
  spec = make_pad(3, 4, 0.02, 0.03)
  spec.site("pad_unit_2_3").name = "missing"
  with pytest.raises(ValueError, match="Incomplete"):
    pad_geometry_from_sites(spec, "pad")
  spec = make_pad(3, 4, 0.02, 0.03)
  spec.site("pad_contact_center").pos[2] += 0.01
  with pytest.raises(ValueError, match="centroid"):
    pad_geometry_from_sites(spec, "pad")


def test_unit_on_an_offset_child_cannot_be_misread_as_pad_local():
  spec = make_pad(3, 4, 0.02, 0.03)
  original = spec.site("pad_unit_0_0")
  pos = original.pos.copy()
  original.name = "unused_site"
  child = spec.body("pad").add_body(name="offset_child", pos=[0.0, 0.0, 0.1])
  child.add_site(name="pad_unit_0_0", pos=pos)
  for source in (spec, spec.compile()):
    with pytest.raises(ValueError, match="attached directly"):
      pad_geometry_from_sites(source, "pad")


def test_scripts_and_tools_never_import_a_cli_script():
  root = Path(__file__).resolve().parents[2]
  scripts = list((root / "scripts").rglob("*.py"))
  script_modules = {p.stem for p in scripts}
  paths = scripts + list((root / "src/scr_phybot_mjab/tools").rglob("*.py"))
  for path in paths:
    for node in ast.walk(ast.parse(path.read_text())):
      modules = (
        [node.module or ""]
        if isinstance(node, ast.ImportFrom)
        else ([item.name for item in node.names] if isinstance(node, ast.Import) else [])
      )
      assert not any(m.split(".")[0] in script_modules for m in modules), path
