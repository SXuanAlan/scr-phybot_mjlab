"""Failure data semantics, persistent draws, and C-referenced unit mechanics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scr_phybot_mjab.microspine.affordance import MicrospineAffordance
from scr_phybot_mjab.microspine.empirical_limit_surface import (
  DEFAULT_TABLE,
  EmpiricalLimitSurface,
  shift_wrench_reference,
)
from scr_phybot_mjab.microspine.hold_judge import HoldJudge
from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units


@pytest.fixture(scope="module")
def surface():
  return EmpiricalLimitSurface()


def hold_judge():
  """The 3x4 CAD unit grid, as offsets from the contact-array centre C."""
  x, y = torch.meshgrid(torch.linspace(-0.0315, 0.0315, 3), torch.linspace(-0.042, 0.042, 4), indexing="ij")
  return HoldJudge(SimpleNamespace(x=-x.flatten().numpy(), y=-y.flatten().numpy()), DEFAULT_TABLE, "cpu")


def term():
  a = object.__new__(MicrospineAffordance)
  a.hold_judge = hold_judge()
  a.unit_strength_quantile = torch.full((2, 4, 12), 0.5)
  a.unit_loading_quantile = torch.full((2, 4, 12), 0.99)
  return a


def test_numpy_torch_lookup_agree_and_lateral_reflection_is_explicit(surface):
  rng = np.random.default_rng(4)
  s = rng.uniform(-16, 31, 300).astype(np.float32)
  l = rng.uniform(-31, 31, 300).astype(np.float32)
  u = rng.random(300).astype(np.float32)
  expected = surface.query_numpy(s, l, u)
  actual = surface.query(*(torch.tensor(x) for x in (s, l, u)))
  for x, y in zip(expected, actual):
    np.testing.assert_allclose(x, y.numpy(), atol=3e-6)
  for x, y in zip(actual, surface.query(torch.tensor(s), -torch.tensor(l), torch.tensor(u))):
    torch.testing.assert_close(x, y)


def test_inverse_cdf_is_monotone_and_singleton_cells_have_pooled_scatter(surface):
  u = torch.linspace(0, 1, 1025)
  cap = surface.query(torch.full_like(u, 10), torch.zeros_like(u), u)[0]
  assert torch.all(cap.diff() >= 0)
  # S=10,L=0 has only one direct sample. Pooling must not claim zero variance.
  assert cap.max() - cap.min() > 1


def test_unsupported_shear_lateral_and_nonfinite_loads_still_fail(surface):
  ft = torch.tensor([30.0, 30.01, -15.01, 10.0, 10.0, float("nan"), 10.0])
  fl = torch.tensor([0.0, 0.0, 0.0, 30.01, 0.0, 0.0, 0.0])
  fn = torch.tensor([-1.0, -20.0, -20.0, -20.0, float("-inf"), 0.0, float("nan")])
  result = surface.margins(ft, fl, fn, torch.full_like(ft, 0.9), torch.ones_like(ft))
  assert result[2].tolist() == [True, False, False, False, False, False, False]
  assert torch.all(result[0][1:] < 0)


def test_preload_is_not_a_compression_ceiling_or_extra_pull_off_capacity(surface):
  fn = torch.tensor([-10.0, -10.01, -13.2, -20.0, -100.0])
  ft, fl, u = torch.full_like(fn, 10.0), torch.zeros_like(fn), torch.full_like(fn, 0.5)
  margin, cap, domain, early, _ = surface.margins(ft, fl, fn, u, torch.ones_like(fn))
  assert domain.all() and (margin > 0).all() and not early.any()
  torch.testing.assert_close(cap, cap[0].expand_as(cap))
  # The same unit still fails when pulled beyond its sampled normal capacity.
  failed = surface.margins(ft, fl, cap + 0.1, u, torch.ones_like(fn))
  assert (failed[0] < 0).all() and failed[2].all()


def test_legacy_table_preload_metadata_does_not_restore_compression_ceiling(surface, tmp_path):
  arrays = dict(surface.arrays, schema_version=np.int32(1), normal_min_n=np.float32(-10))
  arrays.pop("protocol_preload_n", None)
  path = tmp_path / "legacy.npz"
  np.savez_compressed(path, **arrays)
  legacy = EmpiricalLimitSurface(path)
  assert legacy.protocol_preload_n == 10.0
  x = torch.tensor([10.0])
  result = legacy.margins(x, x * 0, x * -2, x * 0 + 0.5, x * 0 + 1)
  assert result[2].item() and result[0].item() > 0


def test_early_failure_is_separate_from_a_normal_threshold(surface):
  ft, fl, u = (torch.tensor([25.0]), torch.tensor([30.0]), torch.tensor([0.9]))
  cap, early, domain, _ = surface.query(ft, fl, u)
  assert early.item() > 0
  fn = torch.tensor([-20.0])
  failed = surface.margins(ft, fl, fn, u, torch.zeros_like(u))
  survived = surface.margins(ft, fl, fn, u, torch.ones_like(u))
  torch.testing.assert_close(failed[1], survived[1])
  assert failed[3].item() and failed[0].item() < 0
  assert not survived[3].item()


def test_draws_change_only_for_new_latches_and_hold_during_queries():
  a = term()
  before_s, before_l = a.unit_strength_quantile.clone(), a.unit_loading_quantile.clone()
  mask = torch.zeros((2, 4), dtype=torch.bool)
  mask[1, 2] = True
  torch.manual_seed(123)
  a._redraw_empirical_latches(mask)
  torch.testing.assert_close(a.unit_strength_quantile[~mask], before_s[~mask])
  torch.testing.assert_close(a.unit_loading_quantile[~mask], before_l[~mask])
  assert torch.unique(a.unit_strength_quantile[mask]).numel() == 12
  after = a.unit_strength_quantile.clone()
  for force in (80.0, 100.0, 80.0):
    ft = torch.full((2, 4), force)
    z = torch.zeros_like(ft)
    a._judge_units(ft, z, z, z, z, z)
  torch.testing.assert_close(a.unit_strength_quantile, after)
  a._redraw_empirical_latches(torch.zeros_like(mask))
  torch.testing.assert_close(a.unit_strength_quantile, after)


def test_real_c_offset_and_new_grid_conserve_world_force_and_moment():
  judge = hold_judge()
  force = torch.tensor([120.0, 12.0, 6.0], dtype=torch.float64)
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_PAD_GEOMETRY

  c = PHYBOT_C2_PAD_GEOMETRY["contact_center_in_pad_m"]
  b_to_c = torch.tensor([-c[2], -c[0], c[1]], dtype=torch.float64)  # [rake,lateral,outward]
  mb = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
  mc = shift_wrench_reference(force, mb, b_to_c)
  x, y = judge.unit_x.double(), judge.unit_y.double()
  loads = distribute_wrench_to_units(
    *force, *mc, x, y, float(x.square().sum()), float(y.square().sum()), float(x.square().sum() + y.square().sum())
  )
  f = torch.stack(loads, -1)
  r = torch.stack((x, y, torch.zeros_like(x)), -1)
  torch.testing.assert_close(f.sum(0), force)
  torch.testing.assert_close(torch.cross(r, f, dim=-1).sum(0), mc)
  torch.testing.assert_close(torch.cross(r + b_to_c, f, dim=-1).sum(0), mb)


def test_empirical_unit_load_indices_match_physical_cad_sites():
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec

  model = get_spec().compile()
  judge = hold_judge()
  for ee in ("left_hand_spine", "right_hand_spine", "left_foot_spine", "right_foot_spine"):
    c = model.site(ee + "_contact_center").pos
    for row in range(3):
      for col in range(4):
        offset = model.site(f"{ee}_unit_{row}_{col}").pos - c
        k = row * 4 + col
        np.testing.assert_allclose(
          [judge.unit_x[k], judge.unit_y[k]], [-offset[2], -offset[0]], atol=5e-8
        )  # CAD export precision


def test_moment_ablation_uses_same_draws_and_detects_peeling():
  a = term()
  ft = torch.full((2, 4), 120.0)
  z = torch.zeros_like(ft)
  clean = a._judge_units(ft, z, z, z, z, z)
  peeled = a._judge_units(ft, z, z, z, torch.full_like(ft, 3.0), z)
  assert torch.all(peeled.margin < clean.margin)
  assert torch.all(peeled.violation)
  assert a.unit_load_tln.shape == (2, 4, 12, 3)


def test_build_preserves_right_censoring_and_excludes_early_normal_points():
  from scr_phybot_mjab.tools.microspine.empirical_fitting import conditional_quantiles

  rows = [
    dict(result="detached_normal", fz_cmd_n="2"),
    dict(result="held", fz_last_held_n="10"),
    dict(result="detached_drag", fz_cmd_n="-10"),
  ]
  q, tail = conditional_quantiles(rows, np.array([0.1, 0.49, 0.51, 0.9]))
  np.testing.assert_array_equal(q, [2, 2, 10, 10])
  np.testing.assert_array_equal(tail, [False, False, True, True])


def test_native_weld_rotational_rows_are_a_c_couple_not_a_second_lever_arm():
  import mujoco

  from scr_phybot_mjab.microspine.constraint import weld_rotational_force_to_world_moment

  model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0" jacobian="dense"/>
    <worldbody><body name="pad"><freejoint/><geom type="box" size=".02 .02 .02" mass="1"/>
    </body></worldbody><equality><weld body1="pad" torquescale="1"/></equality></mujoco>""")
  c = np.array([0.0, -0.04, 0.075])
  model.eq_data[0, :3] = c
  model.eq_data[0, 3:6] = c
  model.eq_data[0, 6:10] = [1, 0, 0, 0]
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  multiplier = np.array([20.0, -30.0, 80.0, 0.4, -0.6, 1.0])
  generalized = data.efc_J.reshape(data.nefc, model.nv).T @ multiplier
  force_load = -multiplier[:3]
  moment_b_load = -generalized[3:6]
  converted = -weld_rotational_force_to_world_moment(
    torch.tensor(multiplier[3:]),
    torch.tensor([1.0, 0, 0, 0], dtype=torch.float64),
    torch.tensor([1.0, 0, 0, 0], dtype=torch.float64),
    1.0,
  ).numpy()
  np.testing.assert_allclose(moment_b_load - np.cross(c, force_load), converted, atol=1e-10)
