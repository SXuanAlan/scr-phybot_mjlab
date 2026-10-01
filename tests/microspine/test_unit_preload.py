"""Pre-latch compression, biased contact loads, and synchronized wrench history."""

from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.sensor import ContactMatch

from scr_phybot_mjab.microspine.affordance import MicrospineAffordance
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg
from scr_phybot_mjab.microspine.contact import get_sphere_geom_names
from scr_phybot_mjab.microspine.contact_sensor import WrenchContactData, WrenchContactSensorCfg
from scr_phybot_mjab.microspine.release_reason import ReleaseReason


@pytest.fixture
def affordance(request):
  cfg = MicrospineCfg(
    ee_body_names=("hand", "foot"),
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    initial_on=False,
    engage_unit_preload_n=10.0,
    engage_consecutive_steps=3,
    reengage_cooldown_steps=2,
  )
  ls = cfg.limit_surface
  ls.unit_layout = getattr(request, "param", "all")
  spec = mujoco.MjSpec()
  for ee in cfg.ee_body_names:
    body = spec.worldbody.add_body(name="robot/" + ee)
    body.add_site(name=f"robot/{ee}_contact_center")
    for row in range(3):
      for col in range(4):
        body.add_site(name=f"robot/{ee}_unit_{row}_{col}", pos=[(col - 1.5) * 0.028, 0, (row - 1) * 0.0315])
  model = spec.compile()
  quat = torch.zeros(2, model.nbody, 4)
  quat[..., 0] = 1
  robot = SimpleNamespace(
    find_bodies=lambda names: ([model.body("robot/" + n).id for n in names], names),
    data=SimpleNamespace(body_link_quat_w=quat, body_link_lin_vel_w=torch.zeros(2, model.nbody, 3)),
  )
  # Model order is foot before hand, unlike the affordance's EE order.
  names = get_sphere_geom_names(cfg)
  sensor = SimpleNamespace(
    cfg=SimpleNamespace(reduce="netforce"),
    primary_names=list(names[4:]) + list(names[:4]),
    data=WrenchContactData(
      found=torch.ones(2, 8),
      force=torch.zeros(2, 8, 3),
      torque=torch.zeros(2, 8, 3),
      reference_pos=torch.zeros(2, 8, 3),
      pos=torch.tensor([[-0.042, 0, -0.0315], [0.042, 0, -0.0315], [-0.042, 0, 0.0315], [0.042, 0, 0.0315]]).repeat(
        2, 2, 1
      ),
    ),
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    scene={"robot": robot, "contact": sensor},
    sim=SimpleNamespace(mj_model=model),
  )
  term = MicrospineAffordance(MicrospineAffordanceCfg(microspine_cfg=cfg, contact_sensor_name="contact"), env)
  term._wall_inward_normal = lambda: torch.tensor([[0.0, -1.0, 0.0]]).expand(2, -1)
  term._face_wall_cos = lambda: torch.ones(2, 2)
  term._virtual_spine_fraction = lambda: torch.ones(2, 2)
  term._sync_retraction = lambda: None
  term._limit_surface_release = lambda: torch.zeros(2, 2, dtype=torch.bool)
  return term, sensor.data


def compress(data, pressures):
  pressure = torch.as_tensor(pressures).expand(2, 2, 4)
  # Input is EE-major; the sensor reports foot before hand.
  data.force[..., 1] = -pressure[:, [1, 0]].reshape(2, 8)


@pytest.mark.parametrize(
  "pressures,expected",
  [
    ([30.0, 30.0, 30.0, 30.0], True),  # Exactly 10 N on each of 12 units.
    ([29.9, 29.9, 29.9, 29.9], False),
    ([27.0, 27.0, 45.0, 45.0], False),  # 144 N total; least-loaded row gets 7.5 N/unit.
    ([50.0, 22.0, 50.0, 22.0], False),  # Bending about the other pad axis.
    ([120.0, 0.0, 0.0, 0.0], False),  # A single corner cannot seat the whole pad.
    ([-30.0, -30.0, -30.0, -30.0], False),  # Tension never counts as compression.
    ([float("inf")] * 4, False),
  ],
)
def test_each_unit_must_meet_preload(affordance, pressures, expected):
  term, data = affordance
  compress(data, pressures)
  assert (term._unit_preload_condition() == expected).all()
  if expected:
    torch.testing.assert_close(term.unit_preload_n, torch.full((2, 2, 12), 10.0))


@pytest.mark.parametrize("affordance", ["corners"], indirect=True)
@pytest.mark.parametrize(
  "pressures,expected",
  [
    ([10.0] * 4, True),
    ([9.9] * 4, False),
    ([6.0, 6.0, 16.0, 16.0], False),
    ([40.0, 0.0, 0.0, 0.0], False),
    ([-10.0] * 4, False),
  ],
)
def test_four_corner_preload_and_hold_use_only_four_physical_units(affordance, pressures, expected):
  term, data = affordance
  compress(data, pressures)
  assert (term._unit_preload_condition() == expected).all()
  assert term.unit_preload_n.shape == term.unit_strength_quantile.shape == (2, 2, 4)
  assert tuple(n.rsplit("_", 2)[-2:] for n in term.unit_site_names) == (["0", "0"], ["0", "3"], ["2", "0"], ["2", "3"])
  # 200 N shear exceeds four units' measured domain (30 N/unit), although
  # it would fit the old twelve-unit array. Disabled units cannot help hold.
  ft = torch.full((2, 2), 200.0)
  z = torch.zeros_like(ft)
  verdict = term._judge_units(ft, z, z, z, z, z)
  assert verdict.violation.all() and (verdict.reason == ReleaseReason.EMPIRICAL_DOMAIN).all()
  torch.testing.assert_close(term.unit_load_tln[..., 0], torch.full((2, 2, 4), 50.0))


def test_sensor_order_and_environment_batches_remain_independent(affordance):
  term, data = affordance
  compress(data, [[[30.0] * 4, [0.0] * 4], [[0.0] * 4, [33.0] * 4]])
  assert term._unit_preload_condition().tolist() == [[True, False], [False, True]]


@pytest.mark.parametrize("affordance", ["all", "corners"], indirect=True)
@pytest.mark.parametrize("unit_pressure", [10.0, 13.2, 20.0])
def test_preload_at_or_above_ten_can_latch_and_hold(affordance, unit_pressure):
  term, data = affordance
  count = term.hold_judge.unit_count
  compress(data, [unit_pressure * count / 4] * 4)
  for _ in range(3):
    term._update_command()
  assert term.engaged.all()
  term.unit_loading_quantile.fill_(1.0)
  term.unit_strength_quantile.fill_(0.5)
  shear = torch.full((2, 2), 10.0 * count)
  zero = torch.zeros_like(shear)
  normal = torch.full_like(shear, -unit_pressure * count)
  verdict = term._judge_units(shear, zero, normal, zero, zero, zero)
  assert not verdict.violation.any() and term.unit_empirical_supported.all()


def test_zero_time_reset_does_not_count_as_preload_or_consume_release_event(affordance):
  term, data = affordance
  term.cfg.skip_zero_time_update = True
  term._update_metrics = lambda: None
  compress(data, [40.0] * 4)
  term._engage_count.fill_(1)
  term.forced_release_this_step[0, 0] = True
  term.compute(0.0)
  assert (term._engage_count == 1).all()
  assert term.forced_release_this_step[0, 0]
  assert not term.engaged.any()
  term._update_command()
  assert not term.engaged.any()
  term._update_command()
  assert term.engaged.all()


def test_contact_torque_is_shifted_from_its_centroid_and_rotated_with_the_wall(affordance):
  term, data = affordance
  compress(data, [33.0] * 4)
  # Move the reported centroid without changing the physical wrench.
  offset = torch.tensor([0.012, 0.0, -0.008])
  data.pos += offset
  data.torque -= torch.linalg.cross(offset.expand_as(data.force), data.force)
  term._unit_preload_condition()
  before = term.unit_preload_n.clone()
  basis = term._engage_frame()
  # Change the world frame and translate it far from the pad's C.
  rot = torch.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
  for field in ("force", "torque"):
    setattr(data, field, getattr(data, field) @ rot.T)
  for field in ("pos", "reference_pos"):
    setattr(data, field, getattr(data, field) @ rot.T + torch.tensor([1.0, 2.0, 3.0]))
  term._engage_frame = lambda: tuple(axis @ rot.T for axis in basis)
  assert term._unit_preload_condition().all()
  torch.testing.assert_close(term.unit_preload_n, before, atol=1e-4, rtol=1e-5)


def test_history_uses_simultaneous_application_and_reference_positions(affordance):
  term, data = affordance
  compress(data, [30.0] * 4)
  data.force_history = data.force.unsqueeze(2).repeat(1, 1, 2, 1)
  data.torque_history = data.torque.unsqueeze(2).repeat(1, 1, 2, 1)
  data.pos_history = data.pos.unsqueeze(2).repeat(1, 1, 2, 1)
  data.reference_pos_history = data.reference_pos.unsqueeze(2).repeat(1, 1, 2, 1)
  # The pad moved between samples; C must move with the contact positions.
  data.pos_history[:, :, 1, 2] += 0.1
  data.reference_pos_history[:, :, 1, 2] += 0.1
  data.force.zero_()
  data.pos.zero_()  # The live contact has disappeared.
  data.found.zero_()
  assert term._unit_preload_condition().all()
  torch.testing.assert_close(term.unit_preload_n, torch.full((2, 2, 12), 10.0), atol=1e-5, rtol=1e-5)
  # A real shift of the contact centroid, with no compensating reference
  # motion, bends the pad even though its time-averaged force is still 120 N.
  data.pos_history[:, :, 1, 2] += 0.01
  assert not term._unit_preload_condition().any()
  data.pos_history = None
  with pytest.raises(ValueError, match="synchronized"):
    term._unit_preload_condition()


def test_debounce_unloading_release_and_cooldown(affordance):
  term, data = affordance
  compress(data, [33.0] * 4)
  for _ in range(2):
    term._update_command()
    assert not term.engaged.any()
  compress(data, [0.0] * 4)
  term._update_command()
  assert not term._engage_count.any()
  compress(data, [33.0] * 4)
  for _ in range(3):
    term._update_command()
  assert term.engaged.all() and term.engaged_this_step.all()
  # Once latched, neither unloading nor stale pre-latch history drops it.
  term._update_command()
  assert not term.unit_preload_n.any()
  compress(data, [0.0] * 4)
  term._update_command()
  assert term.engaged.all() and not term.engaged_this_step.any()
  # The hold judge still owns release; preload cannot bypass its cooldown.
  term._limit_surface_release = lambda: torch.ones(2, 2, dtype=torch.bool)
  term._update_command()
  assert not term.engaged.any()
  term._limit_surface_release = lambda: torch.zeros(2, 2, dtype=torch.bool)
  compress(data, [33.0] * 4)
  for _ in range(2):
    term._update_command()
    assert not term._engage_count.any()
  for _ in range(3):
    term._update_command()
  assert term.engaged.all()
  term._resample_command(torch.tensor([0]))
  assert not term.unit_preload_n[0].any()
  assert not term.pad_contact_wrench_tln[0].any()
  assert not term.engaged[0].any() and term.engaged[1].all()


def test_wrench_sensor_records_matching_histories_and_resets_only_selected_envs():
  spec = mujoco.MjSpec()
  pad = spec.worldbody.add_body(name="pad")
  pad.add_geom(name="corner", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0])
  pad.add_site(name="C")
  spec.worldbody.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
  sensor = WrenchContactSensorCfg(
    name="contact",
    primary=ContactMatch(mode="geom", pattern="corner"),
    secondary=ContactMatch(mode="geom", pattern="wall"),
    history_length=2,
    reference_sites={"corner": "C"},
  ).build()
  sensor.edit_spec(spec, {})
  model = spec.compile()
  raw = SimpleNamespace(
    sensordata=torch.zeros(2, model.nsensordata), site_xpos=torch.zeros(2, model.nsite, 3), time=torch.zeros(2)
  )
  sensor.initialize(model, None, raw, "cpu")
  for value in (1.0, 2.0, 3.0):
    for field, scale in (("found", 1), ("force", 10), ("torque", 100), ("pos", 1000)):
      item = model.sensor(f"contact_corner_{field}")
      raw.sensordata[:, item.adr[0] : item.adr[0] + item.dim[0]] = value * scale
    raw.site_xpos[:] = -value
    sensor.update(0.002)
  for field, scale in (("force", 10), ("torque", 100), ("pos", 1000), ("reference_pos", -1)):
    history = getattr(sensor.data, field + "_history")
    torch.testing.assert_close(history[0, 0, :, 0], torch.tensor([3.0, 2.0]) * scale)
  sensor.reset(torch.tensor([0]))
  for field in ("force", "torque", "pos", "reference_pos"):
    history = getattr(sensor.data, field + "_history")
    assert not history[0].any() and history[1].abs().sum() > 0
