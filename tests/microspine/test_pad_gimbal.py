"""CAD pads: distinct pivot/contact frames, inertia and passive joint behavior."""

import math

import mujoco
import numpy as np
import pytest

from scr_phybot_mjab.assets.phybot_c2 import phybot_c2_constants as C

GIMBAL_JOINTS = tuple(f"{spine}_gimbal_{axis}" for spine in C.SPINE_BODY_NAMES for axis in ("x", "y", "z"))


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  assert C.PHYBOT_C2_PAD_GIMBAL, "suite expects the gimbal-era plant"
  return C.get_spec().compile()


def test_gimbal_joints_present_with_bench_params(model):
  travel = C.PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD
  for name in GIMBAL_JOINTS:
    j = model.joint(name)
    assert j.type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert np.allclose(j.range, [-travel, travel])
    assert j.stiffness[0] == pytest.approx(C.PHYBOT_C2_PAD_GIMBAL_STIFFNESS)
    assert j.damping[0] == pytest.approx(C.PHYBOT_C2_PAD_GIMBAL_DAMPING)
    assert model.jnt_limited[j.id] == 1
    assert tuple(model.jnt_solref[j.id]) == pytest.approx(C.PHYBOT_C2_PAD_GIMBAL_STOP_SOLREF)
    # Spring zero IS the mounting orientation.
    assert model.qpos_spring[model.jnt_qposadr[j.id]] == pytest.approx(0.0)
  for spine in C.SPINE_BODY_NAMES:
    body = model.body(spine)
    assert body.mass[0] == pytest.approx(C.PHYBOT_C2_PAD_MASS)
    assert (body.inertia > 0.0).all()


def test_gimbal_pivot_and_contact_array_have_measured_offset(model):
  for spine in C.SPINE_BODY_NAMES:
    spheres = [model.body(f"{spine}_c_{i}_{j}") for i in range(2) for j in range(2)]
    centroid = np.mean([b.pos for b in spheres], axis=0)
    proxy = model.geom(f"{spine}_c_0_0")
    assert proxy.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    contact = centroid + np.array([0.0, -proxy.size[1], 0.0])
    pivot = model.site(f"{spine}_ball_center").pos
    assert np.allclose(pivot, 0, atol=1e-12)
    assert np.allclose(contact - pivot, C.PHYBOT_C2_PAD_GEOMETRY["contact_center_in_pad_m"], atol=5e-8)
    assert np.allclose(model.site(f"{spine}_contact_center").pos, contact, atol=5e-8)
    units = np.array([model.site(f"{spine}_unit_{row}_{col}").pos for row in range(3) for col in range(4)])
    assert np.allclose(units.mean(axis=0), contact, atol=5e-8)
    grid = units.reshape(3, 4, 3)
    assert np.allclose(np.diff(grid[:, :, 0], axis=1), 0.028, atol=5e-8)
    assert np.allclose(np.diff(grid[:, :, 2], axis=0), 0.0315, atol=5e-8)
    for axis in ("x", "y", "z"):
      j = model.joint(f"{spine}_gimbal_{axis}")
      assert np.allclose(j.pos, pivot, atol=1e-12), (spine, axis)


def test_forearm_extension_and_unchanged_leg_pivots(model):
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  for side in ("left", "right"):
    forearm = model.body(f"{side}_elbow_pitch")
    rot = d.xmat[forearm.id].reshape(3, 3)
    b = d.site_xpos[model.site(f"{side}_hand_spine_ball_center").id]
    e = d.site_xpos[model.site(f"{side}_forearm_end").id]
    # CAD sleeve stop z=35 mm; seat centre y=12.25, z=90.5 mm.
    assert np.allclose(rot.T @ (b - e), [0.01225, 0, -0.0555], atol=1e-10)
    assert np.allclose(rot.T @ (e - d.xpos[forearm.id]), [0, 0, -0.188], atol=1e-7)
    assert np.allclose(model.body(f"{side}_foot_spine").pos, [0.068, 0, -0.15935], atol=1e-12)
    mount = model.body(f"{side}_forearm_pad_mount")
    assert mount.parentid[0] == forearm.id
    assert model.body(f"{side}_hand_spine").parentid[0] == forearm.id
    assert mount.mass[0] == pytest.approx(C.PHYBOT_C2_PAD_GEOMETRY["forearm_mount"]["mass_kg"])


def test_contact_center_moves_about_fixed_ball_center(model):
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  spine = "left_hand_spine"
  b_id = model.site(f"{spine}_ball_center").id
  c_id = model.site(f"{spine}_contact_center").id
  pivot = d.site_xpos[b_id].copy()
  contact = d.site_xpos[c_id].copy()
  d.qpos[model.jnt_qposadr[model.joint(f"{spine}_gimbal_x").id]] = math.radians(3)
  mujoco.mj_forward(model, d)
  assert np.allclose(d.site_xpos[b_id], pivot, atol=1e-12)
  assert np.linalg.norm(d.site_xpos[c_id] - contact) > 0.004
  assert np.linalg.norm(d.site_xpos[c_id] - pivot) == pytest.approx(
    np.linalg.norm(C.PHYBOT_C2_PAD_GEOMETRY["contact_center_in_pad_m"])
  )


def test_dummy_hand_is_removed_from_the_mesh_not_just_hidden():
  import trimesh

  for side in ("left", "right"):
    path = C.PHYBOT_C2_XML.parent.parent / "new_meshes" / f"{side}_forearm_no_hand.stl"
    mesh = trimesh.load_mesh(path)
    assert mesh.bounds[0, 2] == pytest.approx(-0.188, abs=1e-7)
    assert mesh.is_watertight
    assert len(mesh.faces) < 20_000  # Original dummy-hand assemblies had >52k.


def test_gimbal_zero_matches_legacy_fixed_pose(monkeypatch):
  """Default-pose FK identical to the rigid-attachment plant (1e-12 m)."""

  def sphere_positions(enable: bool) -> dict[str, np.ndarray]:
    monkeypatch.setattr(C, "PHYBOT_C2_PAD_GIMBAL", enable)
    m = C.get_spec().compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return {m.body(b).name: d.xpos[b].copy() for b in range(m.nbody) if "_spine_c_" in m.body(b).name}

  legacy = sphere_positions(False)
  gimbal = sphere_positions(True)
  assert set(legacy) == set(gimbal)
  for name, pos in legacy.items():
    assert np.allclose(pos, gimbal[name], atol=1e-12), name


def _bench_model() -> mujoco.MjModel:
  """Fresh model with gravity and contacts off: pure spring/stop dynamics."""
  m = C.get_spec().compile()
  m.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY | mujoco.mjtDisableBit.mjDSBL_CONTACT
  return m


def test_gimbal_recenters_when_unloaded():
  """Deflected 3 deg with no load, the pad returns to zero without ringing."""
  m = _bench_model()
  data = mujoco.MjData(m)
  j = m.joint("left_hand_spine_gimbal_x")
  adr = m.jnt_qposadr[j.id]
  dof = m.jnt_dofadr[j.id]
  data.qpos[adr] = math.radians(3.0)
  overshoot = 0.0
  for _ in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, data)
    overshoot = max(overshoot, -float(data.qpos[adr]))
  assert abs(data.qpos[adr]) < math.radians(0.1), "did not recenter in 0.5 s"
  assert abs(data.qvel[dof]) < 0.05
  assert overshoot < math.radians(1.5), "underdamped: recenter rings"


def test_gimbal_hard_stops_hold():
  """A deflection past the stop settles back inside the +-4 deg travel."""
  m = _bench_model()
  data = mujoco.MjData(m)
  j = m.joint("right_foot_spine_gimbal_z")
  adr = m.jnt_qposadr[j.id]
  data.qpos[adr] = math.radians(6.0)
  for _ in range(int(0.2 / m.opt.timestep)):
    mujoco.mj_step(m, data)
  assert abs(data.qpos[adr]) <= C.PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD + 1e-3
