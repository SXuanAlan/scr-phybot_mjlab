"""Moving grips must conserve reaction forces and use body-relative anchors."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from scr_phybot_mjab.microspine.config import MicrospineCfg, MicrospineSurfaceCfg
from scr_phybot_mjab.microspine.constraint import weld_rotational_force_to_world_moment
from scr_phybot_mjab.microspine.surfaces import MicrospineSurfaces, conjugate, multiply


def quat(axis, angle):
  q = np.zeros(4)
  q[0], q[1:] = np.cos(angle / 2), np.asarray(axis) * np.sin(angle / 2)
  return q


def dynamic_pair():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep="0.001" jacobian="dense"/>
    <worldbody>
      <body name="pad" pos="0 0 1"><freejoint/><geom size=".03" mass="1" contype="0" conaffinity="0"/></body>
      <body name="brick" pos=".1 0 1"><freejoint/><geom size=".03" mass="3" contype="0" conaffinity="0"/></body>
    </worldbody>
    <equality><weld body1="pad" body2="brick" solref=".005 1" torquescale=".7"/></equality>
  </mujoco>""")
  return model, mujoco.MjData(model)


def test_two_free_bodies_move_together_then_separate_on_release():
  m, d = dynamic_pair()
  d.xfrc_applied[m.body("brick").id, 0] = 4
  for _ in range(200):
    mujoco.mj_step(m, d)
  assert d.qpos[7] > 0.115
  assert d.qpos[9] < 0.85  # Neither body has been fixed to the world.
  np.testing.assert_allclose(d.qpos[7:10] - d.qpos[:3], [0.1, 0, 0], atol=0.001)
  np.testing.assert_allclose(d.qfrc_constraint[:3] + d.qfrc_constraint[6:9], 0, atol=1e-9)
  d.eq_active[:] = False
  d.xfrc_applied[m.body("pad").id, 0] = -20
  for _ in range(150):
    mujoco.mj_step(m, d)
  assert d.qpos[7] - d.qpos[0] > 0.2


@pytest.mark.parametrize("angle", [0.0, 0.8, 2.1])
def test_weld_moment_matches_mujoco_jacobian_in_rotated_target_frame(angle):
  m, d = dynamic_pair()
  q2 = quat([0, 0, 1], angle)
  relative = quat([1, 0, 0], 0.4)
  q1 = multiply(torch.tensor(q2), torch.tensor(relative)).numpy()
  # Nonzero orientation error tests the cross product term, not only the identity.
  q1 = multiply(torch.tensor(quat([0, 1, 0], 0.13)), torch.tensor(q1)).numpy()
  d.qpos[3:7], d.qpos[10:14] = q1, q2
  m.eq_data[0, 6:10] = conjugate(torch.tensor(relative)).numpy()
  mujoco.mj_forward(m, d)
  multipliers = np.array([0.7, -1.2, 0.9])
  jac = d.efc_J.reshape(d.nefc, m.nv)
  jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  mujoco.mj_jacBody(m, d, jp, jr, m.body("pad").id)
  expected = np.linalg.solve(jr[:, 3:6].T, jac[3:6, 3:6].T @ multipliers)
  anchor = multiply(torch.tensor(q2), torch.tensor(relative))
  actual = weld_rotational_force_to_world_moment(
    torch.tensor(multipliers), torch.tensor(q1), anchor, 0.7, torch.tensor(q2)
  )
  np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10)


def surface_fixture():
  surface = object.__new__(MicrospineSurfaces)
  pos = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
  q = torch.tensor(quat([0, 0, 1], np.pi / 2), dtype=torch.float32).expand(1, 2, 4)
  surface.pose = lambda: (pos, q)
  surface.local_point = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
  surface.local_normal = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
  surface.local_tangent = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  surface.extents = torch.tensor([[0.1, 0.08], [0.1, 0.08]])
  return surface


def test_opposing_faces_rotate_with_brick_and_reject_overhanging_footprint():
  s = surface_fixture()
  p, n, u, v = s.frame()
  torch.testing.assert_close(n[0, 0], -n[0, 1])
  vertices = p[:, :, None] + 0.09 * u[:, :, None] + 0.07 * v[:, :, None]
  assert s.contains(vertices).all()
  vertices[:, 0] += 0.02 * u[:, 0, None]
  assert s.contains(vertices).tolist() == [[False, True]]


def test_velocity_includes_rotation_at_contact_point():
  s = object.__new__(MicrospineSurfaces)
  p, q = torch.zeros(1, 1, 3), torch.tensor([[[1.0, 0, 0, 0]]])
  s.pose = lambda: (p, q)
  data = SimpleNamespace(
    body_link_lin_vel_w=torch.tensor([[[1.0, 0, 0]]]), body_link_ang_vel_w=torch.tensor([[[0.0, 0, 2.0]]])
  )
  s.targets = [(SimpleNamespace(data=data), 0)]
  torch.testing.assert_close(s.velocity_at(torch.tensor([[[0.0, 0.2, 0.0]]])), torch.tensor([[[0.6, 0, 0]]]))


def test_surface_bindings_require_one_face_per_hand():
  surface = MicrospineSurfaceCfg(
    ee_body_name="left",
    entity_name="brick",
    body_name="brick",
    geom_name="box",
    point=(0, 0.2, 0),
    normal=(0, 1, 0),
    tangent=(1, 0, 0),
    half_extents=(0.1, 0.1),
  )
  cfg = MicrospineCfg(
    ee_body_names=("left", "right"), spheres_per_ee_rows=2, spheres_per_ee_cols=2, surfaces=(surface,)
  )
  with pytest.raises(ValueError, match="every configured EE"):
    cfg.ordered_surfaces()


def test_surface_constructor_validates_real_mujoco_body_ids():
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="brick/brick"><geom name="brick/box" size=".1"/></body>
    </worldbody></mujoco>""")
  surface = MicrospineSurfaceCfg(
    ee_body_name="left",
    entity_name="brick",
    body_name="brick",
    geom_name="box",
    point=(0, 0.1, 0),
    normal=(0, 1, 0),
    tangent=(1, 0, 0),
    half_extents=(0.1, 0.1),
  )
  cfg = MicrospineCfg(ee_body_names=("left",), spheres_per_ee_rows=2, spheres_per_ee_cols=2, surfaces=(surface,))
  entity = SimpleNamespace(find_bodies=lambda _: ([0], ["brick"]))
  env = SimpleNamespace(scene={"brick": entity}, sim=SimpleNamespace(mj_model=model), device="cpu")
  binding = MicrospineSurfaces(env, cfg)
  assert binding.targets == [(entity, 0)]
