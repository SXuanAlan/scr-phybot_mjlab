"""Physical contact checks for the four CAD-sized corner blocks."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import pad_contact_center, pad_contact_face_centers, sphere_names
from scr_phybot_mjab.microspine.pad_proxies import ProxyWallGaps
from scr_phybot_mjab.microspine.spec import add_attachment_equalities


@pytest.fixture(scope="module")
def bench():
  asset = get_spec().compile()
  names = tuple(f"{side}_{limb}_spine" for limb in ("hand", "foot") for side in ("left", "right"))
  cfg = MicrospineCfg(ee_body_names=names, spheres_per_ee_rows=2, spheres_per_ee_cols=2)
  spec = mujoco.MjSpec()
  spec.option.gravity = [0, -9.81, 0]
  # Wall outward normal +y; its local -x face remains the task convention.
  q = [2**-0.5, 0, 0, -(2**-0.5)]
  wall = spec.worldbody.add_body(name="wall/wall_body", pos=[0, -0.5, 0], quat=q)
  wall.add_geom(
    name="wall/wall_geom",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=[-0.5, 0, 0],
    quat=[2**-0.5, 0, -(2**-0.5), 0],
    size=[0.5, 0.6, 0.1],
  )
  for i, ee in enumerate(names):
    contact = asset.site(ee + "_contact_center").pos
    body = spec.worldbody.add_body(name="robot/" + ee, pos=[0.25 * i, -contact[1] - 0.0002, 0])
    body.add_freejoint()
    for name in sphere_names(ee, 2, 2):
      child = body.add_body(name="robot/" + name, pos=asset.body(name).pos)
      child.add_geom(
        name="robot/" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=asset.geom(name).size, contype=0, conaffinity=0
      )
      spec.add_pair(geomname1="robot/" + name, geomname2="wall/wall_geom", condim=3, margin=0, gap=0)
      spec.add_sensor(
        name="force_" + name,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,
        objname="robot/" + name,
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="wall/wall_geom",
        intprm=[23, 3, 1],
      )  # found + force + torque + pos; netforce
  add_attachment_equalities(spec, cfg)
  model = spec.compile()
  return asset, cfg, model, q


def test_box_faces_center_on_c(bench):
  asset, cfg, _, _ = bench
  for ee in cfg.ee_body_names:
    points = pad_contact_face_centers(asset, cfg, ee, robot_prefix="")
    c = asset.site(ee + "_contact_center").pos
    np.testing.assert_allclose(points.mean(0), c, atol=5e-8)
    for i, (row, col) in enumerate(((0, 0), (0, 3), (2, 0), (2, 3))):
      np.testing.assert_allclose(points[i], asset.site(f"{ee}_unit_{row}_{col}").pos, atol=1e-10)


def test_authored_c_is_used_instead_of_rounding_the_corner_centroid():
  spec = get_spec()
  ee = "left_hand_spine"
  cfg = MicrospineCfg(ee_body_names=(ee,), spheres_per_ee_rows=2, spheres_per_ee_cols=2)
  centre = spec.site(ee + "_contact_center").pos.copy()
  # A slightly unequal proxy grid must not change the declared wrench frame.
  spec.body(ee + "_c_0_0").pos += [0.0001, 0, 0]
  np.testing.assert_array_equal(pad_contact_center(spec, cfg, ee, ""), centre)


def test_netforce_sensor_sums_every_point_on_a_flat_block(bench):
  _, cfg, model, _ = bench
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for ee in cfg.ee_body_names:
    for name in sphere_names(ee, 2, 2):
      gid = model.geom("robot/" + name).id
      adr = model.sensor("force_" + name).adr[0]
      observed = data.sensordata[adr : adr + 4]
      forces = []
      for i, contact in enumerate(data.contact):
        if gid not in contact.geom:
          continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        direction = 1 if contact.geom[0] == gid else -1
        forces.append(direction * contact.frame.reshape(3, 3).T @ force[:3])
      assert observed[0] == len(forces) == 4
      np.testing.assert_allclose(observed[1:], np.sum(forces, axis=0), atol=1e-9)
      assert -observed[2] > 0  # Compression points primary -> wall (-y).


def test_netforce_centroid_and_torque_recover_the_contact_moment_at_c(bench):
  asset, cfg, model, _ = bench
  data = mujoco.MjData(model)
  # A small tilt and sliding produce uneven corner pressure and shear.
  for i in range(4):
    data.qpos[7 * i + 3 : 7 * i + 7] = [np.cos(0.003), np.sin(0.003), 0, 0]
    data.qvel[6 * i : 6 * i + 3] = [0.2, 0, -0.1]
  mujoco.mj_forward(model, data)
  for ee in cfg.ee_body_names:
    bid = model.body("robot/" + ee).id
    center = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ asset.site(ee + "_contact_center").pos
    for name in sphere_names(ee, 2, 2):
      gid = model.geom("robot/" + name).id
      adr = model.sensor("force_" + name).adr[0]
      observed = data.sensordata[adr : adr + 10]
      moment = np.zeros(3)
      for i, contact in enumerate(data.contact):
        if gid not in contact.geom:
          continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        direction = 1 if contact.geom[0] == gid else -1
        rotation = direction * contact.frame.reshape(3, 3).T
        moment += rotation @ local[3:] + np.cross(contact.pos - center, rotation @ local[:3])
      reconstructed = observed[4:7] + np.cross(observed[7:10] - center, observed[1:4])
      np.testing.assert_allclose(reconstructed, moment, atol=1e-9)


@pytest.mark.parametrize("angle_deg", [0.0, 3.0, 25.0])
def test_box_wall_gap_matches_physical_contacts_and_retraction(bench, angle_deg):
  _, cfg, model, wall_quat = bench
  data = mujoco.MjData(model)
  angle = np.deg2rad(angle_deg)
  for i in range(4):
    data.qpos[i * 7 + 3 : i * 7 + 7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
  mujoco.mj_forward(model, data)
  # Seat the closest edge only 0.2 mm into the wall at every angle. Rotating
  # about B without translating would start the 25-degree case 5 cm inside it.
  for i, ee in enumerate(cfg.ee_body_names):
    gids = [model.geom("robot/" + name).id for name in sphere_names(ee, 2, 2)]
    nearest = min(data.geom_xpos[g, 1] - np.abs(data.geom_xmat[g].reshape(3, 3)[1]) @ model.geom_size[g] for g in gids)
    data.qpos[i * 7 + 1] += -0.0002 - nearest
  mujoco.mj_forward(model, data)
  env = SimpleNamespace(
    device="cpu",
    sim=SimpleNamespace(
      mj_model=model,
      data=SimpleNamespace(
        geom_xpos=torch.tensor(data.geom_xpos, dtype=torch.float32).unsqueeze(0),
        geom_xmat=torch.tensor(data.geom_xmat, dtype=torch.float32).unsqueeze(0),
      ),
    ),
    scene={
      "wall": SimpleNamespace(
        data=SimpleNamespace(
          root_link_pos_w=torch.tensor([[0.0, -0.5, 0.0]]),
          root_link_quat_w=torch.tensor([wall_quat]),
        )
      )
    },
  )
  proxy_gaps = ProxyWallGaps(model, cfg, "cpu")
  gaps = proxy_gaps(env)[0].numpy()
  ids = proxy_gaps.geom_ids.tolist()
  for gid, gap in zip(ids, gaps):
    contacts = [c.dist for c in data.contact if gid in c.geom]
    if contacts:
      assert gap == pytest.approx(min(contacts), abs=3e-7)  # float32 world transforms
  if angle_deg == 0:
    assert data.ncon == 64  # Four contact points per block, 16 blocks total.
    np.testing.assert_allclose(gaps, -0.0002, atol=1e-7)
  # The executor retracts geom_pos, leaving the four surface-reference body
  # frames in place. At these poses 50 mm must clear every box corner.
  before = data.xpos.copy()
  original = model.geom_pos[ids].copy()
  try:
    model.geom_pos[ids, 1] += 0.05
    model.geom_sameframe[ids] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    np.testing.assert_allclose(data.xpos, before)
  finally:
    model.geom_pos[ids] = original
