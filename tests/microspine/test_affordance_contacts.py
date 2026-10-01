from types import SimpleNamespace

import torch

from scr_phybot_mjab.microspine.contact_features import PadContactReader


def reader(sensor, ee_body_names, rows, cols, inward=None):
  env = SimpleNamespace(num_envs=1, device="cpu", scene={"contact": sensor})
  mcfg = SimpleNamespace(spheres_per_ee_rows=rows, spheres_per_ee_cols=cols, ee_body_names=ee_body_names)
  return PadContactReader(env, "contact", mcfg, lambda: inward)


def test_resolved_substep_contact_is_recovered_from_force_history():
  force_history = torch.zeros(1, 2, 4, 3)
  force_history[0, 0, 1, 0] = 8.0
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.zeros(1, 2),
      force=torch.zeros(1, 2, 3),
      force_history=force_history,
    ),
    primary_names=["paw_c_0_0", "paw_c_0_1"],
  )

  normal_force, found_count = reader(sensor, ("paw",), 1, 2).proxy_normal_forces()

  torch.testing.assert_close(normal_force, torch.tensor([[2.0, 0.0]]))
  torch.testing.assert_close(found_count, torch.tensor([[1.0]]))


def test_sensor_rows_permuted_to_ee_major_order():
  """The ContactSensor expands primaries in MODEL order (find_geoms), not
  pattern order — on the C2 that delivered [feet, hands] rows to an
  affordance slicing [hands, feet]: the hand engage gate read the foot
  sensors (zeroed while latched) and never fired."""
  # Sensor rows arrive foot-first; the hand's 5 N press sits in row 1.
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[0.0, 1.0]]),
      force=torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]),
      force_history=None,
    ),
    primary_names=["robot/foot_c_0_0", "robot/hand_c_0_0"],
  )

  normal_force, found_count = reader(sensor, ("hand", "foot"), 1, 1).proxy_normal_forces()

  # EE-major order (hand, foot): the press must land on the HAND.
  torch.testing.assert_close(normal_force, torch.tensor([[5.0, 0.0]]))
  torch.testing.assert_close(found_count, torch.tensor([[1.0, 0.0]]))


def test_netforce_projects_total_block_force_in_a_tilted_wall_frame():
  inward = torch.tensor([[[0.6, 0.0, 0.8]]])
  # Four simultaneous point contacts have already been summed by netforce.
  # Force history retains an earlier 20 N compression, even with no live contact.
  force = torch.tensor([[[12.0, -7.0, 16.0]]])
  history = torch.stack([torch.zeros_like(force), force], dim=2)
  sensor = SimpleNamespace(
    cfg=SimpleNamespace(reduce="netforce", global_frame=False),
    data=SimpleNamespace(
      found=torch.zeros(1, 1), force=torch.zeros_like(force), force_history=history, normal=None, tangent=None
    ),
    primary_names=["robot/paw_c_0_0"],
  )
  normal, found = reader(sensor, ("paw",), 1, 1, inward).proxy_normal_forces()
  torch.testing.assert_close(normal, torch.tensor([[10.0]]))
  torch.testing.assert_close(found, torch.ones(1, 1))


def test_sphere_geom_ids_resolve_per_ee_in_ee_major_order():
  import mujoco

  from scr_phybot_mjab.microspine.config import MicrospineCfg
  from scr_phybot_mjab.microspine.contact import resolve_sphere_geom_ids
  from scr_phybot_mjab.microspine.spec import add_contact_pairs

  cfg = MicrospineCfg(
    ee_body_names=("left_paw", "right_paw"),
    spheres_per_ee_rows=1,
    spheres_per_ee_cols=2,
    wall_geom_name="wall_geom",
  )

  spec = mujoco.MjSpec()
  wall = spec.worldbody.add_body(name="wall_body")
  wall.add_geom(name="wall_geom", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
  for ee in cfg.ee_body_names:
    body = spec.worldbody.add_body(name=f"robot/{ee}")
    body.add_freejoint()
    for i in range(2):
      child = body.add_body(name=f"robot/{ee}_c_0_{i}", pos=[0.01 * i, 0, 0])
      child.add_geom(
        name=f"robot/{ee}_c_0_{i}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.005, 0, 0],
      )
  add_contact_pairs(spec, cfg)
  model = spec.compile()

  ids = resolve_sphere_geom_ids(model, cfg)
  # One list per EE, one geom per sphere, all distinct, names match EE-major.
  assert len(ids) == 2 and all(len(ee) == 2 for ee in ids)
  flat = [g for ee in ids for g in ee]
  assert len(set(flat)) == 4
  for ee_name, ee_geoms in zip(cfg.ee_body_names, ids):
    for sphere_i, gid in enumerate(ee_geoms):
      expected = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{ee_name}_c_0_{sphere_i}")
      assert gid == expected
