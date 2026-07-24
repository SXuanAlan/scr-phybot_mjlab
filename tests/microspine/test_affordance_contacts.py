from types import SimpleNamespace

import torch

from scr_phybot_mjab.microspine.affordance import MicrospineAffordance


def test_resolved_substep_contact_is_recovered_from_force_history():
  term = object.__new__(MicrospineAffordance)
  term._num_ee = 1
  term._mcfg = SimpleNamespace(spheres_per_ee_rows=1, spheres_per_ee_cols=2)
  term.cfg = SimpleNamespace(contact_sensor_name="contact")

  force_history = torch.zeros(1, 2, 4, 3)
  force_history[0, 0, 1, 0] = 8.0
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.zeros(1, 2),
      force=torch.zeros(1, 2, 3),
      force_history=force_history,
    )
  )
  term._env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"contact": sensor},
  )

  normal_force, found_count = term._sensor_contact_features()

  torch.testing.assert_close(normal_force, torch.tensor([[2.0, 0.0]]))
  torch.testing.assert_close(found_count, torch.tensor([[1.0]]))


def test_sphere_geom_ids_resolve_per_ee_in_ee_major_order():
  import mujoco

  from scr_phybot_mjab.microspine.config import MicrospineCfg
  from scr_phybot_mjab.microspine.contact import (
    add_contact_pairs,
    resolve_sphere_geom_ids,
  )

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
      expected = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{ee_name}_c_0_{sphere_i}"
      )
      assert gid == expected
