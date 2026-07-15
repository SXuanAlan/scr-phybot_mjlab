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
