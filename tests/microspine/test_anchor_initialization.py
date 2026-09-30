"""RSI anchors and actual new-latch seating use distinct lifecycles."""
from types import SimpleNamespace

import torch

from scr_phybot_mjab.microspine.constraint import MicrospineConstraint


def constraint_fixture():
  term = object.__new__(MicrospineConstraint)
  term._num_envs, term._num_ee, term._device = 2, 2, 'cpu'
  term._mcfg = SimpleNamespace(anchor_project_to_wall=False, reanchor_after_grace=True,
                              anchor_reseat_mode='down', anchor_reseat_step_cap=.001)
  term._lock_rotation = True
  term._reseat_max_age = 13
  term._eq_ids = term._ee_body_ids = torch.arange(2)
  term._check_expansion = lambda: None
  term._down_wall_dir = lambda: torch.tensor([0., 0., -1.]).expand(2, 2, -1)
  pad_pos = torch.zeros(2, 2, 3)
  pad_quat = torch.zeros(2, 2, 4)
  pad_quat[..., 0] = 1.
  command = torch.zeros(2, 2, dtype=torch.bool)
  term.pad_center_world = lambda: pad_pos
  term._asset = SimpleNamespace(data=SimpleNamespace(body_link_quat_w=pad_quat))
  term._env = SimpleNamespace(sim=SimpleNamespace(
    model=SimpleNamespace(eq_data=torch.zeros(2, 2, 11)),
    data=SimpleNamespace(eq_active=command.clone())))
  term._get_engage_command = lambda _: command
  term.engaged = command.clone()
  term.last_engage_command = command.clone()
  term.engaged_this_step = command.clone()
  term.disengaged_this_step = command.clone()
  term.current_disengage_time = torch.zeros(2, 2)
  term.anchor_pos_w = pad_pos.clone()
  term.anchor_quat_w = pad_quat.clone()
  term._engage_age = torch.zeros(2, 2, dtype=torch.long)
  term._needs_init = torch.ones(2, dtype=torch.bool)
  return term, pad_pos, pad_quat, command


def test_mature_birth_is_immediate_and_preserves_other_world_seating():
  term, positions, _, command = constraint_fixture()
  command[0, 0] = True
  term.engaged[1] = True
  term.anchor_pos_w[1] = 2.
  term._engage_age[1] = 3
  term.engaged_this_step[:] = True
  positions[0] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
  term.initialize_seeded_anchors(torch.tensor([0]))
  torch.testing.assert_close(term.anchor_pos_w[0, 0], positions[0, 0])
  assert (term.anchor_pos_w[1] == 2.).all()
  assert (term._engage_age[1] == 3).all()
  assert (term._engage_age[0] > term._reseat_max_age).all()
  assert term._env.sim.data.eq_active.tolist() == [[True, False], [True, True]]
  assert not term.engaged_this_step[0].any()
  assert term.engaged_this_step[1].all()


def test_new_latch_follows_only_capped_down_rake_then_freezes_pose():
  term, positions, quats, command = constraint_fixture()
  command[:] = True
  term._apply_state(command)
  term.engaged = command.clone()
  positions[:] = torch.tensor([.1, .2, -.05])
  quats[..., 0], quats[..., 3] = .8, .6
  for age in range(1, term._reseat_max_age + 1):
    term._apply_state(command)
    torch.testing.assert_close(term.anchor_pos_w[..., :2], torch.zeros(2, 2, 2))
    torch.testing.assert_close(term.anchor_pos_w[..., 2], torch.full((2, 2), -.001*age))
    # Existing real-latch behavior: orientation conforms in the same finite
    # window. This is separate from the 1 mm translation cap.
    torch.testing.assert_close(term.anchor_quat_w, quats)
  anchor, quat = term.anchor_pos_w.clone(), term.anchor_quat_w.clone()
  positions[:] *= 2
  quats[..., 0], quats[..., 3] = .6, .8
  term._apply_state(command)
  torch.testing.assert_close(term.anchor_pos_w, anchor)
  torch.testing.assert_close(term.anchor_quat_w, quat)


def test_mature_birth_does_not_reseat_translation_or_orientation():
  term, positions, quats, command = constraint_fixture()
  command[:] = True
  term.initialize_seeded_anchors(torch.arange(2))
  anchor, quat = term.anchor_pos_w.clone(), term.anchor_quat_w.clone()
  positions[..., 2] = -.01
  quats[..., 0], quats[..., 3] = .8, .6
  term._apply_state(command)
  torch.testing.assert_close(term.anchor_pos_w, anchor)
  torch.testing.assert_close(term.anchor_quat_w, quat)
