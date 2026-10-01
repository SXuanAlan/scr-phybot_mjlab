"""The latch state machine's pure tensor steps and the engage-frame geometry."""

import torch

from scr_phybot_mjab.microspine import latch_state
from scr_phybot_mjab.microspine.engage_gate import engage_frame, face_wall_cos, to_engage_frame


def test_count_consecutive_restarts_on_any_failed_step():
  count = torch.tensor([0, 2, 5])
  assert latch_state.count_consecutive(torch.tensor([True, True, False]), count).tolist() == [1, 3, 0]


def test_hold_guard_waits_longer_only_for_seeded_latches():
  age = torch.tensor([0, 1, 3, 3])
  seeded = torch.tensor([False, False, False, True])
  assert latch_state.in_hold_guard(age, seeded, 2, None).tolist() == [True, True, False, False]
  assert latch_state.in_hold_guard(age, seeded, 2, 6).tolist() == [True, True, False, True]


def test_engage_wins_over_release_and_release_clears():
  engaged = torch.tensor([False, True, True, False])
  engage = torch.tensor([True, True, False, False])
  release = torch.tensor([False, True, True, True])
  assert latch_state.next_latch_state(engaged, engage, release).tolist() == [True, True, False, False]


def test_cooldown_counts_down_and_restarts_on_release():
  cooldown = torch.tensor([0, 1, 3])
  released = torch.tensor([True, False, False])
  assert latch_state.tick_cooldown(cooldown, released, 4).tolist() == [4, 0, 2]
  assert latch_state.tick_cooldown(cooldown, released, 0).tolist() == [0, 0, 2]


def test_engage_age_starts_at_zero_and_is_zero_while_released():
  age = torch.tensor([5, 5, 5])
  engaged = torch.tensor([True, True, False])
  engage = torch.tensor([True, False, False])
  assert latch_state.advance_engage_age(age, engaged, engage).tolist() == [0, 6, 0]


def test_engage_frame_is_right_handed_and_falls_back_for_a_perpendicular_rake():
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  # Wall normal +y (outward). The -z rake lies in the wall plane; a -y rake
  # is perpendicular to it and falls back to world -z.
  n_out = torch.tensor([[[0.0, 1.0, 0.0]]])
  for rake in ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)):
    t_hat, l_hat, n_hat = engage_frame([identity], torch.tensor(rake), n_out)
    torch.testing.assert_close(t_hat, torch.tensor([[[0.0, 0.0, -1.0]]]))
    torch.testing.assert_close(torch.linalg.cross(n_hat, t_hat), l_hat)
    torch.testing.assert_close(n_hat, n_out)
  load = torch.tensor([[[3.0, 4.0, -5.0]]])
  f_t, f_l, f_n = to_engage_frame(load, (t_hat, l_hat, n_hat))
  assert (f_t.item(), f_l.item(), f_n.item()) == (5.0, -3.0, 4.0)


def test_face_wall_cos_per_wall_and_per_pad():
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  # A quarter turn about z maps the -y face axis to +x.
  turned = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
  face = torch.tensor([0.0, -1.0, 0.0])
  wall = torch.tensor([[0.0, -1.0, 0.0]])
  torch.testing.assert_close(face_wall_cos([identity, turned], face, wall), torch.tensor([[1.0, 0.0]]))
  per_pad = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]])
  torch.testing.assert_close(face_wall_cos([identity, turned], face, per_pad), torch.tensor([[1.0, 1.0]]))
