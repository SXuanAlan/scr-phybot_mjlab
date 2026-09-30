"""Release rewards use pre-release unit loads and cannot pay latch churn."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units_numpy
from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import microspine_release_load_reward


def release_env(tangent_loads, *, count=4, dt=0.02):
  forces = torch.tensor(tangent_loads, dtype=torch.float32)
  batch = len(forces)
  loads = torch.zeros(batch, 4, count, 3)
  loads[:, 0, :, :2] = forces[:, None, :]
  # Other pads continue carrying weight; the reward must only price releases.
  loads[:, 1:, :, 0] = 25.0
  aff = SimpleNamespace(
    unit_load_tln=loads,
    engaged=torch.tensor([[False, True, True, True]]).expand(batch, -1).clone(),
    forced_release_this_step=torch.tensor([[True, False, False, False]]).expand(batch, -1).clone(),
    release_reason=torch.tensor([[2, 0, 0, 0]]).expand(batch, -1).clone(),
    dr_release_pending=torch.zeros(batch, 4, dtype=torch.bool),
  )
  motion = SimpleNamespace(
    ref_engaged=aff.engaged.clone(),
    ref_edge_countdowns=torch.ones(batch, 4, 2),
    cycle_time_s=5.0,
  )
  terms = {"microspine_affordance": aff, "climb_motion": motion}
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=terms.__getitem__),
    episode_length_buf=torch.full((batch,), 100),
    step_dt=dt,
  )
  return env, aff, motion


def event_score(env, **params):
  return microspine_release_load_reward()(env, **params) * env.step_dt


@pytest.mark.parametrize("count", [4, 12])
def test_near_zero_release_pays_and_both_shear_signs_cost_more_than_lateral(count):
  env, aff, _ = release_env([[0, 0], [0.5, 0.5], [4, 0], [-4, 0], [0, 4], [0, -4]], count=count)
  aff.release_reason[3, 0] = 1  # Loaded reverse-shear unhooks are not free.
  score = event_score(env)
  assert score[0].item() == pytest.approx(1.0)
  assert 0 < score[1] < score[0]
  assert score[2] < score[4] < 0
  torch.testing.assert_close(score[2], score[3])
  torch.testing.assert_close(score[4], score[5])


def test_zero_pad_resultant_cannot_hide_loaded_units_under_twist():
  from scr_phybot_mjab.tools.microspine.geometry import phybot_pad_geometry

  geometry = phybot_pad_geometry().select_layout("corners")
  loads = distribute_wrench_to_units_numpy([0, 0, 0, 0, 0, 1.0], geometry)
  np.testing.assert_allclose(loads.sum(axis=0), 0.0, atol=1e-8)
  env, aff, _ = release_env([[0, 0], [0, 0]])
  aff.unit_load_tln[1, 0] = torch.tensor(loads, dtype=torch.float32)
  score = event_score(env)
  assert score[0] > 0 and score[1] < 0


def test_normal_force_is_not_an_unloading_target():
  env, aff, _ = release_env([[0, 0]] * 3)
  aff.unit_load_tln[:, 0, :, 2] = torch.tensor([-20.0, 0.0, 5.0])[:, None]
  torch.testing.assert_close(event_score(env), torch.ones(3))


def test_bonus_needs_a_real_scheduled_release_and_three_remaining_supports():
  env, aff, motion = release_env([[0, 0]] * 5)
  aff.forced_release_this_step[0] = False  # No transition, even at zero load.
  motion.ref_engaged[1, 0] = True  # Still scheduled to hold.
  motion.ref_engaged[2, 0] = True
  motion.ref_edge_countdowns[2, 0, 0] = 0.04  # 0.2 s before RELEASE: eligible.
  motion.ref_engaged[3, 0] = True
  motion.ref_edge_countdowns[3, 0, 1] = 0.0  # Near regrip, not release.
  aff.engaged[4, 1] = False  # Only two other pads remain.
  assert event_score(env).tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0])


def test_off_schedule_and_unsafe_loaded_releases_are_still_penalized():
  env, aff, motion = release_env([[4, 0]] * 3)
  motion.ref_engaged[0, 0] = True
  aff.engaged[1, 1] = False
  aff.forced_release_this_step[2] = False
  score = event_score(env)
  assert score[0] < 0 and score[1] < 0 and score[2] == 0


def test_dr_events_and_recovery_do_not_pay_and_empirical_failures_are_not_bonuses():
  env, aff, _ = release_env([[20, 0], [0, 0], [0, 0], [0, 0]])
  aff.release_reason[:, 0] = torch.tensor([6, 7, 8, 2])
  aff.dr_release_pending[3, 1] = True
  torch.testing.assert_close(event_score(env), torch.zeros(4))


def test_one_bonus_spans_release_lead_and_swing_and_rearms_next_cycle():
  env, aff, motion = release_env([[0, 0]])
  reward = microspine_release_load_reward()
  motion.ref_engaged[0, 0] = True
  motion.ref_edge_countdowns[0, 0, 0] = 0.04
  assert (reward(env) * env.step_dt).item() == pytest.approx(1.0)
  motion.ref_engaged[0, 0] = False
  # Another falling edge in the same OFF window must not earn another bonus.
  assert reward(env).item() == 0.0
  aff.unit_load_tln[0, 0, :, 0] = 4.0
  assert reward(env).item() < 0.0  # Repeat loaded releases still cost.
  # Close the swing and enter a normal hold, then the next release window.
  aff.forced_release_this_step.zero_()
  motion.ref_engaged[0, 0] = True
  motion.ref_edge_countdowns[0, 0, 0] = 1.0
  assert reward(env).item() == 0.0
  aff.forced_release_this_step[0, 0] = True
  aff.unit_load_tln[0, 0, :, 0] = 0.0
  motion.ref_engaged[0, 0] = False
  assert (reward(env) * env.step_dt).item() == pytest.approx(1.0)


def test_a_bad_first_release_also_consumes_the_window_bonus():
  env, aff, _ = release_env([[4, 0]])
  reward = microspine_release_load_reward()
  assert reward(env).item() < 0.0
  aff.unit_load_tln[0, 0, :, 0] = 0.0
  assert reward(env).item() == 0.0


def test_partial_reset_and_rsi_do_not_open_an_existing_window_for_bonus():
  env, aff, motion = release_env([[0, 0], [0, 0]])
  reward = microspine_release_load_reward()
  torch.testing.assert_close(reward(env) * env.step_dt, torch.ones(2))
  env.episode_length_buf[0] = 0
  assert reward(env).eq(0).all()
  env.episode_length_buf[0] = 100
  assert reward(env).eq(0).all()  # Born in this swing: still consumed.
  aff.forced_release_this_step[0] = False
  motion.ref_engaged[0, 0] = True
  assert reward(env).eq(0).all()
  motion.ref_engaged[0, 0] = False
  aff.forced_release_this_step[0, 0] = True
  assert (reward(env) * env.step_dt).tolist() == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_event_amount_is_independent_of_policy_dt(dt):
  env, _, _ = release_env([[0, 0], [4, 0]], dt=dt)
  score = event_score(env)
  assert score.tolist() == pytest.approx([1.0, np.exp(-4) - 1.0])


def test_nonfinite_or_extreme_loads_stay_bounded_and_never_pay():
  env, aff, _ = release_env([[float("nan"), 0], [float("inf"), 0], [1e20, 0], [0, 0]])
  aff.unit_load_tln[3, 0, :, 2] = float("nan")
  score = event_score(env)
  assert torch.isfinite(score).all()
  torch.testing.assert_close(score, torch.full((4,), -4.0))
