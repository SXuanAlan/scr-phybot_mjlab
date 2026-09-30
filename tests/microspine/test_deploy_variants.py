"""Contracts for the deploy PD gains, the s2r DR package and forced releases."""

from types import SimpleNamespace

import pytest
import torch

from scr_phybot_mjab.microspine.affordance import (
  RELEASE_REASON_DR_FORCED,
  MicrospineAffordance,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.events import random_microspine_release

_DR_EVENTS = ("body_friction", "base_mass")


def _gains(cfg):
  return {
    act.target_names_expr[0]: (act.stiffness, act.damping)
    for act in cfg.scene.entities["robot"].articulation.actuators
  }


def test_default_gains_are_pd100():
  gains = _gains(phybot_c2_climb_wall_env_cfg(play=False))
  # pd100 table (2026-08-02, v4 revival): the shipped lineage (v3/model_3300
  # -> deploypd-v4) trained on the full upper chain INCLUDING shoulder_yaw
  # at 100/10 and legs at 100/10; the robot aligns its drivers to these.
  for expr in (
    "waist_yaw",
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
    ".*_elbow_pitch",
  ):
    assert gains[expr] == (100.0, 10.0)
  for expr in (".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw", ".*_knee"):
    assert gains[expr] == (100.0, 10.0)
  for expr in (".*_ankle_pitch", ".*_ankle_roll"):
    assert gains[expr] == (50.0, 5.0)
  # Target-space safety contract survives the variant.
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  assert cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor == 0.98


def test_domain_rand_switch_turns_the_package_off():
  cfg = phybot_c2_climb_wall_env_cfg(play=False, domain_rand=False)
  # Gains are a fixed property of the task, not tied to the DR switch.
  assert _gains(cfg)["waist_yaw"] == (100.0, 10.0)
  for name in _DR_EVENTS:
    assert name not in cfg.events
  assert cfg.observations["actor"].enable_corruption is False
  assert "student" not in cfg.observations


def test_empirical_default_preserves_noncontact_dr():
  # Measured-distribution capacity replaces the old contact randomization;
  # observation, encoder, friction and mass randomization remain independent.
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  for name in ("body_friction", "base_mass", "encoder_bias"):
    assert name in cfg.events
  assert cfg.events["base_mass"].params["ranges"] == (-2.0, 2.0)
  assert cfg.events["base_mass"].params["operation"] == "add"
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
  assert cfg.events["encoder_bias"].mode == "startup"
  assert cfg.observations["actor"].enable_corruption is True
  expected_noise = {
    "base_lin_vel": 0.5,
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "joint_pos_rel": 0.01,
    "joint_vel_rel": 1.5,
  }
  for term, half_range in expected_noise.items():
    noise = cfg.observations["actor"].terms[term].noise
    assert noise is not None
    assert (noise.n_min, noise.n_max) == (-half_range, half_range)
  assert "random_release" not in cfg.events
  assert "push_robot" not in cfg.events


def test_play_is_deterministic():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  for name in _DR_EVENTS:
    assert name not in cfg.events
  assert cfg.observations["actor"].enable_corruption is False


def test_release_drill_requires_explicit_opt_in():
  cfg = phybot_c2_climb_wall_env_cfg(play=False, robustness_release=True)
  drill = cfg.events["random_release"]
  assert drill.mode == "interval"
  assert drill.interval_range_s == (6.0, 12.0)  # gentle rung; (3, 8) later
  for name in _DR_EVENTS:
    assert name in cfg.events
  fresh = phybot_c2_climb_wall_env_cfg(play=False)
  assert "random_release" not in fresh.events
  play = phybot_c2_climb_wall_env_cfg(play=True)
  assert "random_release" not in play.events


def _make_affordance_stub(num_envs=2, num_ee=4, cooldown=10):
  aff = object.__new__(MicrospineAffordance)
  aff._mcfg = SimpleNamespace(
    grip_enabled=True,
    reengage_cooldown_steps=cooldown,
    dr_reseat_hold_timeout_steps=75,
    spring_field_enabled=False,
  )
  aff._num_ee = num_ee
  aff._env = SimpleNamespace(device="cpu")  # backs the read-only .device
  shape = (num_envs, num_ee)
  aff._engaged = torch.ones(shape, dtype=torch.bool)
  aff._cooldown_steps = torch.zeros(shape, dtype=torch.long)
  aff._engage_count = torch.ones(shape, dtype=torch.long)
  aff._violation_count = torch.ones(shape, dtype=torch.long)
  aff._steps_since_engage = torch.ones(shape, dtype=torch.long)
  aff.forced_release_this_step = torch.zeros(shape, dtype=torch.bool)
  aff.release_reason = torch.zeros(shape, dtype=torch.int8)
  aff._seeded = torch.zeros(shape, dtype=torch.bool)
  aff.released_was_seeded = torch.zeros(shape, dtype=torch.bool)
  aff.dr_release_pending = torch.zeros(shape, dtype=torch.bool)
  aff._reseat_hold_steps = torch.zeros(num_envs, dtype=torch.long)
  aff._seat_scale = torch.full(shape, 0.5)
  aff._seat_charge = torch.full(shape, 0.3)
  aff.sync_calls = 0

  def _sync():
    aff.sync_calls += 1

  aff._sync_latched_contact_margins = _sync
  return aff


def test_force_release_mirrors_real_disengage():
  aff = _make_affordance_stub()
  # A same-step limit-surface release on env 0 EE 3 must survive the OR.
  aff.forced_release_this_step[0, 3] = True
  aff.release_reason[0, 3] = 2
  env_ids = torch.tensor([0])
  mask = torch.zeros(1, 4, dtype=torch.bool)
  mask[0, 1] = True
  aff.force_release(env_ids, mask)
  assert not aff._engaged[0, 1]
  assert aff._engaged[0, 0] and aff._engaged[0, 2] and aff._engaged[0, 3]
  assert aff._cooldown_steps[0, 1] == 10
  assert aff._engage_count[0, 1] == 0 and aff._violation_count[0, 1] == 0
  assert bool(aff.forced_release_this_step[0, 1])
  assert bool(aff.forced_release_this_step[0, 3])  # OR semantics
  assert int(aff.release_reason[0, 1]) == RELEASE_REASON_DR_FORCED
  assert int(aff.release_reason[0, 3]) == 2  # untouched
  assert float(aff._seat_scale[0, 1]) == 1.0
  assert float(aff._seat_charge[0, 1]) == 0.0
  # Env 1 untouched.
  assert bool(aff._engaged[1].all())
  assert aff.sync_calls == 1


def test_seat_charge_fraction_property():
  # v60: the public charge readout behind the first_charge rung.
  aff = _make_affordance_stub()
  aff._ls_cfg = SimpleNamespace(capacity_ramp_time_s=0.25)
  # The stub's 0.3 s of shear-loaded time overfills the 0.25 s ramp ->
  # clamped mature, never >1.
  assert bool((aff.seat_charge_fraction == 1.0).all())
  aff._seat_charge = torch.full((2, 4), 0.1)
  assert float(aff.seat_charge_fraction[0, 0]) == pytest.approx(0.4)
  # Ramp disabled -> every grip mature by definition.
  aff._ls_cfg = SimpleNamespace(capacity_ramp_time_s=0.0)
  assert bool((aff.seat_charge_fraction == 1.0).all())


def test_force_release_report_false_stays_silent():
  aff = _make_affordance_stub()
  env_ids = torch.tensor([0])
  mask = torch.zeros(1, 4, dtype=torch.bool)
  mask[0, 2] = True
  aff.force_release(env_ids, mask, report=False)
  assert not aff._engaged[0, 2]
  assert aff._cooldown_steps[0, 2] == 10
  assert not bool(aff.forced_release_this_step.any())
  assert int(aff.release_reason[0, 2]) == 0


def test_force_release_ignores_unlatched():
  aff = _make_affordance_stub()
  aff._engaged[0, 1] = False
  env_ids = torch.tensor([0])
  mask = torch.zeros(1, 4, dtype=torch.bool)
  mask[0, 1] = True
  aff.force_release(env_ids, mask)
  # Releasing an unlatched EE is a no-op (no cooldown, no report, no sync).
  assert aff._cooldown_steps[0, 1] == 0
  assert not bool(aff.forced_release_this_step.any())
  assert aff.sync_calls == 0


def test_random_release_guard_is_a_correctness_requirement():
  # minimum_supports=2 with a reason-blind unsafe_support penalty: an event
  # that could drop below 2 latches bills the policy for a DR fault.
  with pytest.raises(ValueError):
    random_microspine_release(
      env=None, env_ids=None, only_if_num_engaged_ge=2, max_release_per_env=1
    )


def test_random_release_only_touches_eligible_envs():
  aff = _make_affordance_stub(num_envs=3)
  aff._engaged[1] = torch.tensor([True, True, False, False])  # only 2 latched
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: aff)
  )
  torch.manual_seed(0)
  random_microspine_release(env, env_ids=torch.tensor([0, 1, 2]))
  # Envs 0 and 2 lose exactly one latch; env 1 (guard) keeps both.
  assert int(aff._engaged[0].sum()) == 3
  assert int(aff._engaged[1].sum()) == 2
  assert int(aff._engaged[2].sum()) == 3
  assert bool(aff.forced_release_this_step[0].any())
  assert not bool(aff.forced_release_this_step[1].any())
