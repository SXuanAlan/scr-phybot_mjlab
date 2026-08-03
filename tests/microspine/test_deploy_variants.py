"""Contracts for the deploy-PD / DR / distillation variants (2026-08-01)."""

from types import SimpleNamespace

import pytest
import torch

from scr_phybot_mjab.microspine.affordance import (
  RELEASE_REASON_DR_FORCED,
  MicrospineAffordance,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_distill_env_cfg,
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_distill_runner_cfg,
  phybot_c2_ppo_runner_cfg,
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


def test_default_dr_is_the_full_v3_package():
  # v4 revival (2026-08-02): the shipped lineage carries v3's FULL
  # walk-parity DR from birth — from-scratch-with-noise beats
  # resume-then-inject (the graded R1/R2 ladder measured a ~0.015 permanent
  # clean-exam tax on the noise-naive 18c line). Byte-matched against the
  # deploypd-v3-pd100 run's params/env.yaml.
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
  assert "random_release" not in cfg.events  # R4 stays an opt-in switch
  assert "push_robot" not in cfg.events


def test_play_is_deterministic():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  for name in _DR_EVENTS:
    assert name not in cfg.events
  assert cfg.observations["actor"].enable_corruption is False


def test_release_drill_is_opt_in_only():
  cfg = phybot_c2_climb_wall_env_cfg(play=False, robustness_release=True)
  assert cfg.events["random_release"].mode == "interval"
  for name in _DR_EVENTS:
    assert name in cfg.events
  play = phybot_c2_climb_wall_env_cfg(play=True, robustness_release=True)
  assert "random_release" not in play.events


def test_student_group_contract():
  cfg = phybot_c2_climb_wall_distill_env_cfg(play=False)
  student = cfg.observations["student"]
  # Deployable terms only, in the exported (C++) order.
  assert list(student.terms.keys()) == [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "motion_command",
  ]
  assert student.history_length == 10
  assert student.enable_corruption is True
  # Clean teacher labels; the physical DR stays on.
  assert cfg.observations["actor"].enable_corruption is False
  for name in _DR_EVENTS:
    assert name in cfg.events
  # Curricula pinned to single stages (teacher-graduated difficulty).
  assert len(cfg.curriculum["gravity_angle"].params["angle_stages"]) == 1
  assert len(cfg.curriculum["tilt_peel"].params["stages"]) == 1
  play = phybot_c2_climb_wall_distill_env_cfg(play=True)
  assert play.observations["student"].enable_corruption is False


def test_distill_runner_cfg_contract():
  distill = phybot_c2_distill_runner_cfg()
  ppo = phybot_c2_ppo_runner_cfg()
  # The teacher model cfg must equal the teacher run's actor cfg: the ckpt's
  # actor_state_dict (normalizer + std_param included) is loaded strict.
  assert distill.teacher == ppo.actor
  assert distill.obs_groups == {"student": ("student",), "teacher": ("actor",)}
  assert distill.algorithm.class_name == "Distillation"
  assert distill.teacher_checkpoint is None
  # Rollout exploration std: teacher-converged scale, not the PPO birth 0.6.
  assert distill.student.distribution_cfg["init_std"] == 0.3
  assert distill.student.obs_normalization is True
  # The scripts' policy-set discriminator: distillation cfgs expose `student`,
  # PPO cfgs must not — {"actor": True} on a distill ckpt is a silent no-op.
  assert hasattr(distill, "student")
  assert not hasattr(ppo, "student")


def _make_affordance_stub(num_envs=2, num_ee=4, cooldown=10):
  aff = object.__new__(MicrospineAffordance)
  aff._mcfg = SimpleNamespace(
    grip_enabled=True, reengage_cooldown_steps=cooldown
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
