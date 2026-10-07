"""Contracts for the deploy PD gains and the s2r DR package."""

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

_DR_EVENTS = ("body_friction", "base_mass")


def _gains(cfg):
  return {
    act.target_names_expr[0]: (act.stiffness, act.damping) for act in cfg.scene.entities["robot"].articulation.actuators
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
  assert "push_robot" not in cfg.events


def test_play_is_deterministic():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  for name in _DR_EVENTS:
    assert name not in cfg.events
  assert cfg.observations["actor"].enable_corruption is False
