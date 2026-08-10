"""Config contract of the walk->climb transition (mount) task.

The transition lives in its own PARALLEL package
(``tasks/climb_transition``) and builds its cfg by mutating the climb
builder's output. Its load-bearing promises, asserted here: CONTRACT PARITY
with climb (identical observation/action terms, so climb checkpoints
warm-start strict), one-shot playback via the MountMotionCommand subclass,
floor enabled through the spec_fn wrap, blind-delivery scatter on standing
starts only — and ZERO edits to the climb_wall package itself.
"""

import mujoco
import pytest

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  configure_phybot_c2_floor_geom,
)
from scr_phybot_mjab.tasks.climb_transition.config.phybot_c2.env_cfgs import (
  _TRANSITION_SPAWN_XY_NOISE,
  _TRANSITION_SPAWN_YAW_NOISE,
  _transition_spec_fn,
  phybot_c2_climb_transition_env_cfg,
)
from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
  MountMotionCommandCfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionCommandCfg


def test_transition_keeps_the_climb_contract() -> None:
  """Same obs terms in the same order, same commands, same action — the
  precondition for strict warm-starting from climb checkpoints."""
  climb = phybot_c2_climb_wall_env_cfg()
  trans = phybot_c2_climb_transition_env_cfg()

  assert tuple(trans.observations["actor"].terms) == tuple(
    climb.observations["actor"].terms
  )
  assert tuple(trans.observations["critic"].terms) == tuple(
    climb.observations["critic"].terms
  )
  assert tuple(trans.commands) == (
    "climb_velocity",
    "microspine_affordance",
    "climb_motion",
  )
  assert tuple(trans.actions) == tuple(climb.actions)
  assert (
    trans.actions["joint_pos"].actuator_names
    == climb.actions["joint_pos"].actuator_names
  )


def test_transition_motion_command_is_the_mount_subclass() -> None:
  cfg = phybot_c2_climb_transition_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert isinstance(motion, MountMotionCommandCfg)
  assert motion.motion_file.endswith("climb_mount_prologue.npy")
  assert motion.sampling_mode == "uniform"
  assert motion.joint_pos_noise == pytest.approx(0.02)
  assert motion.spawn_xy_noise == _TRANSITION_SPAWN_XY_NOISE
  assert motion.spawn_yaw_noise == pytest.approx(_TRANSITION_SPAWN_YAW_NOISE)
  # v4: the bias window aims at the latch stretch and the press shaping is
  # boosted with an anneal-back-to-parity curriculum.
  assert motion.spawn_bias_segments == ("hand_place", "hand_set_hold")
  assert cfg.rewards["motion_scheduled_press"].weight == pytest.approx(3.0)
  anneal = cfg.curriculum["press_bootstrap_anneal"]
  assert anneal.params["weight_stages"][-1]["weight"] == pytest.approx(0.4)
  assert anneal.params["steps_per_iter"] == 96
  # A mount has no climb rate; the command obs lane reads constant zero.
  assert cfg.commands["climb_velocity"].ranges.lin_vel_z == (0.0, 0.0)
  # Floor comes in through the spec_fn wrap (introspectable partial).
  assert cfg.scene.spec_fn.func is _transition_spec_fn


def test_transition_strips_cycle_training_aids() -> None:
  cfg = phybot_c2_climb_transition_env_cfg()
  # Default gravity IS the vertical-wall operating point; the tilt-peel
  # band is statically configured at the calibrated values.
  assert "wall_angle" not in cfg.curriculum
  assert "tilt_peel" not in cfg.curriculum
  # v1 trains the nominal mount; the release drill is a later R-series
  # rung. This also removes the only reseat_hold producer, so the terminal
  # clock cannot be frozen short of the handoff frame.
  assert "random_release" not in cfg.events
  assert "mount_handoff_ready" in cfg.metrics
  # The base DR package stays (contract parity includes s2r hardening).
  for dr_event in ("encoder_bias", "body_friction", "base_mass"):
    assert dr_event in cfg.events


def test_transition_play_is_deterministic_and_noise_free() -> None:
  cfg = phybot_c2_climb_transition_env_cfg(play=True)
  motion = cfg.commands["climb_motion"]
  # Evaluation delivers the canonical frame-0 stance: no spawn scatter, no
  # joint noise, clock from the standing start.
  assert motion.sampling_mode == "start"
  assert motion.joint_pos_noise == 0.0
  assert motion.spawn_xy_noise == (0.0, 0.0)
  assert motion.spawn_yaw_noise == 0.0
  # Play keeps climb-parity press weight and no bootstrap curriculum.
  assert cfg.rewards["motion_scheduled_press"].weight == pytest.approx(0.4)
  assert "press_bootstrap_anneal" not in cfg.curriculum
  for dr_event in ("encoder_bias", "body_friction", "base_mass"):
    assert dr_event not in cfg.events


def test_climb_wall_package_is_untouched() -> None:
  """The parallel-architecture promise: climb's cfg carries NONE of the
  transition machinery."""
  cfg = phybot_c2_climb_wall_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert type(motion) is ClimbMotionCommandCfg  # not the mount subclass
  assert not hasattr(motion, "spawn_xy_noise")
  assert motion.motion_file.endswith("climb_cycle_moment_safe_paced.npy")
  assert "wall_angle" in cfg.curriculum and "tilt_peel" in cfg.curriculum
  assert "mount_handoff_ready" not in cfg.metrics
  # The mount cfg's own noise knobs default OFF at the class level.
  fields = MountMotionCommandCfg.__dataclass_fields__
  assert fields["spawn_yaw_noise"].default == 0.0


def test_floor_geom_toggle_controls_contact_bits() -> None:
  """The spec_fn wrap's ground truth, on a minimal synthetic spec."""

  def build(enable: bool) -> mujoco.MjsGeom:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    geom = body.add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[1.0, 1.0, 0.1],
    )
    configure_phybot_c2_floor_geom(spec, enable_contact=enable)
    return geom

  enabled = build(True)
  assert enabled.contype == 1 and enabled.conaffinity == 1
  disabled = build(False)
  assert disabled.contype == 0 and disabled.conaffinity == 0
