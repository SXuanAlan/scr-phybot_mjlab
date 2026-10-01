"""Sim-to-real randomization of the climbing task: observation noise and startup DR events."""

from __future__ import annotations

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import MICROSPINE_AFFORDANCE_COMMAND

DOMAIN_RAND_EVENTS = ("body_friction", "base_mass", "encoder_bias", "random_release")
"""Every randomization event; play removes them all."""


def actor_obs_noise() -> dict[str, Unoise]:
  """Walk's sensor noise, carried from the start of training."""
  return {
    "base_lin_vel": Unoise(n_min=-0.5, n_max=0.5),
    "base_ang_vel": Unoise(n_min=-0.2, n_max=0.2),
    "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
    "joint_pos_rel": Unoise(n_min=-0.01, n_max=0.01),
    "joint_vel_rel": Unoise(n_min=-1.5, n_max=1.5),
  }


def domain_rand_events() -> dict[str, EventTermCfg]:
  return {
    "body_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        # Walk's selection: every collision geom except the feet. Proxy and
        # toe wall contacts are pair-driven, so the microspine calibration
        # stays untouched.
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(r"^(?!(left|right)_ankle_roll_collision$).*_collision$",),
        ),
        "operation": "abs",
        "ranges": (0.6, 1.0),
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
        # +-2 kg, narrower than walk's +-5. The bound has not been re-derived
        # for the empirical unit model.
        "ranges": (-2.0, 2.0),
        "operation": "add",
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        # +-0.01 rad is a 4-8 mm unobservable EE offset against the ~5 mm
        # spine seating travel.
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
  }


def random_release_event() -> EventTermCfg:
  """An optional recovery drill: random unscheduled latch releases."""
  return EventTermCfg(
    mode="interval",
    interval_range_s=(6.0, 12.0),
    func=mdp.random_microspine_release,
    params={"command_name": MICROSPINE_AFFORDANCE_COMMAND},
  )
