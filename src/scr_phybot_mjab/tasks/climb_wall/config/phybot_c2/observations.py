"""Climbing observations. Term order is the checkpoint and export layout contract."""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import (
  MICROSPINE_AFFORDANCE_COMMAND,
  MICROSPINE_GRIP_EVENT,
  MOTION_COMMAND,
  VELOCITY_COMMAND,
)
from scr_phybot_mjab.tasks.common.observations import projected_gravity_from_framequat_sensor
from scr_phybot_mjab.tasks.common.phybot_c2 import (
  BASE_ANG_VEL_SENSOR,
  BASE_LIN_VEL_SENSOR,
  BASE_QUAT_SENSOR,
  policy_joint_cfg,
)


def proprio_terms() -> dict[str, ObservationTermCfg]:
  """The robot's own sensing: base and joint state and the last three actions."""
  return {
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": BASE_LIN_VEL_SENSOR}),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": BASE_ANG_VEL_SENSOR}),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor, params={"sensor_name": BASE_QUAT_SENSOR}
    ),
    "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": policy_joint_cfg()}),
    "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": policy_joint_cfg()}),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    "prev_action": ObservationTermCfg(func=mdp.prev_action),  # a_{t-1}
    "prev_prev_action": ObservationTermCfg(func=mdp.prev_prev_action),  # a_{t-2}
  }


def latch_terms(microspine_cfg: MicrospineCfg) -> dict[str, ObservationTermCfg]:
  """Each pad's latch state and the seating fraction of its spine field."""
  return {
    "microspine_engaged_state": ObservationTermCfg(
      func=mdp.microspine_engaged_state, params={"command_name": MICROSPINE_AFFORDANCE_COMMAND}
    ),
    "microspine_contact_fraction": ObservationTermCfg(
      func=mdp.microspine_contact_fraction,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "spheres_per_ee": microspine_cfg.spheres_per_ee_rows * microspine_cfg.spheres_per_ee_cols,
      },
    ),
  }


def motion_command_term() -> ObservationTermCfg:
  """The reference command: phase clock and engage-schedule chart."""
  return ObservationTermCfg(func=mdp.generated_commands, params={"command_name": MOTION_COMMAND})


def add_noise(terms: dict[str, ObservationTermCfg], noise: dict[str, UniformNoiseCfg] | None) -> None:
  for name, term_noise in (noise or {}).items():
    terms[name].noise = term_noise


def climb_observations(
  microspine_cfg: MicrospineCfg, actor_noise: dict[str, UniformNoiseCfg] | None
) -> dict[str, ObservationGroupCfg]:
  """The actor group, corrupted with ``actor_noise`` when given, and its privileged critic."""
  terms = {
    **proprio_terms(),
    # Robot height above spawn: the critic needs it to value climbing.
    "height_above_spawn": ObservationTermCfg(func=mdp.height_above_spawn),
    # Commanded world-frame velocity; the command manager owns the target, so
    # observation and reward stay synchronized.
    "velocity_command": ObservationTermCfg(func=mdp.velocity_command, params={"command_name": VELOCITY_COMMAND}),
    **latch_terms(microspine_cfg),
    "microspine_disengage_time": ObservationTermCfg(
      func=mdp.ee_disengage_time_obs, params={"event_term_name": MICROSPINE_GRIP_EVENT, "clip": 1.5}
    ),
    "microspine_cooldown": ObservationTermCfg(
      func=mdp.microspine_cooldown_obs,
      params={"command_name": MICROSPINE_AFFORDANCE_COMMAND, "cooldown_steps": microspine_cfg.reengage_cooldown_steps},
    ),
    # Cycle phase clock (cos, sin) and the clip's engage schedule: per-EE
    # engage bits, per-EE countdowns to the next release or regrip and
    # per-foot toe-brace flags, 16 dims in all.
    "motion_command": motion_command_term(),
  }
  add_noise(terms, actor_noise)
  actor = ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=actor_noise is not None)
  # The critic copies the finished actor terms, so it is built last.
  return {"actor": actor, "critic": mdp.privileged_critic_group(actor, command_name=MICROSPINE_AFFORDANCE_COMMAND)}
