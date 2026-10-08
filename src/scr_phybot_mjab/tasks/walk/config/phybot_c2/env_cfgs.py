"""PHYBot C2 walking and standing on a flat floor.

Follows MJLab's official Unitree G1 velocity pipeline with the C2 asset.
One configuration serves three tasks (see variants.py): walk, stand and
amp. Each section lives in its own module and is built directly in its
final form; this module composes them and applies the play overrides.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_ACTION_SCALE, PHYBOT_C2_POLICY_JOINT_NAMES

from .commands import TWIST_COMMAND, walk_commands
from .curricula import walk_curriculum
from .events import walk_events
from .observations import walk_observations
from .rewards import walk_rewards
from .scene import walk_robot_cfg, walk_scene, walk_sim, walk_viewer
from .terminations import walk_terminations
from .variants import WalkVariant


def phybot_c2_walk_env_cfg(play: bool = False, variant: WalkVariant = "walk") -> ManagerBasedRlEnvCfg:
  """The walking configuration of one variant; ``play`` is unlimited, clean and deterministic."""
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=20.0,
    scene=walk_scene(walk_robot_cfg(real_pd_damping=variant == "amp")),
    sim=walk_sim(),
    viewer=walk_viewer(),
    observations=walk_observations(amp=variant == "amp"),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES,
        preserve_order=True,
        scale=PHYBOT_C2_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands=walk_commands(variant),
    events=walk_events(variant),
    rewards=walk_rewards(variant),
    terminations=walk_terminations(),
    curriculum=walk_curriculum(variant),
    metrics={"mean_action_acc": MetricsTermCfg(func=base_mdp.mean_action_acc)},
  )
  if play:
    apply_play_overrides(cfg, variant)
  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg, variant: WalkVariant) -> None:
  """Unlimited episodes without observation noise, pushes or curricula.

  walk previews the curriculum's second-stage command range. stand starts
  exactly at its keyframe on the nominal model, with its randomization
  events pinned to single values.
  """
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  if variant == "walk":
    ranges = cfg.commands[TWIST_COMMAND].ranges
    ranges.lin_vel_x = (-1.5, 2.0)
    ranges.ang_vel_z = (-0.7, 0.7)
  elif variant == "stand":
    for name in ("reset_robot_joints", "reset_robot_upper_joints"):
      cfg.events[name].params["position_range"] = (0.0, 0.0)
      cfg.events[name].params["velocity_range"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["body_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["foot_friction"].params["ranges"] = (0.7, 0.7)
    cfg.events["base_mass"].params["ranges"] = (0.0, 0.0)
