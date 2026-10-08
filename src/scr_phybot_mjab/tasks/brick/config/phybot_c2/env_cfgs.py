"""PHYBot C2 dual-pad brick pickup, carry and placement.

The robot picks a brick off the floor with a microspine pad on each hand,
carries it and places it on a goal, tracking a measured-phase reference.
The variants differ in the training stage, the randomization and whether
the pads can grip. Each section lives in its own module and is built
directly in its final form; this module composes them and applies the play
overrides.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.brick.actions import BrickJointPositionActionCfg
from scr_phybot_mjab.tasks.brick.geometry import DEFAULT_UNIT_LAYOUT, active_unit_count

from .commands import brick_commands
from .events import brick_events
from .microspine import brick_microspine_cfg
from .observations import brick_observations
from .rewards import brick_rewards
from .scene import brick_robot_cfg, brick_scene, brick_sim, brick_viewer
from .terminations import brick_terminations


def brick_env_cfg(
  play: bool = False,
  *,
  stage: str = "full",
  randomize: bool = False,
  grip_enabled: bool = True,
  brick: BrickCfg | None = None,
  unit_layout: str = DEFAULT_UNIT_LAYOUT,
) -> ManagerBasedRlEnvCfg:
  """The brick task configuration.

  Args:
    play: evaluation mode (one env, noise-free brick pose estimates).
    stage: the reference phase training starts from: "full", "hold",
      "engage" or "place".
    randomize: startup randomization of brick density, foot friction and
      servo gains, reset scatter and pose-estimate dropouts.
    grip_enabled: whether the pads can latch at all (False is the no-grip
      ablation).
    brick: the brick's size, mass and placement.
    unit_layout: the active microspine units, "corners" or "all".
  """
  active_unit_count(unit_layout)
  brick = BrickCfg() if brick is None else brick
  mc = brick_microspine_cfg(brick, grip_enabled, unit_layout)
  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=30.0,
    scene=brick_scene(brick, mc, brick_robot_cfg()),
    sim=brick_sim(),
    viewer=brick_viewer(),
    observations=brick_observations(),
    actions={
      "joint_pos": BrickJointPositionActionCfg(
        entity_name="robot", actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True, scale=0.25
      )
    },
    commands=brick_commands(mc, brick, stage, randomize, unit_layout),
    events=brick_events(mc, randomize),
    rewards=brick_rewards(),
    terminations=brick_terminations(),
  )
  if play:
    apply_play_overrides(cfg)
  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """One env and noise-free brick pose estimates; the randomization events stay."""
  cfg.scene.num_envs = 1
  cfg.observations["actor"].enable_corruption = False
  command = cfg.commands["brick_task"]
  command.pose_position_noise_m = 0.0
  command.pose_rotation_noise_rad = 0.0
  command.max_delay_steps = 0
  command.dropout_probability = 0.0
