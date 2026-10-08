"""Position residuals around the measured-phase reference trajectory."""

from dataclasses import dataclass

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES

__all__ = (
  "BrickJointPositionAction",
  "BrickJointPositionActionCfg",
)


@dataclass(kw_only=True)
class BrickJointPositionActionCfg(JointPositionActionCfg):
  use_default_offset: bool = False

  def __post_init__(self):
    super().__post_init__()
    if self.use_default_offset or self.offset != 0.0:
      raise ValueError("Brick actions use the phase reference as their position offset")

  def build(self, env):
    return BrickJointPositionAction(self, env)


class BrickJointPositionAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if tuple(self.target_names) != tuple(PHYBOT_C2_POLICY_JOINT_NAMES):
      raise ValueError("Brick residual actions require the reference policy joint order")

  def process_actions(self, actions):
    # The base implementation scales/clips the residual and records raw
    # actions. The reference is sampled once per control step, before physics.
    super().process_actions(actions)
    self._processed_actions += self._env.command_manager.get_term("brick_task").joint_pos_ref
