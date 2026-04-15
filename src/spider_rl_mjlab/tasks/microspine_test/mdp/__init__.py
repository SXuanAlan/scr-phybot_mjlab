"""MDP components for the microspine test task."""

from spider_rl_mjlab.tasks.microspine_test.mdp.contact_force_vis import ContactForceVis  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.mdp.distributed_force import apply_distributed_pressure  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.mdp.events import apply_timed_force  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.mdp.freeze_policy import FreezeAfterPhase1  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.mdp.phase_force import PhaseForce  # noqa: F401
from spider_rl_mjlab.tasks.microspine_test.mdp.rewards import (  # noqa: F401
  contact_continuous_time,
  contact_lost_penalty,
  disengage_penalty,
  engage_progress,
  engaged_bonus,
  engaged_time,
  engagement_drag,
  friction_up_penalty,
  hand_height,
  hand_on_wall,
  hand_to_wall_dist,
  is_engaged_obs,
  palm_orientation,
  base_forward_penalty,
  torso_upright,
)
