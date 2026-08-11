"""PHYBot C2 walk->climb transition task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_climb_transition_env_cfg
from .rl_cfg import phybot_c2_climb_transition_ppo_runner_cfg

# The climb contract verbatim against the acyclic mount prologue: floor on,
# terminal-hold clock, blind-delivery spawn scatter. Same obs/action dims as
# Mjlab-Climb-Wall-Phybot-C2 by construction, so climb checkpoints
# warm-start via scripts/train_warmstart.py.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Phybot-C2",
  env_cfg=phybot_c2_climb_transition_env_cfg(),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)
