"""Launch a from-scratch climb teacher (no R4 release drill).

The default climb task ships with the R4 unscheduled-release drill in its DR
package; the from-scratch protocol starts WITHOUT it and adds it only after
exact-90 is mastered (see config/phybot_c2/__init__.py). This launcher
registers the ``-From0`` variant (identical recipe, ``robustness_release=
False``) and delegates to the standard mjlab train CLI, so the usual
``--agent.* / --env.* / --gpu-ids`` flags all apply:

  python scripts/train_climb_from0.py Mjlab-Climb-Wall-Phybot-C2-From0 \
      --agent.run-name perunit-v5-from0 --gpu-ids 0
"""

import dataclasses

import scr_phybot_mjab.tasks  # noqa: F401 - registers the base tasks
from mjlab.scripts.train import main
from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-From0",
  env_cfg=phybot_c2_climb_wall_env_cfg(robustness_release=False),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True, robustness_release=False),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint", "perunit_from0"),
  ),
)

if __name__ == "__main__":
  main()
