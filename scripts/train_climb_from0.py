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
import json
import os

import scr_phybot_mjab.tasks  # noqa: F401 - registers the base tasks
from mjlab.scripts.train import main
from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)

_env_cfg = phybot_c2_climb_wall_env_cfg(robustness_release=False)

# Competence-gated curriculum surgery: the campaign advances stages on
# demonstrated learning, not on the wall clock. These env vars override the
# stage tables at launch (JSON, same dict shapes as the curriculum params)
# so a resume can extend a stage the policy has not mastered yet.
_angle = os.environ.get("FROM0_ANGLE_STAGES")
if _angle:
  _env_cfg.curriculum["wall_angle"].params["angle_stages"] = json.loads(
    _angle
  )
  print("[from0] angle stages:", json.loads(_angle))
_tilt = os.environ.get("FROM0_TILT_STAGES")
if _tilt:
  _env_cfg.curriculum["tilt_peel"].params["stages"] = json.loads(_tilt)
  print("[from0] tilt stages:", json.loads(_tilt))
# Moment-billing override for controlled comparisons (the 2x2 ignition
# matrix: start-angle x billing). FROM0_MOMENT_BILLING=worst_edge with
# FROM0_MOMENT_ARM=0.5 reproduces the v3/v4-lineage physics exactly.
_billing = os.environ.get("FROM0_MOMENT_BILLING")
if _billing:
  _ls = _env_cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
  _ls.moment_billing_mode = _billing
  _arm = os.environ.get("FROM0_MOMENT_ARM")
  if _arm:
    _ls.moment_arm_fraction = float(_arm)
  print(f"[from0] moment billing: {_billing} arm={_ls.moment_arm_fraction}")

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-From0",
  env_cfg=_env_cfg,
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True, robustness_release=False),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint", "perunit_from0"),
  ),
)

if __name__ == "__main__":
  main()
