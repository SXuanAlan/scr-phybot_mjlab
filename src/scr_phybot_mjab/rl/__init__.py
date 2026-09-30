"""RL wiring extensions (distillation runner + cfgs) for scr_phybot_mjab."""

from scr_phybot_mjab.rl.distillation import (
  MjlabDistillationRunner,
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
)

__all__ = (
  "MjlabDistillationRunner",
  "RslRlDistillationAlgorithmCfg",
  "RslRlDistillationRunnerCfg",
)

# The segment-routed KL distillation (unified mount+climb student) lives in
# scr_phybot_mjab.rl.kl_distill and is imported lazily by its consumers —
# the runner resolves "scr_phybot_mjab.rl.kl_distill:KLDistillPPO" by
# dotted path, so nothing here needs to re-export it.
