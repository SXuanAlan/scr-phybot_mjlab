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
