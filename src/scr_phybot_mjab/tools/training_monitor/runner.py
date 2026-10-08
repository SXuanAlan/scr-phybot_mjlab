"""A PPO runner whose checkpoints carry the live wall-angle curriculum and the RSI failure sampler."""

from __future__ import annotations

from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner

from scr_phybot_mjab.tasks.climb_wall.mdp.live_curriculum import LiveCurriculumControl


class MonitoredRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    sampler = env.command_manager.get_term("climb_motion").phase_sampler
    controller = env._live_climb_curriculum
    infos = {
      **(infos or {}),
      "climb_monitor": {
        "phase_failed": sampler.failed.detach().cpu(),
        "active_stage": controller.active_stage,
        "stage_started_iter": controller.stage_started_iter,
        "control": controller.control,
      },
    }
    # Publish complete checkpoint files so concurrent evaluations never load
    # a partially serialized optimizer/model.
    destination = Path(path)
    temporary = destination.with_name("." + destination.name + ".tmp")
    upload = self.cfg["upload_model"]
    self.cfg["upload_model"] = False
    try:
      super().save(str(temporary), infos)
      temporary.replace(destination)
    finally:
      self.cfg["upload_model"] = upload
    if upload:
      self.logger.save_model(str(destination), self.current_learning_iteration)

  def load(self, *args, **kwargs):
    infos = super().load(*args, **kwargs)
    env = self.env.unwrapped
    saved = (infos or {}).get("climb_monitor")
    if saved:
      sampler = env.command_manager.get_term("climb_motion").phase_sampler
      sampler.failed.copy_(saved["phase_failed"].to(env.device))
      controller: LiveCurriculumControl = env._live_climb_curriculum
      controller.active_stage = saved["active_stage"]
      controller.stage_started_iter = saved["stage_started_iter"]
      controller.control = saved["control"]
      controller.last_iteration = -1
    # Saved model_N already completed update N; continue at N+1.
    self.current_learning_iteration += 1
    env.reset()
    return infos
