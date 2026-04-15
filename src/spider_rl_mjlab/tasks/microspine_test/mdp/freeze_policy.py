"""Cancel joint control after Phase 1.

During Phase 2 / Phase 3 the test rig probes microspine adhesion under
external force. The arm should be **passive** — only adhesion + wall contact
+ friction determine survival, not the policy pushing back.

mjlab's ``step`` events fire AFTER physics substeps, so writing to
``env.sim.data.ctrl`` from a step event has no effect — the ActionManager's
``apply_action`` (called every substep) re-writes ctrl from the policy's
processed action at the start of the next env step.

To actually neutralize the policy, this class **monkey-patches** the
JointPositionAction term's ``process_actions`` at env-build time. Each env
step, after the term computes its ``_processed_actions`` from the raw
policy action, we overwrite them with the current joint ``qpos`` whenever
the env is past phase 1. Position actuator force is then
``kp*(qpos - qpos) - kv*qvel ≈ 0`` (residual damping is passive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class FreezeAfterPhase1:
  """Startup/step hybrid: monkey-patches the joint action term.

  On ``__init__`` (env build), wraps the JointPositionAction term's
  ``process_actions`` so that, past ``phase1_duration``, ``_processed_actions``
  is forced to the current ``qpos`` for each joint the term controls.

  Params (cfg.params):
    phase1_duration: Seconds of Phase 1; after this, joint control is cut.
  """

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    # NOTE: EventManager constructs before ActionManager, so we cannot access
    # env.action_manager here. The monkey-patch is deferred to the first
    # __call__ (which fires at step time, after all managers exist).
    self._env = env
    self._phase1_duration: float = cfg.params["phase1_duration"]
    self._patched: bool = False
    self._qposadr: torch.Tensor | None = None

  def _install_patch(self, env: "ManagerBasedRlEnv") -> None:
    am = env.action_manager
    term = None
    for _, t in am._terms.items():
      if t.__class__.__name__.startswith("JointPosition"):
        term = t
        break
    assert term is not None, (
      f"FreezeAfterPhase1: no JointPositionAction term found in "
      f"{list(am._terms.keys())}"
    )

    orig_process = term.process_actions
    phase1_duration = self._phase1_duration
    target_ids = term._target_ids

    def wrapped(actions: torch.Tensor) -> None:
      orig_process(actions)
      elapsed = env.episode_length_buf.float() * env.step_dt
      past = elapsed >= phase1_duration  # (E,)
      if not bool(past.any()):
        return
      # Use the entity's joint_pos (in joint_id order) and slice by target_ids.
      joint_pos = term._entity.data.joint_pos[:, target_ids]  # (E, K)
      pa = term._processed_actions  # (E, K)
      pa[past] = joint_pos[past]

    term.process_actions = wrapped
    self._patched = True

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    if not self._patched:
      self._install_patch(env)
    # After patch installed, all work happens inside wrapped process_actions.

  def reset(self, env_ids: torch.Tensor) -> None:
    pass
