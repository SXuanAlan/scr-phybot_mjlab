"""Teacher-student distillation wiring for mjlab (rsl_rl 5.x).

mjlab ships no distillation support: its runner cfg dataclasses only know
actor/critic and ``MjlabOnPolicyRunner`` only None-strips those two model
cfgs. This module adds the cfg dataclasses and a runner that composes
mjlab's checkpoint/export behavior with rsl_rl's ``DistillationRunner``.

The teacher is a finished PPO checkpoint: ``Distillation.load`` maps a
checkpoint containing ``actor_state_dict`` onto the teacher when called with
``load_cfg=None``, which is exactly what ``teacher_checkpoint`` triggers at
construction time. This keeps the distillation run's ``experiment_name``
fully independent of the teacher's log directory (mjlab's resume search only
scans ``log_root/experiment_name``); ``--agent.resume`` stays available for
resuming the distillation run itself (a distillation checkpoint carries
``student_state_dict`` and loads through the same call unambiguously).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg
from mjlab.rl.runner import MjlabOnPolicyRunner


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Maps onto ``rsl_rl.algorithms.Distillation`` constructor kwargs."""

  class_name: str = "Distillation"
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Runner cfg for student-teacher distillation.

  ``teacher`` must match the teacher run's actor cfg field-for-field
  (hidden dims, obs normalization, distribution cfg): the checkpoint's
  ``actor_state_dict`` — normalizer statistics and ``distribution.std_param``
  included — is loaded strict.
  """

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"student": ("student",), "teacher": ("actor",)},
  )
  teacher_checkpoint: str | None = None
  """PPO checkpoint to load as the frozen teacher at construction time."""


class MjlabDistillationRunner(MjlabOnPolicyRunner, DistillationRunner):
  """mjlab checkpointing/export + rsl_rl distillation.

  MRO: this class -> MjlabOnPolicyRunner (None-strip, save/load env-state,
  legacy-key migration, ONNX export) -> DistillationRunner (teacher-loaded
  assertion in ``learn``) -> OnPolicyRunner. ``get_policy()`` returns the
  student, so mjlab's jit/onnx export paths export the deployable model.

  Note: loading the teacher through ``MjlabOnPolicyRunner.load`` also
  restores ``common_step_counter`` from the teacher's env_state. The distill
  env pins its curricula to single stages, so this is inert here — revisit
  if a distillation env ever carries a live curriculum schedule.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Same None-strip MjlabOnPolicyRunner applies to actor/critic: MLPModel
    # rejects unexpected cnn/rnn/distribution kwargs.
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    teacher_checkpoint = train_cfg.get("teacher_checkpoint")
    if teacher_checkpoint:
      self.load(str(teacher_checkpoint), load_cfg=None, map_location=device)
