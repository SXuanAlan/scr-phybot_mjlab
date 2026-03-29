"""RL configuration for friction test (minimal — this task is for physics validation, not training)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def friction_test_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a minimal RL runner config (needed for registration, not for training)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(32,),
      activation="elu",
      obs_normalization=False,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(32,),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=1,
      num_mini_batches=1,
      learning_rate=1e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    experiment_name="friction_test",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1,
  )
