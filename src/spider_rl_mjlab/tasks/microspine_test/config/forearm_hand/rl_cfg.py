"""RL configuration for forearm+hand microspine test task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def forearm_hand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for forearm+hand microspine test."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 2.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.03,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.5e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.025,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    experiment_name="forearm_hand_microspine_test",
    logger="tensorboard",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
