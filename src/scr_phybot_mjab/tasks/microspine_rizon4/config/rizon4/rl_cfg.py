"""Minimal RL config for Rizon4 microspine task registration."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def rizon4_microspine_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a small runner config so play/list tooling can load the task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=4,
      num_mini_batches=2,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="rizon4_microspine",
    logger="wandb",
    wandb_project="scr_phybot_mjab-rizon4-microspine",
    wandb_tags=("rizon4", "microspine"),
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
