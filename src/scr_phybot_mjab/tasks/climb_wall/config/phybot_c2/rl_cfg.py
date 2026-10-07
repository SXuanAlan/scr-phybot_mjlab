"""RL configuration for PHYBot C2 wall-climbing task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def phybot_c2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PHYBot C2 wall-climbing task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        # An unbounded Gaussian without an action clip, as in walk and squat:
        # oversized actions have physical consequences, so the smoothness and
        # limit rewards regulate magnitude, and SoftJointPositionAction bounds
        # the joint targets. A runaway shows as Policy/mean_std rising past
        # ~1.5 without receding; a healthy run peaks near 1.3 early on.
        # init_std 0.6 rather than the conventional 1.0: the engaged spines
        # carry the robot, and 1.0-wide exploration at birth overloads the
        # anchors past the hold envelope (hold margin -1.3 against a healthy
        # -0.65), so the policy learns to avoid engaging at all.
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    # Asymmetric actor-critic: the critic env group = actor terms + privileged
    # microspine state (see env_cfgs.py). Each side normalizes its own group.
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Safe without an action clip: a tight +-1 clip makes std growth free of
      # consequences and lets entropy run it away. Unclipped, std stays at
      # 0.2-0.5 over 14k iterations at this coefficient.
      entropy_coef=0.003,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="phybot_c2_climb_wall",
    logger="wandb",
    wandb_project="scr_phybot_mjab-climb-wall",
    wandb_tags=("phybot_c2",),
    save_interval=25,
    # One release -> cooldown -> re-engage -> hold-confirmation sequence takes
    # roughly 0.7 seconds.
    # Keep the full transition inside a PPO rollout instead of splitting its
    # credit across four 24-step fragments.
    num_steps_per_env=96,
    max_iterations=12_000,
  )
