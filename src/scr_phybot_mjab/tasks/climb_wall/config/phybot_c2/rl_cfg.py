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
        # IsaacLab-style unbounded Gaussian with NO wrapper action clip (same
        # recipe as walk/squat and the healthy 14k-iter formal-hybridPD-yaw50
        # run): oversized actions carry real physical consequences, so the
        # smoothness/limit rewards regulate magnitude. Robot safety is bounded
        # in target space by SoftJointPositionAction, not in action space.
        # Runaway fingerprint (2026-07-13 redesign run, std 0.6->48 under a
        # tight +-1 clip): Policy/mean_std rising past ~1.5 without receding.
        # A healthy run peaks at ~1.3 during the early dig-in plateau.
        # init_std stays at 0.6, NOT the walk/IsaacLab-conventional 1.0: a
        # wall-hung robot's engaged microspines carry real load, and 1.0-wide
        # birth exploration overloads the anchors past the release envelope
        # (hold_margin -1.3 vs healthy -0.65), teaching the policy to avoid
        # engaging at all. Verified by a 4-arm ablation on 2026-07-15.
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
      # Safe without an action clip: the 2026-07-13 saturation runaway needed
      # the tight +-1 clip to make std growth consequence-free. At this value
      # the unclipped formal-hybridPD-yaw50 run kept std at 0.2-0.5 for 14k
      # iterations.
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
