"""AMP RL configuration for the PHYBot C2 walk task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES

_AMP_MOTION_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"


@dataclass
class AmpDatasetCfg:
  amp_data_path: str = str(_AMP_MOTION_DIR)
  datasets: dict[str, float] = field(
    default_factory=lambda: {
      "c2_stand_zero": 4.0,
      "mini_walk_slow": 2.0,
      "mini_walk_forward_0": 1.0,
      "mini_walk_forward_1": 1.0,
      "mini_walk_turns": 0.5,
    }
  )
  slow_down_factor: int = 1
  velocity_representation: str = "body_fixed"
  amp_joint_names: list[str] = field(default_factory=lambda: list(PHYBOT_C2_POLICY_JOINT_NAMES))


@dataclass
class AmpDiscriminatorCfg:
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  reward_scale: float = 1.0
  loss_type: str = "BCEWithLogits"
  empirical_normalization: bool = True


@dataclass
class AmpRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  style_weight: float = 0.4
  dataset: AmpDatasetCfg = field(default_factory=AmpDatasetCfg)
  discriminator: AmpDiscriminatorCfg = field(default_factory=AmpDiscriminatorCfg)


def phybot_c2_walk_amp_ppo_runner_cfg() -> AmpRslRlOnPolicyRunnerCfg:
  return AmpRslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="AMP_PPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="phybot_c2_walk_amp",
    logger="tensorboard",
    wandb_project="scr_phybot_mjab-walk",
    wandb_tags=("phybot_c2", "walk", "amp"),
    upload_model=False,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=12_000,
    style_weight=0.4,
  )
