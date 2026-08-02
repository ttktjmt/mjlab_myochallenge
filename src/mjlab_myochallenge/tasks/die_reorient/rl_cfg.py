"""RL training configuration for the Die Reorientation task."""

from dataclasses import dataclass, field

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class DieReorientRlCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for MyoHand Die Reorientation."""

    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        )
    )
    critic: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.015,
            max_grad_norm=1.0,
        )
    )
    wandb_project: str = "mjlab_myochallenge"
    experiment_name: str = "myohand_die_reorient"
    save_interval: int = 200
    num_steps_per_env: int = 64
    max_iterations: int = 30_000
