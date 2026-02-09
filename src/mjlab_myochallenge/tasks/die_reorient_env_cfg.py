"""MyoHand Die Reorientation task environment configuration.

Based on MyoChallenge 2022 Die Reorientation Phase 1 task.
"""

from dataclasses import dataclass, field
from copy import deepcopy
import torch

from mjlab.envs import mdp, ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.lab_api.math import sample_uniform

from mjlab_myochallenge.robot.myohand_constants import (
    DEFAULT_MYOHAND_CFG,
    VIEWER_CONFIG,
    SIM_CFG,
)

# Scene configuration
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=4,  # Number of parallel environments
    extent=2.0,
)


# ============ Rotation Utilities ============


def euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).

    Args:
        euler: Tensor of shape (..., 3) with [roll, pitch, yaw] in radians

    Returns:
        Quaternion tensor of shape (..., 4) with [w, x, y, z]
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).

    Args:
        quat: Tensor of shape (..., 4) with [w, x, y, z]

    Returns:
        Euler angles tensor of shape (..., 3) with [roll, pitch, yaw] in radians
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between two quaternions.

    Args:
        q1: First quaternion (..., 4)
        q2: Second quaternion (..., 4)

    Returns:
        Angular distance in radians (...,)
    """
    # Normalize quaternions
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)

    # Compute dot product
    dot = torch.abs(torch.sum(q1 * q2, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angular distance
    return 2 * torch.acos(dot)


def die_reorient_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create MyoHand Die Reorientation environment configuration.

    Args:
        play: If True, creates a play configuration (no randomization, longer episodes)

    Returns:
        ManagerBasedRlEnvCfg: Environment configuration for die reorientation task
    """

    ################# Observations #################

    # Initialize cached body/site IDs (will be populated on first use)
    _body_site_ids = {}

    def _get_body_site_ids(env: ManagerBasedRlEnv) -> dict:
        """Get and cache body/site IDs from MuJoCo model."""
        if not _body_site_ids:
            model = env.sim.model
            # Initialize all keys with -1 (not found)
            _body_site_ids["object_bid"] = -1
            _body_site_ids["goal_bid"] = -1
            _body_site_ids["object_sid"] = -1
            _body_site_ids["goal_sid"] = -1

            try:
                # Try to find die/object body
                for name in ["Object", "object", "die", "Die"]:
                    try:
                        _body_site_ids["object_bid"] = model.body(name).id
                        break
                    except Exception:
                        pass

                # Try to find goal/target body
                for name in ["target", "Target", "goal", "Goal"]:
                    try:
                        _body_site_ids["goal_bid"] = model.body(name).id
                        break
                    except Exception:
                        pass

                # Try to find object site
                for name in ["object_o", "object", "Object"]:
                    try:
                        _body_site_ids["object_sid"] = model.site(name).id
                        break
                    except Exception:
                        pass

                # Try to find goal site
                for name in ["target_o", "target", "goal"]:
                    try:
                        _body_site_ids["goal_sid"] = model.site(name).id
                        break
                    except Exception:
                        pass

            except Exception as e:
                print(f"Warning: Could not find all body/site IDs: {e}")

        return _body_site_ids

    def get_die_position(env: ManagerBasedRlEnv) -> torch.Tensor:
        """Get die center position."""
        ids = _get_body_site_ids(env)
        if ids["object_sid"] >= 0:
            # Use site position if available (more accurate)
            pos_idx = ids["object_sid"]
            return env.sim.data.site_xpos[:, pos_idx, :]
        elif ids["object_bid"] >= 0:
            # Fall back to body position
            pos_idx = ids["object_bid"]
            return env.sim.data.xpos[:, pos_idx, :]
        else:
            # Last resort: assume die is last body
            return env.sim.data.xpos[:, -1, :]

    def get_die_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
        """Get die orientation as quaternion (w, x, y, z)."""
        ids = _get_body_site_ids(env)
        if ids["object_bid"] >= 0:
            quat_idx = ids["object_bid"]
        else:
            # Assume die is last body with quaternion
            # qpos typically: [hand_joints..., die_pos_x, die_pos_y, die_pos_z, die_quat_w, die_quat_x, die_quat_y, die_quat_z]
            quat_idx = -1

        # Get quaternion from xquat (body quaternions)
        return env.sim.data.xquat[:, quat_idx, :]

    def get_goal_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
        """Get goal orientation as quaternion."""
        # Goal is stored in environment state
        if not hasattr(env, "_goal_quat"):
            # Initialize with identity quaternion
            env._goal_quat = torch.zeros((env.num_envs, 4), device=env.device)
            env._goal_quat[:, 0] = 1.0  # w = 1
        return env._goal_quat

    policy_terms = {
        # Hand joint positions (exclude die: last 7 DOFs if free body)
        "hand_qpos": ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:, :-7],
            noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
        ),
        # Hand joint velocities
        "hand_qvel": ObservationTermCfg(
            func=lambda env: env.sim.data.qvel[:, :-6],
            noise=Unoise(n_min=-0.1, n_max=0.1) if not play else None,
        ),
        # Die position
        "die_pos": ObservationTermCfg(
            func=get_die_position,
            noise=Unoise(n_min=-0.001, n_max=0.001) if not play else None,
        ),
        # Die orientation as Euler angles (more compact than quat)
        "die_euler": ObservationTermCfg(
            func=lambda env: quat_to_euler(get_die_quat(env)),
        ),
        # Goal orientation as Euler angles
        "goal_euler": ObservationTermCfg(
            func=lambda env: quat_to_euler(get_goal_quat(env)),
        ),
        # Previous action (for temporal information)
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    #################### Actions ###################

    # MyoHand muscle actuators: controlled via TendonEffortActionCfg
    actions: dict[str, ActionTermCfg] = {
        "myohand": mdp.TendonEffortActionCfg(
            entity_name="myohand",
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
        )
    }

    #################### Events ####################

    def reset_die_and_goal(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
    ) -> None:
        """Reset die and goal orientations."""
        n = len(env_ids)

        # Sample random goal orientations (Phase 1: limited range +-90 degrees)
        if play:
            # Fixed goal for playing
            goal_euler = torch.zeros((n, 3), device=env.device)
            goal_euler[:, 2] = 0.785  # 45 degrees around Z
        else:
            # Random goal within Phase 1 constraints (±90 degrees)
            goal_euler = sample_uniform(-1.57, 1.57, (n, 3), device=env.device)

        # Convert to quaternion and store
        goal_quat = euler_to_quat(goal_euler)
        if not hasattr(env, "_goal_quat"):
            env._goal_quat = torch.zeros((env.num_envs, 4), device=env.device)
            env._goal_quat[:, 0] = 1.0  # Initialize with identity
        env._goal_quat[env_ids] = goal_quat

        # Reset hand joints to small random values (open hand)
        hand_dof = env.sim.data.qpos.shape[1] - 7  # Exclude die (3 pos + 4 quat)
        env.sim.data.qpos[env_ids, :hand_dof] = sample_uniform(
            -0.05, 0.05, (n, hand_dof), device=env.device
        )

        # Reset die position (on palm, approximately)
        die_pos_start = hand_dof  # Die position starts after hand joints
        env.sim.data.qpos[env_ids, die_pos_start : die_pos_start + 3] = (
            torch.tensor(
                [0.015, 0.025, 0.025],  # x, y, z on palm
                device=env.device,
            )
            .unsqueeze(0)
            .expand(n, -1)
        )

        # Reset die orientation to identity (or slight random)
        die_quat_start = die_pos_start + 3
        if play:
            # Identity quaternion
            env.sim.data.qpos[env_ids, die_quat_start : die_quat_start + 4] = (
                torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],  # w, x, y, z
                    device=env.device,
                )
                .unsqueeze(0)
                .expand(n, -1)
            )
        else:
            # Small random orientation
            small_euler = sample_uniform(-0.2, 0.2, (n, 3), device=env.device)
            env.sim.data.qpos[env_ids, die_quat_start : die_quat_start + 4] = (
                euler_to_quat(small_euler)
            )

        # Reset all velocities to zero
        env.sim.data.qvel[env_ids, :] = 0.0

    events = {
        "reset_scene": EventTermCfg(
            func=reset_die_and_goal,
            mode="reset",
        ),
    }

    #################### Rewards ###################

    def orientation_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
        """Reward for die orientation matching goal orientation.

        Uses quaternion distance.
        """
        die_quat = get_die_quat(env)
        goal_quat = get_goal_quat(env)

        # Compute angular distance
        ang_dist = quat_distance(die_quat, goal_quat)

        # Convert to reward: 1.0 when aligned, 0 when 90 degrees off
        # Use exponential decay
        reward = torch.exp(-ang_dist / 0.5)

        return reward

    def position_reward(env: ManagerBasedRlEnv, std: float = 0.05) -> torch.Tensor:
        """Reward for keeping die on palm (not dropping)."""
        die_pos = get_die_position(env)

        # Target position is on palm center
        target_pos = torch.tensor([0.015, 0.025, 0.025], device=env.device)
        target_pos = target_pos.unsqueeze(0).expand(env.num_envs, -1)

        # Distance from target position
        pos_dist = torch.norm(die_pos - target_pos, dim=-1)

        # Gaussian reward
        return torch.exp(-(pos_dist**2) / (2 * std**2))

    def action_regularization(env: ManagerBasedRlEnv) -> torch.Tensor:
        """Penalize large muscle activations."""
        # Get last action from action manager
        try:
            act = env.action_manager.get_term("myohand").raw_action
        except (AttributeError, KeyError):
            # Fallback: use zeros
            act = torch.zeros((env.num_envs, 1), device=env.device)

        return -torch.mean(act**2, dim=-1)

    rewards = {
        "orientation": RewardTermCfg(
            func=orientation_reward,
            weight=10.0,
        ),
        "position": RewardTermCfg(
            func=position_reward,
            weight=1.0,
            params={"std": 0.05},
        ),
        "action_reg": RewardTermCfg(
            func=action_regularization,
            weight=0.001,
        ),
    }

    ################# Terminations #################

    def check_die_dropped(env: ManagerBasedRlEnv, drop_th: float = 0.1) -> torch.Tensor:
        """Check if die has dropped (fallen off hand)."""
        die_pos = get_die_position(env)

        # Check if die is too far from palm in X-Y plane or too low in Z
        palm_center = torch.tensor([0.015, 0.025, 0.025], device=env.device)
        palm_center = palm_center.unsqueeze(0).expand(env.num_envs, -1)

        # Distance from palm center
        xy_dist = torch.norm(die_pos[:, :2] - palm_center[:, :2], dim=-1)
        z_pos = die_pos[:, 2]

        # Dropped if too far from palm or below ground
        dropped = (xy_dist > drop_th) | (z_pos < 0.0)

        return dropped

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "die_dropped": TerminationTermCfg(
            func=lambda env: check_die_dropped(env, drop_th=0.1),
            time_out=False,
        ),
    }

    ################# Configuration ################

    cfg = ManagerBasedRlEnvCfg(
        scene=deepcopy(SCENE_CFG),
        viewer=deepcopy(VIEWER_CONFIG),
        sim=deepcopy(SIM_CFG),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        decimation=5,  # Control at 100 Hz (500 Hz sim / 5)
        episode_length_s=6.0 if not play else 20.0,  # 6 seconds for training, 20 for play
    )

    # Set the entity
    cfg.scene.entities = {"myohand": DEFAULT_MYOHAND_CFG}

    return cfg


@dataclass
class DieReorientRlCfg(RslRlOnPolicyRunnerCfg):
    """RL training configuration for Die Reorientation task."""

    policy: RslRlPpoActorCriticCfg = field(
        default_factory=lambda: RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(256, 128, 64),
            activation="elu",
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
    wandb_project: str = "mjlab_myochallenge"
    experiment_name: str = "myohand_die_reorient"
    save_interval: int = 200
    num_steps_per_env: int = 24
    max_iterations: int = 50_000
