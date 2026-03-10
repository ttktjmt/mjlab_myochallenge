"""Reward terms for MyoChallenge tasks."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .utils import get_die_quat, get_die_slide_qpos, get_goal_quat, quat_distance


def orientation_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward for die orientation matching the goal.

    Uses quaternion distance from Warp FK (xquat).
    Returns exp(-angular_distance / 0.5): 1.0 when aligned, ~0 at 90°.
    """
    ang_dist = quat_distance(get_die_quat(env), get_goal_quat(env))
    return torch.exp(-ang_dist / 0.5)


def position_reward(env: ManagerBasedRlEnv, std: float = 0.05) -> torch.Tensor:
    """Gaussian reward for keeping the die on the palm.

    Uses die slide-joint displacement from default palm position (0 = on palm).
    """
    slides = get_die_slide_qpos(env)  # (num_envs, 3): [x, y, z] displacement
    dist = torch.norm(slides, dim=-1)
    return torch.exp(-(dist**2) / (2 * std**2))


def action_regularization(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalise large muscle activations (negative reward)."""
    try:
        act = env.action_manager.get_term("myohand").raw_action
    except (AttributeError, KeyError):
        act = torch.zeros((env.num_envs, 1), device=env.device)
    return -torch.mean(act**2, dim=-1)
