"""Termination terms for the MyoChallenge 2024 Bimanual Manipulation task."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from .utils import get_goal_pos, get_object_pos


def object_dropped(env: ManagerBasedRlEnv, drop_z_th: float = 0.3) -> torch.Tensor:
    """True when the object has fallen below the drop threshold.

    The original BimanualEnvV1 terminates when object z < 0.3 m (below
    the table surface level).

    Args:
        drop_z_th: Minimum object z (m) before terminating.
    """
    obj_pos = get_object_pos(env)  # (num_envs, 3)
    return obj_pos[:, 2] < drop_z_th


def goal_held(
    env: ManagerBasedRlEnv,
    goal_dist_th_m: float = 0.12,
    hold_steps: int = 25,
) -> torch.Tensor:
    """True when object-goal distance stays within threshold for ``hold_steps``."""
    goal_dist = torch.norm(get_object_pos(env) - get_goal_pos(env), dim=-1)
    is_success = goal_dist < goal_dist_th_m

    if not hasattr(env, "_bimanual_success_hold_count"):
        env._bimanual_success_hold_count = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    count = torch.where(
        is_success,
        env._bimanual_success_hold_count + 1,
        torch.zeros_like(env._bimanual_success_hold_count),
    )
    # A hold can never be longer than the episode itself, so clamping drops any
    # count carried over from a previous episode (there is no per-term reset hook).
    env._bimanual_success_hold_count = torch.minimum(count, env.episode_length_buf)
    return env._bimanual_success_hold_count >= hold_steps
