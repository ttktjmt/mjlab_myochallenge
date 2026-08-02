"""Termination terms for MyoChallenge tasks."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from .utils import get_die_quat, get_die_slide_qpos, get_goal_quat, quat_distance


def die_dropped(env: ManagerBasedRlEnv, drop_th: float = 0.1) -> torch.Tensor:
    """True when the die has slipped off the palm.

    Measures the die's slide-joint displacement from the default palm position
    (0 = on palm).  Triggers when XY displacement exceeds ``drop_th`` or the
    die falls below the palm in Z.

    Args:
        drop_th: Maximum XY displacement (m) before triggering.
    """
    slides = get_die_slide_qpos(env)  # (num_envs, 3): [x, y, z] displacement
    xy_dist = torch.norm(slides[:, :2], dim=-1)
    z_drop = slides[:, 2] < -drop_th  # fallen more than drop_th below palm
    return (xy_dist > drop_th) | z_drop


def goal_held(
    env: ManagerBasedRlEnv,
    orientation_th_rad: float = 0.35,
    position_th_m: float = 0.04,
    hold_steps: int = 25,
) -> torch.Tensor:
    """True when die orientation+position stay within thresholds for ``hold_steps``.

    This gives a concrete success criterion for faster-first-success tuning and
    allows measuring success episodes directly from termination causes.
    """
    ang_dist = quat_distance(get_die_quat(env), get_goal_quat(env))
    slide_dist = torch.norm(get_die_slide_qpos(env), dim=-1)
    is_success = (ang_dist < orientation_th_rad) & (slide_dist < position_th_m)

    if not hasattr(env, "_die_success_hold_count"):
        env._die_success_hold_count = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    count = torch.where(
        is_success,
        env._die_success_hold_count + 1,
        torch.zeros_like(env._die_success_hold_count),
    )
    # A hold can never be longer than the episode itself, so clamping drops any
    # count carried over from a previous episode (there is no per-term reset hook).
    env._die_success_hold_count = torch.minimum(count, env.episode_length_buf)
    return env._die_success_hold_count >= hold_steps
