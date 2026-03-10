"""Termination terms for MyoChallenge tasks."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .utils import get_die_slide_qpos


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
