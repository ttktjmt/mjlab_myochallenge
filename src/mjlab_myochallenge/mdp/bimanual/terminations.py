"""Termination terms for the MyoChallenge 2024 Bimanual Manipulation task."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .utils import get_bimanual_model_ids, get_object_pos


def object_dropped(env: ManagerBasedRlEnv, drop_z_th: float = 0.3) -> torch.Tensor:
    """True when the object has fallen below the drop threshold.

    The original BimanualEnvV1 terminates when object z < 0.3 m (below
    the table surface level).

    Args:
        drop_z_th: Minimum object z (m) before terminating.
    """
    obj_pos = get_object_pos(env)  # (num_envs, 3)
    return obj_pos[:, 2] < drop_z_th
