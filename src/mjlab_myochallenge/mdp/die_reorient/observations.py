"""Observation terms for MyoChallenge tasks."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from .utils import (
    get_die_position,
    get_die_quat,
    get_goal_quat,
    get_model_ids,
    quat_to_euler,
)


def hand_qpos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hand joint positions (all DOFs except the die's 6-DOF joint block)."""
    ids = get_model_ids(env)
    adr = ids["die_qposadr"]
    return env.sim.data.qpos[:, :adr]


def hand_qvel(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hand joint velocities (all DOFs except the die's 6 velocity DOFs)."""
    ids = get_model_ids(env)
    ndof = ids["die_ndof"]
    return env.sim.data.qvel[:, :-ndof]


def die_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Die centre world position (num_envs, 3) from Warp forward kinematics."""
    return get_die_position(env)


def die_euler(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Die orientation as Euler angles (roll, pitch, yaw) from Warp FK quaternion."""
    return quat_to_euler(get_die_quat(env))


def goal_euler(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Goal orientation as Euler angles (roll, pitch, yaw)."""
    return quat_to_euler(get_goal_quat(env))
