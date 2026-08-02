"""Observation terms for the MyoChallenge 2024 Bimanual Manipulation task."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from .utils import (
    get_bimanual_model_ids,
    get_goal_pos,
    get_object_pos,
)


def myoarm_qpos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """MyoArm joint positions (all DOFs up to prosthesis start)."""
    ids = get_bimanual_model_ids(env)
    end = ids["myo_qpos_end"]
    return env.sim.data.qpos[:, :end]


def myoarm_qvel(env: ManagerBasedRlEnv) -> torch.Tensor:
    """MyoArm joint velocities (all DOFs up to prosthesis start)."""
    ids = get_bimanual_model_ids(env)
    end = ids["myo_qpos_end"]  # qpos and qvel have same count for hinge joints
    return env.sim.data.qvel[:, :end]


def mpl_qpos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Prosthesis (MPL) joint positions."""
    ids = get_bimanual_model_ids(env)
    return env.sim.data.qpos[:, ids["prosth_qpos_start"] : ids["prosth_qpos_end"]]


def mpl_qvel(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Prosthesis (MPL) joint velocities."""
    ids = get_bimanual_model_ids(env)
    return env.sim.data.qvel[:, ids["prosth_qpos_start"] : ids["prosth_qpos_end"]]


def object_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Manipulation object world position (num_envs, 3)."""
    return get_object_pos(env)


def object_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Manipulation object orientation quaternion (w,x,y,z) (num_envs, 4)."""
    ids = get_bimanual_model_ids(env)
    idx = ids["manip_bid"] if ids["manip_bid"] >= 0 else -1
    return env.sim.data.xquat[:, idx, :]


def goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Goal pillar position stored on env by reset event (num_envs, 3)."""
    return get_goal_pos(env)


def object_to_goal_vec(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Vector from object to goal position (num_envs, 3)."""
    return get_goal_pos(env) - get_object_pos(env)
