"""Reward terms for the MyoChallenge 2024 Bimanual Manipulation task.

Reward design mirrors the original BimanualEnvV1:
  - reach_reward    : negative distance from MyoArm palm to object
  - pass_reward     : negative distance from MPL palm to object
  - goal_reward     : shaped reward for object proximity to goal
  - action_reg      : penalise large muscle activations
"""

import torch
from mjlab.envs import ManagerBasedRlEnv

from .utils import get_bimanual_model_ids, get_goal_pos, get_object_pos


def _palm_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """MyoArm palm (S_grasp site) world position (num_envs, 3)."""
    ids = get_bimanual_model_ids(env)
    sid = ids["palm_sid"]
    if sid >= 0:
        return env.sim.data.site_xpos[:, sid, :]
    return get_object_pos(env)  # fallback


def _rpalm_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Prosthesis palm centre (average of palm_thumb and palm_pinky sites) (num_envs, 3)."""
    ids = get_bimanual_model_ids(env)
    sid1, sid2 = ids["rpalm1_sid"], ids["rpalm2_sid"]
    if sid1 >= 0 and sid2 >= 0:
        return 0.5 * (
            env.sim.data.site_xpos[:, sid1, :] + env.sim.data.site_xpos[:, sid2, :]
        )
    return get_object_pos(env)  # fallback


def reach_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Shaped reward for MyoArm palm approaching the object.

    Returns -(dist + log(dist + ε)) to provide dense gradient.
    """
    obj = get_object_pos(env)
    palm = _palm_pos(env)
    dist = torch.norm(palm - obj, dim=-1)
    return -(dist + torch.log(dist + 1e-6))


def pass_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Shaped reward for MPL palm approaching the object.

    Returns -(dist + log(dist + ε)).
    """
    obj = get_object_pos(env)
    rpalm = _rpalm_pos(env)
    dist = torch.norm(rpalm - obj, dim=-1)
    return -(dist + torch.log(dist + 1e-3))


def goal_reward(env: ManagerBasedRlEnv, proximity_th: float = 0.17) -> torch.Tensor:
    """Shaped reward for object proximity to the goal pillar.

    Returns a Gaussian centred on the goal (peaks at 1.0 when aligned,
    tapers smoothly with distance).
    """
    obj = get_object_pos(env)
    goal = get_goal_pos(env)
    dist = torch.norm(obj - goal, dim=-1)
    return torch.exp(-dist / proximity_th)


def action_regularization(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalise large MyoArm muscle activations (negative reward)."""
    try:
        act = env.action_manager.get_term("myoarm").raw_action
    except (AttributeError, KeyError):
        act = torch.zeros((env.num_envs, 1), device=env.device)
    return -torch.mean(act**2, dim=-1)
