"""Event (reset) terms for the MyoChallenge 2024 Bimanual Manipulation task.

Reset logic mirrors BimanualEnvV1.reset():
  1. Randomise start and goal pillar positions within ±shift bounds.
  2. Set arm joint positions from keyframe[2] (standard grasp-ready pose).
  3. Place object at start_pos + [0, 0, 0.1] (above start pillar).
  4. Zero all velocities.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import sample_uniform

from .utils import get_bimanual_model_ids

# Default pillar centres (metres)
_START_CENTER = torch.tensor([-0.4, -0.25, 1.05])
_GOAL_CENTER = torch.tensor([0.4, -0.25, 1.05])

# Randomisation radii (x, y, z — z is fixed)
_START_SHIFTS = torch.tensor([0.055, 0.055, 0.0])
_GOAL_SHIFTS = torch.tensor([0.098, 0.098, 0.0])

# Keyframe index used for arm initialisation (matches BimanualEnvV1)
_INIT_KEY_IDX = 2


def reset_bimanual(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    play: bool = False,
) -> None:
    """Reset bimanual scene: arm poses, start/goal positions, and object position.

    Args:
        env_ids: Indices of environments being reset.
        play:    If True, use fixed start/goal positions (no randomisation).
    """
    n = len(env_ids)
    ids = get_bimanual_model_ids(env)

    dev = env.device
    start_center = _START_CENTER.to(dev)
    goal_center = _GOAL_CENTER.to(dev)
    start_shifts = _START_SHIFTS.to(dev)
    goal_shifts = _GOAL_SHIFTS.to(dev)

    # ---- randomise start/goal pillar positions ----
    if play:
        start_pos = start_center.unsqueeze(0).expand(n, -1)
        goal_pos = goal_center.unsqueeze(0).expand(n, -1)
    else:
        rand_s = sample_uniform(-1.0, 1.0, (n, 3), device=dev)
        rand_g = sample_uniform(-1.0, 1.0, (n, 3), device=dev)
        start_pos = start_center + rand_s * start_shifts
        goal_pos = goal_center + rand_g * goal_shifts

    # Store goal position on env for reward/obs access
    if not hasattr(env, "_bimanual_goal_pos"):
        env._bimanual_goal_pos = torch.zeros((env.num_envs, 3), device=dev)
    env._bimanual_goal_pos[env_ids] = goal_pos

    # ---- arm joint positions from keyframe[2] ----
    # key_qpos shape: (n_keys, nq)
    key_qpos = env.sim.mj_model.key_qpos
    n_keys = key_qpos.shape[0]
    key_idx = min(_INIT_KEY_IDX, n_keys - 1)
    init_qpos = torch.tensor(key_qpos[key_idx], device=dev, dtype=torch.float32)

    obj_adr = ids["obj_qposadr"]  # start of freejoint qpos block

    # Set MyoArm + Prosthesis joints (all DOFs before freejoint)
    if obj_adr > 0:
        env.sim.data.qpos[env_ids, :obj_adr] = init_qpos[:obj_adr].unsqueeze(0)

    # ---- object position: start_pos + 0.1 m height ----
    obj_start = start_pos.clone()
    obj_start[:, 2] = obj_start[:, 2] + 0.1

    # freejoint qpos: [x, y, z, qw, qx, qy, qz]
    # Use same orientation as keyframe (identity-ish: last 4 values)
    obj_quat = torch.tensor(key_qpos[key_idx, obj_adr + 3 : obj_adr + 7], device=dev)
    env.sim.data.qpos[env_ids, obj_adr : obj_adr + 3] = obj_start
    env.sim.data.qpos[env_ids, obj_adr + 3 : obj_adr + 7] = obj_quat.unsqueeze(0)

    # ---- zero all velocities ----
    env.sim.data.qvel[env_ids, :] = 0.0

    # ---- update mocap (pillar) body positions ----
    start_bid = ids["start_bid"]
    goal_bid = ids["goal_bid"]
    if start_bid >= 0:
        env.sim.mj_model.body_pos[start_bid] = start_pos[0].cpu().numpy()
    if goal_bid >= 0:
        env.sim.mj_model.body_pos[goal_bid] = goal_pos[0].cpu().numpy()
