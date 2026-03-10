"""Event (reset) terms for MyoChallenge tasks."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import sample_uniform

from .utils import euler_to_quat, get_model_ids


def reset_die_and_goal(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    play: bool = False,
) -> None:
    """Reset die pose and sample a new goal orientation.

    Die joint structure: 3 slide (x/y/z) + 3 hinge (rx/ry/rz).
    Slide values are displacements FROM the MJCF default palm position,
    so setting them to 0 places the die on the palm.

    Args:
        env_ids: Indices of environments being reset.
        play:    Fixed 45° goal and identity die orientation (no randomisation).
    """
    n = len(env_ids)
    ids = get_model_ids(env)
    qposadr = ids["die_qposadr"]   # first slide joint (x)
    ndof = ids["die_ndof"]         # 6  (3 slide + 3 hinge)
    hand_dof = qposadr             # hand joints occupy qpos[0:qposadr]

    # ---- goal orientation ----
    if play:
        goal_euler = torch.zeros((n, 3), device=env.device)
        goal_euler[:, 2] = 0.785  # 45° around Z
    else:
        goal_euler = sample_uniform(-1.57, 1.57, (n, 3), device=env.device)

    if not hasattr(env, "_goal_quat"):
        env._goal_quat = torch.zeros((env.num_envs, 4), device=env.device)
        env._goal_quat[:, 0] = 1.0
    env._goal_quat[env_ids] = euler_to_quat(goal_euler)

    # ---- hand joints: start from "palm up" pose (pro_sup = -1.5) ----
    # The original myosuite env sets qpos[0] = -1.5 (pro_sup, pronation/supination)
    # so that the palm faces up and supports the die.  All other joints default to 0.
    hand_base = torch.zeros((n, hand_dof), device=env.device)
    hand_base[:, 0] = -1.5  # pro_sup: palm-up orientation
    noise = sample_uniform(-0.05, 0.05, (n, hand_dof), device=env.device)
    env.sim.data.qpos[env_ids, :hand_dof] = hand_base + noise

    # ---- die position (slide DOFs = 0 → resting at default palm position) ----
    env.sim.data.qpos[env_ids, qposadr : qposadr + 3] = 0.0

    # ---- die orientation (hinge DOFs) ----
    rot_start = qposadr + 3
    if play:
        env.sim.data.qpos[env_ids, rot_start : rot_start + 3] = 0.0
    else:
        env.sim.data.qpos[env_ids, rot_start : rot_start + 3] = sample_uniform(
            -0.2, 0.2, (n, 3), device=env.device
        )

    # ---- zero all velocities ----
    env.sim.data.qvel[env_ids, :] = 0.0
