"""Shared utilities for MyoChallenge MDP terms.

Provides rotation helpers and MuJoCo model ID resolution used across
observation, reward, termination, and event term modules.

Die joint structure (myoChallengeDieReorientP1):
  The die body has 6 joints — 3 slide (x/y/z) + 3 hinge (rx/ry/rz) — NOT a
  free joint.  Slide values are displacements FROM the model's default palm
  position, so qpos[die_qposadr : die_qposadr+3] == 0  means the die is
  sitting on the palm.
"""

import torch

# Module-level cache: mj_model object id → resolved IDs
_cached_ids: dict[int, dict[str, int]] = {}

# Body/site name suffixes (mjlab prefixes "{entity}/" after spec.attach())
_OBJECT_BODY_SUFFIXES = ("/Object", "/object", "/die", "/Die")
_GOAL_BODY_SUFFIXES = ("/target", "/Target", "/goal", "/Goal")
_OBJECT_SITE_SUFFIXES = ("/object_o",)
_GOAL_SITE_SUFFIXES = ("/target_o",)


def get_model_ids(env) -> dict[str, int]:
    """Resolve and cache die/goal body, site, and qpos IDs.

    Reads ``env.sim.mj_model`` and matches names by suffix to handle the
    ``entity_name/`` prefix that mjlab adds on ``spec.attach()``.

    Keys returned:
      object_bid  - body index of the die
      goal_bid    - body index of the goal marker
      object_sid  - site index of the die centre site
      goal_sid    - site index of the goal centre site
      die_qposadr - qpos address of the die's first joint (x-slide)
      die_ndof    - number of die qpos DOFs (6 for 3 slide + 3 hinge)
    """
    m = env.sim.mj_model
    model_id = id(m)
    if model_id in _cached_ids:
        return _cached_ids[model_id]

    ids: dict[str, int] = {
        "object_bid": -1,
        "goal_bid": -1,
        "object_sid": -1,
        "goal_sid": -1,
        "die_qposadr": -1,
        "die_ndof": 0,
    }

    for i in range(m.nbody):
        name = m.body(i).name
        if ids["object_bid"] < 0 and name.endswith(_OBJECT_BODY_SUFFIXES):
            ids["object_bid"] = i
            # Resolve qpos address from the body's first joint
            jntadr = int(m.body_jntadr[i])
            jntnum = int(m.body_jntnum[i])
            if jntadr >= 0 and jntnum > 0:
                ids["die_qposadr"] = int(m.jnt_qposadr[jntadr])
                ids["die_ndof"] = jntnum
        if ids["goal_bid"] < 0 and name.endswith(_GOAL_BODY_SUFFIXES):
            ids["goal_bid"] = i

    for i in range(m.nsite):
        name = m.site(i).name
        if ids["object_sid"] < 0 and name.endswith(_OBJECT_SITE_SUFFIXES):
            ids["object_sid"] = i
        if ids["goal_sid"] < 0 and name.endswith(_GOAL_SITE_SUFFIXES):
            ids["goal_sid"] = i

    _cached_ids[model_id] = ids
    return ids


def get_die_position(env) -> torch.Tensor:
    """Return die centre world position (num_envs, 3) from Warp FK."""
    ids = get_model_ids(env)
    if ids["object_bid"] >= 0:
        return env.sim.data.xpos[:, ids["object_bid"], :]
    return env.sim.data.xpos[:, -1, :]


def get_die_quat(env) -> torch.Tensor:
    """Return die orientation quaternion (w, x, y, z) (num_envs, 4) from Warp FK."""
    ids = get_model_ids(env)
    quat_idx = ids["object_bid"] if ids["object_bid"] >= 0 else -1
    return env.sim.data.xquat[:, quat_idx, :]


def get_die_slide_qpos(env) -> torch.Tensor:
    """Return die slide-joint values (num_envs, 3): displacement from default palm pos."""
    ids = get_model_ids(env)
    adr = ids["die_qposadr"]
    return env.sim.data.qpos[:, adr : adr + 3]


def get_goal_quat(env) -> torch.Tensor:
    """Return goal orientation quaternion (num_envs, 4).

    Stored on the env by the reset event; defaults to identity.
    """
    if not hasattr(env, "_goal_quat"):
        env._goal_quat = torch.zeros((env.num_envs, 4), device=env.device)
        env._goal_quat[:, 0] = 1.0  # w = 1 (identity)
    return env._goal_quat


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) → quaternion (w, x, y, z).

    Args:
        euler: (..., 3) tensor [roll, pitch, yaw] in radians.
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) → Euler angles (roll, pitch, yaw).

    Args:
        quat: (..., 4) tensor [w, x, y, z].

    Returns:
        (..., 3) tensor [roll, pitch, yaw] in radians.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance (radians) between two quaternions.

    Args:
        q1: (..., 4)
        q2: (..., 4)

    Returns:
        (...,) angular distance in radians.
    """
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)
    dot = torch.clamp(torch.abs(torch.sum(q1 * q2, dim=-1)), 0.0, 1.0)
    return 2 * torch.acos(dot)
