"""Shared utilities for the MyoChallenge 2024 Bimanual Manipulation MDP.

Bimanual model joint layout (before mjlab spec.attach() prefixing):
  joints 0-37   : MyoArm (sternoclavicular_r2 … md5_flexion) — 38 joints
  joints 38-63  : Prosthesis (prosthesis/Lshoulder_fe … prosthesis/pinky_DIP) — 26 joints
  joint  64     : manip_object/freejoint (type=0, free) — 7 qpos / 6 qvel DOFs

After scene assembly, mjlab prefixes all names with the entity key ("bimanual/"),
so body names become e.g. "bimanual/manip_object".  We resolve IDs dynamically
by matching name suffixes so the code works regardless of scene prefix.
"""

import mujoco
import torch

# Module-level cache keyed on mj_model object id
_cached_ids: dict[int, dict] = {}

# Name patterns (suffix matching)
_MANIP_BODY_SUFFIX = "manip_object"
_START_BODY_SUFFIX = "start"
_GOAL_BODY_SUFFIX = "goal"
_TOUCH_SITE_SUFFIX = "touch_site"
_PALM_SITE_SUFFIX = "S_grasp"
_RPALM1_SITE_SUFFIX = "prosthesis/palm_thumb"
_RPALM2_SITE_SUFFIX = "prosthesis/palm_pinky"


def get_bimanual_model_ids(env) -> dict:
    """Resolve and cache bimanual body/site/qpos addresses.

    Returns a dict with keys:
      manip_bid      – body index of manip_object
      start_bid      – body index of start pillar
      goal_bid       – body index of goal pillar
      touch_sid      – site index of touch_site (on object)
      palm_sid       – site index of MyoArm palm grasp site
      rpalm1_sid     – site index of prosthesis palm_thumb site
      rpalm2_sid     – site index of prosthesis palm_pinky site
      obj_qposadr    – qpos address of manip_object freejoint (pos x)
      obj_qveladr    – qvel address of manip_object freejoint (vel x)
      myo_qpos_end   – qpos address just before prosthesis joints
      prosth_qpos_start – qpos address of first prosthesis joint
      prosth_qpos_end   – qpos address just after last prosthesis joint
      mpl_ctrl_ids   – local ctrl indices for MPL position actuators (tensor)
    """
    m = env.sim.mj_model
    mid = id(m)
    if mid in _cached_ids:
        return _cached_ids[mid]

    ids: dict = {
        "manip_bid": -1,
        "start_bid": -1,
        "goal_bid": -1,
        "touch_sid": -1,
        "palm_sid": -1,
        "rpalm1_sid": -1,
        "rpalm2_sid": -1,
        "obj_qposadr": -1,
        "obj_qveladr": -1,
        "myo_qpos_end": -1,
        "prosth_qpos_start": -1,
        "prosth_qpos_end": -1,
        "mpl_ctrl_ids": None,
    }

    # --- body IDs ---
    for i in range(m.nbody):
        name = m.body(i).name
        if name.endswith(_MANIP_BODY_SUFFIX):
            ids["manip_bid"] = i
        elif name.endswith(_START_BODY_SUFFIX):
            ids["start_bid"] = i
        elif name.endswith(_GOAL_BODY_SUFFIX):
            ids["goal_bid"] = i

    # --- site IDs ---
    for i in range(m.nsite):
        name = m.site(i).name
        if name.endswith(_TOUCH_SITE_SUFFIX):
            ids["touch_sid"] = i
        elif name.endswith(_PALM_SITE_SUFFIX):
            ids["palm_sid"] = i
        elif name.endswith(_RPALM1_SITE_SUFFIX):
            ids["rpalm1_sid"] = i
        elif name.endswith(_RPALM2_SITE_SUFFIX):
            ids["rpalm2_sid"] = i

    # --- freejoint qpos/qvel addresses ---
    for i in range(m.njnt):
        if int(m.jnt_type[i]) == mujoco.mjtJoint.mjJNT_FREE:
            # This is the manip_object freejoint
            ids["obj_qposadr"] = int(m.jnt_qposadr[i])
            ids["obj_qveladr"] = int(m.jnt_dofadr[i])

    # --- prosthesis qpos range ---
    # Find the first joint whose name contains "prosthesis"
    prosth_start_adr = None
    prosth_end_adr = None
    for i in range(m.njnt):
        name = m.joint(i).name
        if "prosthesis" in name and int(m.jnt_type[i]) != mujoco.mjtJoint.mjJNT_FREE:
            adr = int(m.jnt_qposadr[i])
            if prosth_start_adr is None or adr < prosth_start_adr:
                prosth_start_adr = adr
            if prosth_end_adr is None or adr > prosth_end_adr:
                prosth_end_adr = adr + 1  # each hinge joint = 1 qpos DOF

    ids["prosth_qpos_start"] = prosth_start_adr if prosth_start_adr is not None else -1
    ids["prosth_qpos_end"] = prosth_end_adr if prosth_end_adr is not None else -1
    ids["myo_qpos_end"] = prosth_start_adr if prosth_start_adr is not None else -1

    # --- MPL position actuator ctrl indices ---
    # MPL actuators have dyntype != mjDYN_MUSCLE (= 4)
    # They correspond to the prosthesis/ position actuators
    mpl_ctrl = []
    for i in range(m.nu):
        dyn = int(m.actuator_dyntype[i])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if dyn != mujoco.mjtDyn.mjDYN_MUSCLE and "prosthesis" in name:
            mpl_ctrl.append(i)
    ids["mpl_ctrl_ids"] = torch.tensor(mpl_ctrl, device=env.device, dtype=torch.long)

    _cached_ids[mid] = ids
    return ids


def get_object_pos(env) -> torch.Tensor:
    """Return manip_object world position (num_envs, 3) from xpos."""
    ids = get_bimanual_model_ids(env)
    if ids["manip_bid"] >= 0:
        return env.sim.data.xpos[:, ids["manip_bid"], :]
    return env.sim.data.xpos[:, -1, :]


def get_object_quat(env) -> torch.Tensor:
    """Return manip_object orientation quaternion (w,x,y,z) (num_envs, 4)."""
    ids = get_bimanual_model_ids(env)
    idx = ids["manip_bid"] if ids["manip_bid"] >= 0 else -1
    return env.sim.data.xquat[:, idx, :]


def get_goal_pos(env) -> torch.Tensor:
    """Return current goal position stored on env by reset event (num_envs, 3)."""
    if not hasattr(env, "_bimanual_goal_pos"):
        env._bimanual_goal_pos = torch.zeros((env.num_envs, 3), device=env.device)
        env._bimanual_goal_pos[:, 0] = 0.4
        env._bimanual_goal_pos[:, 1] = -0.25
        env._bimanual_goal_pos[:, 2] = 1.05
    return env._bimanual_goal_pos


def get_mpl_ctrl_range(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ctrl_lo, ctrl_hi) tensors for MPL position actuators."""
    if not hasattr(env, "_mpl_ctrl_lo"):
        ids = get_bimanual_model_ids(env)
        ctrl_ids = ids["mpl_ctrl_ids"].cpu().numpy()
        ctrl_range = env.sim.mj_model.actuator_ctrlrange[ctrl_ids]
        env._mpl_ctrl_lo = torch.tensor(ctrl_range[:, 0], device=env.device)
        env._mpl_ctrl_hi = torch.tensor(ctrl_range[:, 1], device=env.device)
    return env._mpl_ctrl_lo, env._mpl_ctrl_hi
