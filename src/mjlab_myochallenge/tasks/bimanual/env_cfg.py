"""Environment configuration for MyoChallenge 2024 Bimanual Manipulation.

Follows the mjlab factory-function pattern: ``bimanual_env_cfg(play=False)``
returns a fully composed ``ManagerBasedRlEnvCfg``.

Task: Transfer a gelatin box from a start pillar to a goal pillar using
coordinated control of:
  - MyoArm: 63 muscle actuators (tendon effort, [0, 1])
  - Prosthesis (MPL): 17 position actuators (MplJointPositionAction, [-1, 1])

Total action dim: 80 (63 MyoArm + 17 MPL)

Observations (per policy):
  - myoarm_qpos  : 38 dims
  - myoarm_qvel  : 38 dims
  - mpl_qpos     : 26 dims
  - mpl_qvel     : 26 dims
  - object_pos   :  3 dims
  - object_quat  :  4 dims
  - goal_pos     :  3 dims
  - last_action  : 80 dims
  Total: ~218 dims
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_myochallenge.mdp.bimanual import events as bevt
from mjlab_myochallenge.mdp.bimanual import observations as bobs
from mjlab_myochallenge.mdp.bimanual import rewards as brwd
from mjlab_myochallenge.mdp.bimanual import terminations as bterm
from mjlab_myochallenge.mdp.bimanual.actions import MplJointPositionActionCfg
from mjlab_myochallenge.models.bimanual import (
    BIMANUAL_SIM_CFG,
    BIMANUAL_VIEWER_CONFIG,
    DEFAULT_BIMANUAL_CFG,
)


def bimanual_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create MyoChallenge 2024 Bimanual Manipulation environment configuration.

    Args:
        play: If True, disables observation noise and domain randomisation,
              fixes start/goal positions, and extends the episode length.

    Returns:
        ManagerBasedRlEnvCfg ready for ``register_mjlab_task``.
    """

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    policy_terms = {
        "myoarm_qpos": ObservationTermCfg(
            func=bobs.myoarm_qpos,
            noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
        ),
        "myoarm_qvel": ObservationTermCfg(
            func=bobs.myoarm_qvel,
            noise=Unoise(n_min=-0.1, n_max=0.1) if not play else None,
        ),
        "mpl_qpos": ObservationTermCfg(
            func=bobs.mpl_qpos,
            noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
        ),
        "mpl_qvel": ObservationTermCfg(
            func=bobs.mpl_qvel,
            noise=Unoise(n_min=-0.1, n_max=0.1) if not play else None,
        ),
        "object_pos": ObservationTermCfg(
            func=bobs.object_pos,
            noise=Unoise(n_min=-0.002, n_max=0.002) if not play else None,
        ),
        "object_quat": ObservationTermCfg(func=bobs.object_quat),
        "goal_pos": ObservationTermCfg(func=bobs.goal_pos),
        "object_to_goal": ObservationTermCfg(func=bobs.object_to_goal_vec),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    obs_cfg = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ------------------------------------------------------------------
    # Actions
    # MyoArm: 63 muscle actuators via tendon effort (activation [0, 1])
    # MPL:    17 position actuators via custom ctrl-direct action ([-1, 1])
    # ------------------------------------------------------------------
    actions_cfg: dict[str, ActionTermCfg] = {
        "myoarm": mdp.TendonEffortActionCfg(
            entity_name="bimanual",
            actuator_names=(".*_tendon",),
            scale=1.0,
            offset=0.0,
        ),
        "mpl": MplJointPositionActionCfg(
            entity_name="bimanual",
            scale=1.0,
        ),
    }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    events_cfg = {
        "reset_scene": EventTermCfg(
            func=bevt.reset_bimanual,
            mode="reset",
            params={"play": play},
        ),
    }

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    rewards_cfg = {
        "reach": RewardTermCfg(
            func=brwd.reach_reward,
            weight=0.3,
        ),
        "pass": RewardTermCfg(
            func=brwd.pass_reward,
            weight=1.2,
        ),
        "goal": RewardTermCfg(
            func=brwd.goal_reward,
            weight=3.5,
        ),
        "action_reg": RewardTermCfg(
            func=brwd.action_regularization,
            weight=0.002,
        ),
    }

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------
    terminations_cfg = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "object_dropped": TerminationTermCfg(
            func=bterm.object_dropped,
            time_out=False,
            params={"drop_z_th": 0.22},
        ),
        "goal_held": TerminationTermCfg(
            func=bterm.goal_held,
            time_out=False,
            params={"goal_dist_th_m": 0.12, "hold_steps": 25},
        ),
    }

    # ------------------------------------------------------------------
    # Assemble environment config
    # ------------------------------------------------------------------
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"bimanual": deepcopy(DEFAULT_BIMANUAL_CFG)},
            num_envs=4,
            extent=3.0,
        ),
        viewer=deepcopy(BIMANUAL_VIEWER_CONFIG),
        sim=deepcopy(BIMANUAL_SIM_CFG),
        observations=obs_cfg,
        actions=actions_cfg,
        events=events_cfg,
        rewards=rewards_cfg,
        terminations=terminations_cfg,
        decimation=5,  # 100 Hz control (500 Hz sim / 5)
        episode_length_s=30.0 if play else 12.0,
    )

    return cfg
