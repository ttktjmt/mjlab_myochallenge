"""Environment configuration for MyoChallenge 2022 Die Reorientation (Phase 1).

Follows the mjlab factory-function pattern: `die_reorient_env_cfg(play=False)`
returns a fully composed `ManagerBasedRlEnvCfg`.  MDP logic lives in
`mjlab_myochallenge.mdp.myochallenge`; the model specification is provided by
`mjlab_myochallenge.models.myohand`.
"""

from copy import deepcopy

from mjlab.envs import mdp, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_myochallenge.mdp.die_reorient import observations, rewards, terminations, events
from mjlab_myochallenge.models.myohand import DEFAULT_MYOHAND_CFG, VIEWER_CONFIG, SIM_CFG


def die_reorient_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create MyoHand Die Reorientation environment configuration.

    Args:
        play: If True, disables observation noise and domain randomisation,
              fixes the goal pose, and extends the episode length.

    Returns:
        ManagerBasedRlEnvCfg ready for ``register_mjlab_task``.
    """

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    policy_terms = {
        "hand_qpos": ObservationTermCfg(
            func=observations.hand_qpos,
            noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
        ),
        "hand_qvel": ObservationTermCfg(
            func=observations.hand_qvel,
            noise=Unoise(n_min=-0.1, n_max=0.1) if not play else None,
        ),
        "die_pos": ObservationTermCfg(
            func=observations.die_pos,
            noise=Unoise(n_min=-0.001, n_max=0.001) if not play else None,
        ),
        "die_euler": ObservationTermCfg(func=observations.die_euler),
        "goal_euler": ObservationTermCfg(func=observations.goal_euler),
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
    # Actions  (39 MyoHand muscle actuators via tendon transmission)
    # ------------------------------------------------------------------
    actions_cfg: dict[str, ActionTermCfg] = {
        "myohand": mdp.TendonEffortActionCfg(
            entity_name="myohand",
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
        )
    }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    events_cfg = {
        "reset_scene": EventTermCfg(
            func=events.reset_die_and_goal,
            mode="reset",
            params={"play": play},
        ),
    }

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    rewards_cfg = {
        "orientation": RewardTermCfg(
            func=rewards.orientation_reward,
            weight=10.0,
        ),
        "position": RewardTermCfg(
            func=rewards.position_reward,
            weight=1.0,
            params={"std": 0.05},
        ),
        "action_reg": RewardTermCfg(
            func=rewards.action_regularization,
            weight=0.001,
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
        "die_dropped": TerminationTermCfg(
            func=terminations.die_dropped,
            time_out=False,
            params={"drop_th": 0.1},
        ),
    }

    # ------------------------------------------------------------------
    # Assemble environment config
    # ------------------------------------------------------------------
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={"myohand": deepcopy(DEFAULT_MYOHAND_CFG)},
            num_envs=4,
            extent=2.0,
        ),
        viewer=deepcopy(VIEWER_CONFIG),
        sim=deepcopy(SIM_CFG),
        observations=obs_cfg,
        actions=actions_cfg,
        events=events_cfg,
        rewards=rewards_cfg,
        terminations=terminations_cfg,
        decimation=5,  # 100 Hz control (500 Hz sim / 5)
        episode_length_s=20.0 if play else 6.0,
    )

    return cfg
