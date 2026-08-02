from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .bimanual.env_cfg import bimanual_env_cfg
from .bimanual.rl_cfg import BimanualRlCfg
from .die_reorient.env_cfg import die_reorient_env_cfg
from .die_reorient.rl_cfg import DieReorientRlCfg

register_mjlab_task(
    task_id="Myosuite-Manipulation-DieReorient-Myohand-v0",
    env_cfg=die_reorient_env_cfg(play=False),
    play_env_cfg=die_reorient_env_cfg(play=True),
    rl_cfg=DieReorientRlCfg(max_iterations=30_000),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Myosuite-Manipulation-Bimanual-Myoarm-v0",
    env_cfg=bimanual_env_cfg(play=False),
    play_env_cfg=bimanual_env_cfg(play=True),
    rl_cfg=BimanualRlCfg(max_iterations=60_000),
    runner_cls=MjlabOnPolicyRunner,
)
