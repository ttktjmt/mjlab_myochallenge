from mjlab.tasks.registry import register_mjlab_task
from rsl_rl.runners import OnPolicyRunner

from .die_reorient.env_cfg import die_reorient_env_cfg
from .die_reorient.rl_cfg import DieReorientRlCfg

register_mjlab_task(
    task_id="Myosuite-Manipulation-DieReorient-Myohand",
    env_cfg=die_reorient_env_cfg(play=False),
    play_env_cfg=die_reorient_env_cfg(play=True),
    rl_cfg=DieReorientRlCfg(max_iterations=50_000),
    runner_cls=OnPolicyRunner,
)
