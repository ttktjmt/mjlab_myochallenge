"""
Script to record RL agent visualization to .viser file.

Usage:
  uv run scripts/record.py Myosuite-Manipulation-DieReorient-Myohand \
    --wandb-run-path team/project/run-id \
    --output-file recordings/<filename>.viser
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
import viser

# Import mjlab_myochallenge to trigger auto-registration of MyoSuite environments
try:
    import mjlab_myochallenge  # noqa: F401
except ImportError:
    pass  # MyoSuite not available, skip registration

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sim.sim import Simulation
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser.scene import ViserMujocoScene
from rsl_rl.runners import OnPolicyRunner


@dataclass(frozen=True)
class RecordConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    output_file: str = "recordings/output.viser"
    num_steps: int = 1000
    env_idx: int = 0
    frame_skip: int = 2
    sleep_duration: float = 0.016
    random_goals: bool = True
    distance: float = 5.0
    azimuth: float = 45.0
    elevation: float = -20.0


def run_record(task: str, cfg: RecordConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check if this is a MyoSuite environment
    is_myosuite = task.startswith("Myosuite")

    env_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )

    if is_tracking_task:
        if DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError(
                    "Tracking tasks require `registry_name` when using dummy agents."
                )
            registry_name = str(cfg.registry_name)
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            assert env_cfg.commands is not None
            motion_cmd = env_cfg.commands["motion"]
            assert isinstance(motion_cmd, MotionCommandCfg)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            assert env_cfg.commands is not None
            motion_cmd = env_cfg.commands["motion"]
            assert isinstance(motion_cmd, MotionCommandCfg)

            if cfg.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            else:
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

        # Set random goal sampling if requested
        if cfg.random_goals:
            assert env_cfg.commands is not None
            motion_cmd = env_cfg.commands["motion"]
            assert isinstance(motion_cmd, MotionCommandCfg)
            motion_cmd.sampling_mode = "uniform"
            print("[INFO]: Using random goal sampling (uniform mode)")

    log_dir: Optional[Path] = None
    resume_path: Optional[Path] = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Handle MyoSuite environments differently
    if is_myosuite:
        # For MyoSuite, we use the environment directly without wrapper
        env.clip_actions = agent_cfg.clip_actions
        pass
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if DUMMY_MODE:
        action_shape = None
        if hasattr(env, "single_action_space"):
            action_shape = getattr(env.single_action_space, "shape", None)  # type: ignore
        else:
            action_shape = getattr(env.action_space, "shape", None)  # type: ignore
            if (
                isinstance(action_shape, tuple)
                and len(action_shape) > 1
                and getattr(env, "num_envs", 1) == action_shape[0]
            ):
                action_shape = action_shape[1:]

        env_device = env.device if hasattr(env, "device") else torch.device("cpu")

        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    per_env_shape = (
                        action_shape if isinstance(action_shape, tuple) else ()
                    )
                    return torch.zeros(
                        (getattr(env, "num_envs", 1),) + per_env_shape,
                        device=env_device,
                    )

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    per_env_shape = (
                        action_shape if isinstance(action_shape, tuple) else ()
                    )
                    return (
                        2
                        * torch.rand(
                            (getattr(env, "num_envs", 1),) + per_env_shape,
                            device=env_device,
                        )
                        - 1
                    )

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task)
        if runner_cls is None:
            runner_cls = OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    # Start recording
    print(f"[INFO]: Starting recording to {cfg.output_file}")
    print(f"[INFO]: Recording {cfg.num_steps} steps (frame skip: {cfg.frame_skip})")

    # Create viser server and scene
    server = viser.ViserServer(label="mjlab-recording", verbose=False)
    sim = env.sim
    assert isinstance(sim, Simulation)

    scene = ViserMujocoScene.create(
        server=server,
        mj_model=sim.mj_model,
        num_envs=env.num_envs,
    )
    scene.env_idx = cfg.env_idx

    # Get serializer
    serializer = server.get_scene_serializer()

    # Initialize environment
    obs, _ = env.reset()

    # Recording loop
    frame_count = 0
    for step in range(cfg.num_steps):
        # Get action from policy
        action = policy(obs)

        # Detach action to avoid gradient tracking issues
        if isinstance(action, torch.Tensor):
            action = action.detach()

        # Step environment
        obs, _, _, _, _ = env.step(action)

        # Update visualization (only every Nth frame)
        if step % cfg.frame_skip == 0:
            with server.atomic():
                scene.update(sim.wp_data)
                server.flush()

            # Insert sleep for animation timing
            serializer.insert_sleep(cfg.sleep_duration)
            frame_count += 1

            # Progress indicator
            if (step + 1) % 100 == 0:
                print(
                    f"[INFO]: Recorded {step + 1}/{cfg.num_steps} steps ({frame_count} frames)"
                )

    # Save recording
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recording_data = serializer.serialize()
    output_path.write_bytes(recording_data)

    print(f"[INFO]: Recording saved to {output_path}")
    print(f"[INFO]: Total frames: {frame_count}")
    print(f"[INFO]: Duration: ~{frame_count * cfg.sleep_duration:.1f} seconds")
    print()
    print("[INFO]: To view the recording:")
    print("   1. Run: viser-build-client --out-dir viser-client")
    print(f"   2. Open: viser-client/?playbackPath=../{output_path}")

    # Cleanup
    server.stop()
    env.close()


def main():
    from mjlab.tasks.registry import list_tasks

    # Parse first argument to choose the task
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
    )

    # Parse the rest of the arguments
    env_cfg = load_env_cfg(chosen_task, play=False)
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        RecordConfig,
        args=remaining_args,
        default=RecordConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del env_cfg, agent_cfg, remaining_args

    run_record(chosen_task, args)


if __name__ == "__main__":
    main()
