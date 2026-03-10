# mjlab_myochallenge

MyoChallenge task implementations using [mjlab](https://github.com/mujocolab/mjlab).

Currently: **MyoChallenge 2022 Die Reorientation** — manipulate a die held in a MyoHand to match a target orientation.

## Setup

```bash
uv sync
```

The MJCF model is loaded at runtime from the installed myosuite package.

## Usage

**Train:**
```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
```

**Evaluate:**
```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file <path>
```

**Debug (no checkpoint needed):**
```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random
```

**CPU-only (no GPU):**
```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --gpu-ids None
```

## macOS: Native Viewer

Using `--viewer native` requires `mjpython`:

```bash
.venv/bin/mjpython -m mjlab.scripts.play Myosuite-Manipulation-DieReorient-Myohand --agent random --viewer native
```

If you get a `dlopen` error, create a symlink for the shared library:
```bash
UV_PYTHON_PATH=$(readlink .venv/bin/python | sed 's|/bin/python3.12||')
ln -sf "${UV_PYTHON_PATH}/lib/libpython3.12.dylib" .venv/lib/libpython3.12.dylib
```

## Project Structure

```
src/mjlab_myochallenge/
├── models/myohand.py          # Model spec (loads MJCF from myosuite)
├── mdp/myochallenge/          # MDP terms
│   ├── observations.py
│   ├── rewards.py
│   ├── terminations.py
│   └── events.py
└── tasks/myochallenge/die_reorient/
    ├── env_cfg.py             # Environment config
    └── rl_cfg.py              # PPO config
```

## References

- [MyoChallenge 2022](https://sites.google.com/view/myochallenge)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [mjlab](https://github.com/mujocolab/mjlab)
