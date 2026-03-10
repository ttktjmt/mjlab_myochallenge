# mjlab_myochallenge

MyoChallenge task implementations using [mjlab](https://github.com/mujocolab/mjlab).

Tasks:
- **MyoChallenge 2022 Die Reorientation** — manipulate a die held in a MyoHand to match a target orientation.
- **MyoChallenge 2024 Bimanual Manipulation** — transfer a YCB object between pillars using a MyoArm and an MPL prosthetic hand.

## Setup

```bash
uv sync
```

The MJCF model is loaded at runtime from the installed myosuite package.

## Usage

**Train:**
```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
uv run train Myosuite-Manipulation-Bimanual-2024 --env.scene.num-envs 2048
```

**Evaluate:**
```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file <path>
uv run play Myosuite-Manipulation-Bimanual-2024 --checkpoint-file <path>
```

**Debug (no checkpoint needed):**
```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random
uv run play Myosuite-Manipulation-Bimanual-2024 --agent zero
uv run play Myosuite-Manipulation-Bimanual-2024 --agent random
```

**CPU-only (no GPU):**
```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --gpu-ids None
uv run train Myosuite-Manipulation-Bimanual-2024 --gpu-ids None
```

## macOS: Native Viewer

Using `--viewer native` requires `mjpython`:

```bash
.venv/bin/mjpython -m mjlab.scripts.play Myosuite-Manipulation-DieReorient-Myohand --agent random --viewer native
.venv/bin/mjpython -m mjlab.scripts.play Myosuite-Manipulation-Bimanual-2024 --agent random --viewer native
```

If you get a `dlopen` error, create a symlink for the shared library:
```bash
UV_PYTHON_PATH=$(readlink .venv/bin/python | sed 's|/bin/python3.12||')
ln -sf "${UV_PYTHON_PATH}/lib/libpython3.12.dylib" .venv/lib/libpython3.12.dylib
```

## Project Structure

```
src/mjlab_myochallenge/
├── models/
│   ├── myohand.py             # MyoHand model spec (die reorient)
│   └── bimanual.py            # MyoArm + MPL model spec (bimanual)
├── mdp/
│   ├── die_reorient/          # MDP terms for die reorientation
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── terminations.py
│   │   ├── events.py
│   │   └── utils.py
│   └── bimanual/              # MDP terms for bimanual manipulation
│       ├── actions.py         # Custom MPL position action
│       ├── observations.py
│       ├── rewards.py
│       ├── terminations.py
│       ├── events.py
│       └── utils.py
└── tasks/
    ├── die_reorient/          # Myosuite-Manipulation-DieReorient-Myohand
    │   ├── env_cfg.py
    │   └── rl_cfg.py
    └── bimanual/              # Myosuite-Manipulation-Bimanual-2024
        ├── env_cfg.py
        └── rl_cfg.py
```

## References

- [MyoChallenge 2022](https://sites.google.com/view/myochallenge)
- [MyoChallenge 2024](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2024)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [mjlab](https://github.com/mujocolab/mjlab)
