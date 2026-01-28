# mjlab_myochallenge

MyoSuite MyoHand Die Reorientation task implementation using [mjlab](https://github.com/mujocolab/mjlab).

## Overview

This package provides an mjlab-compatible implementation of the MyoChallenge 2022 Die Reorientation task. The task involves manipulating a die held in a dexterous MyoHand to match a target orientation without dropping it.

**Task ID**: `Myosuite-Manipulation-DieReorient-Myohand`

# mjlab_myochallenge — Quick Commands

Install (editable):

```bash
pip install -e .
```

Train:

```bash
uv run train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
# or
python -m mjlab.scripts.train Myosuite-Manipulation-DieReorient-Myohand --env.scene.num-envs 2048
```

**CPU-only mode** (for systems without GPU):

```bash
python -m mjlab.scripts.train Myosuite-Manipulation-DieReorient-Myohand --gpu-ids None
```

Play / Evaluate:

```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file [path-to-checkpoint]
# or
python -m mjlab.scripts.play Myosuite-Manipulation-DieReorient-Myohand --checkpoint-file [path-to-checkpoint]
```

Debug (quick agents):

```bash
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent zero
uv run play Myosuite-Manipulation-DieReorient-Myohand --agent random
```

## Troubleshooting

### macOS: Native Viewer with `uv` Virtual Environments

When using the native MuJoCo viewer (`--viewer native`) on macOS with a `uv`-created virtual environment, you may encounter this error:

```
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

**Solution:**

1. Use `mjpython` instead of the regular Python interpreter:
   ```bash
   .venv/bin/mjpython -m mjlab.scripts.play Myosuite-Manipulation-DieReorient-Myohand --agent random --viewer native
   ```

2. If you get a `dlopen` error about missing `libpython3.12.dylib`, create a symlink:
   ```bash
   # Find your uv Python installation path
   UV_PYTHON_PATH=$(readlink .venv/bin/python | sed 's|/bin/python3.12||')
   
   # Create symlink
   mkdir -p .venv/lib
   ln -sf "${UV_PYTHON_PATH}/lib/libpython3.12.dylib" .venv/lib/libpython3.12.dylib
   ```

This issue occurs because `uv` creates virtual environments with symlinks to a centralized Python installation, and `mjpython` needs the shared library to be accessible from the venv's `lib` directory.

## References

- [MyoChallenge 2022](https://sites.google.com/view/myochallenge)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [mjlab](https://github.com/mujocolab/mjlab)
- [mjlab_upkie](https://github.com/MarcDcls/mjlab_upkie)
- [mjlab_cartpole](https://github.com/Gregwar/mjlab_cartpole)
