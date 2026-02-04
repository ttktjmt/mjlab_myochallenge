#!/usr/bin/env python3
"""Compile MyoHand XML model to MJB binary format.

This script compiles the myohand_die_fixed.xml model to a binary .mjb file
for faster loading and to lock the MuJoCo version.
"""

import os
from pathlib import Path

import mujoco


def main():
    """Compile XML to MJB."""
    # Get the assets directory
    script_dir = Path(__file__).parent
    assets_dir = script_dir.parent / "robot" / "assets"
    xml_path = assets_dir / "myohand_die_fixed.xml"
    mjb_path = assets_dir / "myohand_die.mjb"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Compiling {xml_path.name} to {mjb_path.name}...")

    # Change to assets directory so relative paths work
    original_dir = os.getcwd()
    os.chdir(assets_dir)

    try:
        # Load the XML model
        model = mujoco.MjModel.from_xml_path(str(xml_path.name))

        # Save as MJB
        mujoco.mj_saveModel(model, str(mjb_path.name), None)

        print(f"✓ Successfully compiled to {mjb_path}")
        print(f"  Model dimensions:")
        print(f"    - Bodies: {model.nbody}")
        print(f"    - Joints: {model.njnt}")
        print(f"    - Actuators: {model.nu}")
        print(f"    - Tendons: {model.ntendon}")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
