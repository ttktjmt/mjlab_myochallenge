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

    # Load the XML model using absolute path
    model = mujoco.MjModel.from_xml_path(str(xml_path.absolute()))

    # Save as MJB
    mujoco.mj_saveModel(model, str(mjb_path.absolute()), None)

    print(f"✓ Successfully compiled to {mjb_path}")
    print(f"  Model dimensions:")
    print(f"    - Bodies: {model.nbody}")
    print(f"    - Joints: {model.njnt}")
    print(f"    - Actuators: {model.nu}")
    print(f"    - Tendons: {model.ntendon}")


if __name__ == "__main__":
    main()
