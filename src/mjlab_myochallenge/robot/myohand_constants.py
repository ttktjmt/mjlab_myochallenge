from pathlib import Path
import mujoco
import os

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType

# Use fixed MyoHand XML with sidesite attributes to work around MuJoCo 3.4.0 bug
# See ISSUE_REPORT.md for details on the spec.attach() prefix bug
MYOHAND_DIE_XML = Path(os.path.dirname(__file__)) / "assets" / "myohand_die_fixed.xml"

if not MYOHAND_DIE_XML.exists():
    raise FileNotFoundError(f"MyoHand Die XML not found at {MYOHAND_DIE_XML}")


def get_myohand_spec() -> mujoco.MjSpec:
    """Load MyoHand die manipulation model spec."""
    return mujoco.MjSpec.from_file(str(MYOHAND_DIE_XML))


# MyoHand has 39 muscle actuators (ECRL, ECRB, ECU, FCR, FCU, etc.)
# These are defined in the myohand_assets_fixed.xml file
# Use XmlMuscleActuatorCfg to load them from the XML
MUSCLE_ACTUATOR_NAMES = (".*",)  # Match all actuators with regex

# Initial hand pose: palm facing up (fingers open)
# MyoHand joints are controlled by muscle activations
DEFAULT_HAND_QPOS = {
    r".*": 0.0  # All joints start at 0 (open hand position)
}

# Initial die position and orientation
DIE_INIT_POS = (0.015, 0.025, 0.025)  # Resting on palm
DIE_INIT_QUAT = (1.0, 0.0, 0.0, 0.0)  # No rotation initially

# Collision configuration for MyoHand
COLLISION_CFG = CollisionCfg(
    geom_names_expr=tuple([".*"]),  # All geoms can collide
    condim={r".*": 3},  # 3D contact for all geoms
    friction={r".*": (1.0, 0.005, 0.0001)},  # Default friction
)

# Articulation configuration - MyoHand uses muscle actuators via tendons
# Using fixed XML (myohand_die_fixed.xml) that preserves all 39 actuators
MUSCLE_ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        XmlMuscleActuatorCfg(
            target_names_expr=MUSCLE_ACTUATOR_NAMES,
            transmission_type=TransmissionType.TENDON,
        ),
    ),
)

# Default MyoHand entity configuration
DEFAULT_MYOHAND_CFG = EntityCfg(
    spec_fn=get_myohand_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 0),  # MyoHand is fixed at origin
        joint_pos=DEFAULT_HAND_QPOS,
        joint_vel={".*": 0.0},
    ),
    collisions=(COLLISION_CFG,),
    articulation=MUSCLE_ARTICULATION_CFG,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="myohand",
    body_name="radius",  # Center camera on radius (forearm bone)
    distance=0.5,
    elevation=-10.0,
    azimuth=180.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.002,  # 2ms timestep (500 Hz)
        iterations=5,
        ls_iterations=10,
    ),
    nconmax=512,
    njmax=1024,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"myohand": DEFAULT_MYOHAND_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
