from pathlib import Path
import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType

# Load original myohand_die.xml from myosuite's gymnasium registry
import myosuite  # noqa: F401 (triggers env registration)
import gymnasium
MYOHAND_DIE_XML = Path(gymnasium.spec("myoChallengeDieReorientP1-v0").kwargs["model_path"]).resolve()

if not MYOHAND_DIE_XML.exists():
    raise FileNotFoundError(f"MyoHand Die XML not found at {MYOHAND_DIE_XML}")


def _resolve_xml_paths(xml_str: str) -> str:
    """Convert relative meshdir/texturedir/file paths to absolute paths.

    Required because we reload the XML via MjSpec.from_string(), which has
    no base directory for resolving relative paths.
    """
    xml_dir = MYOHAND_DIE_XML.parent
    abs_simhive = (xml_dir / "../../../../simhive/myo_sim").resolve()
    xml_str = xml_str.replace(
        'meshdir="../../../../simhive/myo_sim/"',
        f'meshdir="{abs_simhive}/"',
    )
    xml_str = xml_str.replace(
        'texturedir="../../../../simhive/myo_sim/"',
        f'texturedir="{abs_simhive}/"',
    )
    abs_dice = (abs_simhive / "../../envs/myo/assets/hand/dice.png").resolve()
    xml_str = xml_str.replace(
        'file="../../envs/myo/assets/hand/dice.png"',
        f'file="{abs_dice}"',
    )
    return xml_str


def _patch_tendon_sidesites(xml_str: str) -> str:
    """Add missing sidesite attributes to 3 tendon wraps (EDM, EPL, FPL).

    Without these, MuJoCo's spec.attach() silently drops the corresponding
    actuators. Also adds the MPthumb_site_EPL_side site which is commented
    out in the original myohand_body.xml.
    """
    # Add MPthumb_site_EPL_side (commented out in original myohand_body.xml)
    xml_str = xml_str.replace(
        '<site name="MPthumb_site_EPB_side"',
        '<site name="MPthumb_site_EPL_side" pos="0.0233473 -0.0173314 -0.02"'
        ' class="myohand"/>\n'
        '                                  <site name="MPthumb_site_EPB_side"',
    )
    # EDM_tendon: Fifthpm_wrap missing sidesite
    xml_str = xml_str.replace(
        '<geom geom="Fifthpm_wrap"/>',
        '<geom geom="Fifthpm_wrap" sidesite="Fifthpm_site_EDC5_side"/>',
    )
    # FPL_tendon: FPL_ellipsoid_wrap missing sidesite
    xml_str = xml_str.replace(
        '<geom geom="FPL_ellipsoid_wrap"/>',
        '<geom geom="FPL_ellipsoid_wrap" sidesite="FPL_ellipsoid_site_FPL_side"/>',
    )
    # EPL_tendon: MPthumb_wrap missing sidesite
    xml_str = xml_str.replace(
        '<geom geom="MPthumb_wrap"/>',
        '<geom geom="MPthumb_wrap" sidesite="MPthumb_site_EPL_side"/>',
    )
    return xml_str


def get_myohand_spec() -> mujoco.MjSpec:
    """Load MyoHand die manipulation model spec."""
    spec = mujoco.MjSpec.from_file(str(MYOHAND_DIE_XML))
    xml_str = spec.to_xml()
    xml_str = _resolve_xml_paths(xml_str)
    xml_str = _patch_tendon_sidesites(xml_str)
    return mujoco.MjSpec.from_string(xml_str)


# MyoHand has 39 muscle actuators (ECRL, ECRB, ECU, FCR, FCU, etc.)
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
