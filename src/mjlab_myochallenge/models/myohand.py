"""MyoHand model configuration for mjlab.

Resolves the MJCF path from the installed myosuite package and provides
the EntityCfg used to load the MyoHand die-manipulation model into mjlab's
scene pipeline via spec_fn / MjSpec injection.
"""

from pathlib import Path

import gymnasium
import mujoco
import myosuite  # noqa: F401 — triggers gym env registration
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

MYOHAND_DIE_XML = Path(
    gymnasium.spec("myoChallengeDieReorientP1-v0").kwargs["model_path"]
).resolve()

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


def _disable_body_collision_geoms(spec: mujoco.MjSpec) -> None:
    """Restore contype=0/conaffinity=0 for non-contact scene/body geoms.

    mjlab's CollisionCfg sets contype/conaffinity=1 on all geoms (including
    unnamed ones) when using geom_names_expr=".*".  But the myohand model has
    large unnamed ellipsoid geoms representing the torso/body that deliberately
    have contype=0 in the MJCF.  Leaving them enabled causes massive contact
    forces that launch the die off the palm immediately after reset.

    The imported MyoSuite scene also contains a decorative floor plane and a
    cylindrical arena wall.  In mjswan these show up as force targets, while
    the mjlab scene already adds its own terrain plane, so we keep them visual
    only here as well.
    """
    for geom in spec.geoms:
        if geom.name == "" and geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            geom.contype = 0
            geom.conaffinity = 0
            continue
        if geom.name == "floor" and geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype = 0
            geom.conaffinity = 0
            continue
        if (
            geom.name == ""
            and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            and geom.group == 4
        ):
            geom.contype = 0
            geom.conaffinity = 0


def get_myohand_spec() -> mujoco.MjSpec:
    """Load MyoHand die manipulation model spec."""
    spec = mujoco.MjSpec.from_file(str(MYOHAND_DIE_XML))
    xml_str = spec.to_xml()
    xml_str = _resolve_xml_paths(xml_str)
    xml_str = _patch_tendon_sidesites(xml_str)
    spec = mujoco.MjSpec.from_string(xml_str)
    _disable_body_collision_geoms(spec)
    return spec


DEFAULT_MYOHAND_CFG = EntityCfg(
    spec_fn=get_myohand_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 0),
        joint_pos={r".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # No CollisionCfg override: the MJCF already has correct contype/friction values
    # (contype=0 for structural/bone geoms, contype=1 for skin/contact capsules,
    # and friction=[1.0, 0.005, 0.0001] everywhere).
    articulation=EntityArticulationInfoCfg(
        actuators=(
            XmlActuatorCfg(
                target_names_expr=(".*",),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
    ),
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="myohand",
    body_name="radius",
    distance=0.5,
    elevation=-10.0,
    azimuth=180.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.002,
        iterations=100,
        ls_iterations=50,
    ),
    nconmax=512,
    njmax=1024,
)
