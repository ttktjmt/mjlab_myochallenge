"""Bimanual model configuration for mjlab (MyoChallenge 2024 Manipulation).

Resolves the MJCF path from the installed myosuite package and provides
the EntityCfg used to load the BionicMyoArms bimanual model into mjlab's
scene pipeline via spec_fn / MjSpec injection.

Model summary:
  - MyoArm: 38 joints (qpos 0..37), 63 muscle actuators (tendon-driven)
  - Prosthesis (MPL): 26 joints (qpos 38..63), 17 position actuators
  - Object (gelatin box): freejoint (qpos 64..70, qvel 64..69)
  - Total: nq=71, nv=70, na=63, nu=80
"""

from pathlib import Path
import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType

import myosuite  # noqa: F401 — triggers gym env registration
import gymnasium

BIMANUAL_XML = Path(
    gymnasium.spec("myoChallengeBimanual-v0").kwargs["model_path"]
).resolve()

if not BIMANUAL_XML.exists():
    raise FileNotFoundError(f"Bimanual XML not found at {BIMANUAL_XML}")


def _resolve_xml_paths(xml_str: str) -> str:
    """Convert relative meshdir/texturedir to absolute paths.

    After MjSpec.from_file() + to_xml(), the compiler still has relative
    meshdir/texturedir paths.  All asset file refs (meshes, textures) use
    paths relative to meshdir (simhive/myo_sim/), so fixing those two
    compiler entries is sufficient.
    """
    xml_dir = BIMANUAL_XML.parent
    # simhive/myo_sim/ is 4 levels up from arm/ then into simhive/myo_sim
    abs_myo_sim = (xml_dir / "../../../../simhive/myo_sim").resolve()
    xml_str = xml_str.replace(
        'meshdir="../../../../simhive/myo_sim/"',
        f'meshdir="{abs_myo_sim}/"',
    )
    xml_str = xml_str.replace(
        'texturedir="../../../../simhive/myo_sim/"',
        f'texturedir="{abs_myo_sim}/"',
    )
    return xml_str


def _patch_tendon_sidesites(xml_str: str) -> str:
    """Add missing sidesite attributes to tendon wraps.

    The bimanual model shares the same MyoArm hand tendons as myohand.
    Three wraps are missing sidesites, causing MuJoCo spec.attach() to
    silently drop the corresponding muscle actuators:
      - Fifthpm_wrap  (EDM tendon)
      - FPL_ellipsoid_wrap  (FPL tendon)
      - MPthumb_wrap  (EPL tendon — also requires adding the side site)
    """
    # Add MPthumb_site_EPL_side before MPthumb_site_EPB_side
    xml_str = xml_str.replace(
        '<site name="MPthumb_site_EPB_side"',
        '<site name="MPthumb_site_EPL_side" pos="0.0233473 -0.0173314 -0.02"/>\n'
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
    """Disable large unnamed ellipsoid structural geoms.

    The model has unnamed ellipsoid geoms (body/shoulder volume) that have
    contype=0 in the original MJCF.  Leaving them enabled causes spurious
    contact forces that destabilise the simulation.
    """
    for geom in spec.geoms:
        if geom.name == "" and geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            geom.contype = 0
            geom.conaffinity = 0


def get_bimanual_spec() -> mujoco.MjSpec:
    """Load BionicMyoArms bimanual manipulation model spec."""
    spec = mujoco.MjSpec.from_file(str(BIMANUAL_XML))
    xml_str = spec.to_xml()
    xml_str = _resolve_xml_paths(xml_str)
    xml_str = _patch_tendon_sidesites(xml_str)
    spec = mujoco.MjSpec.from_string(xml_str)
    _disable_body_collision_geoms(spec)
    return spec


DEFAULT_BIMANUAL_CFG = EntityCfg(
    spec_fn=get_bimanual_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 0),
        # joint_pos=None to use the model's existing keyframe (key[0])
        joint_pos=None,
        joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
        actuators=(
            # MyoArm muscle actuators (63) — tendon-driven
            XmlMuscleActuatorCfg(
                target_names_expr=(".*_tendon",),
                transmission_type=TransmissionType.TENDON,
            ),
            # NOTE: MPL position actuators are handled via a custom action term
            # (MplJointPositionAction) that writes ctrl directly, bypassing the
            # XmlPositionActuator name-matching limitation.
        ),
    ),
)

BIMANUAL_VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="bimanual",
    body_name="radius",
    distance=1.5,
    elevation=-20.0,
    azimuth=120.0,
)

BIMANUAL_SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.002,
        iterations=100,
        ls_iterations=50,
    ),
    nconmax=1024,
    njmax=2048,
)
