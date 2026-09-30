"""Pinned ToddlerBot 2XM asset with four independent, force-only spine units.

The upstream motor/gear topology and link inertias are retained. Waist
backlash is idealized as zero-backlash tendon equalities (not a locked waist).
Each 15 g unit+adapter is provisional; there is no identified local contact
moment capacity. Contact points therefore use connect, never weld, anchors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

ROOT = Path(__file__).resolve().parent
XML = ROOT / "upstream/toddlerbot_2xm.xml"
UNIT_MESH = ROOT / "meshes/microspine_unit.stl"
UNIT_NAMES = ("left_foot_unit", "right_foot_unit", "left_hand_unit", "right_hand_unit")
UNIT_MASS = 0.015
UNIT_PRELOAD = 1.5
# Match C2's limb-surface layout. Names retain their task/contact-slot meaning;
# the units move with the shin and forearm, independently of ankle/wrist motion.
# parent, contact center, quaternion (wxyz), schematic adapter depth beyond CAD
UNIT_MOUNTS = (
    ("left_knee_link", (0.048, 0.019, -0.055), (2**-0.5, 0.0, 0.0, 2**-0.5), 0.018),
    ("right_knee_link", (0.048, -0.019, -0.055), (2**-0.5, 0.0, 0.0, 2**-0.5), 0.018),
    ("left_elbow_yaw_link", (-0.040, 0.00335, -0.029), (0.0, 2**-0.5, 2**-0.5, 0.0), 0.012),
    ("right_elbow_yaw_link", (-0.040, -0.00335, -0.029), (0.0, 2**-0.5, 2**-0.5, 0.0), 0.012),
)
MOUNT_SIGNATURE = hashlib.sha256(json.dumps(UNIT_MOUNTS).encode()).hexdigest()

PARAMS = yaml.safe_load((ROOT / "upstream/default.yml").read_text())
OVERRIDES = yaml.safe_load((ROOT / "upstream/robot.yml").read_text())
MOTOR_NAMES = tuple(e.get("name") for e in ET.parse(XML).findall("./actuator/motor"))
MOTORS = {name: {**PARAMS["motors"][name], **OVERRIDES["motors"].get(name, {})} for name in MOTOR_NAMES}
TORQUE_LIMITS = np.array([PARAMS["actuators"][MOTORS[n]["motor"]]["tau_max"] for n in MOTOR_NAMES])
SPEED_LIMITS = np.array([PARAMS["actuators"][MOTORS[n]["motor"]]["q_dot_max"] for n in MOTOR_NAMES])
CORNER_SPEEDS = np.array([PARAMS["actuators"][MOTORS[n]["motor"]]["q_dot_tau_max"] for n in MOTOR_NAMES])
TORQUE_AT_MAX_SPEED = np.array([PARAMS["actuators"][MOTORS[n]["motor"]]["tau_q_dot_max"] for n in MOTOR_NAMES])
STIFFNESS = np.array([MOTORS[n]["kp"] / PARAMS["actuators"]["kp_ratio"] for n in MOTOR_NAMES])


def speed_torque_limits(velocity):
    speed = np.abs(velocity)
    fraction = np.clip((speed - CORNER_SPEEDS) / (SPEED_LIMITS - CORNER_SPEEDS), 0, 1)
    return np.where(speed <= SPEED_LIMITS, TORQUE_LIMITS - fraction * (TORQUE_LIMITS - TORQUE_AT_MAX_SPEED), 0.0)


def get_spec() -> mujoco.MjSpec:
    """Return a fresh robot spec; importing the package never downloads files."""
    tree = ET.parse(XML)
    root = tree.getroot()
    keyframes = root.find("keyframe")
    if keyframes is not None:
        root.remove(keyframes)
    # The same two coupled motors still carry waist torque in the static audit.
    for tendon in root.findall("./tendon/fixed"):
        tendon.attrib.pop("limited", None)
        tendon.attrib.pop("range", None)
        ET.SubElement(
            root.find("equality"),
            "tendon",
            {
                "name": tendon.get("name") + "_ideal",
                "tendon1": tendon.get("name"),
                "polycoef": "0 0 0 0 0",
                "solref": "0.004 1",
            },
        )
    for motor, limit in zip(root.findall("./actuator/motor"), TORQUE_LIMITS):
        motor.set("ctrlrange", f"{-limit} {limit}")
        motor.set("forcerange", f"{-limit} {limit}")
    # Name every physical/visual geom for MJLab and collision diagnostics.
    for i, geom in enumerate(root.findall("./worldbody//geom")):
        if "name" not in geom.attrib:
            geom.set("name", f"toddler_geom_{i}")
        if geom.get("class") == "collision":
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("condim", "3")
            geom.set("friction", "0.2 0.005 0.0001")
    assets = {"assets/" + p.name: p.read_bytes() for p in (ROOT / "upstream/assets").glob("*.stl")}
    assets["assets/microspine_unit.stl"] = UNIT_MESH.read_bytes()
    ET.SubElement(root.find("asset"), "mesh", name="microspine_single_unit", file="microspine_unit.stl")
    for name, (parent, position, quat, adapter_depth) in zip(UNIT_NAMES, UNIT_MOUNTS):
        # Shin: face +x, rake -z. Forearm: face -x, rake +z (toward elbow).
        # Raised arms and descending legs can both engage the same wall.
        body = ET.SubElement(
            root.find(f'.//body[@name="{parent}"]'),
            "body",
            {
                "name": name,
                "pos": " ".join(map(str, position)),
                "quat": " ".join(map(str, quat)),
            },
        )
        size = np.array([0.028, 0.015 + adapter_depth, 0.025])
        inertia = UNIT_MASS * (np.sum(size**2) - size**2) / 12
        ET.SubElement(
            body,
            "inertial",
            pos=f"0 {size[1] / 2} 0",
            mass=str(UNIT_MASS),
            diaginertia=" ".join(map(str, inertia)),
        )
        ET.SubElement(
            body,
            "geom",
            name=name + "_visual",
            type="mesh",
            mesh="microspine_single_unit",
            rgba="0.95 0.5 0.08 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            body,
            "geom",
            name=name + "_mount_visual",
            type="box",
            pos=f"0 {0.015 + adapter_depth / 2} 0",
            size=f"0.014 {adapter_depth / 2} 0.0125",
            rgba="0.15 0.18 0.22 1",
            contype="0",
            conaffinity="0",
        )
        # One point proxy at C. No invented four-corner force/moment allocation.
        proxy = ET.SubElement(body, "body", name=name + "_c_0_0", pos="0 0.003 0")
        ET.SubElement(
            proxy,
            "geom",
            name=name + "_c_0_0",
            type="sphere",
            size="0.003",
            rgba="0.9 0.2 0.1 0.8",
            contype="1",
            conaffinity="1",
            condim="3",
            friction="0.2 0.005 0.0001",
            mass="0",
            group="3",
            solref="0.004 1",
        )
        ET.SubElement(body, "site", name=name + "_contact_center", pos="0 0 0", size="0.002", rgba="1 0.2 0 1")
        ET.SubElement(body, "site", name=name + "_unit_0_0", pos="0 0 0", size="0.001", group="4")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    return spec


def get_sim_spec() -> mujoco.MjSpec:
    spec = get_spec()
    # MJLab installs its own bounded PD servos with these same names.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def get_robot_cfg() -> EntityCfg:
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    from .actuator import ToddlerServoCfg

    actuators = tuple(
        ToddlerServoCfg(
            target_names_expr=(name,),
            stiffness=float(kp),
            damping=float(PARAMS["actuators"][MOTORS[name]["motor"]]["kd_min"]),
            effort_limit=float(limit),
            corner_speed=float(PARAMS["actuators"][MOTORS[name]["motor"]]["q_dot_tau_max"]),
            max_speed=float(PARAMS["actuators"][MOTORS[name]["motor"]]["q_dot_max"]),
            torque_at_max_speed=float(PARAMS["actuators"][MOTORS[name]["motor"]]["tau_q_dot_max"]),
        )
        for name, kp, limit in zip(MOTOR_NAMES, STIFFNESS, TORQUE_LIMITS)
    )
    return EntityCfg(
        spec_fn=get_sim_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.95),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.315053), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        collisions=(
            CollisionCfg(geom_names_expr=(".*collision.*", ".*_c_0_0"), condim=3, friction=(0.2, 0.005, 0.0001)),
        ),
    )
