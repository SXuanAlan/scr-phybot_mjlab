"""Add MuJoCo adhesion actuators to the robot spec.

MuJoCo's adhesion actuator creates constraint-based "sticky" contacts that
resist pull-off forces, solved within the contact solver (zero delay).
See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-adhesion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from spider_rl_mjlab.assets.microspine.config import MicrospineCfg


def add_adhesion_actuators(
    spec: mujoco.MjSpec,
    cfg: MicrospineCfg,
) -> list[str]:
    """Add one adhesion actuator per end-effector body to the MjSpec.

    Each adhesion actuator targets a body (not a geom or joint). When its ctrl
    is set to a value in [0, 1], it creates adhesion forces on all contact
    points involving geoms of that body, with magnitude up to
    ``ctrl * gain`` Newtons.

    Args:
        spec: The MuJoCo specification to modify (before compilation).
        cfg: Microspine configuration with EE body names and gain.

    Returns:
        List of actuator names (e.g. ``["adhesion_left_ankle_roll", ...]``),
        in the same order as ``cfg.ee_body_names``.
    """
    actuator_names: list[str] = []

    for body_name in cfg.ee_body_names:
        act_name = f"adhesion_{body_name}"
        act = spec.add_actuator()
        act.name = act_name
        act.set_to_adhesion(gain=cfg.max_adhesion_gain)
        act.target = body_name
        act.trntype = mujoco.mjtTrn.mjTRN_BODY
        act.ctrllimited = True
        act.ctrlrange = [0.0, 1.0]
        actuator_names.append(act_name)

    return actuator_names
