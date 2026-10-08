"""Microspine grip of the brick task: the two hand pads on the brick's side faces."""

from __future__ import annotations

import math

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.microspine.config import MicrospineCfg, MicrospineLimitSurfaceCfg, MicrospineSurfaceCfg

HANDS = ("left_hand_spine", "right_hand_spine")


def brick_microspine_cfg(brick: BrickCfg, grip_enabled: bool, unit_layout: str) -> MicrospineCfg:
  """Each hand pad latches onto one side face of the brick (normals +-y).

  The grip event takes a shallow copy of this config, so it must be
  complete before that event is built.
  """
  surfaces = tuple(
    MicrospineSurfaceCfg(
      ee_body_name=hand,
      entity_name="brick",
      body_name="brick",
      geom_name="brick_geom",
      point=(0.0, sign * brick.size[1] / 2, 0.0),
      normal=(0.0, sign, 0.0),
      tangent=(1.0, 0.0, 0.0),
      half_extents=(brick.size[0] / 2, brick.size[2] / 2),
    )
    for hand, sign in zip(HANDS, (1.0, -1.0))
  )
  return MicrospineCfg(
    ee_body_names=HANDS,
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    surfaces=surfaces,
    initial_on=False,
    grip_enabled=grip_enabled,
    contact_margin=0.0,
    contact_gap=0.0,
    contact_friction=(0.2, 0.2, 0.005, 0.0001, 0.0001),
    anchor_torquescale=1.0,
    reanchor_after_grace=False,
    engage_unit_preload_n=10.0,
    engage_min_spine_fraction=1.0,
    engage_parallel_cos_threshold=math.cos(math.radians(5)),
    engage_consecutive_steps=3,
    reengage_cooldown_steps=10,
    limit_surface=MicrospineLimitSurfaceCfg(unit_layout=unit_layout, hold_grace_steps=1),
  )
