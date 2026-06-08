"""Terrain configuration helpers for PHYBot walking tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def rough_terrain_cfg(max_init_terrain_level: int = 5) -> TerrainEntityCfg:
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(ROUGH_TERRAINS_CFG),
    max_init_terrain_level=max_init_terrain_level,
  )


def set_rough_terrain_curriculum(
  cfg: ManagerBasedRlEnvCfg,
  enabled: bool,
) -> None:
  terrain_cfg = cfg.scene.terrain
  if terrain_cfg is None or terrain_cfg.terrain_generator is None:
    return
  terrain_cfg.terrain_generator.curriculum = enabled


def configure_play_rough_terrain(
  cfg: ManagerBasedRlEnvCfg,
  *,
  num_cols: int = 5,
  num_rows: int = 5,
  border_width: float = 10.0,
) -> None:
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
  )

  terrain_cfg = cfg.scene.terrain
  if terrain_cfg is None or terrain_cfg.terrain_generator is None:
    return

  terrain_cfg.terrain_generator.curriculum = False
  terrain_cfg.terrain_generator.num_cols = num_cols
  terrain_cfg.terrain_generator.num_rows = num_rows
  terrain_cfg.terrain_generator.border_width = border_width


def configure_flat_terrain(
  cfg: ManagerBasedRlEnvCfg,
  *,
  terrain_cfg_fn: Callable[[Any], None] | None = None,
  spec_fn: Callable[[Any], None] | None = None,
) -> None:
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  if terrain_cfg_fn is not None:
    terrain_cfg_fn(cfg.scene.terrain)
  if spec_fn is not None:
    cfg.scene.spec_fn = spec_fn

  cfg.curriculum.pop("terrain_levels", None)
  cfg.events.pop("randomize_terrain", None)
