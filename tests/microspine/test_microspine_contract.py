"""The microspine interface other code relies on: config fields and runtime attributes.

Config field names are CLI override paths and appear in saved run configs;
the runtime attributes are read by task MDP terms, scripts and tools. A
failure here is a rename or removal that has to be made deliberately,
together with its readers.
"""

import dataclasses
from types import SimpleNamespace

import mujoco
import pytest
import torch

from scr_phybot_mjab.microspine.affordance import MicrospineAffordance
from scr_phybot_mjab.microspine.config import (
  MicrospineAffordanceCfg,
  MicrospineCfg,
  MicrospineLimitSurfaceCfg,
)
from scr_phybot_mjab.microspine.constraint import MicrospineConstraint, make_constraint_event_cfg
from scr_phybot_mjab.microspine.contact import sphere_names
from scr_phybot_mjab.microspine.spec import add_attachment_equalities

MICROSPINE_CFG_FIELDS = {
  "ee_body_names",
  "spheres_per_ee_rows",
  "spheres_per_ee_cols",
  "grip_enabled",
  "affordance_name",
  "wall_geom_name",
  "surfaces",
  "arm_wall_contact_geoms",
  "contact_friction",
  "contact_condim",
  "contact_solref",
  "contact_solimp",
  "contact_margin",
  "contact_gap",
  "latched_contact_retract",
  "engage_unit_preload_n",
  "engage_parallel_cos_threshold",
  "virtual_spine_grid",
  "virtual_spine_travel",
  "engage_min_spine_fraction",
  "engage_max_antirake_vel",
  "engage_consecutive_steps",
  "reengage_cooldown_steps",
  "face_local_axis",
  "wall_entity_name",
  "initial_on",
  "initial_engaged_ee_names",
  "anchor_solref",
  "anchor_solimp",
  "anchor_torquescale",
  "anchor_reseat_step_cap",
  "reanchor_after_grace",
  "limit_surface",
}
LIMIT_SURFACE_CFG_FIELDS = {
  "enable",
  "empirical_table_path",
  "ee_rake_local_axis",
  "hold_judge_guard_steps",
  "unit_layout",
  "seed_hold_judge_guard_steps",
  "hold_grace_steps",
  "violation_consecutive_steps",
}
AFFORDANCE_CFG_FIELDS = {
  "microspine_cfg",
  "skip_zero_time_update",
  "asset_name",
  "contact_sensor_name",
  "physics_event_name",
  "resampling_time_range",
  "debug_vis",
}

# Read by task MDP terms (climb_wall, transition80) and by scripts.
AFFORDANCE_READS = (
  "command",
  "engaged",
  "seeded",
  "cooldown_steps",
  "seed_engaged",
  "set_initial_engaged_mask",
  "set_initial_engaged_ee_names",
  "anchor_load_tnl",
  "anchor_moment_tnl",
  "anchor_effective_load_tnl",
  "hold_margin",
  "hold_margin_raw",
  "forced_release_this_step",
  "engaged_this_step",
  "release_reason",
  "released_was_seeded",
  "face_cos",
  "contact_count",
  "pad_press_force",
  "spine_fraction",
  "unit_site_names",
  "unit_preload_n",
  "pad_contact_wrench_tln",
  "unit_strength_quantile",
  "unit_loading_quantile",
  "unit_load_tln",
  "hold_judge",
  "spine_field",
  "_engaged",
  "_engage_count",
  "_cooldown_steps",
  "_steps_since_engage",
  "_ee_body_ids",
  "_mcfg",
  "_get_executor",
  "_engage_frame",
  "_wall_inward_normal",
)
EXECUTOR_READS = (
  "engaged",
  "engage_age",
  "engaged_this_step",
  "disengaged_this_step",
  "current_disengage_time",
  "anchor_pos_w",
  "anchor_quat_w",
  "anchor_load_w",
  "anchor_moment_w",
  "pad_center_world",
  "compute_anchor_loads",
  "anchor_error_w",
  "assert_now",
  "initialize_seeded_anchors",
  "reset",
  "_ee_body_ids",
  "_eq_ids",
  "_pad_local",
  "_surfaces",
)


def field_names(cls) -> set[str]:
  return {field.name for field in dataclasses.fields(cls)}


def test_config_field_names_are_stable():
  assert field_names(MicrospineCfg) == MICROSPINE_CFG_FIELDS
  assert field_names(MicrospineLimitSurfaceCfg) == LIMIT_SURFACE_CFG_FIELDS
  assert field_names(MicrospineAffordanceCfg) == AFFORDANCE_CFG_FIELDS


@pytest.fixture
def plant():
  """Two pads with a 2x2 proxy grid, a 3x4 unit array and their weld anchors."""
  cfg = MicrospineCfg(ee_body_names=("hand", "foot"), spheres_per_ee_rows=2, spheres_per_ee_cols=2)
  spec = mujoco.MjSpec()
  for ee in cfg.ee_body_names:
    body = spec.worldbody.add_body(name="robot/" + ee)
    body.add_freejoint()
    body.add_site(name=f"robot/{ee}_contact_center")
    for row in range(3):
      for col in range(4):
        body.add_site(name=f"robot/{ee}_unit_{row}_{col}", pos=[(col - 1.5) * 0.028, 0, (row - 1) * 0.0315])
    for i, name in enumerate(sphere_names(ee, 2, 2)):
      proxy = body.add_body(name="robot/" + name, pos=[(i % 2 - 0.5) * 0.08, 0, (i // 2 - 0.5) * 0.06])
      proxy.add_geom(name="robot/" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.005, 0.01])
  add_attachment_equalities(spec, cfg)
  model = spec.compile()
  robot = SimpleNamespace(find_bodies=lambda names: ([model.body("robot/" + n).id for n in names], names))
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot}, sim=SimpleNamespace(mj_model=model))
  return cfg, env


def test_affordance_exposes_what_tasks_read(plant):
  cfg, env = plant
  term = MicrospineAffordance(MicrospineAffordanceCfg(microspine_cfg=cfg, contact_sensor_name="contact"), env)
  assert [name for name in AFFORDANCE_READS if not hasattr(term, name)] == []


def test_executor_exposes_what_tasks_read(plant):
  cfg, env = plant
  executor = MicrospineConstraint(make_constraint_event_cfg(cfg, affordance_name="microspine_affordance"), env)
  assert [name for name in EXECUTOR_READS if not hasattr(executor, name)] == []
  # The weld anchor sits at the authored contact centre C.
  torch.testing.assert_close(executor._pad_local, torch.zeros(2, 3))
