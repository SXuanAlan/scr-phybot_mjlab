"""Checkpoint resolution and the climb wall-angle pin shared by play and the evaluation scripts."""

import mjlab  # noqa: F401
import pytest

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import resolve_wall_angle
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import phybot_c2_walk_env_cfg
from scr_phybot_mjab.utils.runtime import checkpoint_iteration, pin_climb_wall_angle, resolve_checkpoint


@pytest.mark.parametrize(
  "name,iteration",
  [
    ("model_1250.pt", 1250),
    ("logs/run/model_0.pt", 0),
    ("ckpt-7.pt", 7),
    ("policy_42.pt", 42),
    ("policy.pt", None),
    (None, None),
  ],
)
def test_checkpoint_iteration_reads_the_name(name, iteration):
  assert checkpoint_iteration(name) == iteration


def test_wall_angle_pins_to_the_stage_the_checkpoint_reached():
  cfg = phybot_c2_climb_wall_env_cfg()
  stages = cfg.curriculum["wall_angle"].params["angle_stages"]
  pin_climb_wall_angle(cfg, wall_angle_deg=None, checkpoint="logs/run/model_3000.pt")
  expected = float(resolve_wall_angle(3000, stages))
  assert cfg.curriculum["wall_angle"].params["angle_stages"] == [{"iter": 0, "angle_deg": expected}]


def test_wall_angle_override_wins_and_unreadable_names_leave_the_ladder():
  cfg = phybot_c2_climb_wall_env_cfg()
  pin_climb_wall_angle(cfg, wall_angle_deg=72.5, checkpoint="logs/run/model_3000.pt")
  assert cfg.curriculum["wall_angle"].params["angle_stages"] == [{"iter": 0, "angle_deg": 72.5}]

  cfg = phybot_c2_climb_wall_env_cfg()
  ladder = list(cfg.curriculum["wall_angle"].params["angle_stages"])
  pin_climb_wall_angle(cfg, wall_angle_deg=None, checkpoint="policy.pt", checkpoint_hint=None)
  assert cfg.curriculum["wall_angle"].params["angle_stages"] == ladder


def test_tasks_without_a_wall_angle_are_untouched():
  cfg = phybot_c2_walk_env_cfg()
  curriculum = dict(cfg.curriculum)
  pin_climb_wall_angle(cfg, wall_angle_deg=60.0, checkpoint=None)
  assert cfg.curriculum == curriculum


def test_resolve_checkpoint_prefers_a_local_file(tmp_path):
  checkpoint = tmp_path / "model_5.pt"
  checkpoint.write_bytes(b"")
  assert resolve_checkpoint("any_experiment", str(checkpoint), None) == (checkpoint, "model_5.pt")
  with pytest.raises(FileNotFoundError):
    resolve_checkpoint("any_experiment", str(tmp_path / "missing.pt"), None)
  with pytest.raises(ValueError):
    resolve_checkpoint("any_experiment", None, None)
