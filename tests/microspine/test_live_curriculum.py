import json

import pytest

from scr_phybot_mjab.tasks.climb_wall.mdp.live_curriculum import (
  LiveCurriculumControl, atomic_json, initial_control, validate_control,
)


def test_closed_gate_does_not_advance_at_nominal_boundaries(tmp_path):
  path = tmp_path / "curriculum_control.json"
  atomic_json(path, initial_control())
  gate = LiveCurriculumControl(path)
  for iteration in (0, 1499, 1500, 3300, 9999):
    assert gate.resolve(iteration)["angle_deg"] == 45.


def test_delayed_approval_preserves_original_time_in_next_stage(tmp_path):
  path = tmp_path / "curriculum_control.json"
  value = initial_control()
  atomic_json(path, value)
  gate = LiveCurriculumControl(path)
  gate.resolve(1500)
  value['max_stage'] = 2
  atomic_json(path, value)
  assert gate.resolve(2000)['angle_deg'] == 55.
  assert gate.resolve(3300)['angle_deg'] == 55.
  assert gate.resolve(3799)['angle_deg'] == 55.
  assert gate.resolve(3800)['angle_deg'] == 60.


def test_future_boundaries_can_be_extended_but_active_stage_cannot(tmp_path):
  path = tmp_path / "curriculum_control.json"
  value = initial_control()
  value['max_stage'] = 1
  value['angle_stages'][1]['iter'] = 1750
  value['angle_stages'][2]['iter'] = 3550
  atomic_json(path, value)
  gate = LiveCurriculumControl(path)
  assert gate.resolve(1500)['angle_deg'] == 45.
  assert gate.resolve(1750)['angle_deg'] == 55.
  value['angle_stages'][1]['iter'] = 2000
  atomic_json(path, value)
  assert gate.resolve(1751)['angle_deg'] == 55.
  assert gate.last_error


@pytest.mark.parametrize('corruption', ['missing', 'truncated', 'angle', 'early', 'max_stage'])
def test_invalid_control_does_not_advance(tmp_path, corruption):
  path = tmp_path / 'curriculum_control.json'
  value = initial_control()
  value['max_stage'] = 2
  if corruption == 'truncated':
    path.write_text('{')
  elif corruption != 'missing':
    if corruption == 'angle':
      value['angle_stages'][2]['angle_deg'] = 80.
    elif corruption == 'early':
      value['angle_stages'][1]['iter'] = 100
    else:
      value['max_stage'] = True
    path.write_text(json.dumps(value))
  gate = LiveCurriculumControl(path)
  assert gate.resolve(4000)['angle_deg'] == 45.
  assert gate.last_error


def test_control_defaults_are_independent_and_validate():
  first, second = initial_control(), initial_control()
  validate_control(first)
  first['angle_stages'][1]['iter'] = 2000
  assert second['angle_stages'][1]['iter'] == 1500
