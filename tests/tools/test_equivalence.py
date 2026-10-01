"""The equivalence harness itself: canonical dumps and rollout judging."""

from dataclasses import dataclass, field

import numpy as np

from scr_phybot_mjab.tools.equivalence.canonical import apply_renames, canonical_lines
from scr_phybot_mjab.tools.equivalence.compare import Removals, _column_noise, _signal_problem, _value_problems


@dataclass
class _Term:
  weight: float = 1.0
  params: dict = field(default_factory=dict)


def _kernel(env, std=0.3):
  return std


def test_dump_keeps_order_exact_floats_and_defaults():
  lines = canonical_lines({"b": _Term(0.1), "a": _Term(2.0)}, "cfg")
  assert lines.index("cfg[b].weight = f:0.1") < lines.index("cfg[a].weight = f:2.0")
  kernel = canonical_lines(_kernel, "fn")
  assert f"fn = fn:{__name__}._kernel" in kernel
  assert "fn.<defaults>[0] = f:0.3" in kernel


def test_shared_mutable_state_is_written_once():
  shared = {"k": 1}
  lines = canonical_lines({"x": _Term(params=shared), "y": _Term(params=shared)}, "cfg")
  assert "cfg[y].params = alias:cfg[x].params" in lines


def test_closure_values_are_part_of_the_dump():
  def make(scale):
    return lambda env: scale

  assert any(line.endswith("f:0.5") for line in canonical_lines(make(0.5), "fn"))
  assert canonical_lines(make(0.5), "fn") != canonical_lines(make(0.6), "fn")


def test_renames_rewrite_moved_functions_only():
  lines = ["cfg.func = fn:old.mod.kernel", "cfg.name = s:'old.mod.kernel'"]
  renamed = apply_renames(lines, {"old.mod.kernel": "new.mod.kernel"})
  assert renamed == ["cfg.func = fn:new.mod.kernel", "cfg.name = s:'old.mod.kernel'"]


def test_renames_cover_instances_and_enum_members_of_moved_classes():
  lines = [
    "cfg.toe = dataclass:old.mod.Toe",
    "cfg.obj = object:old.mod.Toe",
    "cfg.kind = enum:old.mod.Kind.A",
    "cfg.other = dataclass:old.mod.Toes",
  ]
  renamed = apply_renames(lines, {"old.mod.Toe": "new.mod.Toe", "old.mod.Kind": "new.mod.Kind"})
  assert renamed == [
    "cfg.toe = dataclass:new.mod.Toe",
    "cfg.obj = object:new.mod.Toe",
    "cfg.kind = enum:new.mod.Kind.A",
    "cfg.other = dataclass:old.mod.Toes",
  ]


def test_rollout_judging_tolerates_noise_but_not_a_changed_column():
  rng = np.random.default_rng(0)
  base = rng.normal(size=(150, 8, 3)).astype(np.float32)
  rerun = base + rng.normal(scale=1e-6, size=base.shape).astype(np.float32)
  noise = _column_noise(base, rerun)

  again = base + rng.normal(scale=1e-6, size=base.shape).astype(np.float32)
  again[7, 3, 1] += 5.0  # one threshold flip
  assert _signal_problem("obs", base, again, noise) is None

  changed = again.copy()
  changed[..., 2] *= 1.001
  problem = _signal_problem("obs", base, changed, noise)
  assert problem is not None and "column 2" in problem


def test_clip_comparison_handles_scalars_nested_lists_and_changes():
  clip = {"fps": np.float32(50.0), "names": ["a", "b"], "q": np.zeros((3, 2), np.float32)}
  assert _value_problems(clip, {**clip}, "") == []
  changed = {**clip, "fps": np.float32(60.0), "q": np.ones((3, 2), np.float32)}
  problems = _value_problems(clip, changed, "")
  assert any(p.startswith("/fps") for p in problems) and any(p.startswith("/q") for p in problems)


def test_declared_removals_drop_keys_and_shrink_dict_headers_only():
  base = [
    "env.rewards[r].params = dict[3]",
    "env.rewards[r].params[a] = f:1.0",
    "env.rewards[r].params[gone] = f:0.5",
    "env.rewards[r].params[b] = f:2.0",
    "env.cfg.gone_field = f:0.0",
  ]
  removals = Removals([r"params\[gone\]", r"[./]gone_field$"])
  assert removals.drop_cfg_lines(base) == [
    "env.rewards[r].params = dict[2]",
    "env.rewards[r].params[a] = f:1.0",
    "env.rewards[r].params[b] = f:2.0",
  ]
  assert removals.allows_signal("cmd/term/gone_field") and not removals.allows_signal("cmd/term/kept")
  assert removals.report() == [
    "declared removal 'params\\[gone\\]': 1 base entries",
    "declared removal '[./]gone_field$': 2 base entries",
  ]
