"""The equivalence harness itself: canonical dumps and rollout judging."""

from dataclasses import dataclass, field

import numpy as np

from scr_phybot_mjab.tools.equivalence.canonical import apply_renames, canonical_lines
from scr_phybot_mjab.tools.equivalence.compare import _column_noise, _signal_problem


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
