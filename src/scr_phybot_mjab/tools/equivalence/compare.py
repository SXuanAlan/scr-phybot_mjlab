"""Compare two snapshot directories, tier by tier."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scr_phybot_mjab.tools.equivalence.canonical import apply_renames
from scr_phybot_mjab.tools.equivalence.fingerprint import changed_parts

# Private command attributes may be renamed by a refactor; their effects are
# still checked through observations, rewards and public state.
_PRIVATE_PREFIXES = ("cmd_private/", "reset/cmd_private/")
# Signals stacking one column per manager term, with the meta key naming the
# columns and the log key under which a removed term is declared.
_TERM_STACKS = {
  "reward/terms": ("reward_terms", "log/Episode_Reward/{}"),
  "metrics": ("metric_terms", "log/Episode_Metrics/{}"),
  "reset/metrics": ("metric_terms", "log/Episode_Metrics/{}"),
}
_MAX_DETAIL_LINES = 40
# Rollout noise is calibrated per signal column (trailing axes after steps and
# envs): quiet columns such as reward terms differ by ~1e-8 between same-code
# runs, busy ones such as joint velocities by ~1e-3. A column's floor is the
# 99.9% quantile of its same-code |difference|, which leaves out a rare
# threshold flip. A flip keeps an env off the baseline for the rest of its
# reset segment, so new code may exceed _FLOOR_MARGIN x floor in up to
# _MIN_OUTLIERS entries (one segment) or _OUTLIER_SHARE of the column,
# whichever is larger. A real logic change moves far more entries.
_NOISE_QUANTILE = 0.999
_FLOOR_MARGIN = 10.0
_EPS = 1e-7
_OUTLIER_SHARE = 0.02
_MIN_OUTLIERS = 5


@dataclass(frozen=True)
class Finding:
  tier: str
  item: str
  ok: bool
  detail: str = ""


class Removals:
  """Declared removals: cfg keys and rollout signals the new code no longer has.

  A pattern (``re.search``) excuses base-only cfg lines and missing rollout
  signals whose key it matches. A removed dict entry also shrinks the count
  in its container's ``dict[N]`` header. Anything the new snapshot still has
  is compared as usual, so an over-broad pattern shows up as an addition.
  """

  def __init__(self, patterns: list[str]) -> None:
    self.patterns = [re.compile(p) for p in patterns]
    self.hits: Counter[str] = Counter()

  def _match(self, key: str) -> bool:
    for pattern in self.patterns:
      if pattern.search(key):
        self.hits[pattern.pattern] += 1
        return True
    return False

  def drop_cfg_lines(self, lines: list[str]) -> list[str]:
    kept, entries = [], Counter()
    for line in lines:
      key = line.split(" = ", 1)[0]
      if not self._match(key):
        kept.append(line)
      elif parent := re.fullmatch(r"(.*)\[[^\[\]]*\]", key):
        entries[parent.group(1)] += 1
    out = []
    for line in kept:
      key, _, value = line.partition(" = ")
      header = re.fullmatch(r"dict\[(\d+)\]", value)
      if header and entries[key]:
        line = f"{key} = dict[{int(header.group(1)) - entries[key]}]"
      out.append(line)
    return out

  def allows_signal(self, key: str) -> bool:
    return self._match(key)

  def covers(self, key: str) -> bool:
    """Whether a pattern matches ``key``, without counting it as a hit."""
    return any(pattern.search(key) for pattern in self.patterns)

  def report(self) -> list[str]:
    return [f"declared removal '{p.pattern}': {self.hits[p.pattern]} base entries" for p in self.patterns]


def compare_cfg(base: Path, new: Path, renames: dict[str, str], removed: Removals | None = None) -> list[Finding]:
  findings = []
  for name, base_file, new_file in _paired(base / "cfg", new / "cfg", "*.txt"):
    if new_file is None or base_file is None:
      findings.append(_unpaired("cfg", name, base_file))
      continue
    old = apply_renames(base_file.read_text().splitlines(), renames)
    if removed is not None:
      old = removed.drop_cfg_lines(old)
    cur = new_file.read_text().splitlines()
    diff = list(difflib.unified_diff(old, cur, "base", "new", lineterm="", n=0))
    findings.append(Finding("cfg", name, not diff, "\n".join(diff[:_MAX_DETAIL_LINES])))
  return findings


def compare_model(base: Path, new: Path) -> list[Finding]:
  findings = []
  for name, base_file, new_file in _paired(base / "model", new / "model", "*.json"):
    if new_file is None or base_file is None:
      findings.append(_unpaired("model", name, base_file))
      continue
    old, cur = json.loads(base_file.read_text()), json.loads(new_file.read_text())
    same = old["mjb_sha256"] == cur["mjb_sha256"]
    findings.append(Finding("model", name, same, "" if same else "\n".join(changed_parts(old, cur))))
  return findings


def compare_rollouts(
  base: Path, new: Path, floor: dict[str, np.ndarray], removed: Removals | None = None
) -> list[Finding]:
  """Every rollout signal column must stay within its same-code noise floor."""
  findings = []
  for name, base_file, new_file in _paired(base / "rollout", new / "rollout", "*.npz"):
    if new_file is None or base_file is None:
      findings.append(_unpaired("rollout", name, base_file))
      continue
    problems = _meta_problems(base_file, new_file, removed) + _signal_problems(
      base_file, new_file, name, floor, removed
    )
    findings.append(Finding("rollout", name, not problems, "\n".join(problems[:_MAX_DETAIL_LINES])))
  return findings


def noise_floor(runs: list[Path]) -> dict[str, np.ndarray]:
  """Per-column same-code noise of every rollout signal, keyed ``<file>|<signal>``.

  Floats: 99.9% quantile of |runs[0] - run| per column. Bools/ints: share of
  differing entries per column. The maximum over runs[1:] is kept.
  """
  floor: dict[str, np.ndarray] = {}
  for other in runs[1:]:
    for name, first_file, other_file in _paired(runs[0] / "rollout", other / "rollout", "*.npz"):
      if first_file is None or other_file is None:
        continue
      with np.load(first_file) as first, np.load(other_file) as again:
        for key in first.files:
          if key in again.files and first[key].shape == again[key].shape:
            noise = _column_noise(first[key], again[key])
            slot = f"{name}|{key}"
            floor[slot] = np.maximum(floor[slot], noise) if slot in floor else noise
  return floor


def compare_clips(base: Path, new: Path) -> list[Finding]:
  findings = []
  for name, base_file, new_file in _paired(base / "clips", new / "clips", "*.np[yz]"):
    if new_file is None or base_file is None:
      findings.append(_unpaired("clips", name, base_file))
      continue
    problems = _value_problems(_load_clip(base_file), _load_clip(new_file), "")
    findings.append(Finding("clips", name, not problems, "\n".join(problems[:_MAX_DETAIL_LINES])))
  for name, base_file, new_file in _paired(base / "clips", new / "clips", "*.json"):
    if new_file is not None and base_file is not None:
      old, cur = json.loads(base_file.read_text()), json.loads(new_file.read_text())
      findings.append(Finding("clips", name, old == cur, "" if old == cur else "json content differs"))
  return findings


def _paired(base_dir: Path, new_dir: Path, pattern: str):
  names = sorted({p.name for p in base_dir.glob(pattern)} | {p.name for p in new_dir.glob(pattern)})
  for file_name in names:
    base_file, new_file = base_dir / file_name, new_dir / file_name
    yield file_name, base_file if base_file.exists() else None, new_file if new_file.exists() else None


def _unpaired(tier: str, name: str, base_file: Path | None) -> Finding:
  where = "new snapshot" if base_file is not None else "base snapshot"
  return Finding(tier, name, False, f"missing from the {where}")


def _signal_problems(
  base_file: Path, new_file: Path, name: str, floor: dict[str, np.ndarray], removed: Removals | None = None
) -> list[str]:
  problems = []
  old_meta = json.loads(base_file.with_suffix(".json").read_text())
  new_meta = json.loads(new_file.with_suffix(".json").read_text())
  with np.load(base_file) as old, np.load(new_file) as cur:
    for key in old.files:
      if key not in cur.files:
        if not key.startswith(_PRIVATE_PREFIXES) and not (removed is not None and removed.allows_signal(key)):
          problems.append(f"{key}: missing")
        continue
      base_values, noise = old[key], floor.get(f"{name}|{key}")
      if key in _TERM_STACKS:
        meta_key, _ = _TERM_STACKS[key]
        kept = _kept_columns(old_meta.get(meta_key), new_meta.get(meta_key), key, removed)
        if kept is not None:
          base_values = base_values[..., kept]
          noise = None if noise is None else noise[kept]
      problem = _signal_problem(key, base_values, cur[key], noise)
      if problem:
        problems.append(problem)
    new_keys = [key for key in cur.files if key not in old.files and not key.startswith(_PRIVATE_PREFIXES)]
    problems += [f"{key}: new signal" for key in new_keys]
  return problems


def _signal_problem(key: str, old: np.ndarray, cur: np.ndarray, noise: np.ndarray | None) -> str | None:
  if old.shape != cur.shape or old.dtype != cur.dtype:
    return f"{key}: {old.dtype}{old.shape} -> {cur.dtype}{cur.shape}"
  base, new = _columns(old), _columns(cur)
  noise = np.zeros(base.shape[1]) if noise is None else noise
  entries = base.shape[0]
  allowed = np.maximum(_MIN_OUTLIERS, _OUTLIER_SHARE * entries)
  if old.dtype.kind in "fc":
    exceed = _abs_diff(base, new) > _FLOOR_MARGIN * noise + _EPS * _column_scale(base)
  else:
    exceed = base != new
    allowed = np.maximum(allowed, _FLOOR_MARGIN * noise * entries)
  counts = exceed.sum(axis=0)
  bad = np.flatnonzero(counts > allowed)
  if bad.size == 0:
    return None
  worst = bad[np.argmax(counts[bad])]
  return (
    f"{key}: {bad.size}/{counts.size} columns beyond noise; column {worst} differs in "
    f"{counts[worst]}/{entries} entries (first at {_first_step(key, old, cur)})"
  )


def _column_noise(old: np.ndarray, cur: np.ndarray) -> np.ndarray:
  first, again = _columns(old), _columns(cur)
  if old.dtype.kind in "fc":
    return np.quantile(_abs_diff(first, again), _NOISE_QUANTILE, axis=0)
  return np.mean(first != again, axis=0)


def _columns(values: np.ndarray) -> np.ndarray:
  """(entries, columns) view: leading steps/segments and envs become entries."""
  if values.ndim <= 2:
    return values.reshape(-1, 1)
  return values.reshape(values.shape[0] * values.shape[1], -1)


def _abs_diff(old: np.ndarray, cur: np.ndarray) -> np.ndarray:
  old, cur = np.atleast_1d(old), np.atleast_1d(cur)
  delta = np.abs(old.astype(np.float64) - cur.astype(np.float64))
  delta[np.isnan(old) & np.isnan(cur)] = 0.0
  return np.nan_to_num(delta, nan=np.inf)


def _column_scale(values: np.ndarray) -> np.ndarray:
  finite = np.where(np.isfinite(values), np.abs(values.astype(np.float64)), 0.0)
  return np.maximum(1.0, finite.max(axis=0)) if finite.size else np.ones(values.shape[1])


def _exact_problem(key: str, old: np.ndarray, cur: np.ndarray) -> str | None:
  if old.shape != cur.shape or old.dtype != cur.dtype:
    return f"{key}: {old.dtype}{old.shape} -> {cur.dtype}{cur.shape}"
  distance = _abs_diff(old, cur).max() if old.dtype.kind in "fc" else float(np.count_nonzero(old != cur))
  if distance == 0.0:
    return None
  kind = f"max |d| = {distance:.3g}" if old.dtype.kind in "fc" else f"{int(distance)} entries differ"
  return f"{key}: {kind} (first at {_first_step(key, old, cur)})"


def _first_step(key: str, old: np.ndarray, cur: np.ndarray) -> str:
  if old.ndim == 0:
    return "scalar"
  differs = old != cur
  if old.dtype.kind in "fc":
    differs &= ~(np.isnan(old) & np.isnan(cur))
  rows = np.flatnonzero(differs.reshape(differs.shape[0], -1).any(axis=1))
  unit = "segment" if key.startswith("reset/") else "step"
  return f"{unit} {rows[0]}" if rows.size else "?"


def _meta_problems(base_file: Path, new_file: Path, removed: Removals | None = None) -> list[str]:
  old = json.loads(base_file.with_suffix(".json").read_text())
  cur = json.loads(new_file.with_suffix(".json").read_text())
  stacks = {meta_key: stack for stack, (meta_key, _) in _TERM_STACKS.items()}
  return [
    f"meta {k}: {old[k]!r} -> {cur.get(k)!r}"
    for k in old
    if old[k] != cur.get(k) and not (k in stacks and _kept_columns(old[k], cur.get(k), stacks[k], removed))
  ]


def _kept_columns(
  old_terms: list[str] | None, new_terms: list[str] | None, stack: str, removed: Removals | None
) -> list[int] | None:
  """Base columns of a term stack that remain, in new order, when the only
  change is the removal of declared terms; None otherwise."""
  if removed is None or old_terms is None or new_terms is None or old_terms == new_terms:
    return None
  _, log_key = _TERM_STACKS[stack]
  dropped = [term for term in old_terms if term not in new_terms]
  if [term for term in old_terms if term in new_terms] != new_terms:
    return None
  if not all(removed.covers(log_key.format(term)) for term in dropped):
    return None
  return [old_terms.index(term) for term in new_terms]


def _load_clip(path: Path) -> Any:
  if path.suffix == ".npz":
    with np.load(path, allow_pickle=True) as data:
      return {key: data[key] for key in data.files}
  data = np.load(path, allow_pickle=True)
  return data.item() if data.dtype == object and data.shape == () else data


def _value_problems(old: Any, cur: Any, path: str) -> list[str]:
  if isinstance(old, dict) and isinstance(cur, dict):
    problems = [f"{path}/{k}: missing" for k in old if k not in cur]
    problems += [f"{path}/{k}: new key" for k in cur if k not in old]
    for key in old.keys() & cur.keys():
      problems += _value_problems(old[key], cur[key], f"{path}/{key}")
    return problems
  if isinstance(old, np.ndarray) or isinstance(cur, np.ndarray):
    old_array, cur_array = np.asarray(old), np.asarray(cur)
    if old_array.dtype == object or cur_array.dtype == object:
      same = repr(old_array.tolist()) == repr(cur_array.tolist())
      return [] if same else [f"{path}: object array differs"]
    problem = _exact_problem(path, old_array, cur_array)
    return [problem] if problem else []
  if isinstance(old, (list, tuple)) and isinstance(cur, (list, tuple)):
    if len(old) != len(cur):
      return [f"{path}: length {len(old)} -> {len(cur)}"]
    return [p for i, (a, b) in enumerate(zip(old, cur)) for p in _value_problems(a, b, f"{path}[{i}]")]
  return [] if old == cur else [f"{path}: {old!r} -> {cur!r}"]
