"""Headless failure-force sweep using the Rizon4 OSC arm controller.

This script exercises the same ``PadTeleop`` force-control path as the live
bench tool, but creates no Qt or MuJoCo viewer.  Every trial homes the arm,
engages the pad, ramps an absolute Cartesian force in the engage-frame
``(f_t, f_l, f_n)`` direction, and freezes the measured anchor load on the
single tick where the limit surface forces a release.

The default mode draws reproducible, stratified-random targets across all five
finite limit-surface faces, plus a smaller set of positive-margin interior
targets that must remain engaged.  ``canonical`` and ``planes`` remain
available as deterministic diagnostic modes.

Examples (run from the repository root inside ``env_mjlab``)::

  python scripts/sweep_rizon4_limitsurface.py
  python scripts/sweep_rizon4_limitsurface.py --surface-samples 500 --interior-samples 250 --seed 4
  python scripts/sweep_rizon4_limitsurface.py --mode planes --samples-per-plane 12
  python scripts/sweep_rizon4_limitsurface.py --cases pull-off reverse ultimate
  python scripts/sweep_rizon4_limitsurface.py --dry-run
  python scripts/sweep_rizon4_limitsurface.py --plot-only results/rizon4_limitsurface_sweep.npz

The output stem produces four files by default:

* ``.npz``: complete trial results and per-tick traces;
* ``.csv``: one summary row per trial;
* ``.releases.csv``: released t-n/t-l rows compatible with
  ``scripts/fit_limit_surface.py``;
* ``.png``: headless 2D projections and a 3D release-envelope summary.

Sign convention: +t is down-wall shear, +l is lateral, and +n is pull-off
tension (therefore -n is compression into the wall).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, get_type_hints

import numpy as np

from scr_phybot_mjab.microspine.limit_surface_geom import LIMIT_KEYS, envelope_margin


RELEASE_REASON_NAMES = {
  0: "none",
  1: "reverse-shear",
  2: "tension-cone",
  3: "lateral",
  4: "ultimate-shear",
}


@dataclass(frozen=True)
class SweepDirection:
  name: str
  plane: str
  angle_deg: float
  tln: tuple[float, float, float]
  sample_kind: str = "canonical"
  target_force_N: float = math.nan
  expected_release: bool | None = None


@dataclass
class TraceSample:
  trial_index: int
  step: int
  level_tick: int
  command_force_N: float
  command_t: float
  command_l: float
  command_n: float
  measured_t: float
  measured_l: float
  measured_n: float
  reported_margin: float
  model_margin: float
  engaged: bool
  released: bool
  release_reason: int


@dataclass
class TrialResult:
  trial_index: int
  repeat: int
  name: str
  plane: str
  angle_deg: float
  direction_t: float
  direction_l: float
  direction_n: float
  predicted_force_N: float
  predicted_reason: int
  predicted_reason_name: str
  expected_release: bool
  sample_kind: str
  target_force_N: float
  target_t: float
  target_l: float
  target_n: float
  target_margin: float
  step_N: float
  hold_ticks: int
  max_force_N: float
  baseline_force_N: float
  baseline_t: float
  baseline_l: float
  baseline_n: float
  engaged: bool
  released: bool
  unexpected_disengage: bool
  release_reason: int
  release_reason_name: str
  reason_matches_prediction: bool
  command_force_N: float
  command_t: float
  command_l: float
  command_n: float
  measured_force_N: float
  release_force_N: float
  f_t_N: float
  f_n_N: float
  f_l_N: float
  reported_margin: float
  model_margin: float
  steps: int
  duration_s: float
  status: str
  passed: bool


def _unit(
  name: str,
  plane: str,
  angle_deg: float,
  tln: Sequence[float],
  *,
  sample_kind: str = "canonical",
  target_force_N: float = math.nan,
  expected_release: bool | None = None,
) -> SweepDirection:
  vector = np.asarray(tln, dtype=float)
  norm = float(np.linalg.norm(vector))
  if not math.isfinite(norm) or norm <= 0.0:
    raise ValueError(f"direction {name!r} must be finite and nonzero")
  vector /= norm
  if math.isfinite(target_force_N) and target_force_N <= 0.0:
    raise ValueError(f"target force for {name!r} must be positive")
  return SweepDirection(
    name,
    plane,
    float(angle_deg),
    tuple(float(v) for v in vector),
    sample_kind,
    float(target_force_N),
    expected_release,
  )


def canonical_directions(include_compression: bool = False) -> list[SweepDirection]:
  """Finite rays spanning every face plus coupled 2D/3D loading."""
  directions = [
    _unit("pull-off", "tn", 90.0, (0.0, 0.0, 1.0)),
    _unit("lateral-pos", "tl", 90.0, (0.0, 1.0, 0.0)),
    _unit("lateral-neg", "tl", -90.0, (0.0, -1.0, 0.0)),
    _unit("reverse", "tn", 180.0, (-1.0, 0.0, 0.0)),
    _unit("ultimate", "tn", 0.0, (1.0, 0.0, 0.0)),
    _unit("shear-pull", "tn", 45.0, (1.0, 0.0, 1.0)),
    _unit("shear-lateral-pos", "tl", 45.0, (1.0, 1.0, 0.0)),
    _unit("shear-lateral-neg", "tl", -45.0, (1.0, -1.0, 0.0)),
    _unit("triaxial-pos", "3d", math.nan, (1.0, 1.0, 1.0)),
    _unit("triaxial-neg-l", "3d", math.nan, (1.0, -1.0, 1.0)),
  ]
  if include_compression:
    directions.append(_unit("compression", "tn", -90.0, (0.0, 0.0, -1.0)))
  return directions


def plane_directions(planes: Sequence[str], samples_per_plane: int) -> list[SweepDirection]:
  if samples_per_plane < 4:
    raise ValueError("samples_per_plane must be at least 4")
  directions = []
  for plane in planes:
    if plane not in ("tn", "tl"):
      raise ValueError(f"unknown plane {plane!r}")
    for angle_deg in np.linspace(-180.0, 180.0, samples_per_plane, endpoint=False):
      angle = math.radians(float(angle_deg))
      if plane == "tn":
        vector = (math.cos(angle), 0.0, math.sin(angle))
      else:
        vector = (math.cos(angle), math.sin(angle), 0.0)
      directions.append(
        _unit(
          f"{plane}_{angle_deg:+06.1f}deg",
          plane,
          float(angle_deg),
          vector,
          sample_kind="plane",
        )
      )
  return directions


_RANDOM_FACES = (
  ("reverse", 1, 0.0),
  ("tension", 2, 0.0),
  ("lateral-pos", 3, 1.0),
  ("lateral-neg", 3, -1.0),
  ("ultimate", 4, 0.0),
)


def _balanced_random_faces(count: int, rng: np.random.Generator) -> list[tuple[str, int, float]]:
  faces = list(_RANDOM_FACES) * (count // len(_RANDOM_FACES))
  remainder = count % len(_RANDOM_FACES)
  if remainder:
    indices = rng.choice(len(_RANDOM_FACES), size=remainder, replace=False)
    faces.extend(_RANDOM_FACES[int(index)] for index in indices)
  rng.shuffle(faces)
  return faces


def _random_boundary_point(
  rng: np.random.Generator,
  face: tuple[str, int, float],
  params: Mapping[str, float],
  normal_floor: float,
  edge_fraction: float = 0.05,
) -> np.ndarray:
  name, _, lateral_sign = face

  def interior_uniform(lower: float, upper: float) -> float:
    inset = edge_fraction * (upper - lower)
    return float(rng.uniform(lower + inset, upper - inset))

  if name == "reverse":
    f_t = -params["f_rev_max"]
  elif name == "ultimate":
    f_t = params["f_t_max"]
  else:
    f_t = interior_uniform(-params["f_rev_max"], params["f_t_max"])

  f_t_pos = max(f_t, 0.0)
  cap_n = params["mu_adhesion"] * f_t_pos + params["f_n0"]
  cap_l = params["mu_lateral"] * f_t_pos + params["f_l0"]
  if name == "tension":
    f_n = cap_n
    f_l = interior_uniform(-cap_l, cap_l)
  elif name in ("lateral-pos", "lateral-neg"):
    f_n = interior_uniform(normal_floor, cap_n)
    f_l = lateral_sign * cap_l
  else:
    f_n = interior_uniform(normal_floor, cap_n)
    f_l = interior_uniform(-cap_l, cap_l)
  return np.asarray((f_t, f_l, f_n), dtype=float)


def random_directions(
  params: Mapping[str, float],
  *,
  surface_samples: int,
  interior_samples: int,
  seed: int,
  max_force: float,
  soft_step: float,
  ultimate_step: float,
  normal_floor: float,
  interior_fraction: tuple[float, float],
  interior_margin: float,
) -> list[SweepDirection]:
  """Stratified random probes on all five finite faces and inside the envelope."""
  if surface_samples < 0 or interior_samples < 0:
    raise ValueError("random sample counts must be non-negative")
  if surface_samples + interior_samples == 0:
    raise ValueError("at least one random sample is required")
  fraction_lo, fraction_hi = interior_fraction
  if not (0.0 < fraction_lo <= fraction_hi < 1.0):
    raise ValueError("interior fractions must satisfy 0 < min <= max < 1")
  if not (0.0 < interior_margin < 1.0):
    raise ValueError("interior_margin must be between 0 and 1")
  if not math.isfinite(normal_floor) or normal_floor >= params["f_n0"]:
    raise ValueError("normal_floor must be finite and below f_n0")

  rng = np.random.default_rng(seed)
  directions: list[SweepDirection] = []

  def make_case(
    index: int,
    kind: str,
    face: tuple[str, int, float],
  ) -> SweepDirection:
    face_name, face_reason, _ = face
    for _ in range(500):
      boundary = _random_boundary_point(rng, face, params, normal_floor)
      boundary_force = float(np.linalg.norm(boundary))
      unit = boundary / boundary_force
      predicted_force, predicted_reason = predict_release(unit, params)
      if predicted_reason != face_reason or not math.isclose(
        predicted_force, boundary_force, rel_tol=1.0e-8, abs_tol=1.0e-8
      ):
        continue

      if kind == "surface":
        target = boundary
        target_force = boundary_force
        step = ultimate_step if face_reason == 4 else soft_step
        if target_force + step > max_force:
          continue
        expected_release = True
      else:
        target = boundary * float(rng.uniform(fraction_lo, fraction_hi))
        target_force = float(np.linalg.norm(target))
        if target_force > max_force or _model_margin(target, params) < interior_margin:
          continue
        expected_release = False

      plane = "tl" if face_reason == 3 else "tn"
      transverse = target[1] if plane == "tl" else target[2]
      angle_deg = math.degrees(math.atan2(float(transverse), float(target[0])))
      return _unit(
        f"{kind}-{face_name}-{index + 1:03d}",
        plane,
        angle_deg,
        target,
        sample_kind=kind,
        target_force_N=target_force,
        expected_release=expected_release,
      )
    raise ValueError(
      f"could not sample {kind} point on {face_name} below max_force={max_force:g} N"
    )

  for index, face in enumerate(_balanced_random_faces(surface_samples, rng)):
    directions.append(make_case(index, "surface", face))
  for index, face in enumerate(_balanced_random_faces(interior_samples, rng)):
    directions.append(make_case(index, "interior", face))
  rng.shuffle(directions)
  return directions


def select_directions(
  args: argparse.Namespace, params: Mapping[str, float]
) -> list[SweepDirection]:
  canonical = canonical_directions(include_compression=True)
  if args.cases:
    by_name = {direction.name: direction for direction in canonical}
    return [by_name[name] for name in args.cases]
  if args.mode == "planes":
    return plane_directions(args.planes, args.samples_per_plane)
  if args.mode == "canonical":
    return canonical_directions(args.include_compression)
  return random_directions(
    params,
    surface_samples=args.surface_samples,
    interior_samples=args.interior_samples,
    seed=args.seed,
    max_force=args.max_force,
    soft_step=args.soft_step,
    ultimate_step=args.ultimate_step,
    normal_floor=args.normal_floor,
    interior_fraction=(args.interior_min_fraction, args.interior_max_fraction),
    interior_margin=args.interior_margin,
  )


def predict_release(
  direction_tln: Sequence[float], params: Mapping[str, float]
) -> tuple[float, int]:
  """Return the ideal ray-envelope intersection and the limiting face."""
  f_t, f_l, f_n = (float(value) for value in direction_tln)
  f_t_pos = max(f_t, 0.0)
  candidates: list[tuple[float, int]] = []
  if f_t < -1.0e-12:
    candidates.append((params["f_rev_max"] / -f_t, 1))
  if f_t > 1.0e-12:
    candidates.append((params["f_t_max"] / f_t, 4))
  tension_denom = f_n - params["mu_adhesion"] * f_t_pos
  if tension_denom > 1.0e-12:
    candidates.append((params["f_n0"] / tension_denom, 2))
  lateral_denom = abs(f_l) - params["mu_lateral"] * f_t_pos
  if lateral_denom > 1.0e-12:
    candidates.append((params["f_l0"] / lateral_denom, 3))
  if not candidates:
    return math.inf, 0

  minimum = min(radius for radius, _ in candidates)
  # Match affordance overwrite priority when two faces are crossed together.
  priority = {3: 0, 1: 1, 2: 2, 4: 3}
  tied = [reason for radius, reason in candidates if math.isclose(radius, minimum)]
  return float(minimum), max(tied, key=priority.__getitem__)


def _model_margin(load_tln: Sequence[float], params: Mapping[str, float]) -> float:
  load = np.asarray(load_tln, dtype=float).reshape(1, 3)
  return float(envelope_margin(load, params)[0])


def _trace_sample(
  trial_index: int,
  step: int,
  level_tick: int,
  command_force: float,
  command_tln: np.ndarray,
  reading,
  reason: int,
  params: Mapping[str, float],
) -> TraceSample:
  load = np.asarray(reading.load_tln, dtype=float).copy()
  return TraceSample(
    trial_index=trial_index,
    step=step,
    level_tick=level_tick,
    command_force_N=float(command_force),
    command_t=float(command_tln[0]),
    command_l=float(command_tln[1]),
    command_n=float(command_tln[2]),
    measured_t=float(load[0]),
    measured_l=float(load[1]),
    measured_n=float(load[2]),
    reported_margin=float(reading.margin),
    model_margin=_model_margin(load, params),
    engaged=bool(reading.engaged),
    released=bool(reading.released),
    release_reason=int(reason),
  )


def run_trial(
  pad,
  direction: SweepDirection,
  *,
  trial_index: int,
  repeat: int,
  max_force: float,
  soft_step: float,
  ultimate_step: float,
  hold_ticks: int,
  zero_ticks: int,
  margin_tolerance: float,
  traces: list[TraceSample],
) -> TrialResult:
  started = time.perf_counter()
  params = pad.limit_params
  vector = np.asarray(direction.tln, dtype=float)
  predicted_force, predicted_reason = predict_release(vector, params)
  force_step = ultimate_step if predicted_reason == 4 else soft_step
  nan = float("nan")
  if math.isfinite(direction.target_force_N):
    target_force = float(direction.target_force_N)
    target_tln = target_force * vector
    target_margin = _model_margin(target_tln, params)
    expected_release = bool(direction.expected_release)
    # Surface targets define the random ray and theoretical intersection.  The
    # measured arm load may lag the command, so keep advancing that same ray
    # until measured release or the global safety cap.  Interior probes stop
    # exactly at their planned target.
    trial_max_force = max_force if expected_release else target_force
    if target_force > max_force + 1.0e-9:
      raise ValueError(
        f"trial {direction.name!r} requires {target_force:.1f} N, "
        f"above max_force={max_force:.1f} N"
      )
  else:
    target_force = nan
    target_tln = np.full(3, nan)
    target_margin = nan
    trial_max_force = max_force
    expected_release = math.isfinite(predicted_force) and predicted_force <= trial_max_force

  pad.home()
  pad.press_engage()
  baseline = np.full(3, nan)
  reading = None
  reason = 0
  steps = 0
  baseline_ok = True
  for zero_tick in range(zero_ticks):
    reading = pad.apply((0.0, 0.0, 0.0))
    steps += 1
    reason = int(pad.affordance.release_reason[0, 0].item())
    sample = _trace_sample(
      trial_index, steps, zero_tick, 0.0, np.zeros(3), reading, reason, params
    )
    traces.append(sample)
    baseline = np.array((sample.measured_t, sample.measured_l, sample.measured_n))
    if reading.released or not reading.engaged:
      baseline_ok = False
      break

  baseline_norm = float(np.linalg.norm(baseline))
  if not baseline_ok:
    released = bool(reading is not None and reading.released)
    status = "preload-release" if released else "engage-failed"
    load = baseline
    model_margin = _model_margin(load, params) if np.all(np.isfinite(load)) else nan
    return TrialResult(
      trial_index=trial_index,
      repeat=repeat,
      name=direction.name,
      plane=direction.plane,
      angle_deg=direction.angle_deg,
      direction_t=direction.tln[0],
      direction_l=direction.tln[1],
      direction_n=direction.tln[2],
      predicted_force_N=predicted_force,
      predicted_reason=predicted_reason,
      predicted_reason_name=RELEASE_REASON_NAMES[predicted_reason],
      expected_release=expected_release,
      sample_kind=direction.sample_kind,
      target_force_N=target_force,
      target_t=float(target_tln[0]),
      target_l=float(target_tln[1]),
      target_n=float(target_tln[2]),
      target_margin=target_margin,
      step_N=force_step,
      hold_ticks=hold_ticks,
      max_force_N=trial_max_force,
      baseline_force_N=baseline_norm,
      baseline_t=float(baseline[0]),
      baseline_l=float(baseline[1]),
      baseline_n=float(baseline[2]),
      engaged=False,
      released=released,
      unexpected_disengage=not released,
      release_reason=reason,
      release_reason_name=RELEASE_REASON_NAMES.get(reason, f"reason-{reason}"),
      reason_matches_prediction=released and reason == predicted_reason,
      command_force_N=0.0,
      command_t=0.0,
      command_l=0.0,
      command_n=0.0,
      measured_force_N=baseline_norm,
      release_force_N=baseline_norm if released else nan,
      f_t_N=float(load[0]),
      f_n_N=float(load[2]),
      f_l_N=float(load[1]),
      reported_margin=float(reading.margin) if reading is not None else nan,
      model_margin=model_margin,
      steps=steps,
      duration_s=time.perf_counter() - started,
      status=status,
      passed=False,
    )

  released = False
  unexpected_disengage = False
  command_force = 0.0
  command_tln = np.zeros(3)
  load = baseline.copy()
  reported_margin = float(reading.margin)
  model_margin = _model_margin(load, params)
  reason = 0

  while command_force < trial_max_force - 1.0e-12:
    command_force = min(command_force + force_step, trial_max_force)
    command_tln = command_force * vector
    for level_tick in range(hold_ticks):
      reading = pad.apply(tuple(float(value) for value in command_tln))
      steps += 1
      reason_now = int(pad.affordance.release_reason[0, 0].item())
      sample = _trace_sample(
        trial_index,
        steps,
        level_tick,
        command_force,
        command_tln,
        reading,
        reason_now,
        params,
      )
      traces.append(sample)
      load = np.array((sample.measured_t, sample.measured_l, sample.measured_n))
      reported_margin = sample.reported_margin
      model_margin = sample.model_margin
      if reading.released:
        released = True
        reason = reason_now
        break
      if not reading.engaged:
        unexpected_disengage = True
        break
    if released or unexpected_disengage:
      break

  finite = bool(
    np.all(np.isfinite(load))
    and math.isfinite(reported_margin)
    and math.isfinite(model_margin)
  )
  measured_force = float(np.linalg.norm(load))
  release_force = measured_force if released else nan
  reason_matches = released and reason == predicted_reason

  if unexpected_disengage:
    status, passed = "unexpected-disengage", False
  elif not finite:
    status, passed = "nonfinite", False
  elif released and reading.engaged:
    status, passed = "release-state-inconsistent", False
  elif expected_release and not released:
    status, passed = "missing-release", False
  elif not expected_release and released:
    status, passed = "unexpected-release", False
  elif direction.sample_kind == "interior" and model_margin < 0.0:
    status, passed = "interior-outside-envelope", False
  elif released and abs(model_margin) > margin_tolerance:
    status, passed = "release-off-envelope", False
  elif direction.name == "ultimate" and load[0] <= 350.0:
    status, passed = "ultimate-load-too-low", False
  elif released:
    status, passed = "released", True
  elif direction.sample_kind == "interior":
    status, passed = "held-interior", True
  else:
    status, passed = "held-to-max", True

  return TrialResult(
    trial_index=trial_index,
    repeat=repeat,
    name=direction.name,
    plane=direction.plane,
    angle_deg=direction.angle_deg,
    direction_t=direction.tln[0],
    direction_l=direction.tln[1],
    direction_n=direction.tln[2],
    predicted_force_N=predicted_force,
    predicted_reason=predicted_reason,
    predicted_reason_name=RELEASE_REASON_NAMES[predicted_reason],
    expected_release=expected_release,
    sample_kind=direction.sample_kind,
    target_force_N=target_force,
    target_t=float(target_tln[0]),
    target_l=float(target_tln[1]),
    target_n=float(target_tln[2]),
    target_margin=target_margin,
    step_N=force_step,
    hold_ticks=hold_ticks,
    max_force_N=trial_max_force,
    baseline_force_N=baseline_norm,
    baseline_t=float(baseline[0]),
    baseline_l=float(baseline[1]),
    baseline_n=float(baseline[2]),
    engaged=bool(reading.engaged),
    released=released,
    unexpected_disengage=unexpected_disengage,
    release_reason=reason,
    release_reason_name=RELEASE_REASON_NAMES.get(reason, f"reason-{reason}"),
    reason_matches_prediction=reason_matches,
    command_force_N=command_force,
    command_t=float(command_tln[0]),
    command_l=float(command_tln[1]),
    command_n=float(command_tln[2]),
    measured_force_N=measured_force,
    release_force_N=release_force,
    f_t_N=float(load[0]),
    f_n_N=float(load[2]),
    f_l_N=float(load[1]),
    reported_margin=reported_margin,
    model_margin=model_margin,
    steps=steps,
    duration_s=time.perf_counter() - started,
    status=status,
    passed=passed,
  )


def _npz_arrays(
  results: Sequence[TrialResult],
  traces: Sequence[TraceSample],
  params: Mapping[str, float],
  metadata: Mapping[str, object],
) -> dict[str, np.ndarray]:
  def result_array(field: str, dtype=None):
    return np.asarray([getattr(result, field) for result in results], dtype=dtype)

  direction_tln = np.asarray(
    [[result.direction_t, result.direction_l, result.direction_n] for result in results],
    dtype=float,
  ).reshape(-1, 3)
  command_tln = np.asarray(
    [[result.command_t, result.command_l, result.command_n] for result in results],
    dtype=float,
  ).reshape(-1, 3)
  target_tln = np.asarray(
    [[result.target_t, result.target_l, result.target_n] for result in results],
    dtype=float,
  ).reshape(-1, 3)
  load_tln = np.asarray(
    [[result.f_t_N, result.f_l_N, result.f_n_N] for result in results], dtype=float
  ).reshape(-1, 3)
  baseline_tln = np.asarray(
    [[result.baseline_t, result.baseline_l, result.baseline_n] for result in results],
    dtype=float,
  ).reshape(-1, 3)
  trace_command_tln = np.asarray(
    [[sample.command_t, sample.command_l, sample.command_n] for sample in traces],
    dtype=float,
  ).reshape(-1, 3)
  trace_load_tln = np.asarray(
    [[sample.measured_t, sample.measured_l, sample.measured_n] for sample in traces],
    dtype=float,
  ).reshape(-1, 3)
  released_loads = load_tln[result_array("released", bool)]
  interior_loads = load_tln[result_array("sample_kind", "U16") == "interior"]

  return {
    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    "param_names": np.asarray(LIMIT_KEYS, dtype="U32"),
    "params": np.asarray([params[key] for key in LIMIT_KEYS], dtype=float),
    "name": result_array("name", "U64"),
    "plane": result_array("plane", "U8"),
    "angle_deg": result_array("angle_deg", float),
    "repeat": result_array("repeat", int),
    "direction_tln": direction_tln,
    "predicted_force_N": result_array("predicted_force_N", float),
    "predicted_reason": result_array("predicted_reason", int),
    "expected_release": result_array("expected_release", bool),
    "sample_kind": result_array("sample_kind", "U16"),
    "target_force_N": result_array("target_force_N", float),
    "target_tln": target_tln,
    "target_margin": result_array("target_margin", float),
    "engaged": result_array("engaged", bool),
    "released": result_array("released", bool),
    "release_reason": result_array("release_reason", int),
    "reason_matches_prediction": result_array("reason_matches_prediction", bool),
    "command_force_N": result_array("command_force_N", float),
    "command_tln": command_tln,
    "measured_force_N": result_array("measured_force_N", float),
    "load_tln": load_tln,
    "baseline_tln": baseline_tln,
    "reported_margin": result_array("reported_margin", float),
    "model_margin": result_array("model_margin", float),
    "status": result_array("status", "U40"),
    "passed": result_array("passed", bool),
    "steps": result_array("steps", int),
    "duration_s": result_array("duration_s", float),
    # Compatible with LimitSurfaceView.save().
    "releases": released_loads.reshape(-1, 3),
    "interior_loads": interior_loads.reshape(-1, 3),
    "trace_trial_index": np.asarray([sample.trial_index for sample in traces], dtype=int),
    "trace_step": np.asarray([sample.step for sample in traces], dtype=int),
    "trace_level_tick": np.asarray([sample.level_tick for sample in traces], dtype=int),
    "trace_command_force_N": np.asarray(
      [sample.command_force_N for sample in traces], dtype=float
    ),
    "trace_command_tln": trace_command_tln,
    "trace_load_tln": trace_load_tln,
    "trace_reported_margin": np.asarray(
      [sample.reported_margin for sample in traces], dtype=float
    ),
    "trace_model_margin": np.asarray(
      [sample.model_margin for sample in traces], dtype=float
    ),
    "trace_engaged": np.asarray([sample.engaged for sample in traces], dtype=bool),
    "trace_released": np.asarray([sample.released for sample in traces], dtype=bool),
    "trace_release_reason": np.asarray(
      [sample.release_reason for sample in traces], dtype=int
    ),
  }


def save_outputs(
  output: Path,
  results: Sequence[TrialResult],
  traces: Sequence[TraceSample],
  params: Mapping[str, float],
  metadata: Mapping[str, object],
) -> tuple[Path, Path, Path]:
  output = output if output.suffix == ".npz" else output.with_suffix(".npz")
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **_npz_arrays(results, traces, params, metadata))

  summary_csv = output.with_suffix(".csv")
  fieldnames = list(TrialResult.__dataclass_fields__)
  with summary_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(result) for result in results)

  releases_csv = output.with_name(output.stem + ".releases.csv")
  release_fields = [
    "plane",
    "angle_deg",
    "preload_N",
    "stroke_mm",
    "release_force_N",
    "f_t_N",
    "f_n_N",
    "f_l_N",
    "name",
    "repeat",
    "command_force_N",
    "release_reason",
    "release_reason_name",
    "model_margin",
    "sample_kind",
    "target_t_N",
    "target_n_N",
    "target_l_N",
    "target_margin",
  ]
  with releases_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=release_fields)
    writer.writeheader()
    for result in results:
      if not result.released or result.plane not in ("tn", "tl"):
        continue
      writer.writerow({
        "plane": result.plane,
        "angle_deg": result.angle_deg,
        "preload_N": result.baseline_force_N,
        "stroke_mm": "",
        "release_force_N": result.release_force_N,
        "f_t_N": result.f_t_N,
        "f_n_N": result.f_n_N,
        "f_l_N": result.f_l_N,
        "name": result.name,
        "repeat": result.repeat,
        "command_force_N": result.command_force_N,
        "release_reason": result.release_reason,
        "release_reason_name": result.release_reason_name,
        "model_margin": result.model_margin,
        "sample_kind": result.sample_kind,
        "target_t_N": result.target_t,
        "target_n_N": result.target_n,
        "target_l_N": result.target_l,
        "target_margin": result.target_margin,
      })
  return output, summary_csv, releases_csv


def load_saved_sweep(path: Path) -> tuple[list[TrialResult], dict[str, float]]:
  """Load a saved NPZ plus its sibling summary CSV for plot-only rendering."""
  npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
  csv_path = npz_path.with_suffix(".csv")
  if not npz_path.is_file():
    raise FileNotFoundError(f"saved sweep not found: {npz_path}")
  if not csv_path.is_file():
    raise FileNotFoundError(f"summary CSV required for plotting: {csv_path}")

  with np.load(npz_path, allow_pickle=False) as data:
    names = [str(name) for name in data["param_names"]]
    params = {
      name: float(value)
      for name, value in zip(names, data["params"], strict=True)
    }

  hints = get_type_hints(TrialResult)
  results: list[TrialResult] = []
  with csv_path.open(newline="") as stream:
    for row in csv.DictReader(stream):
      values: dict[str, object] = {}
      for name, kind in hints.items():
        raw = row[name]
        if kind is bool:
          values[name] = raw.strip().lower() in ("1", "true", "yes")
        elif kind is int:
          values[name] = int(raw)
        elif kind is float:
          values[name] = float(raw)
        else:
          values[name] = raw
      results.append(TrialResult(**values))
  if not results:
    raise ValueError(f"no trial rows found in {csv_path}")
  return results, params


def save_diagnostic_plot(
  path: Path,
  results: Sequence[TrialResult],
  params: Mapping[str, float],
) -> Path:
  """Render the detailed 2D-projection plus 3D diagnostic layout."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  path.parent.mkdir(parents=True, exist_ok=True)
  p = params
  colors = {
    0: "#7A828A",
    1: "#CC79A7",
    2: "#009E73",
    3: "#0072B2",
    4: "#D55E00",
  }
  labels = {
    1: "reverse shear",
    2: "tension cone",
    3: "lateral",
    4: "ultimate shear",
  }

  ft_curve = np.concatenate((
    np.linspace(-p["f_rev_max"], 0.0, 12, endpoint=False),
    np.linspace(0.0, p["f_t_max"], 90),
  ))
  cap_n = p["mu_adhesion"] * np.maximum(ft_curve, 0.0) + p["f_n0"]
  cap_l = p["mu_lateral"] * np.maximum(ft_curve, 0.0) + p["f_l0"]

  finite_results = [
    result
    for result in results
    if np.all(np.isfinite((result.f_t_N, result.f_l_N, result.f_n_N)))
  ]
  released = [
    result
    for result in finite_results
    if result.released and result.sample_kind != "interior"
  ]
  interior_held = [
    result for result in finite_results if result.status == "held-interior"
  ]
  held = [result for result in finite_results if result.status == "held-to-max"]
  failed = [
    result
    for result in finite_results
    if (
      (not result.released and result.status not in ("held-to-max", "held-interior"))
      or (result.released and result.sample_kind == "interior")
    )
  ]
  measured_ft = np.asarray([result.f_t_N for result in finite_results], dtype=float)
  measured_fl = np.asarray([result.f_l_N for result in finite_results], dtype=float)
  measured_fn = np.asarray([result.f_n_N for result in finite_results], dtype=float)

  x_values = np.concatenate((ft_curve, measured_ft))
  y_values = np.concatenate((-cap_l, cap_l, measured_fl))
  z_values = np.concatenate((cap_n, measured_fn))
  x_pad = 0.08 * max(float(np.ptp(x_values)), 1.0)
  y_pad = 0.08 * max(float(np.ptp(y_values)), 1.0)
  z_floor = min(-0.30 * float(cap_n.max()), float(z_values.min()) - 0.10 * max(np.ptp(z_values), 1.0))
  z_top = max(float(cap_n.max()), float(z_values.max()))
  z_pad = 0.08 * max(z_top - z_floor, 1.0)
  x_limits = (float(x_values.min()) - x_pad, float(x_values.max()) + x_pad)
  y_limits = (float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)
  z_limits = (z_floor - 0.02 * z_pad, z_top + z_pad)

  with plt.rc_context({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#202124",
    "axes.labelcolor": "#202124",
    "xtick.color": "#30343A",
    "ytick.color": "#30343A",
    "grid.color": "#CBD2D9",
    "grid.alpha": 0.55,
    "axes.titleweight": "bold",
  }):
    fig = plt.figure(figsize=(18.0, 6.4), constrained_layout=True, facecolor="white")
    ax_tn = fig.add_subplot(1, 3, 1)
    ax_tl = fig.add_subplot(1, 3, 2)
    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")

    ax_tn.plot(ft_curve, cap_n, color="#005A8D", lw=2.0, label="theoretical boundary")
    ax_tn.axvline(-p["f_rev_max"], color="#CC79A7", lw=1.4, ls="--")
    ax_tn.axvline(p["f_t_max"], color="#D55E00", lw=1.4, ls="--")
    ax_tn.fill_between(ft_curve, z_floor, cap_n, color="#56B4E9", alpha=0.10)

    ax_tl.plot(ft_curve, cap_l, color="#005A8D", lw=2.0, label="theoretical boundary")
    ax_tl.plot(ft_curve, -cap_l, color="#005A8D", lw=2.0)
    ax_tl.axvline(-p["f_rev_max"], color="#CC79A7", lw=1.4, ls="--")
    ax_tl.axvline(p["f_t_max"], color="#D55E00", lw=1.4, ls="--")
    ax_tl.fill_between(ft_curve, -cap_l, cap_l, color="#56B4E9", alpha=0.10)

    for reason in (1, 2, 3, 4):
      points = [result for result in released if result.release_reason == reason]
      if points:
        ax_tn.scatter(
          [result.f_t_N for result in points],
          [result.f_n_N for result in points],
          s=52,
          color=colors[reason],
          edgecolor="white",
          linewidth=0.8,
          zorder=5,
          label=labels[reason],
        )
        ax_tl.scatter(
          [result.f_t_N for result in points],
          [result.f_l_N for result in points],
          s=52,
          color=colors[reason],
          edgecolor="white",
          linewidth=0.8,
          zorder=5,
          label=labels[reason],
        )

    if interior_held:
      ax_tn.scatter(
        [result.f_t_N for result in interior_held],
        [result.f_n_N for result in interior_held],
        marker="D",
        s=34,
        color="#4C6A92",
        edgecolor="white",
        linewidth=0.6,
        label="interior held",
      )
      ax_tl.scatter(
        [result.f_t_N for result in interior_held],
        [result.f_l_N for result in interior_held],
        marker="D",
        s=34,
        color="#4C6A92",
        edgecolor="white",
        linewidth=0.6,
        label="interior held",
      )
    if held:
      ax_tn.scatter(
        [result.f_t_N for result in held],
        [result.f_n_N for result in held],
        marker="x",
        s=55,
        color=colors[0],
        label="held to max",
      )
      ax_tl.scatter(
        [result.f_t_N for result in held],
        [result.f_l_N for result in held],
        marker="x",
        s=55,
        color=colors[0],
        label="held to max",
      )
    if failed:
      ax_tn.scatter(
        [result.f_t_N for result in failed],
        [result.f_n_N for result in failed],
        marker="X",
        s=52,
        color="#C43C39",
        label="failed / censored",
      )
      ax_tl.scatter(
        [result.f_t_N for result in failed],
        [result.f_l_N for result in failed],
        marker="X",
        s=52,
        color="#C43C39",
        label="failed / censored",
      )

    for axis, ylabel, title, ylim in (
      (ax_tn, r"$f_n\;[\mathrm{N}]$", r"$t$–$n$ projection", z_limits),
      (ax_tl, r"$f_l\;[\mathrm{N}]$", r"$t$–$l$ projection", y_limits),
    ):
      axis.set_xlim(*x_limits)
      axis.set_ylim(*ylim)
      axis.set_xlabel(r"$f_t\;[\mathrm{N}]$")
      axis.set_ylabel(ylabel)
      axis.set_title(title)
      axis.grid(True)
      axis.spines[["top", "right"]].set_visible(False)
      handles, legend_labels = axis.get_legend_handles_labels()
      if handles:
        unique = dict(zip(legend_labels, handles, strict=True))
        axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8, frameon=False)

    # Five open physical boundary faces, clipped only for display in -f_n.
    lateral_fraction = np.linspace(-1.0, 1.0, 13)
    vertical_fraction = np.linspace(0.0, 1.0, 11)
    T = np.broadcast_to(ft_curve[:, None], (ft_curve.size, lateral_fraction.size))
    L = cap_l[:, None] * lateral_fraction[None, :]
    N = np.broadcast_to(cap_n[:, None], T.shape)
    ax_3d.plot_surface(T, L, N, color="#56B4E9", alpha=0.16, linewidth=0)
    ax_3d.plot_wireframe(
      T, L, N, rstride=8, cstride=2, color="#0072B2", alpha=0.42, linewidth=0.55
    )
    for sign in (-1.0, 1.0):
      Tw = np.broadcast_to(ft_curve[:, None], (ft_curve.size, vertical_fraction.size))
      Lw = sign * np.broadcast_to(cap_l[:, None], Tw.shape)
      Nw = z_floor + (cap_n[:, None] - z_floor) * vertical_fraction[None, :]
      ax_3d.plot_surface(Tw, Lw, Nw, color="#56B4E9", alpha=0.075, linewidth=0)
      ax_3d.plot_wireframe(
        Tw, Lw, Nw, rstride=8, cstride=3, color="#0072B2", alpha=0.26, linewidth=0.45
      )

    end_lateral_fraction = np.linspace(-1.0, 1.0, 13)
    for index in (0, -1):
      Te = np.full(
        (vertical_fraction.size, end_lateral_fraction.size), ft_curve[index]
      )
      Le = cap_l[index] * np.broadcast_to(end_lateral_fraction, Te.shape)
      Ne = z_floor + (cap_n[index] - z_floor) * np.broadcast_to(
        vertical_fraction[:, None], Te.shape
      )
      ax_3d.plot_surface(Te, Le, Ne, color="#56B4E9", alpha=0.075, linewidth=0)
      ax_3d.plot_wireframe(
        Te, Le, Ne, rstride=2, cstride=3, color="#0072B2", alpha=0.26, linewidth=0.45
      )

    for reason in (1, 2, 3, 4):
      points = [result for result in released if result.release_reason == reason]
      if points:
        ax_3d.scatter(
          [result.f_t_N for result in points],
          [result.f_l_N for result in points],
          [result.f_n_N for result in points],
          s=42,
          color=colors[reason],
          edgecolor="white",
          linewidth=0.7,
          depthshade=False,
          label=labels[reason],
        )
    if held:
      ax_3d.scatter(
        [result.f_t_N for result in held],
        [result.f_l_N for result in held],
        [result.f_n_N for result in held],
        marker="x",
        s=45,
        color=colors[0],
        depthshade=False,
        label="held to max",
      )
    if interior_held:
      ax_3d.scatter(
        [result.f_t_N for result in interior_held],
        [result.f_l_N for result in interior_held],
        [result.f_n_N for result in interior_held],
        marker="D",
        s=34,
        color="#4C6A92",
        edgecolor="white",
        linewidth=0.6,
        depthshade=False,
        label="interior held",
      )
    if failed:
      ax_3d.scatter(
        [result.f_t_N for result in failed],
        [result.f_l_N for result in failed],
        [result.f_n_N for result in failed],
        marker="X",
        s=45,
        color="#C43C39",
        depthshade=False,
        label="failed / censored",
      )
    ax_3d.set_xlim(*x_limits)
    ax_3d.set_ylim(*y_limits)
    ax_3d.set_zlim(*z_limits)
    ax_3d.set_xlabel(r"$f_t\;[\mathrm{N}]$", labelpad=8)
    ax_3d.set_ylabel(r"$f_l\;[\mathrm{N}]$", labelpad=8)
    ax_3d.set_zlabel(r"$f_n\;[\mathrm{N}]$", labelpad=8)
    ax_3d.set_title("3D release envelope")
    ax_3d.view_init(elev=22, azim=-55)
    ax_3d.set_box_aspect((
      max(x_limits[1] - x_limits[0], 1.0),
      max(y_limits[1] - y_limits[0], 1.0),
      max(z_limits[1] - z_limits[0], 1.0),
    ))
    ax_3d.grid(True, alpha=0.35)
    handles, legend_labels = ax_3d.get_legend_handles_labels()
    if handles:
      unique = dict(zip(legend_labels, handles, strict=True))
      ax_3d.legend(
        unique.values(), unique.keys(), loc="upper left", fontsize=8, frameon=False
      )

    passed = sum(result.passed for result in results)
    omitted = len(results) - len(finite_results)
    omitted_note = f"; {omitted} non-finite omitted" if omitted else ""
    fig.suptitle(
      f"Rizon4 OSC force-failure sweep — {passed}/{len(results)} trials passed"
      f"{omitted_note}",
      fontsize=15,
      fontweight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
  return path


def save_sweep_plot(
  path: Path,
  results: Sequence[TrialResult],
  params: Mapping[str, float],
) -> Path:
  """Render four publication-style 3D views of theory and measured points."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch

  path.parent.mkdir(parents=True, exist_ok=True)
  p = params
  theory_color = "#8ECAE6"
  release_color = "#E66100"
  interior_color = "#3F5F7F"
  failed_color = "#C43C39"

  finite_results = [
    result
    for result in results
    if np.all(np.isfinite((result.f_t_N, result.f_l_N, result.f_n_N)))
  ]
  releases = [result for result in finite_results if result.released]
  interior = [
    result
    for result in finite_results
    if result.sample_kind == "interior" and not result.released
  ]
  failures = [result for result in finite_results if not result.passed]

  release_loads = np.asarray(
    [[result.f_t_N, result.f_l_N, result.f_n_N] for result in releases],
    dtype=float,
  ).reshape(-1, 3)
  interior_loads = np.asarray(
    [[result.f_t_N, result.f_l_N, result.f_n_N] for result in interior],
    dtype=float,
  ).reshape(-1, 3)
  failure_loads = np.asarray(
    [[result.f_t_N, result.f_l_N, result.f_n_N] for result in failures],
    dtype=float,
  ).reshape(-1, 3)

  f_t = np.concatenate((
    np.linspace(-p["f_rev_max"], 0.0, 14, endpoint=False),
    np.linspace(0.0, p["f_t_max"], 72),
  ))
  cap_n = p["mu_adhesion"] * np.maximum(f_t, 0.0) + p["f_n0"]
  cap_l = p["mu_lateral"] * np.maximum(f_t, 0.0) + p["f_l0"]

  all_loads = np.concatenate((release_loads, interior_loads, failure_loads), axis=0)
  data_t = all_loads[:, 0] if all_loads.size else f_t
  data_l = all_loads[:, 1] if all_loads.size else np.concatenate((-cap_l, cap_l))
  data_n = all_loads[:, 2] if all_loads.size else cap_n
  x_span = p["f_t_max"] + p["f_rev_max"]
  y_bound = max(float(cap_l.max()), float(np.max(np.abs(data_l))))
  z_floor = min(
    -0.30 * float(cap_n.max()),
    float(data_n.min()) - 0.08 * max(float(np.ptp(data_n)), 1.0),
  )
  z_top = max(float(cap_n.max()), float(data_n.max()))
  x_limits = (
    min(-p["f_rev_max"], float(data_t.min())) - 0.035 * x_span,
    max(p["f_t_max"], float(data_t.max())) + 0.035 * x_span,
  )
  y_limits = (-1.06 * y_bound, 1.06 * y_bound)
  z_pad = 0.07 * max(z_top - z_floor, 1.0)
  z_limits = (z_floor - 0.02 * z_pad, z_top + z_pad)

  lateral_fraction = np.linspace(-1.0, 1.0, 25)
  vertical_fraction = np.linspace(0.0, 1.0, 21)
  top_t = np.broadcast_to(f_t[:, None], (f_t.size, lateral_fraction.size))
  top_l = cap_l[:, None] * lateral_fraction[None, :]
  top_n = np.broadcast_to(cap_n[:, None], top_t.shape)

  def draw_envelope(axis) -> None:
    surface_style = {
      "color": theory_color,
      "alpha": 0.27,
      "edgecolor": "none",
      "linewidth": 0.0,
      "shade": False,
      "antialiased": True,
    }
    axis.plot_surface(top_t, top_l, top_n, **surface_style)
    for sign in (-1.0, 1.0):
      wall_t = np.broadcast_to(
        f_t[:, None], (f_t.size, vertical_fraction.size)
      )
      wall_l = sign * np.broadcast_to(cap_l[:, None], wall_t.shape)
      wall_n = z_floor + (cap_n[:, None] - z_floor) * vertical_fraction[None, :]
      axis.plot_surface(wall_t, wall_l, wall_n, **surface_style)
    for index in (0, -1):
      end_t = np.full(
        (vertical_fraction.size, lateral_fraction.size), f_t[index]
      )
      end_l = cap_l[index] * np.broadcast_to(lateral_fraction, end_t.shape)
      end_n = z_floor + (cap_n[index] - z_floor) * np.broadcast_to(
        vertical_fraction[:, None], end_t.shape
      )
      axis.plot_surface(end_t, end_l, end_n, **surface_style)

  with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelcolor": "#202124",
    "xtick.color": "#30343A",
    "ytick.color": "#30343A",
  }):
    fig = plt.figure(
      figsize=(13.2, 10.4), constrained_layout=True, facecolor="white"
    )
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    for axis, azimuth in zip(axes, (30, 120, 210, 300), strict=True):
      draw_envelope(axis)
      if interior_loads.size:
        axis.scatter(
          interior_loads[:, 0],
          interior_loads[:, 1],
          interior_loads[:, 2],
          s=8,
          marker="o",
          color=interior_color,
          alpha=0.42,
          edgecolors="none",
          depthshade=False,
        )
      if release_loads.size:
        axis.scatter(
          release_loads[:, 0],
          release_loads[:, 1],
          release_loads[:, 2],
          s=9,
          marker="o",
          color=release_color,
          alpha=0.82,
          edgecolors="none",
          depthshade=False,
        )
      if failure_loads.size:
        axis.scatter(
          failure_loads[:, 0],
          failure_loads[:, 1],
          failure_loads[:, 2],
          s=18,
          marker="x",
          color=failed_color,
          alpha=0.9,
          depthshade=False,
        )

      axis.set_xlim(*x_limits)
      axis.set_ylim(*y_limits)
      axis.set_zlim(*z_limits)
      axis.set_xlabel(r"$f_t$ [N] (shear +)", labelpad=10, fontsize=13)
      axis.set_ylabel(r"$f_l$ [N] (lateral)", labelpad=10, fontsize=13)
      axis.set_zlabel(r"$f_n$ [N] (pull-off +)", labelpad=11, fontsize=13)
      axis.set_title(rf"azim {azimuth}$^\circ$", fontsize=12, pad=3)
      axis.view_init(elev=22, azim=azimuth)
      axis.set_box_aspect((1.18, 1.05, 0.88))
      axis.tick_params(labelsize=10, pad=1)
      axis.grid(True)
      for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.set_facecolor((0.96, 0.97, 0.98, 1.0))
        coordinate_axis.pane.set_edgecolor((0.72, 0.75, 0.78, 0.8))
        coordinate_axis._axinfo["grid"].update({
          "color": (0.70, 0.73, 0.76, 0.55),
          "linewidth": 0.55,
        })

    fig.suptitle(
      "Microspine 3D Limit Surface — Theoretical Envelope and Measured Force Samples",
      fontsize=17,
      fontweight="bold",
    )
    legend_handles = [
      Patch(facecolor=theory_color, edgecolor="none", alpha=0.35, label="theory envelope"),
      Line2D(
        [0], [0], marker="o", linestyle="none", markersize=7,
        markerfacecolor=release_color, markeredgecolor="none", label="measured release"
      ),
    ]
    if interior:
      legend_handles.append(Line2D(
        [0], [0], marker="o", linestyle="none", markersize=7,
        markerfacecolor=interior_color, markeredgecolor="none", alpha=0.55,
        label="interior held",
      ))
    if failures:
      legend_handles.append(Line2D(
        [0], [0], marker="x", linestyle="none", markersize=7,
        color=failed_color, label="failed / censored",
      ))
    fig.legend(
      handles=legend_handles,
      loc="upper right",
      bbox_to_anchor=(0.985, 0.985),
      fontsize=11,
      frameon=True,
      framealpha=0.92,
      edgecolor="#D5D9DD",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
  return path


def print_direction_plan(
  directions: Sequence[SweepDirection], params: Mapping[str, float], max_force: float
) -> None:
  print("#  name                         kind       target/direction (t,l,n)          expected   boundary")
  for index, direction in enumerate(directions, 1):
    force, reason = predict_release(direction.tln, params)
    prediction = (
      f"{force:6.1f}N {RELEASE_REASON_NAMES[reason]}"
      if math.isfinite(force)
      else f">{max_force:.0f} N (unbounded/censored)"
    )
    if math.isfinite(direction.target_force_N):
      t, l, n = (
        direction.target_force_N * np.asarray(direction.tln, dtype=float)
      )
      probe = f"({t:+6.1f},{l:+6.1f},{n:+6.1f})"
      expected = "release" if direction.expected_release else "hold"
    else:
      t, l, n = direction.tln
      probe = f"dir=({t:+.2f},{l:+.2f},{n:+.2f})"
      expected = "auto"
    print(
      f"{index:2d} {direction.name:<28} {direction.sample_kind:<10} "
      f"{probe:<29} {expected:<9} {prediction}"
    )


def print_result(result: TrialResult, total: int) -> None:
  marker = "PASS" if result.passed else "FAIL"
  prediction = (
    f"pred={result.predicted_force_N:.1f}N"
    if math.isfinite(result.predicted_force_N)
    else "pred=inf"
  )
  measured = (
    f"release={result.release_force_N:.1f}N/{result.release_reason_name}"
    if result.released
    else f"held={result.command_force_N:.1f}N"
  )
  print(
    f"[{result.trial_index + 1:02d}/{total:02d}] {marker} "
    f"{result.name:<23} {prediction:<13} {measured:<34} "
    f"margin={result.model_margin:+.3f} ({result.status})",
    flush=True,
  )


def run_sweep(args: argparse.Namespace) -> int:
  if args.plot_only is not None:
    source = Path(args.plot_only)
    results, params = load_saved_sweep(source)
    output = save_sweep_plot(source.with_suffix(".png"), results, params)
    releases = sum(result.released for result in results)
    interior = sum(result.sample_kind == "interior" for result in results)
    print(
      f"Replotted {releases} releases and {interior} interior points: {output}"
    )
    return 0

  if args.max_force <= 0.0 or args.soft_step <= 0.0 or args.ultimate_step <= 0.0:
    raise ValueError("force limits and steps must be positive")
  if args.repeats <= 0 or args.zero_ticks <= 0 or args.settle_ticks <= 0:
    raise ValueError("repeats and tick counts must be positive")
  if args.margin_tolerance < 0.0:
    raise ValueError("margin_tolerance must be non-negative")

  # Dry-run uses the deterministic midpoint values configured by PadTeleop.
  dry_params = {
    "f_t_max": 400.0,
    "f_rev_max": 20.0,
    "mu_adhesion": 0.4,
    "f_n0": 15.0,
    "mu_lateral": 0.6,
    "f_l0": 15.0,
  }
  if args.dry_run:
    directions = select_directions(args, dry_params)
    if not directions:
      raise ValueError("no sweep directions selected")
    print_direction_plan(directions, dry_params, args.max_force)
    return 0

  # Lazy import keeps --help/--dry-run usable without CUDA/mjlab startup.
  from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop

  pad = None
  results: list[TrialResult] = []
  traces: list[TraceSample] = []
  interrupted = False
  runtime_error: str | None = None
  plot_error: str | None = None
  params: Mapping[str, float] = dry_params
  total = 0
  try:
    pad = PadTeleop(device=args.device)
    params = dict(pad.limit_params)
    directions = select_directions(args, params)
    if not directions:
      raise ValueError("no sweep directions selected")
    total = len(directions) * args.repeats
    hold_ticks = max(
      args.settle_ticks,
      int(pad.affordance._ls_cfg.violation_consecutive_steps) + 1,
    )
    print("== Rizon4 headless limit-surface force sweep ==")
    print(
      f"device={args.device} trials={total} max_force={args.max_force:.1f}N "
      f"hold={hold_ticks} ticks margin_tol={args.margin_tolerance:.3f}"
    )
    print_direction_plan(directions, params, args.max_force)
    print()
    for repeat in range(args.repeats):
      for direction in directions:
        trial_index = len(results)
        result = run_trial(
          pad,
          direction,
          trial_index=trial_index,
          repeat=repeat,
          max_force=args.max_force,
          soft_step=args.soft_step,
          ultimate_step=args.ultimate_step,
          hold_ticks=hold_ticks,
          zero_ticks=args.zero_ticks,
          margin_tolerance=args.margin_tolerance,
          traces=traces,
        )
        results.append(result)
        print_result(result, total)
  except KeyboardInterrupt:
    interrupted = True
    print("\nInterrupted; saving partial results.", file=sys.stderr)
  except Exception as exc:
    runtime_error = f"{type(exc).__name__}: {exc}"
    print(f"\nSweep failed; saving partial results: {runtime_error}", file=sys.stderr)
  finally:
    if pad is not None:
      pad.env.close()

  metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "device": args.device,
    "argv": sys.argv,
    "mode": args.mode,
    "seed": args.seed,
    "surface_samples": args.surface_samples,
    "interior_samples": args.interior_samples,
    "normal_floor_N": args.normal_floor,
    "interior_fraction": [args.interior_min_fraction, args.interior_max_fraction],
    "interior_margin": args.interior_margin,
    "samples_per_plane": args.samples_per_plane,
    "repeats": args.repeats,
    "max_force_N": args.max_force,
    "soft_step_N": args.soft_step,
    "ultimate_step_N": args.ultimate_step,
    "settle_ticks": args.settle_ticks,
    "zero_ticks": args.zero_ticks,
    "margin_tolerance": args.margin_tolerance,
    "plot_enabled": not args.no_plot,
    "interrupted": interrupted,
    "runtime_error": runtime_error,
  }
  if results:
    output, summary_csv, releases_csv = save_outputs(
      Path(args.output), results, traces, params, metadata
    )
    print(f"\nSaved: {output}")
    print(f"       {summary_csv}")
    print(f"       {releases_csv}")
    if not args.no_plot:
      try:
        plot = save_sweep_plot(output.with_suffix(".png"), results, params)
        print(f"       {plot}")
      except Exception as exc:
        plot_error = f"{type(exc).__name__}: {exc}"
        print(
          f"Plot failed (sweep data are saved): {plot_error}",
          file=sys.stderr,
        )

  if interrupted:
    return 130
  if runtime_error is not None:
    return 2
  if not results:
    return 2

  passed = sum(result.passed for result in results)
  released = [result for result in results if result.released]
  print(f"\nOVERALL: {passed}/{len(results)} trials passed")
  if released:
    abs_margins = np.abs([
      result.model_margin for result in released if math.isfinite(result.model_margin)
    ])
    if abs_margins.size:
      print(
        f"release |margin|: median={np.median(abs_margins):.3f}, "
        f"p90={np.quantile(abs_margins, 0.9):.3f}"
      )
  if plot_error is not None:
    return 2
  return 0 if passed == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--plot-only",
    metavar="NPZ",
    help="Rebuild the sibling PNG from a saved NPZ/CSV without starting mjlab.",
  )
  parser.add_argument(
    "--output",
    default="results/rizon4_limitsurface_sweep.npz",
    help="NPZ output path; CSV and PNG outputs use the same stem.",
  )
  parser.add_argument(
    "--mode",
    choices=("random", "canonical", "planes"),
    default="random",
    help="Random probes are the default; canonical and planes retain deterministic diagnostics.",
  )
  parser.add_argument("--surface-samples", type=int, default=500)
  parser.add_argument("--interior-samples", type=int, default=250)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--normal-floor",
    type=float,
    default=-100.0,
    help="Lower +n sampling-window bound in N; it is not a physical failure face.",
  )
  parser.add_argument("--interior-min-fraction", type=float, default=0.20)
  parser.add_argument("--interior-max-fraction", type=float, default=0.75)
  parser.add_argument(
    "--interior-margin",
    type=float,
    default=0.15,
    help="Minimum normalized theoretical margin for random interior targets.",
  )
  parser.add_argument("--planes", nargs="+", choices=("tn", "tl"), default=("tn", "tl"))
  parser.add_argument("--samples-per-plane", type=int, default=12)
  parser.add_argument(
    "--cases",
    nargs="+",
    choices=tuple(direction.name for direction in canonical_directions(True)),
    help="Run selected named cases instead of --mode.",
  )
  parser.add_argument(
    "--include-compression",
    action="store_true",
    help="Add pure -n compression as an expected no-release control.",
  )
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--max-force", type=float, default=600.0)
  parser.add_argument("--soft-step", type=float, default=2.0)
  parser.add_argument("--ultimate-step", type=float, default=5.0)
  parser.add_argument("--settle-ticks", type=int, default=3)
  parser.add_argument("--zero-ticks", type=int, default=3)
  parser.add_argument("--margin-tolerance", type=float, default=0.20)
  parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Skip the default headless PNG summary plot.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print directions and theoretical failure forces without starting mjlab.",
  )
  return parser


def main() -> None:
  try:
    code = run_sweep(build_parser().parse_args())
  except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    code = 2
  raise SystemExit(code)


if __name__ == "__main__":
  main()
