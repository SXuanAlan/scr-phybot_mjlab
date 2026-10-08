"""Offline fitting of protocol-conditioned empirical unit distributions."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from scr_phybot_mjab.microspine.empirical_limit_surface import DEFAULT_TABLE, EmpiricalLimitSurface
from scr_phybot_mjab.tools.paths import DEFAULT_UNIT_TRIALS, EMPIRICAL_UNIT_DIR

EARLY = {"detached_drag", "detached_lateral"}


def read_trials(path):
  with Path(path).open() as stream:
    rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
  ids = [int(r["trial"]) for r in rows]
  if len(set(ids)) != len(ids):
    raise ValueError("Duplicate trial IDs; cannot treat duplicates as independent")
  for row in rows:
    if row["result"] not in EARLY | {"detached_normal", "held"}:
      raise ValueError(f"Unclassified result: {row['result']}")
    if float(row["fn_n"]) != 10:
      raise ValueError("Separate preload protocols before fitting")
  return rows


def conditional_quantiles(rows, u, weights=None):
  """Kaplan-Meier inverse CDF for trials reaching normal loading.

  Remaining survival mass beyond the last observation uses the largest
  observed load as a conservative lower bound, with a separate censor flag.
  Early loading failures are NOT fabricated normal-failure observations.
  """
  if weights is None:
    weights = np.ones(len(rows))
  observations = [
    (
      float(r["fz_cmd_n"] if r["result"] == "detached_normal" else r["fz_last_held_n"]),
      r["result"] == "detached_normal",
      w,
    )
    for r, w in zip(rows, weights)
    if r["result"] not in EARLY and w > 1e-10
  ]
  if not observations:
    return None
  values = sorted(set(x for x, _, _ in observations))
  risk, survival = sum(w for _, _, w in observations), 1.0
  event_values, cdf = [], []
  for value in values:
    events = sum(w for x, failed, w in observations if x == value and failed)
    censored = sum(w for x, failed, w in observations if x == value and not failed)
    if events:
      survival *= max(0, 1 - events / max(risk, events))
      event_values.append(value)
      cdf.append(1 - survival)
    risk -= events + censored
  q = np.full(len(u), values[-1], dtype=float)
  tail = np.ones(len(u), dtype=bool)
  if cdf:
    idx = np.searchsorted(cdf, u, side="left")
    known = idx < len(cdf)
    q[known] = np.asarray(event_values)[idx[known]]
    tail[known] = False
  return q, tail


def kernel_weights(rows, s, l, bandwidth):
  xy = np.array([(float(r["side"]) * float(r["s_n"]), float(r["l_n"])) for r in rows])
  distance = ((xy - (s, l)) / np.asarray(bandwidth)) ** 2
  return np.exp(-0.5 * distance.sum(axis=1))


def choose_bandwidth(rows):
  """Select conditional-distribution smoothing using calibration rows ONLY.

  Leave an entire placement/trial out; minimize integrated quantile loss.
  Conditional normal data determine bandwidth; early outcomes remain a
  separately weighted protocol-success probability, not synthetic thresholds.
  """
  u = (np.arange(63) + 0.5) / 63
  scores = []
  for hs in (2.5, 5.0, 7.5, 10.0):
    for hl in (2.5, 5.0, 7.5, 10.0):
      losses = []
      for i, row in enumerate(rows):
        if row["result"] != "detached_normal":
          continue
        weights = kernel_weights(rows, float(row["side"]) * float(row["s_n"]), float(row["l_n"]), (hs, hl))
        # Exclude coincident recorded placements too, rather than leaking
        # duplicate placement measurements into the bandwidth selection.
        for j, other in enumerate(rows):
          if other["x_mm"] == row["x_mm"] and other["y_mm"] == row["y_mm"]:
            weights[j] = 0
        q, _ = conditional_quantiles(rows, u, weights)
        error = float(row["fz_cmd_n"]) - q
        losses.append(float(2 * np.mean(np.maximum(u * error, (u - 1) * error))))
      scores.append(
        dict(shear_bandwidth_n=hs, lateral_bandwidth_n=hl, loo_integrated_quantile_loss_n=float(np.mean(losses)))
      )
  best = min(scores, key=lambda r: r["loo_integrated_quantile_loss_n"])
  return (best["shear_bandwidth_n"], best["lateral_bandwidth_n"]), scores


def build_table(rows, output, source_sha, bins=1024, bandwidth=None):
  cells = defaultdict(list)
  for row in rows:
    cells[float(row["side"]) * float(row["s_n"]), float(row["l_n"])].append(row)
  s_axis, l_axis = np.arange(-15.0, 31.0, 5.0), np.arange(0.0, 31.0, 5.0)
  u = (np.arange(bins) + 0.5) / bins
  points, qs, tails, early, audit = [], [], [], [], []
  for (s, l), group in sorted(cells.items()):
    result = conditional_quantiles(group, u)
    count = Counter(r["result"] for r in group)
    audit.append(
      dict(
        shear_n=s,
        lateral_n=l,
        count=dict(count),
        trial_ids=[int(r["trial"]) for r in group],
        terminal_command_n=[float(r["fz_cmd_n"]) for r in group if r["result"] == "detached_normal"],
        first_release_command_n=[
          float(r["fz_first_release_n"])
          for r in group
          if r["fz_first_release_n"] and np.isfinite(float(r["fz_first_release_n"]))
        ],
        censored_lower_bounds_n=[float(r["fz_last_held_n"]) for r in group if r["result"] == "held"],
      )
    )
    if result is None:
      continue
    points.append((s, l))
    qs.append(result[0])
    tails.append(result[1])
    early.append(sum(count[k] for k in EARLY) / len(group))
  grid = np.stack(np.meshgrid(s_axis, l_axis, indexing="ij"), axis=-1)
  q = LinearNDInterpolator(points, np.asarray(qs))(grid)
  e = LinearNDInterpolator(points, np.asarray(early))(grid)
  c = LinearNDInterpolator(points, np.asarray(tails, dtype=float))(grid)
  supported = np.isfinite(q).all(axis=-1)
  if bandwidth is not None:
    # Pool weighted *observations*, never average failure loads. This retains
    # observed scatter at sparse/singleton cells and yields a full inverse CDF.
    for i, s in enumerate(s_axis):
      for j, l in enumerate(l_axis):
        weights = kernel_weights(rows, s, l, bandwidth)
        q[i, j], c[i, j] = conditional_quantiles(rows, u, weights)
        e[i, j] = sum(w for row, w in zip(rows, weights) if row["result"] in EARLY) / weights.sum()
  else:
    # A cell with only early failures has an observed loading-success rate
    # zero, even though its conditional normal distribution is unidentified.
    for i, s in enumerate(s_axis):
      for j, l in enumerate(l_axis):
        group = cells.get((s, l), [])
        if group and all(r["result"] in EARLY for r in group):
          q[i, j] = -10
          c[i, j] = 0
          e[i, j] = 1
          supported[i, j] = True
  observed = np.array([tuple(p) in cells for p in grid.reshape(-1, 2)]).reshape(grid.shape[:2])
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    schema_version=np.int32(2),
    source_sha256=np.array(source_sha),
    shear_n=s_axis,
    lateral_n=l_axis,
    protocol_preload_n=np.float32(10),
    quantile_u=u,
    normal_quantile_n=np.nan_to_num(q, nan=-10).astype(np.float32),
    early_probability=np.nan_to_num(e, nan=1).astype(np.float32),
    censored_tail=np.nan_to_num(c, nan=0) > 1e-7,
    supported=supported,
    observed_cell=observed,
    trial_ids=np.array([int(r["trial"]) for r in rows], dtype=np.int32),
  )
  return audit


def holdout_report(rows, table):
  """Independent placements, same panel/protocol; retrospective spatial split."""
  metrics = []
  u = (np.arange(table.nq) + 0.5) / table.nq
  for row in rows:
    s, l = float(row["side"]) * float(row["s_n"]), float(row["l_n"])
    q, p, valid, tail = table.query_numpy(s, l, u)
    if not valid.all():
      metrics.append(dict(trial=int(row["trial"]), supported=False))
      continue
    early = row["result"] in EARLY
    item = dict(
      trial=int(row["trial"]),
      supported=True,
      result=row["result"],
      early_brier=float((p[0] - early) ** 2),
      early_probability=float(p[0]),
    )
    if row["result"] == "detached_normal":
      y = float(row["fz_cmd_n"])
      # Whole trials are held out. Capped tails mean these errors describe
      # the conservative operational table, not uncensored tail accuracy.
      lo, med, hi = np.quantile(q, [0.1, 0.5, 0.9])
      item.update(
        observed_n=y,
        predicted_median_n=float(med),
        absolute_error_n=float(abs(med - y)),
        in_10_90_band=bool(lo <= y <= hi),
        median_overpredicts=bool(med > y),
        prediction_has_capped_tail=bool(tail.any()),
      )
    metrics.append(item)
  normal = [r for r in metrics if "absolute_error_n" in r]
  valid = [r for r in metrics if r["supported"]]
  return dict(
    trials=metrics,
    total=len(rows),
    supported=len(valid),
    terminal_count=len(normal),
    terminal_mae_n=float(np.mean([r["absolute_error_n"] for r in normal])),
    terminal_10_90_coverage=float(np.mean([r["in_10_90_band"] for r in normal])),
    terminal_median_overprediction_fraction=float(np.mean([r["median_overpredicts"] for r in normal])),
    early_brier=float(np.mean([r["early_brier"] for r in valid])),
  )


def build_empirical_model(data_path=DEFAULT_UNIT_TRIALS, output=DEFAULT_TABLE, analysis_dir=EMPIRICAL_UNIT_DIR):
  """Write one runtime table; keep calibration, baseline and reports in analysis_dir."""
  data_path, output, analysis_dir = Path(data_path), Path(output), Path(analysis_dir)
  analysis_dir.mkdir(parents=True, exist_ok=True)
  rows = read_trials(data_path)
  sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
  # Fixed geometric split before computing prediction scores; all repetitions
  # at a held-out placement stay together. Not a preregistered future test.
  calibration = [r for r in rows if float(r["x_mm"]) < 540.0]
  validation = [r for r in rows if float(r["x_mm"]) >= 540.0]
  cal_path = analysis_dir / (output.stem + "_calibration.npz")
  build_table(calibration, cal_path, sha)
  raw_validation_report = holdout_report(validation, EmpiricalLimitSurface(cal_path))
  raw_path = analysis_dir / (output.stem + "_raw_cells.npz")
  build_table(rows, raw_path, sha)
  bandwidth, bandwidth_scores = choose_bandwidth(calibration)
  build_table(calibration, cal_path, sha, bandwidth=bandwidth)
  validation_report = holdout_report(validation, EmpiricalLimitSurface(cal_path))
  # Compare prediction quality on the SAME terminal placements; each table's
  # support mask and full denominators remain available in its own report.
  raw_by_trial = {r["trial"]: r for r in raw_validation_report["trials"] if "observed_n" in r}
  paired = [r for r in validation_report["trials"] if "observed_n" in r and r["trial"] in raw_by_trial]
  paired_raw = [raw_by_trial[r["trial"]] for r in paired]
  paired_comparison = dict(
    terminal_count=len(paired),
    trial_ids=[r["trial"] for r in paired],
    raw_mae_n=float(np.mean([r["absolute_error_n"] for r in paired_raw])),
    kernel_mae_n=float(np.mean([r["absolute_error_n"] for r in paired])),
    raw_10_90_coverage=float(np.mean([r["in_10_90_band"] for r in paired_raw])),
    kernel_10_90_coverage=float(np.mean([r["in_10_90_band"] for r in paired])),
  )
  cells = build_table(rows, output, sha, bandwidth=bandwidth)
  listed = {r["detail_csv"] for r in rows}
  traces = {p.name for p in data_path.parent.glob("*_T*.csv")}
  report = dict(
    schema_version=1,
    source=str(data_path),
    source_sha256=sha,
    result_counts=dict(Counter(r["result"] for r in rows)),
    cells=cells,
    unlisted_detail_files=sorted(traces - listed),
    missing_detail_files=sorted(listed - traces),
    protocol=dict(
      preload_per_unit_n=10,
      normal_step_n=0.5,
      dwell_per_step_s=3,
      failure_value="fz_cmd_n, including actual ramp command at the event",
      partial_release="fz_first_release_n retained separately; not terminal failure",
    ),
    assumptions=[
      "same substrate and 10 N unit seating protocol",
      "Gaussian kernel pooling of observed outcomes; bandwidth selected by calibration-only leave-placement-out "
      "quantile loss",
      "negative lateral reflected from positive lateral; unmeasured symmetry",
      "same quantile rank across directions; unmeasured directional correlation",
      "early loading failure frequency indexed by intended S,L protocol; not a normal threshold",
      "censored tail capped conservatively at last observation, with explicit flag",
      "bilinear conditional-quantile interpolation; missing cells filled inside data hull",
      "independent unit draws; no measured between-unit or between-pad correlation",
      "equal-stiffness allocation at C; first unit violation releases whole pad",
      "no shear/lateral extrapolation beyond measured S[-15,30], abs(L)[0,30]",
      "10 N seating is a minimum engagement preload, not a compression limit; normal failure is an upper pull-off "
      "bound",
      "additional compression does not increase fitted pull-off capacity; structural compression failure is not "
      "identified",
      "no dwell-time extrapolation, fatigue identification or full-pad physical validation",
    ],
    validation=dict(
      split="retrospective x_mm < 540 calibration, x_mm >= 540 validation; same panel",
      calibration_trial_ids=[int(r["trial"]) for r in calibration],
      validation_trial_ids=[int(r["trial"]) for r in validation],
      **validation_report,
    ),
    raw_cell_baseline_validation=raw_validation_report,
    paired_terminal_validation=paired_comparison,
    estimator=dict(
      kind="kernel-weighted conditional Kaplan-Meier inverse CDF",
      bandwidth_n=bandwidth,
      calibration_only_bandwidth_scores=bandwidth_scores,
      model_development_note="Sparse-cell baseline coverage motivated pooling; spatial split is retrospective "
      "development evidence, not a preregistered confirmation test",
    ),
    runtime_table="refit on all listed trials AFTER split-only validation; do not validate this on its training rows",
  )
  (analysis_dir / (output.stem + ".json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(
    json.dumps(
      dict(
        table=str(output),
        result_counts=report["result_counts"],
        calibration_n=len(calibration),
        validation_n=len(validation),
        heldout_mae_n=validation_report["terminal_mae_n"],
        heldout_10_90_coverage=validation_report["terminal_10_90_coverage"],
      ),
      indent=2,
    )
  )

  return report
