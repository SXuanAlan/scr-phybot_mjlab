"""Offline, auditable first-slip estimates for the physical robot-arm CSVs.

These are signal-based estimates, not ground-truth microscopic failure causes.
The acquisition's terminal result and first-release field are comparison data,
never inputs to the motion/force event detector. No simulation table is changed.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

LOAD_STEPS = ("drag", "lateral", "normal")
NUMERIC_COLUMNS = (
    "t_wall_s", "tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
    "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz", "fx_tcp", "fy_tcp", "fz_tcp",
    "cmd_fx", "cmd_fy", "cmd_fz", "disp_mm",
)


@dataclass(frozen=True)
class DetectorConfig:
    dt_s: float = 0.01
    jump_mm: float = 0.15
    jump_window_s: float = 0.14
    force_drop_n: float = 0.75
    force_drop_fraction: float = 0.06
    quiet_speed_mm_s: float = 0.25
    burst_speed_mm_s: float = 1.0
    creep_mm_per_s: float = 0.20
    reengage_speed_mm_s: float = 0.40
    reengage_dwell_s: float = 0.35
    force_pre_window_s: tuple[float, float] = (-0.08, -0.02)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spine_components(tcp: np.ndarray) -> np.ndarray:
    """The acquisition targets imply S=(-Fx_tcp+Fy_tcp)/sqrt(2), L=(Fx+Fy)/sqrt(2)."""
    return np.column_stack(((tcp[:, 1] - tcp[:, 0]) / np.sqrt(2),
                            (tcp[:, 0] + tcp[:, 1]) / np.sqrt(2), tcp[:, 2]))


def prepare_trace(path: Path, config: DetectorConfig) -> dict:
    rows = csv_rows(path)
    if not rows:
        raise ValueError(f"Empty detail file: {path}")
    a = {k: np.array([float(row[k]) for row in rows]) for k in NUMERIC_COLUMNS}
    steps = np.array([row["step"] for row in rows])
    take = np.flatnonzero(np.isin(steps, LOAD_STEPS))
    info = {"raw_rows": len(rows), "detail_sha256": sha256(path), "file": path.name}
    if not len(take):
        return {**info, "t": np.array([])}
    # The logger writes the preceding phase's detach/timeout event on the FIRST
    # unload row. Keep that instant only; never inspect the subsequent unloading.
    boundary = int(take[-1]) + 1
    if boundary < len(rows) and steps[boundary] == "unload":
        take = np.append(take, boundary)
    raw_t = a["t_wall_s"][take]
    if not np.isfinite(raw_t).all() or np.any(np.diff(raw_t) <= 0):
        raise ValueError(f"Non-monotonic/non-finite per-trial clock: {path}")
    if np.max(np.diff(raw_t)) > 0.10:
        raise ValueError(f"Loading trace contains a >100 ms logging gap: {path}")
    t = np.arange(raw_t[0], raw_t[-1] + config.dt_s / 10, config.dt_s)
    near = np.clip(np.searchsorted(raw_t, t, side="right") - 1, 0, len(raw_t) - 1)
    step = steps[take][near].copy()
    step[step == "unload"] = steps[take[-2]]

    def interp(columns):
        result = np.column_stack([np.interp(t, raw_t, a[k][take]) for k in columns])
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite loading signal: {path}, {columns}")
        return result

    pos_world = interp(["tcp_x", "tcp_y", "tcp_z"]) * 1000
    q0 = np.array([a[k][take[0]] for k in ("cmd_qx", "cmd_qy", "cmd_qz", "cmd_qw")])
    pos = spine_components(Rotation.from_quat(q0).inv().apply(pos_world - pos_world[0]))
    pos = median_filter(pos, size=(3, 1))
    force = median_filter(spine_components(interp(["fx_tcp", "fy_tcp", "fz_tcp"])), size=(3, 1))
    command = spine_components(interp(["cmd_fx", "cmd_fy", "cmd_fz"]))
    # A 60 ms centered velocity suppresses position quantization; event times
    # remain estimates at about 0.1 s resolution, not exact 10 ms ground truth.
    velocity = np.linalg.norm((np.roll(pos, -3, axis=0) - np.roll(pos, 3, axis=0)) / .06, axis=1)
    velocity[:3], velocity[-3:] = velocity[3], velocity[-4]
    rot = Rotation.from_quat(interp(["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]))
    tilt = np.degrees((Rotation.from_quat(q0).inv() * rot).magnitude())
    return {**info, "t": t, "step": step, "pos": pos, "force": force, "command": command,
            "speed": velocity, "tilt_deg": tilt, "logged_disp": interp(["disp_mm"])[:, 0],
            "controller_t": raw_t[-1], "sample_dt_median_s": float(np.median(np.diff(raw_t))),
            "sample_dt_max_s": float(np.max(np.diff(raw_t)))}


def detect_events(trace: dict, config: DetectorConfig, scale: float = 1.0) -> list[dict]:
    """Find the earliest irreversible slip, then require re-engagement to count another.

    The two abrupt-event routes are force loss + motion, or a jump after a quiet
    hold. A third, explicitly less precise, route marks sustained normal-stage
    creep. Initial loading compliance with monotonic force is not a force loss.
    """
    t = trace["t"]
    if len(t) < 40:
        return []
    p, f, c, v = (trace[k] for k in ("pos", "force", "command", "speed"))
    n = len(t)
    movement = np.linalg.norm(np.roll(p, -12, axis=0) - np.roll(p, 2, axis=0), axis=1)
    candidate_ids = set(np.flatnonzero(movement >= config.jump_mm * scale))
    normal = np.flatnonzero(trace["step"] == "normal")
    creep_ids = set()
    if len(normal) > 150:
        # A gradual drift has no unique changepoint. Report an operational
        # threshold crossing and flag it for review instead of inventing one.
        baseline = float(np.median(v[normal[30:100]]))
        drift = np.linalg.norm(np.roll(p, -100, axis=0) - p, axis=1)
        for i in normal[100:-100]:
            if drift[i] < max(config.creep_mm_per_s * scale, 4 * baseline):
                continue
            window_speed = v[i:i + 100]
            median_speed = float(np.median(window_speed))
            # A future abrupt slip must not become an artificial "creep onset"
            # one second earlier. Require already-started, sustained smooth
            # motion, not just a net displacement inside a look-ahead window.
            sustained = np.mean(window_speed > max(.10 * scale, 2 * baseline)) >= .7
            underway = np.median(v[i - 15:i + 5]) > max(.10 * scale, 2 * baseline)
            smooth = np.max(window_speed) < max(1.0 * scale, 4 * median_speed)
            if sustained and underway and smooth and median_speed > max(.15 * scale, 3 * baseline):
                creep_ids.add(int(i))
    candidate_ids.update(creep_ids)
    residual = np.linalg.norm(f - c, axis=1)
    held = ((v < config.reengage_speed_mm_s)
            & (residual < np.maximum(1.5, 0.10 * np.linalg.norm(c, axis=1))))
    dwell = int(round(config.reengage_dwell_s / config.dt_s))
    runs = np.convolve(held.astype(int), np.ones(dwell, dtype=int), mode="valid")
    stable_starts = np.flatnonzero(runs == dwell)
    events = []
    accept_after = 30
    for i in sorted(candidate_ids):
        if i < accept_after or i < 30 or i >= n - 5:
            continue
        hi = min(i + 12, n - 1)
        before = slice(i - 30, i - 5)
        base_speed = float(np.median(v[before]))
        directions = np.sign(c[i])
        # Seating can transiently perturb the compressive normal force. Only
        # tangential drops establish loss during drag/lateral loading.
        directions[2] = float(trace["step"][i] == "normal" and c[i, 2] > .3)
        prev = np.max(f[i - 20:i] * directions, axis=0)
        future = np.min(f[i:hi + 1] * directions, axis=0)
        drops = prev - future
        strong_drop = bool(np.any(drops > np.maximum(
            config.force_drop_n * scale, config.force_drop_fraction * np.abs(c[i]))))
        steady_command = np.linalg.norm(c[i] - c[i - 25]) < .12
        quiet_jump = (base_speed < config.quiet_speed_mm_s * scale
                      and np.max(v[i:hi + 1]) > max(config.burst_speed_mm_s * scale, 8 * base_speed)
                      and (steady_command or trace["step"][i] == "normal"))
        abrupt = movement[i] >= config.jump_mm * scale and (strong_drop or quiet_jump)
        if abrupt:
            post = np.median(p[hi:min(n, hi + 15)], axis=0)
            pre = np.median(p[i - 8:i], axis=0)
            if np.linalg.norm(post - pre) < .12 * scale:
                continue
            # The look-ahead window brackets a jump. Refine toward its velocity
            # onset, rather than reporting a point 120 ms before the jump.
            fast = np.flatnonzero(v[i:hi + 1] > max(.7 * scale, 2 * base_speed))
            onset = i + int(fast[0]) if len(fast) else i
            evidence = "force_and_motion" if strong_drop else "abrupt_motion"
        elif i in creep_ids:
            onset = i
            evidence = "progressive_motion"
        else:
            continue
        if trace["step"][onset] == "drag" and np.linalg.norm(c[onset, :2]) < .8:
            continue
        subsequent = stable_starts[stable_starts > onset + 20]
        regain = int(subsequent[0]) if len(subsequent) else None
        stop = min(n - 1, onset + 30) if regain is None else regain
        dp = p[stop] - p[max(0, onset - 5)]
        fractions = np.abs(dp) / max(float(np.linalg.norm(dp)), 1e-9)
        direction = ("rake", "lateral", "normal")[int(np.argmax(fractions))]
        if np.max(fractions) < .8:
            direction = "mixed"
        pre_force = np.median(f[max(0, onset - 8):max(1, onset - 1)], axis=0)
        events.append({"index": int(onset), "time_s": float(t[onset]),
                       "elapsed_s": float(t[onset] - t[0]), "phase": str(trace["step"][onset]),
                       "evidence": evidence, "motion_direction": direction,
                       "force_drop_n": float(max(0., np.max(drops))),
                       "jump_mm": float(movement[i]), "baseline_speed_mm_s": base_speed,
                       "fx_pre_n": float(pre_force[0]), "fy_pre_n": float(pre_force[1]),
                       "fz_pre_n": float(pre_force[2]), "fx_cmd_n": float(c[onset, 0]),
                       "fy_cmd_n": float(c[onset, 1]), "fz_cmd_n": float(c[onset, 2]),
                       "tilt_deg": float(trace["tilt_deg"][onset]),
                       "reengaged": regain is not None,
                       "reengage_time_s": float(t[regain]) if regain is not None else None})
        if regain is None:
            break
        accept_after = max(onset + 35, regain + dwell)
    return events


def failure_category(phase: str, signed_s: float, fn: float) -> str:
    if phase == "drag":
        return "reverse_shear" if signed_s < 0 else "forward_shear"
    if phase == "lateral":
        return "lateral_loading"
    if phase == "normal":
        return "preload_loss" if fn <= 0 else "tensile_loading"
    return "unresolved"


def analyze_trial(row: dict, trace: dict, config: DetectorConfig) -> tuple[dict, list[dict]]:
    signed_s, lateral = int(row["side"]) * float(row["s_n"]), float(row["l_n"])
    result = {"trial": int(row["trial"]), "rep": int(row["rep"]), "side": int(row["side"]),
              "s_target_n": signed_s, "l_target_n": lateral, "preload_n": float(row["fn_n"]),
              "x_mm": float(row["x_mm"]), "y_mm": float(row["y_mm"]), "origin": row["origin"],
              "detail_csv": row["detail_csv"], "controller_result": row["result"],
              "controller_fz_cmd_n": float(row["fz_cmd_n"]),
              "controller_first_release_level_n": float(row["fz_first_release_n"]),
              "controller_last_held_level_n": float(row["fz_last_held_n"]),
              "raw_rows": trace["raw_rows"], "detail_sha256": trace["detail_sha256"]}
    events = detect_events(trace, config)
    if row["result"] == "no_contact":
        if len(trace["t"]):
            raise ValueError(f"no_contact trial unexpectedly has load steps: T{row['trial']}")
        return {**result, "category": "no_contact", "first_phase": "none", "evidence": "no_contact",
                "review_required": True, "normal_fit_eligible": False, "event_count": 0,
                "reengaged": False, "sensitivity_phase_stable": False}, []
    if not events:
        # Keep an unresolved trial visible. A controller label is NOT a verified
        # first-slip point, even when it supplies an upper time bound.
        return {**result, "category": "unresolved", "first_phase": "unresolved", "evidence": "unresolved",
                "review_required": True, "normal_fit_eligible": False, "event_count": 0,
                "reengaged": False, "sensitivity_phase_stable": False}, []
    first = events[0]
    alternates = [detect_events(trace, config, scale=s) for s in (.75, 1.25)]
    firsts = [v[0] if v else None for v in alternates]
    phase_stable = all(v is not None and v["phase"] == first["phase"] for v in firsts)
    fz_values = [first["fz_cmd_n"]] + [v["fz_cmd_n"] for v in firsts if v is not None]
    time_values = [first["time_s"]] + [v["time_s"] for v in firsts if v is not None]
    fz_span = max(fz_values) - min(fz_values)
    time_span = max(time_values) - min(time_values)
    # Normal capacity is conditional on reaching the target without earlier
    # failure. Sensitive/gradual onsets remain in the trial table, not the fit.
    review = (not phase_stable or fz_span > .75 or first["evidence"] == "progressive_motion"
              or time_span > 1.0)
    normal_fit = first["phase"] == "normal" and not review
    category = failure_category(first["phase"], signed_s, first["fz_cmd_n"])
    result.update({"category": category, "first_phase": first["phase"],
                   "evidence": first["evidence"], "motion_direction": first["motion_direction"],
                   "first_time_s": first["time_s"], "first_elapsed_s": first["elapsed_s"],
                   "controller_time_s": trace["controller_t"],
                   "time_advance_s": trace["controller_t"] - first["time_s"],
                   "first_fx_pre_n": first["fx_pre_n"], "first_fy_pre_n": first["fy_pre_n"],
                   "first_fz_pre_n": first["fz_pre_n"], "first_fx_cmd_n": first["fx_cmd_n"],
                   "first_fy_cmd_n": first["fy_cmd_n"], "first_fz_cmd_n": first["fz_cmd_n"],
                   "controller_minus_first_fz_n": float(row["fz_cmd_n"]) - first["fz_cmd_n"],
                   "first_force_drop_n": first["force_drop_n"], "first_jump_mm": first["jump_mm"],
                   "first_tilt_deg": first["tilt_deg"], "event_count": len(events),
                   "reengaged": first["reengaged"], "review_required": review,
                   "normal_fit_eligible": normal_fit, "sensitivity_phase_stable": phase_stable,
                   "sensitivity_fz_min_n": min(fz_values), "sensitivity_fz_max_n": max(fz_values),
                   "sensitivity_time_span_s": time_span,
                   "sample_dt_median_s": trace["sample_dt_median_s"],
                   "sample_dt_max_s": trace["sample_dt_max_s"]})
    for scale, alt in zip(("075", "125"), firsts):
        result[f"scale_{scale}_phase"] = alt["phase"] if alt is not None else "unresolved"
        result[f"scale_{scale}_fz_cmd_n"] = alt["fz_cmd_n"] if alt is not None else float("nan")
    for event in events:
        event.update(trial=int(row["trial"]), rep=int(row["rep"]), origin=row["origin"])
    return result, events


def cell_statistics(results: list[dict]) -> list[dict]:
    cells = []
    keys = sorted({(r["s_target_n"], r["l_target_n"]) for r in results})
    for s, l in keys:
        rows = [r for r in results if (r["s_target_n"], r["l_target_n"]) == (s, l)]
        normal = [r for r in rows if r["normal_fit_eligible"]]
        values = np.array([r["first_fz_pre_n"] for r in normal])
        cells.append({"s_n": s, "l_n": l, "n_total": len(rows), "n_normal_used": len(normal),
                      "n_early": sum(r["first_phase"] in ("drag", "lateral") for r in rows),
                      "n_review": sum(r["review_required"] for r in rows),
                      "fz_mean_n": float(np.mean(values)) if len(values) else float("nan"),
                      "fz_sd_n": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                      "fz_min_n": float(np.min(values)) if len(values) else float("nan"),
                      "fz_max_n": float(np.max(values)) if len(values) else float("nan"),
                      "used_trial_ids": " ".join(str(r["trial"]) for r in normal)})
    return cells


def polynomial_features(s, l, degree: int) -> np.ndarray:
    """Even in lateral load; the untested negative half is a stated assumption."""
    s, l = np.broadcast_arrays(np.asarray(s) / 30, np.asarray(l) / 30)
    powers = [(i, j) for total in range(degree + 1) for j in range(0, total + 1, 2)
              for i in [total - j]]
    return np.stack([s ** i * l ** j for i, j in powers], axis=-1)


def fit_surface(cells: list[dict], results: list[dict]) -> dict:
    valid = [cell for cell in cells if cell["n_normal_used"]]
    if len(valid) < 8:
        raise ValueError("Too few supported cells for a descriptive normal-failure surface")
    points = np.array([[v["s_n"], v["l_n"]] for v in valid])
    z = np.array([v["fz_mean_n"] for v in valid])
    # Each cell is one independent target, including S=0 where both side labels
    # repeat the same target. Equal-cell fitting avoids double-weighting S=0.
    fits = []
    heldout_predictions = {}
    for degree in (2, 3, 4):
        a = polynomial_features(points[:, 0], points[:, 1], degree)
        if len(points) < 2 * a.shape[1] or np.linalg.matrix_rank(a) < a.shape[1]:
            continue
        pred = []
        for i in range(len(z)):
            keep = np.arange(len(z)) != i
            coeff = np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
            pred.append(float(a[i] @ coeff))
        fits.append({"degree": degree, "loco_rmse_n": float(np.sqrt(np.mean((np.array(pred) - z) ** 2)))})
        heldout_predictions[degree] = np.asarray(pred)
    best = min(fits, key=lambda x: x["loco_rmse_n"])
    # Prefer the simpler polynomial if its held-out error is within 5%.
    degree = min(v["degree"] for v in fits if v["loco_rmse_n"] <= 1.05 * best["loco_rmse_n"])
    a = polynomial_features(points[:, 0], points[:, 1], degree)
    coeff = np.linalg.lstsq(a, z, rcond=None)[0]
    trial_values = [r for r in results if r["normal_fit_eligible"]]
    trial_a = polynomial_features([r["s_target_n"] for r in trial_values],
                                  [r["l_target_n"] for r in trial_values], degree)
    residuals = np.array([r["first_fz_pre_n"] for r in trial_values]) - trial_a @ coeff
    constant_predictions = (z.sum() - z) / (len(z) - 1)
    total_variation = np.sum((z - z.mean()) ** 2)
    training_r2 = float(1 - np.sum((a @ coeff - z) ** 2) / total_variation) if total_variation > 0 else None
    # Mask against measured positive-L support; the prediction routine uses |L|
    # for the explicitly assumed reflection, never as additional observations.
    hull = Delaunay(points)
    return {"degree": degree, "coefficients": coeff, "hull": hull, "points": points,
            "cv_scores": fits, "cell_rmse_n": float(np.sqrt(np.mean((a @ coeff - z) ** 2))),
            "residual_sd_n": float(np.std(residuals, ddof=1)),
            "n_cells": len(points), "n_trials": len(trial_values), "config": asdict(DetectorConfig()),
            "observed_cell_means_n": z, "loco_predictions_n": heldout_predictions[degree],
            "loco_selected_rmse_n": next(v["loco_rmse_n"] for v in fits if v["degree"] == degree),
            "loco_constant_rmse_n": float(np.sqrt(np.mean((constant_predictions - z) ** 2))),
            "training_r2": training_r2, "n_singleton_cells": sum(v["n_normal_used"] == 1 for v in valid),
            "interpretation": "Exploratory conditional mean regression; no validated physical failure envelope"}


def predict_surface(fit: dict, s, l, mask: bool = True):
    s, l = np.broadcast_arrays(s, l)
    values = polynomial_features(s, l, fit["degree"]) @ fit["coefficients"]
    if mask:
        points = np.column_stack((s.ravel(), np.abs(l).ravel()))
        supported = (fit["hull"].find_simplex(points, tol=1e-9) >= 0).reshape(s.shape)
        values = np.where(supported, values, np.nan)
    return values
