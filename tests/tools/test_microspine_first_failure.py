"""Scientific regressions: first slip, re-catch, seating, and censoring semantics."""

import csv

import numpy as np

from scr_phybot_mjab.tools.microspine.first_failure import (
    NUMERIC_COLUMNS,
    DetectorConfig,
    cell_statistics,
    detect_events,
    fit_surface,
    predict_surface,
    prepare_trace,
    spine_components,
)


def synthetic_trace(slips=(), duration=15.):
    t = np.arange(0, duration, .01)
    command = np.column_stack((10 * np.clip(t / 2, 0, 1), np.zeros(len(t)), np.full(len(t), -3.)))
    normal = t >= 5
    command[normal, 2] = -2.5 + np.floor((t[normal] - 5) / 2)
    force = command.copy()
    pos = np.zeros_like(force)
    pos[:, 0] = .03 * command[:, 0]  # ordinary elastic loading
    for slip in slips:
        pos[:, 0] += .8 * np.clip((t - slip) / .12, 0, 1)
        force[:, 0] -= 4 * np.maximum(0, 1 - np.abs(t - (slip + .12)) / .12)
    v = np.linalg.norm((np.roll(pos, -3, axis=0) - np.roll(pos, 3, axis=0)) / .06, axis=1)
    v[:3], v[-3:] = v[3], v[-4]
    return {"t": t, "force": force, "command": command, "pos": pos, "speed": v,
            "step": np.where(normal, "normal", "drag"), "tilt_deg": np.zeros(len(t))}


def test_earliest_slip_precedes_larger_later_load_and_recatches():
    events = detect_events(synthetic_trace(slips=(8, 12)), DetectorConfig())
    assert len(events) == 2
    assert abs(events[0]["time_s"] - 8) < .15
    assert abs(events[1]["time_s"] - 12) < .15
    assert events[0]["reengaged"]
    assert events[0]["fz_cmd_n"] < events[1]["fz_cmd_n"]
    assert events[0]["fx_pre_n"] > 9.5  # take load before its collapse


def test_initial_elastic_seating_and_command_steps_are_not_failure():
    assert detect_events(synthetic_trace(), DetectorConfig()) == []


def test_slip_during_shear_ramp_counts_before_normal_loading():
    events = detect_events(synthetic_trace(slips=(1.3, 8)), DetectorConfig())
    assert events[0]["phase"] == "drag"
    assert events[0]["time_s"] < 1.5
    assert events[0]["fx_cmd_n"] < 10  # target was never yet reached


def test_continuous_sliding_is_not_counted_as_repeated_release():
    trace = synthetic_trace(slips=(8,))
    i = trace["t"] >= 8
    trace["pos"][i, 0] += 2 * (trace["t"][i] - 8)
    trace["speed"][i] = 2
    events = detect_events(trace, DetectorConfig())
    assert len(events) == 1
    assert not events[0]["reengaged"]


def test_pose_reader_ignores_unloading_and_logged_displacement_resets(tmp_path):
    path = tmp_path / "trial.csv"
    rows = []
    for i in range(250):
        row = dict.fromkeys(NUMERIC_COLUMNS, 0.)
        row.update(t_wall_s=i * .01, cmd_qw=1., tcp_qw=1., step="drag" if i < 100 else "lateral")
        row.update(fz_tcp=-3., cmd_fz=-3., fx_tcp=-7.071068, fy_tcp=7.071068,
                   cmd_fx=-7.071068, cmd_fy=7.071068, disp_mm=2. if i < 100 else 0.)
        if i >= 180:
            row["step"] = "unload"
        if i > 180:
            row["tcp_x"] = (i - 180) * .001  # large, commanded retreat
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*NUMERIC_COLUMNS, "step"])
        writer.writeheader(); writer.writerows(rows)
    trace = prepare_trace(path, DetectorConfig())
    assert trace["controller_t"] == 1.8
    assert np.max(np.abs(trace["pos"])) == 0
    assert detect_events(trace, DetectorConfig()) == []


def test_tcp_rotation_recovers_bench_target_axes():
    tcp = np.array([[-10.606602, 10.606602, -3], [17.677670, 10.606602, -2]])
    np.testing.assert_allclose(spine_components(tcp), [[15, 0, -3], [-5, 20, -2]], atol=1e-5)


def test_early_failures_and_singletons_do_not_invent_normal_samples():
    rows = [dict(s_target_n=0., l_target_n=0., normal_fit_eligible=True, first_fz_pre_n=1.,
                 first_phase="normal", review_required=False, trial=1),
            dict(s_target_n=0., l_target_n=0., normal_fit_eligible=False, first_fz_pre_n=-3.,
                 first_phase="lateral", review_required=False, trial=2)]
    cell = cell_statistics(rows)[0]
    assert cell["n_total"] == 2 and cell["n_normal_used"] == 1 and cell["n_early"] == 1
    assert cell["fz_mean_n"] == 1
    assert np.isnan(cell["fz_sd_n"])


def test_surface_mirrors_explicitly_and_never_extrapolates():
    rows = []
    for s in (0., 5., 10., 15.):
        for l in (0., 5., 10., 15.):
            rows.append(dict(s_target_n=s, l_target_n=l, normal_fit_eligible=True,
                             first_fz_pre_n=.1 * s - .005 * l ** 2, first_phase="normal",
                             review_required=False, trial=len(rows) + 1))
    fit = fit_surface(cell_statistics(rows), rows)
    assert np.isnan(predict_surface(fit, -5., 0.))
    np.testing.assert_allclose(predict_surface(fit, 10., 5.), predict_surface(fit, 10., -5.))
