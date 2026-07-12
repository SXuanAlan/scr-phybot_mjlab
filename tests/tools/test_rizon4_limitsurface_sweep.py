import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "sweep_rizon4_limitsurface.py"
SPEC = spec_from_file_location("rizon4_limitsurface_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
canonical_directions = MODULE.canonical_directions
plane_directions = MODULE.plane_directions
predict_release = MODULE.predict_release
random_directions = MODULE.random_directions
save_sweep_plot = MODULE.save_sweep_plot


P = {
    "f_t_max": 400.0,
    "f_rev_max": 20.0,
    "mu_adhesion": 0.4,
    "f_n0": 15.0,
    "mu_lateral": 0.6,
    "f_l0": 15.0,
}


def _trial_result(**overrides):
    values = {
        "trial_index": 0,
        "repeat": 0,
        "name": "pull-off",
        "plane": "tn",
        "angle_deg": 90.0,
        "direction_t": 0.0,
        "direction_l": 0.0,
        "direction_n": 1.0,
        "predicted_force_N": 15.0,
        "predicted_reason": 2,
        "predicted_reason_name": "tension-cone",
        "expected_release": True,
        "sample_kind": "canonical",
        "target_force_N": math.nan,
        "target_t": math.nan,
        "target_l": math.nan,
        "target_n": math.nan,
        "target_margin": math.nan,
        "step_N": 2.0,
        "hold_ticks": 3,
        "max_force_N": 30.0,
        "baseline_force_N": 0.0,
        "baseline_t": 0.0,
        "baseline_l": 0.0,
        "baseline_n": 0.0,
        "engaged": True,
        "released": True,
        "unexpected_disengage": False,
        "release_reason": 2,
        "release_reason_name": "tension-cone",
        "reason_matches_prediction": True,
        "command_force_N": 16.0,
        "command_t": 0.0,
        "command_l": 0.0,
        "command_n": 16.0,
        "measured_force_N": 16.0,
        "release_force_N": 16.0,
        "f_t_N": 0.0,
        "f_n_N": 16.0,
        "f_l_N": 0.0,
        "reported_margin": -1.0 / 15.0,
        "model_margin": -1.0 / 15.0,
        "steps": 24,
        "duration_s": 0.5,
        "status": "released",
        "passed": True,
    }
    values.update(overrides)
    return MODULE.TrialResult(**values)


class _ReasonValue:
    def __init__(self):
        self.value = 0

    def __getitem__(self, _index):
        return self

    def item(self):
        return self.value


class _FakePad:
    def __init__(self, release_on_violation):
        self.limit_params = P
        self.affordance = SimpleNamespace(release_reason=_ReasonValue())
        self.release_on_violation = release_on_violation
        self.commands = []

    def home(self):
        self.affordance.release_reason.value = 0

    def press_engage(self):
        pass

    def apply(self, command):
        load = np.asarray(command, dtype=float)
        self.commands.append(load.copy())
        margin = MODULE._model_margin(load, P)
        released = self.release_on_violation and margin < -1.0e-9
        if released:
            _, reason = predict_release(load / np.linalg.norm(load), P)
            self.affordance.release_reason.value = reason
        else:
            self.affordance.release_reason.value = 0
        return SimpleNamespace(
            load_tln=load,
            margin=margin,
            engaged=not released,
            released=released,
        )


def test_canonical_predictions_cover_all_release_faces():
    directions = {direction.name: direction for direction in canonical_directions()}
    expected = {
        "pull-off": (15.0, 2),
        "lateral-pos": (15.0, 3),
        "lateral-neg": (15.0, 3),
        "reverse": (20.0, 1),
        "ultimate": (400.0, 4),
    }
    for name, (force, reason) in expected.items():
        predicted_force, predicted_reason = predict_release(directions[name].tln, P)
        assert np.isclose(predicted_force, force)
        assert predicted_reason == reason


def test_random_sample_count_defaults():
    args = MODULE.build_parser().parse_args([])
    assert args.surface_samples == 500
    assert args.interior_samples == 250


def test_plane_sweep_is_normalized_and_treats_pure_compression_as_unbounded():
    directions = plane_directions(("tn", "tl"), samples_per_plane=4)
    assert len(directions) == 8
    assert all(np.isclose(np.linalg.norm(direction.tln), 1.0) for direction in directions)
    compression = next(direction for direction in directions if direction.name == "tn_-090.0deg")
    force, reason = predict_release(compression.tln, P)
    assert math.isinf(force)
    assert reason == 0


def test_random_sweep_is_reproducible_and_samples_surface_plus_interior():
    kwargs = {
        "surface_samples": 15,
        "interior_samples": 7,
        "seed": 23,
        "max_force": 600.0,
        "soft_step": 2.0,
        "ultimate_step": 5.0,
        "normal_floor": -100.0,
        "interior_fraction": (0.2, 0.75),
        "interior_margin": 0.15,
    }
    cases = random_directions(P, **kwargs)
    repeated = random_directions(P, **kwargs)
    different = random_directions(P, **{**kwargs, "seed": 24})

    assert cases == repeated
    assert cases != different
    assert len(cases) == 22
    assert len({case.name for case in cases}) == len(cases)
    assert all(np.isclose(np.linalg.norm(case.tln), 1.0) for case in cases)

    surface = [case for case in cases if case.sample_kind == "surface"]
    interior = [case for case in cases if case.sample_kind == "interior"]
    assert len(surface) == 15
    assert len(interior) == 7
    assert {case.name.split("-")[1] for case in surface} >= {
        "reverse",
        "tension",
        "lateral",
        "ultimate",
    }

    for case in surface:
        target = case.target_force_N * np.asarray(case.tln)
        radius, _ = predict_release(case.tln, P)
        assert case.expected_release is True
        assert np.isclose(radius, case.target_force_N)
        assert abs(MODULE._model_margin(target, P)) < 1.0e-10
        assert case.target_force_N + (5.0 if np.isclose(target[0], 400.0) else 2.0) <= 600.0

    for case in interior:
        target = case.target_force_N * np.asarray(case.tln)
        radius, _ = predict_release(case.tln, P)
        assert case.expected_release is False
        assert case.target_force_N < radius
        assert MODULE._model_margin(target, P) >= 0.15


def test_run_trial_releases_surface_and_holds_exact_interior_target():
    common = {
        "surface_samples": 1,
        "interior_samples": 1,
        "seed": 5,
        "max_force": 600.0,
        "soft_step": 2.0,
        "ultimate_step": 5.0,
        "normal_floor": -100.0,
        "interior_fraction": (0.2, 0.75),
        "interior_margin": 0.15,
    }
    cases = random_directions(P, **common)
    surface = next(case for case in cases if case.sample_kind == "surface")
    interior = next(case for case in cases if case.sample_kind == "interior")

    surface_pad = _FakePad(release_on_violation=True)
    surface_result = MODULE.run_trial(
        surface_pad,
        surface,
        trial_index=0,
        repeat=0,
        max_force=600.0,
        soft_step=2.0,
        ultimate_step=5.0,
        hold_ticks=3,
        zero_ticks=2,
        margin_tolerance=0.2,
        traces=[],
    )
    assert surface_result.passed
    assert surface_result.status == "released"
    assert surface_result.released

    interior_pad = _FakePad(release_on_violation=True)
    interior_result = MODULE.run_trial(
        interior_pad,
        interior,
        trial_index=1,
        repeat=0,
        max_force=600.0,
        soft_step=2.0,
        ultimate_step=5.0,
        hold_ticks=3,
        zero_ticks=2,
        margin_tolerance=0.2,
        traces=[],
    )
    command_norms = np.linalg.norm(interior_pad.commands, axis=1)
    assert interior_result.passed
    assert interior_result.status == "held-interior"
    assert not interior_result.released
    assert np.isclose(command_norms.max(), interior.target_force_N)
    assert np.all(command_norms <= interior.target_force_N + 1.0e-10)


def test_save_sweep_plot_is_headless_and_ignores_nonfinite_trials(tmp_path):
    results = [
        _trial_result(),
        _trial_result(
            trial_index=1,
            name="sensor-error",
            released=False,
            release_reason=0,
            release_reason_name="none",
            measured_force_N=math.nan,
            release_force_N=math.nan,
            f_t_N=math.nan,
            f_l_N=math.inf,
            f_n_N=math.nan,
            reported_margin=math.nan,
            model_margin=math.nan,
            status="nonfinite",
            passed=False,
        ),
        _trial_result(
            trial_index=2,
            name="disengaged",
            released=False,
            unexpected_disengage=True,
            release_reason=0,
            release_reason_name="none",
            f_t_N=4.0,
            f_l_N=2.0,
            f_n_N=3.0,
            release_force_N=math.nan,
            status="unexpected-disengage",
            passed=False,
        ),
        _trial_result(
            trial_index=3,
            name="interior-001",
            sample_kind="interior",
            expected_release=False,
            released=False,
            release_reason=0,
            release_reason_name="none",
            target_force_N=20.0,
            target_t=10.0,
            target_l=5.0,
            target_n=-16.6,
            target_margin=0.3,
            f_t_N=10.0,
            f_l_N=5.0,
            f_n_N=-16.6,
            release_force_N=math.nan,
            status="held-interior",
            passed=True,
        ),
    ]

    output = tmp_path / "nested" / "sweep.png"
    assert save_sweep_plot(output, results, P) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 50_000
