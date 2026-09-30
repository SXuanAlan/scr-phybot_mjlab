"""Evaluation guards: reset seeds and static holds must never prove mastery."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_climb_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_climb_checkpoint", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_static_seeded_hold_exposes_reference_off_failures():
  reference = np.array([[True, True, False, True], [False, True, True, True]])
  metrics = evaluation.schedule_statistics(np.ones_like(reference), reference)
  assert metrics["agreement"] == pytest.approx(.75)
  assert metrics["balanced_accuracy"] == .5
  assert metrics["reference_off_accuracy"] == 0.


def test_reset_seed_edges_do_not_count_as_expected_or_physical_regrips():
  ref = np.array([[False] * 4, [True] * 4, [False] * 4, [True] * 4])
  physical = np.array([[False] * 4, [False] * 4, [False] * 4, [True] * 4])
  expected, matched = evaluation.matched_schedule_edges(ref, physical, np.array([0, 1, 1, 1]), 1)
  np.testing.assert_array_equal(expected, [1] * 4)
  np.testing.assert_array_equal(matched, [1] * 4)


def test_single_physical_latch_cannot_satisfy_multiple_schedule_edges():
  ref = np.array([[False], [True], [False], [True]])
  physical = np.array([[False], [False], [True], [False]])
  expected, matched = evaluation.matched_schedule_edges(ref, physical, np.zeros(4), 2)
  assert expected.tolist() == [2]
  assert matched.tolist() == [1]


@pytest.mark.parametrize("profile", ["canonical", "training-rsi"])
def test_config_pins_requested_angle_and_preserves_empirical_hardware(profile):
  cfg, agent = evaluation.configure_evaluation(evaluation.TASK, 50., 4, profile, 123)
  assert cfg.auto_reset is False
  assert cfg.scene.num_envs == 4
  assert cfg.episode_length_s == 30.
  assert cfg.curriculum["wall_angle"].params["angle_stages"] == [
    {"iter": 0, "angle_deg": 50., "jitter_deg": 0.}]
  assert cfg.curriculum["wall_angle"].params["angle_jitter_deg"] == 0.
  assert set(cfg.terminations) >= {"bad_tracking", "lost_support", "time_out"}
  mc = cfg.commands["microspine_affordance"].microspine_cfg
  assert mc.limit_surface.model == "empirical_unit"
  assert mc.limit_surface.unit_layout == "corners"
  assert mc.engage_unit_preload_n == 10.
  assert cfg.commands["climb_motion"].sampling_mode == ("start" if profile == "canonical" else "uniform")
  assert cfg.observations["actor"].enable_corruption == (profile == "training-rsi")
  assert ("base_mass" in cfg.events) == (profile == "training-rsi")
  assert agent.actor.obs_normalization


def _synthetic_exam(failed_first=False):
  steps, worlds, dt = 60, 2, .5
  phase = np.repeat(((np.arange(steps) + 1) % 18)[:, None], worlds, axis=1)
  ref = np.repeat((phase[..., None] < 15), 4, axis=2)
  new_latch = np.zeros_like(ref)
  new_latch[1:] = ref[1:] & ~ref[:-1]
  ids = np.zeros((steps, worlds), int)
  episodes = []
  for env_id in range(worlds):
    episodes.append({"env_id": env_id, "episode_id": 0, "duration_s": steps * dt,
                     "end_type": "timeout", "termination_reasons": [],
                     "start_phase_frame": 0, "end_phase_frame": int(phase[-1, env_id]),
                     "start_reference_engaged": [True] * 4,
                     "new_latches_by_pad": [3] * 4, "initial_seed_grips": 4,
                     "progress_from_birth_m": .06, "progress_after_settle_m": .06,
                     "reference_progress_after_settle_m": .06})
  if failed_first:
    episodes[0].update(duration_s=1., end_type="terminated", termination_reasons=["lost_support"],
                       new_latches_by_pad=[0] * 4, progress_from_birth_m=-.1,
                       progress_after_settle_m=None, reference_progress_after_settle_m=None)
    # Repeated successful reseeds later in this world cannot rescue its exam.
    ids[2:, 0] = 1
    episodes.append(dict(episodes[1], env_id=0, episode_id=1))
  trajectory = {"episode_id": ids, "reference_engaged": ref, "engaged": ref.copy(),
                "new_latch": new_latch, "phase_frame": phase, "release_reason": np.zeros_like(ref, int)}
  return trajectory, episodes, worlds, dt


def test_curriculum_metrics_grade_each_first_continuous_episode():
  trajectory, episodes, worlds, dt = _synthetic_exam()
  result = evaluation.summarize(trajectory, episodes, worlds, dt, 30., 9.)
  assert result["automatic_candidate_pass"]
  assert result["true_regrip_cycles_min"] == 3
  assert result["reference_cycles_completed_min"] == 3
  assert result["engage_match_min"] == 1.
  assert result["progress_ratio_min"] == 1.
  assert result["first_episode_falls"] == 0
  assert result["full_30s_survival_rate"] == 1.
  assert result["visual_review_required"] is True


def test_later_reset_rollouts_cannot_hide_one_bad_initial_world():
  trajectory, episodes, worlds, dt = _synthetic_exam(failed_first=True)
  result = evaluation.summarize(trajectory, episodes, worlds, dt, 30., 9.)
  assert not result["automatic_candidate_pass"]
  assert result["first_episode_falls"] == 1
  assert result["first_episode_survival_rate"] == .5
  assert result["true_regrip_cycles_min"] == 0
  assert result["reference_cycles_completed_min"] == 0
  assert result["progress_ratio_min"] is None
  assert result["new_latches_by_pad"] == dict(zip(evaluation.PAD_NAMES, [3] * 4))


def test_numeric_divergence_still_publishes_valid_failed_metrics(tmp_path):
  trajectory, episodes, worlds, dt = _synthetic_exam(failed_first=True)
  episodes[0]["progress_after_settle_m"] = float("nan")
  episodes[0]["reference_progress_after_settle_m"] = .06
  result = evaluation.summarize(trajectory, episodes, worlds, dt, 30., 9.)
  assert result["progress_ratio_min"] is None
  assert not result["automatic_candidate_pass"]
  path = tmp_path / "metrics.json"
  evaluation.write_json(path, dict(result, episodes=episodes))
  assert json.loads(path.read_text())["episodes"][0]["progress_after_settle_m"] is None
  assert not path.with_suffix(".json.tmp").exists()


def test_clean_short_smoke_is_not_a_full_30_second_survival_pass():
  trajectory, episodes, worlds, dt = _synthetic_exam()
  trajectory = {key: value[:1] for key, value in trajectory.items()}
  for episode in episodes:
    episode.update(duration_s=.5, end_type="evaluation_horizon", new_latches_by_pad=[0] * 4,
                   progress_after_settle_m=0., reference_progress_after_settle_m=0.)
  result = evaluation.summarize(trajectory, episodes, worlds, dt, .5, 9.)
  assert result["first_episode_survival_rate"] == 1.
  assert result["full_30s_survival_rate"] == 0.
  assert not result["automatic_candidate_pass"]
  assert result["automatic_candidate_checks"]["at_least_30_seconds"] is False
