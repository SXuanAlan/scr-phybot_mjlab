"""Where generated files go."""

from scr_phybot_mjab.tools.paths import checkpoint_eval_dir


def test_checkpoint_analyses_sit_beside_the_checkpoint(tmp_path):
  run = tmp_path / "logs/rsl_rl/phybot_c2_walk/2026-06-12_16-32-35"
  assert checkpoint_eval_dir(run / "model_599.pt") == run.resolve() / "eval" / "model_599"
