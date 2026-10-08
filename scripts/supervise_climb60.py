"""Persistently examine the Corners-A60 run and hold unmastered curriculum stages.

Run this in a separate tmux session from training::

  python scripts/supervise_climb60.py RUN_DIR --gpu 1 --training-log RUN_DIR/train.log
  python scripts/supervise_climb60.py RUN_DIR --once

The training process must already consume RUN_DIR/curriculum_control.json.
This process only writes that control file and monitoring artifacts; it never
starts/restarts training or changes rewards, contacts, or physics. Checkpoints
are evaluated after their modification time has settled. A stage stays locked
until two consecutive complete exams pass and a person/agent has reviewed the
actual engage rasters and rollout videos for both checkpoints. Review templates
are deliberately unsigned and cannot approve themselves.
"""

from __future__ import annotations

from pathlib import Path

from scr_phybot_mjab.tools.training_monitor import supervisor


def main(argv: list[str] | None = None) -> None:
  supervisor.run(supervisor.parse_args(argv, eval_script=Path(__file__).with_name("evaluate_climb_checkpoint.py")))


if __name__ == "__main__":
  main()
