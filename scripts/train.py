"""Train RL agent — delegates to mjlab's train script after registering tasks."""

from mjlab.scripts.train import main

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

if __name__ == "__main__":
  main()
