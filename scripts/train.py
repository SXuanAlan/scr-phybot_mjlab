"""Train RL agent — delegates to mjlab's train script after registering tasks."""

import scr_phybot_mjab.tasks  # noqa: F401 — registers Mjlab-Climb-Wall-Phybot-Big-V2

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
