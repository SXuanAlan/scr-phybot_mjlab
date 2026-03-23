"""Train RL agent — delegates to mjlab's train script after registering tasks."""

import spider_rl_mjlab.tasks  # noqa: F401 — registers Mjlab-ClimbWall-Phybot-C2

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
