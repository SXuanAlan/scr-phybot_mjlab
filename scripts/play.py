"""Play/evaluate RL agent — delegates to mjlab's play script after registering tasks.

Supports --agent zero|random|trained modes:
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --agent zero
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --agent random
  python scripts/play.py Mjlab-ClimbWall-Phybot-C2 --checkpoint-file path/to/model.pt
"""

import spider_rl_mjlab.tasks  # noqa: F401

from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
