"""List registered environments."""

import spider_rl_mjlab.tasks  # noqa: F401

from mjlab.scripts.list_envs import main

if __name__ == "__main__":
  main()
