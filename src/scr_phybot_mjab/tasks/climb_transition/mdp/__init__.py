"""MDP functions for the walk->climb transition task.

The climb task's mdp is re-exported wholesale (the transition env cfg is a
mutated climb cfg and resolves reward/observation/termination functions
through THIS module), then the transition-specific pieces layer on top.
"""

from scr_phybot_mjab.tasks.climb_wall.mdp import *  # noqa: F401, F403

from .metrics import *  # noqa: F403
from .motion import *  # noqa: F403
