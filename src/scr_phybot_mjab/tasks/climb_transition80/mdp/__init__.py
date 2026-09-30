"""MDP functions for the 80-degree transition task.

Climb's mdp is re-exported wholesale (the env cfg is a mutated climb cfg and
resolves its inherited observation/metric functions through THIS module);
the transition80 pieces layer on top. Nothing is imported from
``climb_transition`` — this package is decoupled from it by contract
(tests/microspine/test_climb_transition80_config.py enforces it).
"""

from scr_phybot_mjab.tasks.climb_wall.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F403
from .metrics import *  # noqa: F403
from .motion import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
