"""MDP terms of the 80-degree transition task.

Only this package's own terms: the configs import climb's terms from
``climb_wall.mdp`` where they use them.
"""

from .curriculums import *  # noqa: F403
from .metrics import *  # noqa: F403
from .motion import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
