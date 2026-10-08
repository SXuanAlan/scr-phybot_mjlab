"""MDP terms of the walking tasks: velocity commands, AMP resets, rewards and observations.

Only this package's own terms; the configs import mjlab's generic terms
from ``mjlab.envs.mdp`` and the terms shared with squat from ``tasks.common``.
"""

from .amp_reset import *  # noqa: F403
from .curriculums import *  # noqa: F403
from .events import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .velocity_command import *  # noqa: F403
