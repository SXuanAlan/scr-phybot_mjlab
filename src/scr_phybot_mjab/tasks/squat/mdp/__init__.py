"""MDP terms of the squat task: the phase-driven reference command and the tracking rewards.

Only this package's own terms; the configs import mjlab's generic terms
from ``mjlab.envs.mdp`` and the terms shared with walk from ``tasks.common``.
"""

from .motion_command import *  # noqa: F403
from .rewards import *  # noqa: F403
