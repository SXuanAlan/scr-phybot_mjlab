"""Walk->climb transition (mount) task.

A PARALLEL task package: everything transition-specific lives here. The
climb_wall package is imported, never modified — the env cfg is built by
calling the climb builder and mutating the returned config, and the motion
command is a subclass. Contract parity with climb (identical observation and
action dims) is what lets climb checkpoints warm-start this task strict.
"""

import scr_phybot_mjab.tasks.climb_transition.config.phybot_c2  # noqa: F401
