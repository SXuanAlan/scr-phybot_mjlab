"""The 80-degree walk-to-climb transition task.

A sibling of ``climb_wall`` that reuses climb's scene, microspine plant,
motion command and mdp terms, and adds what a one-shot mount needs: acyclic
engage tables with a terminal hold, blind-delivery spawn scatter, adaptive
and classroom RSI, an engage-progress potential, the contact law and the
engage cone.

Rewards are BeyondMimic's exp-kernel tracking terms plus the potential, a
dwell wage, the rip bill and two regularizers. Every illegal or
wrong-engage state is a termination, with the engage cone as
Constraints-as-Terminations. The joint PD table equals the real robot's
motor firmware table.
"""

import scr_phybot_mjab.tasks.climb_transition80.config.phybot_c2  # noqa: F401
