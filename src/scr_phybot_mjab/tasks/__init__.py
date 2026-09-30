"""Task definitions for scr_phybot_mjab."""

import scr_phybot_mjab.tasks.climb_wall  # noqa: F401
import scr_phybot_mjab.tasks.microspine_rizon4  # noqa: F401
import scr_phybot_mjab.tasks.walk  # noqa: F401

# Imported after climb_wall: the transition task builds on the climb cfg and
# subclasses its motion command.
import scr_phybot_mjab.tasks.climb_transition  # noqa: F401,E402

# Imported after walk: the squat task reuses shared PHYBot C2 walk mdp terms.
import scr_phybot_mjab.tasks.squat  # noqa: F401,E402

# The 80-degree transition task: a parallel package that depends on
# climb_wall only (decoupled from climb_transition by contract).
import scr_phybot_mjab.tasks.climb_transition80  # noqa: F401,E402

import scr_phybot_mjab.tasks.brick  # noqa: F401,E402

