"""Task definitions for scr_phybot_mjab."""

import scr_phybot_mjab.tasks.climb_wall  # noqa: F401
import scr_phybot_mjab.tasks.microspine_rizon4  # noqa: F401
import scr_phybot_mjab.tasks.walk  # noqa: F401

# Imported after walk: the squat task reuses shared PHYBot C2 walk mdp terms.
import scr_phybot_mjab.tasks.squat  # noqa: F401,E402
