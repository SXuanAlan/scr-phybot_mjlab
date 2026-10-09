"""PHYBot C2 climbing facts shared by the env config and the offline reference tools.

Data only: importing this module builds no environment config.
"""

from __future__ import annotations

import json

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_FOOT_TOES
from scr_phybot_mjab.microspine.spec import ToeContact
from scr_phybot_mjab.tasks.climb_wall.motions import CAD_CLIMB_INITIAL_STATE

CLIMB_EE_BODY_NAMES = ("left_hand_spine", "right_hand_spine", "left_foot_spine", "right_foot_spine")
"""The four grip end-effectors, in the order of every per-EE tensor."""

PAD_PROXY_ROWS = 2
PAD_PROXY_COLS = 2
"""Each pad's corner contact proxies, separate from its 12 or 4 active units."""


def climb_initial_state() -> dict:
  """The canonical climbing stance, written next to the clip by the CAD reference generator."""
  return json.loads(CAD_CLIMB_INITIAL_STATE.read_text())


def climb_toe_contacts() -> tuple[ToeContact, ...]:
  """Foot toes that brace the wall during foot regrips: a released pad must never slide on it."""
  return tuple(
    ToeContact(ee_body_name=ee, body_name=body, pos=pos, radius=radius) for ee, body, pos, radius in PHYBOT_C2_FOOT_TOES
  )
