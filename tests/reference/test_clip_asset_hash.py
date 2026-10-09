"""Clips that record the robot MJCF's hash match the current MJCF (AGENTS.md rule 6).

A changed phybot_c2.xml without regenerated clips fails here, before a task
trains on references built for another robot.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_XML
from scr_phybot_mjab.tasks.climb_transition80.motions import CLIMB_MOUNT80_MOTION
from scr_phybot_mjab.tasks.climb_wall.motions import (
  CAD_CLIMB_CORNERS60_MOTION,
  CAD_CLIMB_CORNERS80_MOTION,
  CAD_CLIMB_MOTION,
)

TASKS_DIR = Path(__file__).resolve().parents[2] / "src/scr_phybot_mjab/tasks"
HASHED_CLIPS = (CAD_CLIMB_MOTION, CAD_CLIMB_CORNERS60_MOTION, CAD_CLIMB_CORNERS80_MOTION, CLIMB_MOUNT80_MOTION)


def _recorded_hash(clip: Path) -> str | None:
  data = np.load(clip, allow_pickle=True)
  if data.dtype != object or data.shape != ():
    return None
  value = data.item().get("asset_xml_sha256")
  return None if value is None else str(value)


def test_the_climb_and_mount_clips_record_the_hash():
  recorded = {clip.resolve() for clip in TASKS_DIR.glob("*/motions/*.npy") if _recorded_hash(clip)}
  assert {clip.resolve() for clip in HASHED_CLIPS} <= recorded


@pytest.mark.parametrize(
  "clip",
  sorted(clip for clip in TASKS_DIR.glob("*/motions/*.npy") if _recorded_hash(clip)),
  ids=lambda clip: clip.stem,
)
def test_clip_matches_the_current_robot_xml(clip):
  assert _recorded_hash(clip) == hashlib.sha256(PHYBOT_C2_XML.read_bytes()).hexdigest()
