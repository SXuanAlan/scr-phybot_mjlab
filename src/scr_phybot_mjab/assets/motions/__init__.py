"""Motion resource paths shared by task configuration and offline tools."""

from pathlib import Path

MOTIONS_DIR = Path(__file__).resolve().parent
CAD_CLIMB_MOTION = MOTIONS_DIR / "climb_cad_empirical.npy"
CAD_CLIMB_CORNERS60_MOTION = MOTIONS_DIR / "climb_cad_empirical_corners_a60.npy"
CAD_CLIMB_CORNERS80_MOTION = MOTIONS_DIR / "climb_cad_empirical_corners_a80.npy"
CAD_CLIMB_INITIAL_STATE = MOTIONS_DIR / "climb_cad_empirical_initial.json"
CLIMB_MOUNT80_MOTION = MOTIONS_DIR / "climb_mount_prologue_a80.npy"
