"""The AMP clips of the walk tasks; the AMP dataset reads them by name from this directory."""

from pathlib import Path

AMP_MOTIONS_DIR = Path(__file__).resolve().parent
STAND_ZERO_MOTION = AMP_MOTIONS_DIR / "c2_stand_zero.npy"
