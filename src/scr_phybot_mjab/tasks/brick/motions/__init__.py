"""The brick task's reference clip, beside its audit and preview video (ignored)."""

from pathlib import Path

BRICK_REFERENCE_MOTION = Path(__file__).resolve().parent / "brick_pick_place.npz"
