"""Packaged microspine runtime resources; raw trials and analysis live outside src."""

from pathlib import Path

MICROSPINE_ASSET_DIR = Path(__file__).resolve().parent
EMPIRICAL_UNIT_TABLE = MICROSPINE_ASSET_DIR / "empirical_unit_v1.npz"
