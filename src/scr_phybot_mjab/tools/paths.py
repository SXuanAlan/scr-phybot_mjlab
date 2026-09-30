"""Default locations for offline work in a source checkout.

Runtime imports must use assets resource paths, never these analysis paths.
Command-line tools expose overrides for datasets and analysis destinations.
"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MICROSPINE_RAW_DIR = REPOSITORY_ROOT / "data/microspine/raw"
DEFAULT_UNIT_TRIALS = MICROSPINE_RAW_DIR / "20260905-110314/results.csv"
CLIMB_REVIEW_DIR = REPOSITORY_ROOT / "docs/empirical_climb_review"
LIMIT_SURFACE_ANALYSIS_DIR = CLIMB_REVIEW_DIR / "limit_surface"
EMPIRICAL_UNIT_REPORT = LIMIT_SURFACE_ANALYSIS_DIR / "empirical_unit_v1.json"
FIGURES_DIR = REPOSITORY_ROOT / "docs/figures"
