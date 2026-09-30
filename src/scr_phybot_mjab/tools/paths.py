"""Default locations for offline work in a source checkout.

Runtime imports must use assets resource paths, never these analysis paths.
Command-line tools expose overrides for datasets and analysis destinations.
Generated outputs go under the ignored ``artifacts/``; inputs live in ``data/``.
"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPOSITORY_ROOT / "data"
MICROSPINE_RAW_DIR = DATA_DIR / "microspine/raw"
DEFAULT_UNIT_TRIALS = MICROSPINE_RAW_DIR / "20260905-110314/results.csv"
# Fit provenance of the packaged runtime table assets/microspine/empirical_unit_v1.npz.
EMPIRICAL_UNIT_DIR = DATA_DIR / "microspine/empirical_unit_v1"
EMPIRICAL_UNIT_REPORT = EMPIRICAL_UNIT_DIR / "empirical_unit_v1.json"
CAD_MODEL_DIR = DATA_DIR / "cad_model"

ARTIFACTS_DIR = REPOSITORY_ROOT / "artifacts"
CLIMB_REVIEW_DIR = ARTIFACTS_DIR / "climb_review"
LIMIT_SURFACE_ANALYSIS_DIR = ARTIFACTS_DIR / "limit_surface"

FIGURES_DIR = REPOSITORY_ROOT / "docs/figures"

# Sibling checkout of the robot deploy repository (real-robot logs, log2csv).
DEPLOY_REPO_DIR = REPOSITORY_ROOT.parent / "bdml_phybot_deploy_c2"
