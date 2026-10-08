"""Build the empirical runtime table from raw trials, with separate analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import mjlab  # noqa: F401  # Initialize plugins before project imports.

from scr_phybot_mjab.assets.microspine import EMPIRICAL_UNIT_TABLE
from scr_phybot_mjab.tools.microspine.empirical_fitting import build_empirical_model
from scr_phybot_mjab.tools.paths import DEFAULT_UNIT_TRIALS, EMPIRICAL_UNIT_DIR


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=DEFAULT_UNIT_TRIALS)
  parser.add_argument("--output", type=Path, default=EMPIRICAL_UNIT_TABLE)
  parser.add_argument("--analysis-dir", type=Path, default=EMPIRICAL_UNIT_DIR)
  args = parser.parse_args()
  build_empirical_model(args.data, args.output, args.analysis_dir)


if __name__ == "__main__":
  main()
