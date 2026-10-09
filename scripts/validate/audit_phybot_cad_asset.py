"""Verify the compiled asset against the source CAD and render a comparison."""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from scr_phybot_mjab.tools.microspine.cad_audit import audit_rigid_cad_asset, render_cad_comparison
from scr_phybot_mjab.tools.paths import CLIMB_REVIEW_DIR, FIGURES_DIR


def main(argv: list[str] | None = None):
  argparse.ArgumentParser(description=__doc__).parse_args(argv)
  report = audit_rigid_cad_asset()
  path = CLIMB_REVIEW_DIR / "rigid_cad_asset_audit.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2) + "\n")
  if not report["passed"]:
    raise RuntimeError(f"CAD mismatch: {report['max_compiled_vertex_error_m']} m; see {path}")
  image = FIGURES_DIR / "phybot_c2_cad_rigid_comparison.png"
  render_cad_comparison(image)
  print(f"PASS: maximum compiled vertex error {report['max_compiled_vertex_error_m'] * 1000:.8f} mm")
  print(path)
  print(image)


if __name__ == "__main__":
  main()
