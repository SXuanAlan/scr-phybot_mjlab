"""Source-CAD regressions catch mesh warping, missing pieces and wrong mates."""
import numpy as np

from scr_phybot_mjab.tools.microspine.cad import load_cad_pad_assembly
from scr_phybot_mjab.tools.microspine.cad_audit import audit_rigid_cad_asset


def test_compiled_geometry_is_a_rigid_import_of_every_cad_solid():
  report = audit_rigid_cad_asset()
  assert report["source_solid_count"] == 34
  assert report["passed"], report
  assert len(report["compiled_vertex_errors_m"]) == 58  # 14 per pad + two mounts


def test_step_bearing_seat_and_sleeve_stop_define_mounting():
  cad = load_cad_pad_assembly()
  np.testing.assert_allclose(cad.pivot, [0, .01225, .0905], atol=1e-12)
  np.testing.assert_allclose(cad.features["sleeve_shoulder_m"], [0, 0, .035], atol=1e-12)
  np.testing.assert_allclose(cad.contact_center, [0, -.04750, .07798417], atol=5e-8)
