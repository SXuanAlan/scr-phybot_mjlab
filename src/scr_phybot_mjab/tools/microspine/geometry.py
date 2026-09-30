"""Asset-backed geometry for offline PHYBot pad analyses."""

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import SPINE_BODY_NAMES, get_spec
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry


def phybot_pad_geometry(unit_layout="all"):
  cfg = MicrospineCfg(ee_body_names=SPINE_BODY_NAMES, spheres_per_ee_rows=2, spheres_per_ee_cols=2)
  return matching_pad_geometry(get_spec(), cfg.ee_body_names,
    rake_local_axis=cfg.limit_surface.ee_rake_local_axis,
    face_local_axis=cfg.face_local_axis, unit_layout=unit_layout)
