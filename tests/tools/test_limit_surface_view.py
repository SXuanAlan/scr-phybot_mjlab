import numpy as np, os
from scr_phybot_mjab.tools.limit_surface_teleop.view import LimitSurfaceView

P = {"f_t_max": 400.0, "f_rev_max": 20.0, "mu_adhesion": 0.4,
     "f_n0": 15.0, "mu_lateral": 0.6, "f_l0": 15.0}

def test_release_appends_one_frozen_point_and_saves(tmp_path):
    v = LimitSurfaceView(P, headless=True)   # headless: no GL items
    v.update(np.array([10.0, 0, 5.0]), margin=0.5, released=False)
    assert len(v.releases) == 0
    v.update(np.array([0.0, 0, 15.0]), margin=0.0, released=True)
    assert len(v.releases) == 1
    p = tmp_path / "pts.npz"
    v.save(str(p))
    assert np.load(p)["releases"].shape == (1, 3)
    v.clear()
    assert len(v.releases) == 0

def test_margin_to_color():
    v = LimitSurfaceView(P, headless=True)
    assert v._color(0.5)[1] > v._color(0.5)[0]     # green-ish
    assert v._color(-0.5)[0] > v._color(-0.5)[1]   # red-ish
