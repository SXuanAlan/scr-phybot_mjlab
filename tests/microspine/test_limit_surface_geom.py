import numpy as np
from scr_phybot_mjab.microspine.limit_surface_geom import (
    release_radius, envelope_mesh, envelope_margin, LIMIT_KEYS,
)

P = {"f_t_max": 400.0, "f_rev_max": 20.0, "mu_adhesion": 0.4,
     "f_n0": 15.0, "mu_lateral": 0.6, "f_l0": 15.0}

def test_release_radius_pure_pulloff_hits_f_n0():
    # pure +n (a=0,b=0,c=1): releases via tension at f_n0
    r = release_radius(np.array([0.0]), np.array([0.0]), np.array([1.0]), P)
    assert np.isclose(r[0], P["f_n0"], atol=1e-6)

def test_release_radius_pure_shear_hits_ultimate():
    r = release_radius(np.array([1.0]), np.array([0.0]), np.array([0.0]), P)
    assert np.isclose(r[0], P["f_t_max"], atol=1e-6)

def test_release_radius_pure_compression_is_inf():
    r = release_radius(np.array([0.0]), np.array([0.0]), np.array([-1.0]), P)
    assert not np.isfinite(r[0])

def test_margin_positive_inside_negative_outside():
    inside = envelope_margin(np.array([[0.0, 0.0, 5.0]]), P)   # f_n=5 < 15
    outside = envelope_margin(np.array([[0.0, 0.0, 30.0]]), P)  # f_n=30 > 15
    assert inside[0] > 0 and outside[0] < 0

def test_mesh_shape_and_points_on_surface():
    X, Y, Z = envelope_mesh(P, n_phi=30, n_theta=40)
    assert X.shape == (30, 40)
    finite = np.isfinite(Z)
    loads = np.stack([X[finite], Y[finite], Z[finite]], axis=-1)
    assert np.all(np.abs(envelope_margin(loads, P)) < 1e-3)  # mesh lies on the surface
