"""Pure-numpy microspine limit-surface geometry: release radius, mesh, margin."""
from __future__ import annotations
import math
import numpy as np

LIMIT_KEYS = ("f_t_max", "f_rev_max", "mu_adhesion", "f_n0", "mu_lateral", "f_l0")


def release_radius(a, c_lat, c_norm, p):
    a = np.asarray(a, float); b = np.asarray(c_lat, float); c = np.asarray(c_norm, float)
    inf = np.inf
    ap = np.maximum(a, 0.0)
    r_rev = np.where(a < 0, p["f_rev_max"] / np.maximum(-a, 1e-9), inf)
    r_ult = np.where(a > 0, p["f_t_max"] / np.maximum(a, 1e-9), inf)
    dt = c - p["mu_adhesion"] * ap
    r_ten = np.where(dt > 1e-9, p["f_n0"] / np.maximum(dt, 1e-9), inf)
    dl = np.abs(b) - p["mu_lateral"] * ap
    r_lat = np.where(dl > 1e-9, p["f_l0"] / np.maximum(dl, 1e-9), inf)
    return np.minimum.reduce([r_rev, r_ult, r_ten, r_lat])


def envelope_mesh(p, n_phi=90, n_theta=140, r_clip=None):
    if r_clip is None:
        r_clip = 1.6 * p["f_t_max"]
    phi = np.linspace(0.03, math.pi - 0.03, n_phi)
    th = np.linspace(0.0, 2 * math.pi, n_theta)
    pp, tt = np.meshgrid(phi, th, indexing="ij")
    a = np.cos(pp); b = np.sin(pp) * np.cos(tt); c = np.sin(pp) * np.sin(tt)
    r = release_radius(a, b, c, p)
    r = np.where(np.isfinite(r) & (r < r_clip), r, np.nan)
    return r * a, r * b, r * c


def envelope_margin(load_tln, p):
    L = np.asarray(load_tln, float)
    f_t, f_l, f_n = L[..., 0], L[..., 1], L[..., 2]
    ft_pos = np.maximum(f_t, 0.0)
    cap_n = p["mu_adhesion"] * ft_pos + p["f_n0"]
    cap_l = p["mu_lateral"] * ft_pos + p["f_l0"]
    m1 = (f_t + p["f_rev_max"]) / p["f_rev_max"]
    m2 = (cap_n - f_n) / cap_n
    m3 = (cap_l - np.abs(f_l)) / cap_l
    m4 = (p["f_t_max"] - f_t) / p["f_t_max"]
    return np.minimum.reduce([m1, m2, m3, m4])
