import numpy as np, pytest, torch
from scr_phybot_mjab.microspine.limit_surface_geom import envelope_margin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _ramp_to_release(pt, dir_tln, step, settle, fmax=600.0):
    """Ramp a Cartesian force in the engage-frame direction until the anchor
    releases. Each force level is held `settle` ticks so the release lands at a
    fixed magnitude (the limit-surface soft faces debounce over 2 steps).

    NOTE(deviation): the plan suggests ~5 N/tick. A pure 5 N/tick ramp overshoots
    the soft faces (tension/lateral/reverse caps ~15-20 N) by a full debounce
    window, landing outside the |margin|<0.2 gate. A finer step with a settle
    hold makes every direction land ON the true envelope; shear still uses a
    coarse 5 N step because its ultimate cap is ~400 N.
    """
    d = np.asarray(dir_tln, dtype=float)
    r = None
    f = 0.0
    while f < fmax:
        f += step
        for _ in range(settle):
            r = pt.apply(tuple(f * d))
            if r.released:
                return r
    return r


def test_force_control_maps_full_envelope():
    """Task-space force control reaches failure in EVERY engage-frame direction
    (the win over position-drag) and each release lands on the true envelope."""
    from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop

    pt = PadTeleop()
    try:
        # (t, l, n) engage-frame unit direction, per-tick force step, settle hold.
        cases = {
            "pull_off_+n": ((0.0, 0.0, 1.0), 2.0, 3),
            "lateral_+l": ((0.0, 1.0, 0.0), 2.0, 3),
            "reverse_-t": ((-1.0, 0.0, 0.0), 2.0, 3),
            "shear_+t": ((1.0, 0.0, 0.0), 5.0, 3),  # must reach the ~400 N ultimate
        }
        for name, (dir_tln, step, settle) in cases.items():
            pt.home()
            pt.press_engage()
            assert pt.apply((0.0, 0.0, 0.0)).engaged, f"{name}: not engaged after press"
            r = _ramp_to_release(pt, dir_tln, step, settle)
            assert r is not None and r.released, f"{name}: never released"
            margin = envelope_margin(r.load_tln[None], pt.limit_params)[0]
            assert abs(margin) < 0.2, (
                f"{name}: release load {r.load_tln} off-envelope, margin={margin:.3f}"
            )
        # Shear reaching the ultimate is the whole point: confirm it loaded hard.
        pt.home()
        pt.press_engage()
        r = _ramp_to_release(pt, (1.0, 0.0, 0.0), 5.0, 3)
        assert r.load_tln[0] > 350.0, f"shear only reached f_t={r.load_tln[0]:.1f}"
    finally:
        pt.env.close()
