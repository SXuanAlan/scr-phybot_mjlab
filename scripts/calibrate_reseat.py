"""Grace-window / re-seat mode calibration on the Rizon4 bench.

Re-runs the engage-survival study that originally sized hold_grace_steps
(free-following era: 6 -> 31%, 10 -> 93%, 14 -> 100%) under current physics
(latched-contact retraction, virtual spine field), across

    window in {14, 8, 6}  x  reseat in {free, down, none}

The climb-side phenomenon being calibrated: right after a hand latches, the
press->hang load flip pulls the pad outward; with free re-seat the anchor
follows (12-44 mm measured on motionref12) — something hooked spines cannot
do. The bench emulates that flip by applying 30 N of pull-off tension for
the whole grace window (unjudged by design), then a sustainable hand load
(60 N rake shear + 15 N tension) for 1 s. Cycle scores: engaged, survived,
anchor drift over the grace window (total / outward component).

A final sanity run checks the "down" mode actually follows a down-drag
(60 N rake shear during grace) by the capped amount and nothing else.
"""

import numpy as np

from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop

CYCLES = 8
HOLD_STEPS = 50  # 1 s sustained-load phase


def make_bench(window: int, mode: str) -> PadTeleop:
  def hook(cfg):
    m = cfg.commands["microspine_affordance"].microspine_cfg
    e = cfg.events["microspine_grip"].params["microspine_cfg"]
    for mc in (m, e):
      mc.limit_surface.hold_grace_steps = window
      mc.reanchor_after_grace = mode != "none"
      mc.anchor_reseat_mode = mode if mode != "none" else "free"

  return PadTeleop(device="cuda:0", cfg_hook=hook)


def press_until_latched(t, aff, max_steps: int = 300) -> bool:
  t.home()
  _, _, n_hat = t._engage_frame_world()
  t._hover = None
  f_in = 20.0 * (-n_hat)
  for _ in range(max_steps):
    t._control_tick(f_in)
    if bool(aff.engaged[0, 0].item()):
      return True
  return False


def force_release(t, aff):
  f = 100.0
  while bool(aff.engaged[0, 0].item()) and f < 900.0:
    f += 25.0
    for _ in range(3):
      r = t.apply((0.0, 0.0, f))
      if r.released:
        return


def run_combo(t, aff, ex, window: int, grace_load) -> dict:
  n_out = (-aff._wall_inward_normal()[0]).cpu().numpy()
  t_hat, _, _ = t._engage_frame_world()
  engaged_n = survived = 0
  drifts_total, drifts_out, drifts_down = [], [], []
  for _ in range(CYCLES):
    if not press_until_latched(t, aff):
      continue
    engaged_n += 1
    anchor0 = ex.anchor_pos_w[0, 0].cpu().numpy().copy()
    ok = True
    for _ in range(window):
      r = t.apply(grace_load)
      if r.released:
        ok = False
        break
    d = (ex.anchor_pos_w[0, 0].cpu().numpy() - anchor0) * 1000
    drifts_total.append(float(np.linalg.norm(d)))
    drifts_out.append(float(d @ n_out))
    drifts_down.append(float(d @ np.asarray(t_hat)))
    if ok:
      for _ in range(HOLD_STEPS):
        r = t.apply((60.0, 0.0, 15.0))
        if r.released:
          ok = False
          break
    survived += int(ok)
    force_release(t, aff)
  return {
    "engaged": engaged_n,
    "survived": survived,
    "total": (np.median(drifts_total), np.max(drifts_total))
    if drifts_total
    else (0, 0),
    "out": (np.median(drifts_out), np.max(drifts_out))
    if drifts_out
    else (0, 0),
    "down": (np.median(drifts_down), np.max(drifts_down))
    if drifts_down
    else (0, 0),
  }


def main():
  print("== grid: 30 N pull-off tension during grace (the press->hang flip) ==")
  print(
    f"{'win':>4} {'mode':>5} | {'eng':>5} {'srv':>5} | "
    f"{'total mm p50/max':>16} | {'outward mm p50/max':>18}"
  )
  for window in (14, 8, 6):
    for mode in ("free", "down", "none"):
      t = make_bench(window, mode)
      r = run_combo(
        t,
        t.affordance,
        t.executor,
        window,
        (0.0, 0.0, 30.0),
      )
      print(
        f"{window:>4} {mode:>5} | {r['engaged']:>2}/{CYCLES:<2} "
        f"{r['survived']:>2}/{CYCLES:<2} | "
        f"{r['total'][0]:>7.1f}/{r['total'][1]:<7.1f} | "
        f"{r['out'][0]:>8.1f}/{r['out'][1]:<8.1f}",
        flush=True,
      )
      t.env.close()

  print("== sanity: 60 N down-shear during grace, window 8, mode down ==")
  t = make_bench(8, "down")
  r = run_combo(t, t.affordance, t.executor, 8, (60.0, 0.0, 0.0))
  print(
    f"  engaged {r['engaged']}/{CYCLES} survived {r['survived']}/{CYCLES} "
    f"down-drift p50/max = {r['down'][0]:.1f}/{r['down'][1]:.1f} mm "
    f"(cap = 8 mm), outward p50 = {r['out'][0]:.1f} mm"
  )
  t.env.close()


if __name__ == "__main__":
  main()
