"""Sequential-action physics test on the Rizon4 bench (wang2017 + retraction).

A. Cycle endurance: engage -> hold -> ramp to release -> re-engage, x12,
   a fresh pull direction per cycle; every release must land on the surface
   and every re-engagement must succeed with clean retract/restore state.
B. Continuous direction sweep: latched under constant |F|=100 N rotating in
   the wall plane; release must occur at the azimuth where the envelope
   crosses 100 N (predicted from the LUT), not before/after.
C. Command jitter: C1 oscillating commands below capacity -> zero releases;
   C2 a sustained spike above capacity -> releases within the debounce.
"""
import numpy as np, torch
from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop
from scr_phybot_mjab.microspine.wang_limit_surface import capacity_ratio

t = PadTeleop(device="cuda:0")
lut = torch.tensor(t.limit_params["lut"])
F0 = t.limit_params["f0"]; RFLOOR = t.limit_params["r_floor"]

def charge_anchor(t, steps=15, shear=40.0):
  """Recruitment-era seating: a fresh latch holds only the ramp floor until
  shear-loaded (motionref14). Charge it so threshold tests probe the MATURE
  envelope, which is what the fig8/LUT predictions describe."""
  for _ in range(steps):
    r = t.apply((shear, 0.0, 0.0))
    if r.released:
      return False
  return True


def cap_along(d):
  r = float(capacity_ratio(lut, torch.tensor(d[0]), torch.tensor(d[1]), torch.tensor(d[2])))
  return F0 * max(r, RFLOOR)

def sphere_state():
  import warp as wp
  pos = wp.to_torch(t.env.sim.wp_model.geom_pos)
  gids = t.affordance._sphere_geom_ids
  off = (pos[0, gids] - t.affordance._sphere_pos_default).norm(dim=-1)
  return float(off.max())

rng = np.random.default_rng(3)
ok = True
print("== A. cycle endurance ==")
ratios, engage_fail = [], 0
for cyc in range(12):
  t.press_engage()
  if not bool(t.affordance.engaged[0, 0].item()):
    t.home(); t.press_engage()
    if not bool(t.affordance.engaged[0, 0].item()):
      engage_fail += 1; continue
  if not charge_anchor(t):
    engage_fail += 1; continue
  retract_ok = sphere_state() > 0.04
  margins = []
  for _ in range(30):
    r = t.apply((0.0, 0.0, 0.0)); margins.append(r.margin)
  smooth = min(margins) > 0 and int((np.sign(margins)[1:] != np.sign(margins)[:-1]).sum()) == 0
  d = rng.normal(size=3); d[2] = abs(d[2]); d /= np.linalg.norm(d)
  cap = cap_along(d)
  released = None
  f, df = 0.0, max(1.0, cap / 50.0)
  while f < 1.6 * cap and released is None:
    f += df
    for _ in range(4):
      r = t.apply(tuple(f * d))
      if r.released:
        released = r.load_tln.copy(); break
  if released is None:
    print(f"  cyc{cyc}: NO RELEASE up to 1.6x cap -> FAIL"); ok = False; continue
  mag = float(np.linalg.norm([released[0], released[1], max(released[2], 0)]))
  ratio = mag / cap_along(released / max(np.linalg.norm(released), 1e-9))
  restored = None
  for _ in range(12):
    t.apply((0.0, 0.0, 0.0))
  restored = sphere_state() < 1e-6
  ratios.append(ratio)
  flag = retract_ok and smooth and restored and 0.75 <= ratio <= 1.3
  ok &= flag
  print(f"  cyc{cyc}: cap={cap:5.0f}N release|F|={mag:5.0f}N ratio={ratio:4.2f} "
        f"retract={retract_ok} smooth={smooth} restored={restored} -> {'OK' if flag else 'FAIL'}")
print(f"  engage failures: {engage_fail}/12, ratio p50={np.median(ratios):.2f}")
ok &= engage_fail == 0

print("== B. continuous rotation at |F|=100 N ==")
th = np.linspace(0, 180, 361)
rr = np.array([cap_along((np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0)) for a in th])
pred = float(th[np.argmax(rr < 100.0)])
t.home(); t.press_engage()
charge_anchor(t)
rel_az = None
for a in np.arange(0.0, 181.0, 0.25):
  F = (100.0 * np.cos(np.radians(a)), 100.0 * np.sin(np.radians(a)), 0.0)
  for _ in range(2):
    r = t.apply(F)
    if r.released:
      rel_az = a; break
  if rel_az is not None: break
print(f"  predicted crossing {pred:.0f} deg, released at {rel_az} deg")
b_ok = rel_az is not None and abs(rel_az - pred) <= 12.0
ok &= b_ok
print(f"  -> {'OK' if b_ok else 'FAIL'}")

print("== C. command jitter ==")
t.home(); t.press_engage()
charge_anchor(t)
releases = 0
for i in range(300):
  lat = 80.0 if (i // 5) % 2 == 0 else -80.0
  r = t.apply((150.0, lat, 0.0))
  releases += int(r.released)
c1 = releases == 0
print(f"  C1 below-capacity jitter 300 steps: releases={releases} -> {'OK' if c1 else 'FAIL'}")
ok &= c1
spike_step = None
for i in range(50):
  F = (100.0, 0.0, 100.0) if i >= 10 else (100.0, 0.0, 5.0)
  r = t.apply(F)
  if r.released:
    spike_step = i - 10; break
c2 = spike_step is not None and spike_step <= 6
print(f"  C2 sustained over-capacity spike: released {spike_step} steps after spike onset "
      f"-> {'OK' if c2 else 'FAIL'}")
ok &= c2
print("SEQUENCE BENCH:", "PASS" if ok else "FAIL")
