"""Verify and plot the Wang-2017 microspine limit surface against the paper.

Rebuilds the normalized capacity surface from ``MicrospineLimitSurfaceCfg``
defaults (or the climb task's overrides with --climb), checks its anchor
ratios against the single-tile 3D limit surface of Wang, Jiang & Cutkosky
(IJRR 2017, Fig 8), and renders the two cross-sections plus the full LUT.

Run inside the env_mjlab environment:
  python scripts/plot_wang_limit_surface.py --out wang_surface.png
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from scr_phybot_mjab.tools.microspine.wang_limit_surface import (
  WangPadCfg,
  build_capacity_lut,
  capacity_ratio,
  climb_wang_pad_cfg,
)

# Paper Fig 8 anchor ratios (single tile, normalized by the pure-shear
# capacity F(0)) with loose read-off-the-figure tolerances.
ANCHORS = {
  "pure shear": (1.0, 0.01),
  "peak tension ratio": (0.14, 0.05),
  "peak tension elevation [deg]": (16.0, 6.0),
  "pure lateral": (0.56, 0.15),
  "pure reverse": (0.05, 0.04),
  "pure tension": (0.0, 0.03),
}


def surface_metrics(lut: torch.Tensor) -> dict[str, float]:
  def r(ft: float, fl: float, fn: float) -> float:
    return float(capacity_ratio(lut, torch.tensor(ft), torch.tensor(fl), torch.tensor(fn)))

  phis = [p * 0.5 for p in range(181)]
  fwd = [r(math.cos(math.radians(p)), 0.0, math.sin(math.radians(p))) for p in phis]
  z = [ri * math.sin(math.radians(p)) for ri, p in zip(fwd, phis)]
  i_peak = max(range(len(z)), key=z.__getitem__)
  return {
    "pure shear": fwd[0],
    "peak tension ratio": z[i_peak],
    "peak tension elevation [deg]": phis[i_peak],
    "pure lateral": r(0.0, 1.0, 0.0),
    "pure reverse": r(-1.0, 0.0, 0.0),
    "pure tension": r(0.0, 0.0, 1.0),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default="docs/wang2017_limit_surface.png")
  parser.add_argument(
    "--climb",
    action="store_true",
    help="Use the climb pad instead of WangPadCfg defaults.",
  )
  args = parser.parse_args()
  cfg = climb_wang_pad_cfg() if args.climb else WangPadCfg()

  lut = build_capacity_lut(cfg, "cpu")
  metrics = surface_metrics(lut)

  ok = True
  print("Wang-2017 surface anchors vs paper Fig 8 (normalized by F0):")
  for name, (target, tol) in ANCHORS.items():
    got = metrics[name]
    good = abs(got - target) <= tol
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got:.3f} (paper ~{target:g} ± {tol:g})")

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  model_c, anchor_c, ink = "#2563eb", "#ea580c", "#475569"
  fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

  # --- t-n cross-section (rake shear vs tension), both azimuth halves ------
  ax = axes[0]
  for theta_az in (0.0, 180.0):
    phis = np.linspace(0.0, 90.0, 361)
    sign = 1.0 if theta_az == 0.0 else -1.0
    rr = np.array(
      [
        float(
          capacity_ratio(
            lut,
            torch.tensor(sign * math.cos(math.radians(p))),
            torch.tensor(0.0),
            torch.tensor(math.sin(math.radians(p))),
          )
        )
        for p in phis
      ]
    )
    ax.plot(
      sign * rr * np.cos(np.radians(phis)),
      rr * np.sin(np.radians(phis)),
      color=model_c,
      lw=2,
      label="model boundary" if theta_az == 0.0 else None,
    )
  paper_tn = [(1.0, 0.0), (0.46, 0.13), (0.0, 0.0), (-0.045, 0.0)]
  ax.scatter(*zip(*paper_tn), color=anchor_c, zorder=3, s=45, label="paper anchors")
  ax.axhline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.axvline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.set_xlabel("f_t / F0  (rake shear +)")
  ax.set_ylabel("f_n / F0  (tension +)")
  ax.set_title("t-n plane (Fig 8 bottom-left)")
  ax.set_aspect("equal")
  ax.legend(frameon=False)
  ax.grid(alpha=0.25)

  # --- in-plane cross-section (rake shear vs lateral) at zero tension ------
  ax = axes[1]
  thetas = np.linspace(0.0, 180.0, 361)
  rr = np.array(
    [
      float(
        capacity_ratio(
          lut,
          torch.tensor(math.cos(math.radians(t))),
          torch.tensor(math.sin(math.radians(t))),
          torch.tensor(0.0),
        )
      )
      for t in thetas
    ]
  )
  x = rr * np.cos(np.radians(thetas))
  y = rr * np.sin(np.radians(thetas))
  ax.plot(x, y, color=model_c, lw=2, label="model boundary")
  ax.plot(x, -y, color=model_c, lw=2)
  paper_tl = [(1.0, 0.0), (0.0, 0.56), (0.0, -0.56), (-0.045, 0.0)]
  ax.scatter(*zip(*paper_tl), color=anchor_c, zorder=3, s=45, label="paper anchors")
  ax.axhline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.axvline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.set_xlabel("f_t / F0  (rake shear +)")
  ax.set_ylabel("f_l / F0  (lateral)")
  ax.set_title("wall plane, zero tension (Fig 8 top-left)")
  ax.set_aspect("equal")
  ax.legend(frameon=False)
  ax.grid(alpha=0.25)

  # --- full normalized surface -------------------------------------------
  ax = axes[2]
  im = ax.imshow(
    lut.numpy().T,
    origin="lower",
    aspect="auto",
    extent=(0.0, 180.0, 0.0, 90.0),
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
  )
  fig.colorbar(im, ax=ax, label="R (capacity / F0)")
  ax.set_xlabel("azimuth from rake [deg]  (180 = against rake)")
  ax.set_ylabel("elevation toward tension [deg]")
  ax.set_title("normalized capacity R(theta, phi)")

  fig.suptitle(
    "Microspine limit surface — Wang et al. 2017 model "
    f"(mu={cfg.wang_mu_tip:g}, dpsi={cfg.wang_delta_psi_deg:g}deg, "
    f"beta={cfg.wang_beta_deg:g}deg, alpha={cfg.wang_alpha_deg:g}deg, "
    f"c_i={cfg.wang_c_i:g}, mu_s={cfg.wang_mu_channel:g})"
  )
  fig.tight_layout()
  fig.savefig(args.out, dpi=140)
  print(f"figure saved to {args.out}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
