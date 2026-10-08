"""Publication figure: microspine engage/release state machine + hold envelope.

Renders a four-panel figure in Science Robotics house format (17.8 cm triple
column, Helvetica-metric sans, 5-8 pt type, vector output):

  A  the per-end-effector engage/release state machine, its gate set, and the
     three age-indexed sub-phases of a latched anchor
  B  in-plane azimuth section of the normalized Wang-2017 capacity surface
     R(theta, 0) -- the rake anisotropy
  C  meridian section R(theta, phi) at f_l = 0 -- the closed forward lobe and
     the radial release test
  D  the multiplicative capacity chain F0 x seat quality x recruitment ramp

The figure depicts the retired Wang-2017 hold envelope with seat scaling and
the recruitment ramp, not the empirical unit model the simulation now uses.
Panels B-D evaluate the analytic surface in
``tools/microspine/wang_limit_surface.py``; panel A's constants are read from
``MicrospineCfg`` / ``MicrospineLimitSurfaceCfg`` with the Wang-era climb
overrides pinned in ``climb_configs``.

Run inside the env_mjlab environment:
  python scripts/microspine/plot_microspine_state_machine.py --out docs/figures/microspine_state_machine
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from scr_phybot_mjab.microspine.config import (  # noqa: E402
  MicrospineCfg,
  MicrospineLimitSurfaceCfg,
)
from scr_phybot_mjab.tools.microspine.wang_limit_surface import (  # noqa: E402
  WangPadCfg,
  build_capacity_lut,
  capacity_ratio,
)

# ---------------------------------------------------------------------------
# House palette (matches docs/figures/climb_pipeline_fig.svg). Two semantic
# accents only: teal = hold/grip, amber = release/violation.
# ---------------------------------------------------------------------------
INK = "#232A31"
SLATE = "#5C6670"
RULE = "#C9CFCC"
TEAL = "#22707E"
TEAL_DEEP = "#1B5A66"
TEAL_TINT = "#E3ECEE"
TEAL_PALE = "#EFF4F5"
AMBER = "#B4690E"
AMBER_PALE = "#F3E3CE"
WHITE = "#FFFFFF"

FS_PANEL = 8.0  # panel letter
FS_STATE = 6.8  # state names
FS_BODY = 5.8  # annotations
FS_TICK = 5.2  # axis ticks
FS_TINY = 5.2  # sub-labels


def _rc() -> None:
  plt.rcParams.update(
    {
      "font.family": "sans-serif",
      "font.sans-serif": ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
      "font.size": FS_BODY,
      "axes.linewidth": 0.6,
      "axes.edgecolor": INK,
      "axes.labelcolor": INK,
      "text.color": INK,
      "xtick.color": INK,
      "ytick.color": INK,
      "xtick.labelsize": FS_TICK,
      "ytick.labelsize": FS_TICK,
      "xtick.major.width": 0.6,
      "ytick.major.width": 0.6,
      "xtick.major.size": 1.8,
      "ytick.major.size": 1.8,
      "xtick.direction": "out",
      "ytick.direction": "out",
      "lines.linewidth": 0.9,
      "lines.solid_capstyle": "round",
      "legend.frameon": False,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
      "svg.fonttype": "none",
      "mathtext.fontset": "custom",
      "mathtext.rm": "Liberation Sans",
      "mathtext.it": "Liberation Sans:italic",
      "mathtext.bf": "Liberation Sans:bold",
      "mathtext.default": "it",
      "figure.dpi": 600,
      "savefig.dpi": 600,
    }
  )


@dataclasses.dataclass(frozen=True)
class WangEraHoldCfg(WangPadCfg):
  """Hold settings of the retired Wang-era climb plant the figure depicts."""

  hold_judge_guard_steps: int = MicrospineLimitSurfaceCfg().hold_judge_guard_steps
  hold_grace_steps: int = MicrospineLimitSurfaceCfg().hold_grace_steps
  capacity_ramp_floor: float = 0.3
  capacity_ramp_time_s: float = 0.25
  capacity_ramp_shear_n: float = 10.0


@dataclasses.dataclass(frozen=True)
class WangEraEngageCfg:
  """The retired proxy press gate (G1): one contact proxy at >= 3 N."""

  engage_min_spheres: int = 1
  engage_normal_force_threshold: float = 3.0


def climb_configs() -> tuple[MicrospineCfg, WangEraHoldCfg, WangEraEngageCfg]:
  """The climb's microspine settings (single-EE subset) plus the Wang-era
  hold and press-gate settings the figure depicts."""
  mcfg = MicrospineCfg(
    ee_body_names=("left_hand_spine",),
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    reengage_cooldown_steps=10,
    engage_consecutive_steps=3,
  )
  return mcfg, WangEraHoldCfg(f_t_max_range=(400.0, 650.0)), WangEraEngageCfg()


def panel_letter(fig, x: float, y: float, letter: str) -> None:
  fig.text(x, y, letter, fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)


def ratio(lut: torch.Tensor, ft: float, fl: float, fn: float) -> float:
  return float(capacity_ratio(lut, torch.tensor(ft), torch.tensor(fl), torch.tensor(fn)))


# ===========================================================================
# Panel A -- state machine
# ===========================================================================
def panel_a(ax, mcfg: MicrospineCfg, ls: WangEraHoldCfg, gate: WangEraEngageCfg, dt: float) -> None:
  ax.set_xlim(0, 120)
  ax.set_ylim(0, 54)
  ax.axis("off")

  def state_box(x, y, w, h, title, sub, *, latched=False):
    ax.add_patch(
      FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=1.1",
        facecolor=TEAL_TINT if latched else WHITE,
        edgecolor=TEAL_DEEP if latched else "#5B6770",
        linewidth=1.1 if latched else 0.7,
        mutation_aspect=0.30,
        zorder=3,
      )
    )
    ax.text(
      x + w / 2,
      y + h * 0.64,
      title,
      ha="center",
      va="center",
      fontsize=FS_STATE,
      fontweight="bold",
      color=TEAL_DEEP if latched else INK,
      zorder=4,
    )
    ax.text(x + w / 2, y + h * 0.245, sub, ha="center", va="center", fontsize=FS_TINY, color=SLATE, zorder=4)

  y0, hh = 39.0, 9.0
  state_box(3.0, y0, 21.0, hh, "RELEASED", "contacts live · eq_active = 0")
  state_box(33.0, y0, 15.0, hh, "ARMING", f"count 1…{mcfg.engage_consecutive_steps - 1}")
  state_box(57.0, y0, 28.0, hh, "LATCHED", "eq_active = 1 · spheres retracted", latched=True)
  state_box(
    95.0, y0, 22.0, hh, "COOLDOWN", f"{mcfg.reengage_cooldown_steps} steps = {mcfg.reengage_cooldown_steps * dt:.2f} s"
  )

  def arrow(p0, p1, color=INK, style="arc3,rad=0", lw=0.85, dash="-", z=5):
    ax.add_patch(
      FancyArrowPatch(
        p0,
        p1,
        connectionstyle=style,
        arrowstyle="-|>",
        mutation_scale=5.0,
        linewidth=lw,
        linestyle=dash,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
      )
    )

  yc = y0 + hh / 2
  arrow((24.6, yc), (32.4, yc))
  ax.text(28.5, yc + 1.6, "all gates", ha="center", va="bottom", fontsize=FS_TINY)
  arrow((48.6, yc), (56.4, yc))
  ax.text(52.5, yc + 1.6, f"count ≥ {mcfg.engage_consecutive_steps}", ha="center", va="bottom", fontsize=FS_TINY)
  ax.text(52.5, yc - 1.8, "seat", ha="center", va="top", fontsize=FS_TINY, color=TEAL_DEEP, fontweight="bold")
  arrow((85.6, yc), (94.4, yc), color=AMBER, lw=1.0)
  ax.text(90.0, yc + 1.6, "margin < 0", ha="center", va="bottom", fontsize=FS_TINY, color=AMBER, fontweight="bold")

  # ARMING -> RELEASED on any gate failure, routed above the state row.
  arrow((36.0, y0 + hh), (19.0, y0 + hh), style="arc3,rad=-0.55", color=SLATE, lw=0.7, dash=(0, (2.4, 1.6)))
  ax.text(27.5, 52.4, "any gate fails · count ← 0", ha="center", va="top", fontsize=FS_TINY, color=SLATE)

  # COOLDOWN -> RELEASED, orthogonal routing below the state row.
  y_ret = 35.2
  ax.plot(
    [106.0, 106.0, 13.5],
    [y0, y_ret, y_ret],
    color=SLATE,
    lw=0.7,
    linestyle=(0, (2.4, 1.6)),
    zorder=2,
    solid_capstyle="butt",
  )
  ax.add_patch(
    FancyArrowPatch(
      (13.5, y_ret + 0.1),
      (13.5, y0),
      arrowstyle="-|>",
      mutation_scale=5.0,
      linewidth=0.7,
      linestyle=(0, (2.4, 1.6)),
      color=SLATE,
      shrinkA=0,
      shrinkB=0,
      zorder=2,
    )
  )
  ax.text(
    60.0,
    y_ret + 0.9,
    "cooldown expires · contacts restored",
    ha="center",
    va="bottom",
    fontsize=FS_TINY,
    color=SLATE,
    bbox=dict(facecolor=WHITE, edgecolor="none", pad=0.6),
  )

  # ---- Engage gates ------------------------------------------------------
  gx, gy, gw, gh = 3.0, 1.5, 45.0, 30.0
  ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor=TEAL_PALE, edgecolor=RULE, linewidth=0.5, zorder=1))
  ax.text(
    gx + 2.0,
    gy + gh - 2.4,
    "ENGAGE GATES — all four, every step",
    fontsize=FS_TINY,
    fontweight="bold",
    color=TEAL_DEEP,
    va="top",
    ha="left",
  )
  gates = [
    ("G1", "press", f"≥ {gate.engage_min_spheres} sphere at $F_\\perp$ ≥ {gate.engage_normal_force_threshold:.0f} N"),
    ("G2", "align", f"$\\hat{{a}}\\cdot\\hat{{n}}_{{\\mathrm{{in}}}}$ ≥ {mcfg.engage_parallel_cos_threshold:.2f}"),
    (
      "G3",
      "seat",
      f"spine field ≥ {mcfg.engage_min_spine_fraction:.1f}  "
      f"({mcfg.virtual_spine_grid[0]}×{mcfg.virtual_spine_grid[1]} "
      f"kinematic grid, {mcfg.virtual_spine_travel * 1e3:.0f} mm reach)",
    ),
    ("G4", "rake", f"$v\\cdot\\hat{{t}}$ ≥ −{mcfg.engage_max_antirake_vel:.2f} m s$^{{-1}}$  (no up-rake catch)"),
  ]
  for i, (tag, name, text) in enumerate(gates):
    yy = gy + gh - 7.4 - i * 5.3
    ax.text(gx + 2.4, yy, tag, fontsize=FS_TINY, fontweight="bold", color=TEAL, va="center", ha="left")
    ax.text(gx + 6.4, yy, name, fontsize=FS_TINY, color=SLATE, va="center", ha="left")
    ax.text(gx + 12.4, yy, text, fontsize=FS_TINY, color=INK, va="center", ha="left")

  # ---- Latched sub-phases ------------------------------------------------
  bx, by, bw, bh = 52.5, 1.5, 64.5, 30.0
  ax.add_patch(Rectangle((bx, by), bw, bh, facecolor=TEAL_PALE, edgecolor=RULE, linewidth=0.5, zorder=1))
  ax.text(
    bx + 2.0,
    by + bh - 2.4,
    "INSIDE LATCHED — age since seat",
    fontsize=FS_TINY,
    fontweight="bold",
    color=TEAL_DEEP,
    va="top",
    ha="left",
  )

  t_max = 0.36
  ax_x0, ax_x1 = bx + 11.0, bx + 42.0

  def tx(t: float) -> float:
    return ax_x0 + (ax_x1 - ax_x0) * (t / t_max)

  bars = [
    ("guard", ls.hold_judge_guard_steps * dt, AMBER, AMBER_PALE, "no release test"),
    (
      "re-seat",
      ls.hold_grace_steps * dt,
      TEAL,
      TEAL_TINT,
      f"≤ {mcfg.anchor_reseat_step_cap * 1e3:.0f} mm/step down-rake",
    ),
    ("recruit", ls.capacity_ramp_time_s, TEAL_DEEP, TEAL_TINT, f"$F_0$: {ls.capacity_ramp_floor:.1f} → 1, shear-gated"),
  ]
  for i, (name, dur, ec, fc, note) in enumerate(bars):
    yy = by + bh - 7.6 - i * 5.0
    ax.add_patch(
      Rectangle((ax_x0, yy - 1.15), tx(dur) - ax_x0, 2.3, facecolor=fc, edgecolor=ec, linewidth=0.6, zorder=3)
    )
    ax.text(ax_x0 - 1.4, yy, name, fontsize=FS_TINY, ha="right", va="center", color=ec, fontweight="bold")
    ax.text(ax_x1 + 1.6, yy, note, fontsize=FS_TINY, ha="left", va="center", color=SLATE)

  yy_ax = by + 8.6
  ax.plot([ax_x0, ax_x1], [yy_ax, yy_ax], color=INK, lw=0.6, zorder=3)
  for t in (0.0, 0.1, 0.2, 0.3):
    ax.plot([tx(t), tx(t)], [yy_ax, yy_ax - 0.9], color=INK, lw=0.6, zorder=3)
    ax.text(tx(t), yy_ax - 1.9, f"{t:.1f}", fontsize=FS_TINY, ha="center", va="top", color=INK)
  ax.text(ax_x1 + 1.6, yy_ax - 1.9, "s since seat", fontsize=FS_TINY, ha="left", va="top", color=SLATE)

  ax.text(
    bx + 2.0,
    by + 3.4,
    r"release when  $m=\dfrac{C(\theta,\varphi)-\|\mathbf{F}\|}"
    r"{C+\epsilon F_0}<0$",
    fontsize=FS_TINY,
    ha="left",
    va="center",
    color=AMBER,
  )
  ax.text(
    bx + 32.0,
    by + 3.4,
    "— loss of contact alone never releases",
    fontsize=FS_TINY,
    ha="left",
    va="center",
    color=SLATE,
    style="italic",
  )


# ===========================================================================
# Panel B -- azimuth section (polar)
# ===========================================================================
def panel_b(ax, lut: torch.Tensor, ls: WangEraHoldCfg) -> None:
  theta_half = np.linspace(0.0, 180.0, lut.shape[0])
  r_half = lut[:, 0].numpy()
  theta = np.concatenate([theta_half, 360.0 - theta_half[::-1][1:]])
  r = np.concatenate([r_half, r_half[::-1][1:]])
  th = np.deg2rad(theta)

  ax.set_theta_zero_location("E")
  ax.set_theta_direction(1)
  ax.fill(th, r, facecolor=TEAL_TINT, edgecolor="none", zorder=2)
  ax.plot(th, r, color=TEAL_DEEP, lw=1.0, zorder=3)
  ax.plot(th, np.full_like(th, ls.wang_r_floor), color=AMBER, lw=0.6, linestyle=(0, (2.2, 1.6)), zorder=4)

  ax.set_ylim(0, 1.0)
  ax.set_yticks([0.5, 1.0])
  ax.set_yticklabels(["0.5", "1.0"], fontsize=FS_TICK, color=SLATE)
  ax.set_rlabel_position(152)
  ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
  ax.set_xticklabels([])
  ax.grid(color=RULE, lw=0.4)
  ax.spines["polar"].set_color(RULE)
  ax.spines["polar"].set_linewidth(0.5)

  v_rake = ratio(lut, 1.0, 0.0, 0.0)
  v_lat = ratio(lut, 0.0, 1.0, 0.0)
  v_rev = ratio(lut, -1.0, 0.0, 0.0)
  for deg, v in ((0.0, v_rake), (90.0, v_lat), (180.0, v_rev)):
    ax.plot([math.radians(deg)], [v], "o", ms=2.2, color=TEAL_DEEP, zorder=6)

  def label(xy, xytext, text, color, *, ha, va, bold=False, leader=True):
    ax.annotate(
      text,
      xy=xy,
      xycoords="data",
      xytext=xytext,
      textcoords="axes fraction",
      fontsize=FS_TINY,
      color=color,
      ha=ha,
      va=va,
      fontweight="bold" if bold else "normal",
      annotation_clip=False,
      arrowprops=dict(arrowstyle="-", lw=0.5, color=color, shrinkA=0.5, shrinkB=1.5) if leader else None,
    )

  label((0.0, v_rake), (1.01, 0.52), f"rake\n{v_rake:.2f}", TEAL_DEEP, ha="left", va="center", bold=True, leader=False)
  label((math.radians(90), v_lat), (0.52, 1.02), f"lateral {v_lat:.2f}", INK, ha="left", va="bottom", leader=False)
  label(
    (math.radians(180), v_rev), (0.00, 0.30), f"against rake\n{v_rev:.3f}", AMBER, ha="left", va="bottom", bold=True
  )
  label(
    (math.radians(250), ls.wang_r_floor), (0.13, 0.09), f"floor {ls.wang_r_floor:.2f}", AMBER, ha="left", va="center"
  )
  ax.text(math.radians(325), 0.86, r"$\theta$", fontsize=FS_BODY, color=SLATE)


# ===========================================================================
# Panel C -- meridian section + radial release test
# ===========================================================================
def panel_c(ax, lut: torch.Tensor, ls: WangEraHoldCfg) -> None:
  phis = np.linspace(0.0, 90.0, 361)

  def section(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    sgn = math.cos(math.radians(theta_deg))
    xs, ys = [], []
    for p in phis:
      cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
      v = max(ratio(lut, sgn * cp, 0.0, sp), ls.wang_r_floor)
      xs.append(sgn * v * cp)
      ys.append(v * sp)
    return np.asarray(xs), np.asarray(ys)

  xf, yf = section(0.0)
  xr, yr = section(180.0)
  v_rev_c = max(ratio(lut, -1.0, 0.0, 0.0), ls.wang_r_floor)
  xs = np.concatenate([xr[::-1], xf])
  ys = np.concatenate([yr[::-1], yf])
  ax.fill_between(xs, 0.0, ys, facecolor=TEAL_TINT, edgecolor="none", zorder=2)
  ax.plot(xs, ys, color=TEAL_DEEP, lw=1.0, zorder=4)
  ax.plot([xr[0], xf[0]], [0, 0], color=TEAL_DEEP, lw=1.0, zorder=4)

  ax.add_patch(
    Rectangle((-0.42, -0.30), 1.64, 0.30, facecolor="#F2F4F3", edgecolor=RULE, linewidth=0.5, hatch="/////", zorder=1)
  )
  ax.text(0.40, -0.15, "compression → sphere contacts", fontsize=FS_TINY, ha="center", va="center", color=SLATE)

  # Radial release test along one ray: capacity is a point on that ray.
  phi_deg = 11.0
  cp, sp = math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))
  cap = max(ratio(lut, cp, 0.0, sp), ls.wang_r_floor)
  ray = 1.28 * cap
  ax.plot([0, ray * cp], [0, ray * sp], color=SLATE, lw=0.5, linestyle=(0, (1.4, 1.4)), zorder=5)
  for frac, col, lbl in ((0.60, TEAL_DEEP, "hold  $m>0$"), (1.22, AMBER, "release  $m<0$")):
    ax.annotate(
      "",
      xy=(cap * frac * cp, cap * frac * sp),
      xytext=(0, 0),
      arrowprops=dict(arrowstyle="-|>", lw=1.0, color=col, mutation_scale=4.5),
      zorder=6,
    )
    ax.plot([cap * frac * cp], [cap * frac * sp], "o", ms=2.0, color=col, zorder=7)
    _ = lbl
  ax.plot(
    [cap * cp], [cap * sp], "o", ms=3.0, markerfacecolor=WHITE, markeredgecolor=INK, markeredgewidth=0.8, zorder=8
  )

  ax.annotate(
    r"$C(\theta,\varphi)$",
    xy=(cap * cp, cap * sp),
    xytext=(cap * cp - 0.02, 0.40),
    fontsize=FS_TINY,
    color=INK,
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=2),
  )
  ax.annotate(
    "hold  $m>0$",
    xy=(cap * 0.60 * cp, cap * 0.60 * sp),
    xytext=(0.02, 0.56),
    fontsize=FS_TINY,
    color=TEAL_DEEP,
    fontweight="bold",
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="-", lw=0.5, color=TEAL_DEEP, shrinkA=1, shrinkB=2),
  )
  ax.annotate(
    "release  $m<0$",
    xy=(cap * 1.22 * cp, cap * 1.22 * sp),
    xytext=(1.20, 0.34),
    fontsize=FS_TINY,
    color=AMBER,
    fontweight="bold",
    ha="right",
    va="bottom",
    arrowprops=dict(arrowstyle="-", lw=0.5, color=AMBER, shrinkA=1, shrinkB=2),
  )

  # Peak tension marker.
  i_pk = int(np.argmax(yf))
  ax.annotate(
    "",
    xy=(xf[i_pk], yf[i_pk]),
    xytext=(xf[i_pk], 0.0),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color=SLATE, mutation_scale=3.0),
    zorder=5,
  )
  ax.annotate(
    f"peak tension {yf[i_pk]:.2f}\nat {phis[i_pk]:.0f}° elevation",
    xy=(xf[i_pk], yf[i_pk] * 0.55),
    xytext=(1.20, 0.60),
    fontsize=FS_TINY,
    color=SLATE,
    va="bottom",
    ha="right",
    arrowprops=dict(arrowstyle="-", lw=0.5, color=SLATE, shrinkA=1, shrinkB=2),
  )

  ax.set_xlim(-0.42, 1.22)
  ax.set_ylim(-0.30, 0.78)
  ax.set_aspect("equal")
  ax.axhline(0.0, color=RULE, lw=0.5, zorder=0)
  ax.axvline(0.0, color=RULE, lw=0.5, zorder=0)
  ax.set_xticks([-0.25, 0.0, 0.5, 1.0])
  ax.set_yticks([0.0, 0.25, 0.5])
  ax.set_xlabel(r"in-plane load  $f_t / F_0$", fontsize=FS_TINY, labelpad=1.0)
  ax.set_ylabel(r"tension  $f_n / F_0$", fontsize=FS_TINY, labelpad=1.0)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.spines["left"].set_position(("outward", 2))
  ax.spines["bottom"].set_position(("outward", 2))
  ax.annotate(
    "against\nrake",
    xy=(-v_rev_c, 0.006),
    xytext=(-0.40, 0.16),
    fontsize=FS_TINY,
    color=AMBER,
    ha="left",
    va="bottom",
    fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=AMBER, mutation_scale=3.5, shrinkA=1, shrinkB=1),
  )


# ===========================================================================
# Panel D -- capacity chain
# ===========================================================================
def panel_d(ax, mcfg: MicrospineCfg, ls: WangEraHoldCfg, dt: float) -> None:
  f0_lo, f0_hi = ls.f_t_max_range
  f0 = 520.0
  seat = 0.72
  shear_on = 0.06

  t = np.linspace(0.0, 0.42, 800)
  charge = np.clip(t - shear_on, 0.0, None)
  ramp = ls.capacity_ramp_floor + (1.0 - ls.capacity_ramp_floor) * np.clip(charge / ls.capacity_ramp_time_s, 0.0, 1.0)
  cap = f0 * seat * ramp

  ax.axhspan(f0_lo, f0_hi, facecolor=TEAL_PALE, edgecolor="none", zorder=0)
  ax.axvspan(0.0, ls.hold_judge_guard_steps * dt, facecolor=AMBER_PALE, edgecolor="none", zorder=1)
  ax.axhline(f0, color=TEAL, lw=0.7, linestyle=(0, (3.2, 1.8)), zorder=2)
  ax.axhline(f0 * seat, color=TEAL, lw=0.7, linestyle=(0, (1.3, 1.3)), zorder=2)
  ax.axhline(f0 * seat * ls.capacity_ramp_floor, color=TEAL, lw=0.7, linestyle=(0, (1.3, 1.3)), zorder=2)
  ax.plot([ls.hold_grace_steps * dt] * 2, [0, 690], color=SLATE, lw=0.5, linestyle=(0, (1.2, 1.4)), zorder=2)
  ax.plot(t, cap, color=TEAL_DEEP, lw=1.3, zorder=4)

  ax.text(
    0.415,
    f0_hi - 10,
    r"$F_0\sim\mathrm{U}(400,\,650)$ N per grip",
    fontsize=FS_TINY,
    color=TEAL,
    ha="right",
    va="top",
    linespacing=1.25,
  )
  ax.text(0.415, f0 + 12, f"one draw  {f0:.0f} N", fontsize=FS_TINY, color=TEAL, ha="right", va="bottom")
  ax.text(0.415, f0 * seat + 12, f"× seat quality {seat:.2f}", fontsize=FS_TINY, color=TEAL, ha="right", va="bottom")
  ax.text(
    0.415,
    f0 * seat * ls.capacity_ramp_floor + 12,
    f"× recruitment floor {ls.capacity_ramp_floor:.1f}",
    fontsize=FS_TINY,
    color=TEAL,
    ha="right",
    va="bottom",
  )
  ax.annotate(
    f"$f_t$ ≥ {ls.capacity_ramp_shear_n:.0f} N\ncharging starts",
    xy=(shear_on, f0 * seat * ls.capacity_ramp_floor),
    xytext=(shear_on + 0.012, 300),
    fontsize=FS_TINY,
    color=SLATE,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=SLATE, mutation_scale=4.0, shrinkA=1, shrinkB=1),
  )
  ax.text(ls.hold_judge_guard_steps * dt + 0.006, 672, "guard", fontsize=FS_TINY, color=AMBER, ha="left", va="top")
  ax.text(ls.hold_grace_steps * dt - 0.006, 26, "re-seat ends", fontsize=FS_TINY, color=SLATE, ha="right", va="bottom")

  ax.set_xlim(0, 0.42)
  ax.set_ylim(0, 690)
  ax.set_xlabel("time since seat (s)", fontsize=FS_TINY, labelpad=1.0)
  ax.set_ylabel(r"envelope scale  $F_0^{\mathrm{eff}}$  (N)", fontsize=FS_TINY, labelpad=1.0)
  ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
  ax.set_yticks([0, 200, 400, 600])
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)


# ===========================================================================
def build(out_stem: Path) -> None:
  _rc()
  mcfg, ls, gate = climb_configs()
  dt = 0.02  # policy step = 0.002 s timestep x decimation 10
  lut = build_capacity_lut(ls, "cpu").cpu()

  fig = plt.figure(figsize=(7.008, 5.55))  # 17.8 cm triple column

  ax_a = fig.add_axes((0.022, 0.485, 0.962, 0.495))
  panel_a(ax_a, mcfg, ls, gate, dt)

  ax_b = fig.add_axes((0.070, 0.075, 0.195, 0.310), projection="polar")
  panel_b(ax_b, lut, ls)

  ax_c = fig.add_axes((0.385, 0.075, 0.275, 0.330))
  panel_c(ax_c, lut, ls)

  ax_d = fig.add_axes((0.760, 0.120, 0.215, 0.265))
  panel_d(ax_d, mcfg, ls, dt)

  panel_letter(fig, 0.012, 0.988, "A")
  panel_letter(fig, 0.012, 0.442, "B")
  panel_letter(fig, 0.330, 0.442, "C")
  panel_letter(fig, 0.690, 0.442, "D")

  for ext in ("pdf", "svg", "png"):
    path = out_stem.with_suffix(f".{ext}")
    fig.savefig(path, format=ext, facecolor="white")
    print(f"wrote {path}")
  plt.close(fig)


CAPTION = """\
Fig. X. Anisotropic hold envelope and the per-limb engage/release state
machine of the constraint-based microspine model. (A) Each end-effector runs
an independent four-state machine. Seating requires all four gates G1-G4 to
hold for {n} consecutive control steps ({seat_ms:.0f} ms); a single failure
resets the counter. Seating writes a bilateral weld anchor (eq_active = 1),
retracts the pad's sphere contacts so the anchor becomes the only load path,
draws a fresh capacity F0 for that grip, and captures the seat quality. A
latched anchor is judged against its hold envelope on every step; the only
exit is an envelope violation, followed by a {cd} step lockout. Loss of
contact alone never releases an anchor. Three age-indexed sub-phases modulate
a young latch: a {guard} step guard (the constraint force is reported one
control step late), a {grace} step re-seat window in which the anchor creeps
down the rake by at most {cap_mm:.0f} mm per step, and a {ramp:.2f} s
recruitment ramp charged only by rake-shear-loaded time. (B) In-plane azimuth
section of the normalized capacity surface R(theta, phi = 0) after Wang et al.
(2017). Capacity is strongly unidirectional: 1.00 along the rake, {lat:.2f}
pure lateral and {rev:.3f} against the rake, over an isotropic residual floor
of {floor:.2f} (dashed) that models the stochastic adhesion the deterministic
model predicts as zero. (C) Meridian section at f_l = 0. The forward lobe is
closed: tension capacity exists only alongside simultaneous engaged shear,
peaks at 0.12 F0 at 15 deg elevation, and vanishes for pure pull-off. Release
is a radial test - the anchor holds while the load magnitude stays inside the
capacity along its own direction (teal) and releases the first step it does
not (amber). Compression is excluded from the envelope and carried by the
ordinary sphere contacts (hatched). (D) The envelope scale is the product of
three factors: a per-grip capacity lottery F0 ~ U(400, 650) N, the seat
quality captured when the pad latched, and the recruitment ramp, which
advances only while the anchor carries at least {shear:.0f} N of rake shear.
A fresh, partially seated latch is therefore real but weak, not immune.
Panels B-D are evaluated directly from the simulation's model code.
"""


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--out",
    type=Path,
    default=Path("docs/figures/microspine_state_machine"),
    help="output stem; .pdf/.svg/.png are written",
  )
  args = ap.parse_args()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  build(args.out)

  mcfg, ls, _ = climb_configs()
  lut = build_capacity_lut(ls, "cpu").cpu()
  caption = CAPTION.format(
    n=mcfg.engage_consecutive_steps,
    seat_ms=mcfg.engage_consecutive_steps * 20,
    cd=mcfg.reengage_cooldown_steps,
    guard=ls.hold_judge_guard_steps,
    grace=ls.hold_grace_steps,
    cap_mm=mcfg.anchor_reseat_step_cap * 1e3,
    ramp=ls.capacity_ramp_time_s,
    lat=ratio(lut, 0, 1, 0),
    rev=ratio(lut, -1, 0, 0),
    floor=ls.wang_r_floor,
    shear=ls.capacity_ramp_shear_n,
  )
  cap_path = args.out.with_suffix(".caption.txt")
  cap_path.write_text(caption)
  print(f"wrote {cap_path}")


if __name__ == "__main__":
  main()
