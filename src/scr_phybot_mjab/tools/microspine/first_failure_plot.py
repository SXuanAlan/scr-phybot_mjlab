"""Four-panel presentation reconstructed from the supplied September reference PNG.

This is new plotting source, not the recovered source of
``data/microspine/raw/image-1788640220142.png``. It only draws the supplied
first-failure observations and fitted coefficients; it never refits the data.
The original figure's layout, colors, scale, and camera are visual references.
Its outlier and censored-event labels do not transfer to a different experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .first_failure import predict_surface


def plot_first_failure_surface(results, cells, fit, out: Path):
    """Save the measured pre-slip surface in the reference four-panel layout."""
    with plt.rc_context({
        "font.family": "Liberation Sans", "font.size": 9,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 10,
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "axes.linewidth": .85, "axes.spines.top": True,
        "axes.spines.right": True, "axes.grid": False,
        "xtick.color": "black", "ytick.color": "black",
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        fig = _surface_figure(results, cells, fit)
        for extension in ("png", "pdf", "svg"):
            # Fixed canvas preserves the proportions of the 2148 x 1320 reference.
            fig.savefig(out / f"02_first_failure_surface.{extension}",
                        dpi=200, bbox_inches=None)
        plt.close(fig)


def _surface_figure(results, cells, fit):
    red, blue, gray = "#E32328", "#3279BB", "#808080"
    fig = plt.figure(figsize=(10.74, 6.6), facecolor="white")
    axa = fig.add_axes((.075, .550, .233, .400))
    axb = fig.add_axes((.405, .451, .560, .565), projection="3d")
    axc = fig.add_axes((.075, .105, .233, .275))
    axd = fig.add_axes((.403, .105, .290, .275))
    legend_ax = fig.add_axes((.775, .067, .220, .190))
    legend_ax.axis("off")
    for letter, (left, bottom) in {
        "a": (.019, .968), "b": (.560, .944),
        "c": (.019, .395), "d": (.333, .395),
    }.items():
        fig.text(left, bottom, f"({letter})", weight="bold", fontsize=11)

    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    cmap = plt.get_cmap("RdBu")
    ss, ll = np.meshgrid(np.linspace(-19, 34, 161), np.linspace(-34, 34, 181))
    zz = predict_surface(fit, ss, ll)
    valid = [c for c in cells if c["n_normal_used"]]
    x, y, z, sd = (np.array([c[key] for c in valid])
                    for key in ("s_n", "l_n", "fz_mean_n", "fz_sd_n"))
    early = [r for r in results if r["first_phase"] in ("drag", "lateral")]

    # (a) Targets, measured normal-stage means, and the untested reflection.
    axa.axhspan(-34, 0, color="#ECECEC", zorder=0)
    contours = axa.contour(ss, ll, zz, levels=[-8, -6, -4, -2, 2, 4, 6, 8],
                           colors=gray, linewidths=.65)
    axa.clabel(contours, fmt="%g", fontsize=7)
    if np.nanmin(zz) < 0 < np.nanmax(zz):
        axa.contour(ss, ll, zz, levels=[0], colors=[red], linewidths=1.7)
        axb.contour(ss, ll, zz, levels=[0], colors=[red], linewidths=1.7)
    axa.scatter(x, y, c=z, norm=norm, cmap=cmap, s=32,
                edgecolors="#222222", linewidths=.65, zorder=4)
    reflected = y > 0
    axa.scatter(x[reflected], -y[reflected], s=32, facecolors="none",
                edgecolors=gray, linewidths=.7, zorder=4)
    for cell in cells:
        if cell["n_early"]:
            # Offset flags refer to a target, not to an attained failure force.
            axa.plot(cell["s_n"] + 1.35, cell["l_n"] - .85,
                     "x", color="black", ms=3.8, mew=.8, zorder=5)
    axa.axhline(0, color="#666666", lw=.65)
    axa.set(xlim=(-19, 34), ylim=(-34, 34),
            xlabel=r"$F_x$, drag along spines [N]", ylabel=r"$F_y$, lateral [N]")
    axa.set_xticks([-10, 0, 10, 20, 30])
    axa.set_yticks(np.arange(-30, 31, 10))
    axa.text(.5, .025, "mirrored (not tested)", transform=axa.transAxes,
             ha="center", color=gray, fontsize=8,
             bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=1))

    # (b) A large translucent surface with observed means and vertical SD bars.
    axb.plot_surface(ss, ll, zz, cmap=cmap, norm=norm, rcount=70, ccount=70,
                     alpha=.46, linewidth=.10, edgecolor=(.35, .35, .35, .25),
                     antialiased=True)
    axb.scatter(x, y, z, c=blue, s=20, edgecolors="black", linewidths=.65,
                 depthshade=False)
    for xi, yi, zi, si in zip(x, y, z, sd):
        if np.isfinite(si):
            axb.plot([xi, xi], [yi, yi], [zi - si, zi + si], color=blue, lw=.6)
    if early:
        axb.scatter([r["first_fx_pre_n"] for r in early],
                     [r["first_fy_pre_n"] for r in early],
                     [r["first_fz_pre_n"] for r in early], marker="x", color="black",
                     s=15, linewidths=.65, alpha=.65, depthshade=False)
    axb.set(xlabel=r"$F_x$ [N]", ylabel=r"$F_y$ [N]", zlabel=r"$F_z$ [N]",
             xlim=(-19, 34), ylim=(34, -34), zlim=(-10.5, 11.5))
    axb.set_xticks([-10, 0, 10, 20, 30])
    axb.set_yticks(np.arange(-30, 31, 10))
    axb.set_zticks([-10, -5, 0, 5, 10])
    axb.view_init(elev=24, azim=-125)
    axb.set_box_aspect((1.15, 1.4, .75), zoom=1.13)
    axb.tick_params(labelsize=8, pad=0)
    for axis in (axb.xaxis, axb.yaxis, axb.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#FFFFFF")
        axis._axinfo["grid"].update(color="#D8D8D8", linewidth=.6)
        axis.labelpad = 7
    # 3-D zoom can put valid projected points outside the square axes patch.
    # Keep the full surface and all SD bars visible on the larger figure canvas.
    for artist in [*axb.collections, *axb.lines]:
        artist.set_clip_on(False)

    # (c,d) The identical fitted coefficients, with residual scatter rather than CI.
    sigma = fit["residual_sd_n"]
    for ax, lateral_slice in ((axc, False), (axd, True)):
        xx = np.linspace(-34 if lateral_slice else -19, 34, 401)
        yy = predict_surface(fit, np.zeros_like(xx) if lateral_slice else xx,
                              xx if lateral_slice else np.zeros_like(xx))
        if lateral_slice:
            ax.axvspan(-34, 0, color="#ECECEC", zorder=0)
        ax.fill_between(xx, yy - sigma, yy + sigma, color=red, alpha=.17, linewidth=0)
        ax.plot(xx, yy, color=red, lw=1.7)
        for cell in valid:
            if (cell["s_n"] if lateral_slice else cell["l_n"]) != 0:
                continue
            point = cell["l_n"] if lateral_slice else cell["s_n"]
            err = cell["fz_sd_n"] if np.isfinite(cell["fz_sd_n"]) else None
            ax.errorbar(point, cell["fz_mean_n"], yerr=err, fmt="o", color=blue,
                         mec="black", mew=.7, ms=5.5, capsize=2.6, lw=.9, zorder=5)
            if lateral_slice and point > 0:
                ax.errorbar(-point, cell["fz_mean_n"], yerr=err, fmt="o", color=gray,
                             mfc="none", mew=.7, ms=5.5, capsize=2.6, lw=.8, zorder=5)
        for r in early:
            if (r["s_target_n"] if lateral_slice else r["l_target_n"]) == 0:
                point = r["first_fy_pre_n"] if lateral_slice else r["first_fx_pre_n"]
                ax.plot(point, r["first_fz_pre_n"], "x", color="black",
                         ms=4.3, mew=.9, alpha=.8, zorder=6)
        ax.axhline(0, color="#666666", lw=.65)
        ax.axvline(0, color="#666666", lw=.65)
        ax.set(xlim=(-34, 34) if lateral_slice else (-19, 34), ylim=(-11.5, 11.5),
                xlabel=r"$F_y$, lateral [N]" if lateral_slice else r"$F_x$, drag along spines [N]",
                ylabel=r"$F_z$, normal [N]")
        ax.set_xticks(np.arange(-30 if lateral_slice else -10, 31, 10))
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.text(.03, .95, r"$F_x=0$" if lateral_slice else r"$F_y=0$",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
        if lateral_slice:
            ax.text(.02, .03, "mirrored (not tested)", transform=ax.transAxes,
                    color=gray, fontsize=8,
                    bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=1))
    for ax in (axa, axc, axd):
        ax.tick_params(which="both", direction="in", top=True, right=True,
                       length=3.5, width=.7, labelsize=9)
        ax.xaxis.labelpad = 6
        ax.yaxis.labelpad = 6

    # The reference's red-white-blue scale is centered at zero and spans ±10 N.
    cbax = fig.add_axes((.789, .307, .177, .014))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax,
                 orientation="horizontal", ticks=[-10, -5, 0, 5, 10])
    cbax.tick_params(labelsize=8, length=3, width=.65)
    fig.text(.8775, .330, "$F_z$ just before first slip [N]\n(+ tension, − compression)",
             ha="center", va="bottom", fontsize=8.8)
    handles = [
        Line2D([], [], marker="o", color=blue, mec="black", ms=6, lw=1, label="Cell mean ± SD"),
        Line2D([], [], color=red, lw=1.7, label="Polynomial mean fit"),
        Patch(facecolor=red, alpha=.17, label=r"Fit ± $\sigma$ (residual)"),
        Line2D([], [], marker="x", color="black", ls="none", ms=6, label="Drag / lateral slip"),
        Line2D([], [], marker="o", color=gray, mfc="none", ls="none", ms=6,
               label="Mirrored (symmetry)"),
    ]
    legend_ax.legend(handles=handles, loc="upper left", borderaxespad=0,
                      frameon=False, fontsize=9, handlelength=1.7, labelspacing=.55)
    return fig
