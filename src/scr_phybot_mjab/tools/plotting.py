"""Shared figure styling and artifact export for offline analysis tools."""

from pathlib import Path

INK, SEC, MUT = "#0b0b0b", "#52514e", "#9b9a94"


RC_PARAMS = {
  "font.size": 7.5,
  "axes.titlesize": 8.0,
  "axes.labelsize": 7.5,
  "xtick.labelsize": 6.5,
  "ytick.labelsize": 6.5,
  "legend.fontsize": 6.5,
  "axes.linewidth": 0.6,
  "xtick.major.width": 0.6,
  "ytick.major.width": 0.6,
  "axes.edgecolor": SEC,
  "axes.labelcolor": INK,
  "xtick.color": SEC,
  "ytick.color": SEC,
  "svg.fonttype": "none",
  "pdf.fonttype": 42,
}


def save_three(fig, stem: Path) -> None:
  stem.parent.mkdir(parents=True, exist_ok=True)
  for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300})):
    fig.savefig(stem.with_suffix(suffix), **kwargs)
  print(f"figure saved to {stem}.{{svg,pdf,png}}")


def style_slice(ax, xlabel, ylabel, title):
  ax.grid(alpha=0.25, linewidth=0.5)
  ax.spines[["top", "right"]].set_visible(False)
  ax.axhline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.axvline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=7.5)


def style_3d(ax, xlabel, ylabel, zlabel, box_aspect):
  ax.set_xlabel(xlabel, labelpad=-4)
  ax.set_ylabel(ylabel, labelpad=-1)
  # mplot3d's own z label is unreliable inside tight multi-panel margins;
  # pin it in axes coordinates beside the z spine instead.
  ax.set_zlabel("")
  ax.text2D(
    1.0,
    0.875,
    zlabel,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=6.8,
    color=INK,
  )
  ax.set_box_aspect(box_aspect)
  ax.tick_params(pad=-2, labelsize=5.5)
  for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_alpha(0.04)
  try:  # Private mplot3d styling; purely cosmetic if it goes away.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
      axis._axinfo["grid"].update(linewidth=0.3, color="#c9c8c2")
  except (AttributeError, KeyError):
    pass
