"""Plot the climb-wall curriculum: gravity angle + reward-weight heatmap.

Mirrors the schedule defined in
``scr_phybot_mjab/tasks/climb_wall/climb_wall_env_cfg.py``. Edit the constants
below if the curriculum changes there.

Layout:
  left  — gravity angle vs iteration (line plot)
  right — heatmap, y-axis = reward name, x-axis = iteration, value = weight

Usage:
    python scripts/plot_curriculum.py
Writes ``docs/curriculum_schedule.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

P0_END = 2_000
P1_END = 15_000
P2_END = 45_000
X_MAX = 60_000

# (name, list of (iter, weight) stages, group)
# Only currently ENABLED rewards (non-zero at some stage). Stability/posture
# terms are all held at 0 in the current config and therefore omitted.
REWARD_SCHEDULES: list[tuple[str, list[tuple[int, float]], str]] = [
  # Wall-grip
  ("hands_toward_wall",    [(0, 15.0), (P1_END, 5.0)],                "grip"),
  ("hand_orientation",     [(0, 15.0), (P1_END, 5.0)],                "grip"),
  ("hands_on_wall",        [(0, 20.0)],                               "grip"),
  ("com_near_wall",        [(0, 5.0)],                                "grip"),
  ("both_hands_bonus",     [(0, 30.0), (P0_END, 15.0), (P1_END, 5.0)],"grip"),
  # Gait (off until P0_END)
  ("gait_match",           [(0, 0.0), (P0_END, 8.0)],                 "gait"),
  ("swing_hand_height",    [(0, 0.0), (P0_END, 10.0)],                "gait"),
  ("swing_hand_clearance", [(0, 0.0), (P0_END, -50.0)],               "gait"),
  ("stance_hand_anchor",   [(0, 0.0), (P0_END, -2.0)],                "gait"),
  ("climb_vz",             [(0, 0.0), (P0_END, 20.0)],                "gait"),
  ("swing_phase_force",    [(0, 0.0), (P0_END, 10.0)],                "gait"),
  ("stance_phase_velocity",[(0, 0.0), (P0_END, 10.0)],                "gait"),
]

GRAVITY_STAGES: list[tuple[int, float]] = [(0, 30.0), (P1_END, 30.0), (P2_END, 90.0)]


def step_schedule(stages: list[tuple[int, float]], iters: np.ndarray) -> np.ndarray:
  """reward_weight uses last stage with iter < current_iter — i.e. step function."""
  out = np.full_like(iters, stages[0][1], dtype=float)
  for it, w in stages:
    out = np.where(iters > it, w, out)
  return out


def linear_schedule(stages: list[tuple[int, float]], iters: np.ndarray) -> np.ndarray:
  """gravity_angle linearly interpolates between stage points."""
  xs = np.array([s[0] for s in stages], dtype=float)
  ys = np.array([s[1] for s in stages], dtype=float)
  return np.interp(iters, xs, ys, left=ys[0], right=ys[-1])


def main() -> None:
  iters = np.arange(0, X_MAX + 1, 100)

  # Build weight matrix: rows = rewards (order kept as listed), cols = iters.
  names = [n for n, _, _ in REWARD_SCHEDULES]
  weights = np.stack([step_schedule(s, iters) for _, s, _ in REWARD_SCHEDULES], axis=0)

  fig = plt.figure(figsize=(16, 6.5))
  gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.32)
  ax_grav = fig.add_subplot(gs[0, 0])
  ax_heat = fig.add_subplot(gs[0, 1])

  # --- Left: gravity angle -------------------------------------------------
  g = linear_schedule(GRAVITY_STAGES, iters)
  ax_grav.plot(iters, g, color="black", linewidth=2)
  ax_grav.fill_between(iters, 0, g, alpha=0.1, color="black")
  ax_grav.set_xlabel("iteration")
  ax_grav.set_ylabel("gravity tilt (deg)")
  ax_grav.set_title("Gravity angle", fontsize=11, loc="left", pad=26)
  ax_grav.set_ylim(0, 100)
  ax_grav.set_xlim(0, X_MAX)
  ax_grav.grid(True, alpha=0.25)

  # --- Right: reward-weight heatmap ---------------------------------------
  vmax = max(abs(weights.min()), abs(weights.max()))
  norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
  im = ax_heat.imshow(
    weights,
    aspect="auto",
    origin="upper",
    extent=(iters[0], iters[-1], len(names) - 0.5, -0.5),
    cmap="RdBu_r",
    norm=norm,
    interpolation="nearest",
  )
  ax_heat.set_yticks(np.arange(len(names)))
  ax_heat.set_yticklabels(names, fontsize=9)
  ax_heat.set_xlabel("iteration")
  ax_heat.set_title("Reward weight", fontsize=11, loc="left", pad=26)
  ax_heat.set_xlim(0, X_MAX)

  # Separator between grip and gait groups.
  n_grip = sum(1 for _, _, g in REWARD_SCHEDULES if g == "grip")
  ax_heat.axhline(n_grip - 0.5, color="black", linewidth=1.2)

  # Annotate the weight value in each distinct stage segment so small values
  # (e.g. stance_hand_anchor = -2) are readable even when the colour scale is
  # dominated by ±50.
  stage_edges = sorted({0, P0_END, P1_END, P2_END, X_MAX})
  for row, (_, stages, _) in enumerate(REWARD_SCHEDULES):
    for a, b in zip(stage_edges[:-1], stage_edges[1:]):
      mid = (a + b) / 2
      w = float(step_schedule(stages, np.array([mid]))[0])
      txt = f"{w:g}"
      # Pick readable text colour based on cell colour intensity.
      intensity = abs(w) / vmax
      color = "white" if intensity > 0.55 else "#111"
      ax_heat.text(mid, row, txt, ha="center", va="center",
                   fontsize=8, color=color)

  cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.035)
  cbar.set_label("weight", fontsize=9)

  # --- Phase boundaries on both panels ------------------------------------
  for ax in (ax_grav, ax_heat):
    for x in (P0_END, P1_END, P2_END):
      ax.axvline(x, color="#222", linestyle="--", linewidth=0.7, alpha=0.6)

  # Single-line phase labels placed just above the heatmap top edge. Kept
  # thin so they don't collide with the "Reward weight" title sitting in its
  # default (top-left) position.
  phase_lbls_one_line = [
    (0, P0_END,      "P0 grip"),
    (P0_END, P1_END, "P1 gait@30°"),
    (P1_END, P2_END, "P2 gravity 30°→90°"),
    (P2_END, X_MAX,  "P3 vertical"),
  ]
  for a, b, lbl in phase_lbls_one_line:
    ax_heat.text((a + b) / 2, -0.55, lbl,
                 ha="center", va="bottom", fontsize=9.0, color="#222",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbb", alpha=0.95))

  fig.suptitle("Climb-Wall Curriculum Schedule", fontsize=13, y=1.04)

  out = Path(__file__).resolve().parent.parent / "docs" / "curriculum_schedule.png"
  out.parent.mkdir(exist_ok=True)
  fig.savefig(out, dpi=130, bbox_inches="tight")
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
