"""Classify physical bench trials by first slip and reproduce the four-panel surface.

Example (env_mjlab15):
  python scripts/analyze_microspine_first_failure.py --data data/microspine/raw/20260922-163807

Outputs are analysis artifacts only. Raw CSVs and runtime model tables are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scr_phybot_mjab.tools.paths import LIMIT_SURFACE_ANALYSIS_DIR  # noqa: E402
from scr_phybot_mjab.tools.microspine.first_failure import (  # noqa: E402
    DetectorConfig, analyze_trial, cell_statistics, csv_rows, fit_surface,
    predict_surface, prepare_trace, sha256,
)

from scr_phybot_mjab.tools.microspine.first_failure_plot import (  # noqa: E402
    plot_first_failure_surface as plot_surface,
)

CATEGORIES = {
    "reverse_shear": ("Reverse shear / drag", "#AA4499", "反向拖拽阶段滑移"),
    "forward_shear": ("Forward shear / drag", "#D55E00", "正向拖拽阶段滑移"),
    "lateral_loading": ("Lateral loading", "#DDAA33", "侧向加载阶段失稳"),
    "preload_loss": ("Normal unloading (Fn <= 0)", "#56B4E9", "法向卸压阶段失稳（Fn≤0）"),
    "tensile_loading": ("Normal tension (Fn > 0)", "#0072B2", "法向拉伸阶段失稳（Fn>0）"),
    "unresolved": ("Onset unresolved", "#666666", "首次失效起点未识别"),
    "no_contact": ("No contact", "#CCCCCC", "未建立接触"),
}
COLORS = {key: val[1] for key, val in CATEGORIES.items()}
F_COLORS = ("#2A78D6", "#D55E00", "#189977")
RED = "#D62728"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514E", "axes.labelcolor": "#222222", "axes.axisbelow": True,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def save_figure(fig, out: Path, name: str):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def panel(ax, label, title):
    ax.set_title(f"({label})  {title}", loc="left", fontweight="bold", pad=13)
    ax.grid(alpha=.18, lw=.6)


def plot_classification(results, out):
    counts = Counter(r["category"] for r in results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    fig.subplots_adjust(left=.075, right=.97, bottom=.10, top=.85, wspace=.35, hspace=.40)
    fig.suptitle("First-failure classification | 2026-09-22", x=.075, ha="left", y=.97,
                 fontsize=21, fontweight="bold")
    fig.text(.075, .91, f"{len(results)} trials  |  3 N preload  |  first event across all loading stages  |  "
             f"{sum(r['reengaged'] for r in results)} re-engaged after first slip\n"
             f"Categories describe the loading stage; {sum(r['review_required'] for r in results)} traces need onset review.",
             color="#52514E", linespacing=1.5)
    ax = axes[0, 0]
    names = list(CATEGORIES)
    ax.barh(range(len(names)), [counts[k] for k in names], color=[COLORS[k] for k in names], height=.65)
    ax.set_yticks(range(len(names)), [CATEGORIES[k][0] for k in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts.values()) * 1.25)
    for i, key in enumerate(names):
        ax.text(counts[key] + .8, i, f"{counts[key]}  ({counts[key]/len(results):.1%})", va="center", fontsize=9)
    ax.set_xlabel("Trials")
    panel(ax, "a", "Inferred stage of the first event")

    ax = axes[0, 1]
    for r in results:
        dx = (r["rep"] - 2) * .95
        dy = .55 * r["side"] if r["s_target_n"] == 0 else 0
        ax.scatter(r["s_target_n"] + dx, r["l_target_n"] + dy, s=42,
                   color=COLORS[r["category"]], edgecolors="black" if r["review_required"] else "white",
                   linewidths=.7, zorder=3)
    ax.set(xlim=(-18, 33), ylim=(-2.5, 32.5), xlabel="Target rake force S [N]", ylabel="Target lateral force L [N]")
    ax.set_xticks(np.arange(-15, 31, 5)); ax.set_yticks(np.arange(0, 31, 5))
    panel(ax, "b", "All repetitions at each intended target")
    ax.text(.01, -.23, "Left / middle / right = R1 / R2 / R3; black edge = review.\nS=0 has both side labels (six trials per cell).",
            transform=ax.transAxes, fontsize=9, color="#52514E")

    ax = axes[1, 0]
    old = ("detached_drag", "detached_lateral", "detached_normal", "no_contact")
    matrix = np.array([[sum(r["controller_result"] == a and r["category"] == b for r in results)
                        for b in names] for a in old])
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, matrix.max()))
    for i in range(len(old)):
        for j in range(len(names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > .55 * matrix.max() else "#222222")
    ax.set_yticks(range(len(old)), ["Drag", "Lateral", "Normal", "No contact"])
    ax.set_xticks(range(len(names)), ["Reverse\nshear", "Forward\nshear", "Lateral", "Unload", "Tension", "Review", "No\ncontact"], fontsize=9)
    ax.set_ylabel("Original terminal label")
    ax.set_xlabel("First-event classification")
    panel(ax, "c", "Where terminal labels change")
    ax.grid(False)

    ax = axes[1, 1]
    for category in names:
        rows = [r for r in results if r["category"] == category and "first_fz_cmd_n" in r]
        if rows:
            ax.scatter([r["first_fz_cmd_n"] for r in rows], [r["controller_fz_cmd_n"] for r in rows],
                       c=COLORS[category], s=34, alpha=.6, edgecolors="white", linewidths=.4)
    ax.plot([-3.5, 10.5], [-3.5, 10.5], color="#666666", ls="--", lw=1)
    ax.set(xlim=(-3.5, 10.5), ylim=(-3.5, 10.5), xlabel="Normal command at first event [N]",
           ylabel="Original terminal normal command [N]")
    panel(ax, "d", "Terminal loading can overstate first-slip capacity")
    fig.text(.075, .025, "Force and pose signals determine onset. Smooth creep and threshold-sensitive detections are provisional; "
             "these labels do not identify spine fracture or substrate damage.", fontsize=9, color="#52514E")
    save_figure(fig, out, "01_failure_classification")


def plot_fit_diagnostics(results, cells, fit, out):
    """Expose prediction error and the observations absent from a survivor fit."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    fig.subplots_adjust(left=.075, right=.97, top=.84, bottom=.10, hspace=.47, wspace=.28)
    fig.suptitle("Audit of the conditional polynomial fit", x=.075, ha="left", y=.97,
                 fontsize=21, fontweight="bold")
    fig.text(.075, .905, "The regression estimates a mean among trials that reached normal loading. "
             "It does not fit the full first-failure envelope.\n"
             f"{fit['n_singleton_cells']} / {fit['n_cells']} fitted cells have only one accepted observation; "
             "crosses in the main figure are failures during drag or lateral loading.",
             color="#52514E", fontsize=10, linespacing=1.5)
    ax = axes[0, 0]
    z, predicted = fit["observed_cell_means_n"], fit["loco_predictions_n"]
    valid = [c for c in cells if c["n_normal_used"]]
    for one, marker, label in ((True, "o", "One accepted observation"), (False, "s", "Two or more")):
        keep = np.array([(c["n_normal_used"] == 1) == one for c in valid])
        ax.scatter(z[keep], predicted[keep], marker=marker, s=40, color="#2A78D6",
                   facecolors="none" if one else "#2A78D6", label=label)
    bounds = [min(-3, np.min(predicted)) - .3, max(10, np.max(predicted)) + .3]
    ax.plot(bounds, bounds, "--", color="#777777", lw=1)
    ax.set(xlim=bounds, ylim=bounds, xlabel="Observed cell mean Fz [N]", ylabel="Prediction with this cell held out [N]")
    panel(ax, "a", f"Leave-one-cell-out RMSE = {fit['loco_selected_rmse_n']:.2f} N")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    labels = ["Constant\nbaseline"] + [f"Degree {c['degree']}" for c in fit["cv_scores"]]
    errors = [fit["loco_constant_rmse_n"]] + [c["loco_rmse_n"] for c in fit["cv_scores"]]
    ax.bar(labels, errors, color=["#999999"] + ["#2A78D6" if c["degree"] == fit["degree"] else "#ADC8E7"
                                                  for c in fit["cv_scores"]], width=.65)
    for i, value in enumerate(errors):
        ax.text(i, value + .06, f"{value:.2f}", ha="center", fontsize=10)
    ax.set(ylabel="Held-out cell RMSE [N]", ylim=(0, max(errors) * 1.22))
    panel(ax, "b", "Lowest candidate error is not physical validation")
    for ax, axis, letter in zip(axes[1], (0, 1), ("c", "d")):
        subset = [c for c in cells if (c["l_n"] if axis == 0 else c["s_n"]) == 0]
        xs = np.array([c["s_n"] if axis == 0 else c["l_n"] for c in subset])
        used = np.array([c["n_normal_used"] for c in subset])
        early = np.array([c["n_early"] for c in subset])
        # The third bin includes uncertain normal onsets, unresolved traces and
        # no-contact trials, but never double counts an early slip under review.
        other = np.array([c["n_total"] for c in subset]) - used - early
        ax.bar(xs, used, width=3.6, color="#2A78D6", label="Accepted normal onset")
        ax.bar(xs, early, bottom=used, width=3.6, color="#D55E00", label="Slip before normal ramp")
        ax.bar(xs, other, bottom=used + early, width=3.6, color="#BBBBBB", label="Other / onset under review")
        ax.set(xlabel="Rake target S [N]" if axis == 0 else "Lateral target L [N]", ylabel="All trials at target",
               xticks=xs, ylim=(0, max(c["n_total"] for c in subset) + .8))
        panel(ax, letter, "L = 0: which repeats enter the curve?" if axis == 0 else "S = 0: which repeats enter the curve?")
        for x, c in zip(xs, subset):
            ax.text(x, c["n_total"] + .1, f"{c['n_normal_used']}/{c['n_total']}", ha="center", fontsize=8)
    axes[1, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.26), ncols=3)
    save_figure(fig, out, "05_fit_diagnostics")


def plot_trace(axf, axp, row, trace, events, zoom=False):
    if not len(trace["t"]):
        axf.text(.5, .5, f"T{row['trial']}: no contact", ha="center", transform=axf.transAxes)
        axp.axis("off")
        return
    t = trace["t"] - trace["t"][0]
    stride = max(1, len(t) // 2500)
    for j, name in enumerate(("Rake", "Lateral", "Normal")):
        axf.plot(t[::stride], trace["force"][::stride, j], color=F_COLORS[j], lw=.85, label=f"{name} measured")
        axf.plot(t[::stride], trace["command"][::stride, j], color=F_COLORS[j], ls="--", lw=.75, alpha=.7)
        axp.plot(t[::stride], trace["pos"][::stride, j], color=F_COLORS[j], lw=.9, label=f"{name} motion")
    for ax in (axf, axp):
        for step in ("lateral", "normal"):
            indices = np.flatnonzero(trace["step"] == step)
            if len(indices):
                ax.axvline(t[indices[0]], ls=":", color="#888888", lw=.8)
                if ax is axf and not zoom:
                    ax.text(t[indices[0]], .99, step, transform=ax.get_xaxis_transform(),
                            rotation=90, va="top", ha="right", fontsize=7, color="#666666")
        for j, event in enumerate(events):
            ax.axvline(event["elapsed_s"], color=RED if j == 0 else "#E5A1A1", lw=1.4 if j == 0 else .7)
        ax.axvline(trace["controller_t"] - trace["t"][0], color="#222222", ls="--", lw=1.1)
        ax.grid(alpha=.18, lw=.6)
        if zoom and events:
            onset = events[0]["elapsed_s"]
            ax.set_xlim(max(0, onset - 1), min(t[-1], onset + 2))
        else:
            ax.set_xlim(0, max(1, t[-1] * 1.025))
    axf.set_ylabel("Force [N]"); axp.set_ylabel("TCP motion [mm]")
    axp.set_xlabel("Time since drag start [s]")
    title = (f"T{row['trial']}  R{row['rep']}  S={row['s_target_n']:g}, L={row['l_target_n']:g} N"
             f"  |  {CATEGORIES[row['category']][0]}")
    axf.set_title(title, fontsize=10, loc="left", fontweight="bold")
    evidence = row["evidence"].replace("_", " ")
    axp.set_title(f"{evidence}; re-engaged: {row['reengaged']}; review: {row['review_required']}",
                  fontsize=9, loc="left", color="#52514E")


def plot_examples(results, traces, all_events, out):
    # Fixed, representative trial IDs make regenerated diagnostic panels easy to
    # compare. They include a clean release, an early lateral slip, reverse drag,
    # a normal-stage re-catch, gradual drift and a large terminal overestimate.
    ids = [1, 4, 29, 33, 43, 157]
    available = {r["trial"]: r for r in results}
    ids = [i for i in ids if i in available]
    fig, axes = plt.subplots(len(ids), 2, figsize=(16, 2.55 * len(ids)), squeeze=False)
    fig.subplots_adjust(left=.065, right=.97, top=.91, bottom=.05, hspace=.78, wspace=.2)
    for axs, trial in zip(axes, ids):
        plot_trace(*axs, available[trial], traces[trial], all_events[trial])
    axes[0, 0].legend(ncols=3, fontsize=8, loc="lower left")
    axes[0, 1].legend(ncols=3, fontsize=8, loc="lower left")
    fig.suptitle("First slip, re-engagement and final robot stop", fontsize=20, x=.065, ha="left", y=.975)
    fig.text(.065, .943, "Solid = measured; dashed colored = commanded. Red = first event; pale red = later events; "
             "black dashed = controller stop.\nMotion is reconstructed from TCP pose in a fixed initial frame, avoiding the logged displacement reset between stages.",
             fontsize=10, color="#52514E", linespacing=1.6, va="top")
    save_figure(fig, out, "03_event_examples")


def plot_correction(results, cells, fit, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.subplots_adjust(left=.06, right=.97, top=.78, bottom=.19, wspace=.32)
    fig.suptitle("How first-event processing changes the normal-failure curve", x=.06, ha="left", y=.97,
                 fontsize=18, fontweight="bold")
    fig.text(.06, .86, "Commands are compared with commands below. The main surface uses measured pre-slip force. "
             "All normal-stage fits are conditional on surviving prior loading.", color="#52514E", fontsize=10)
    for ax, axis in zip(axes[:2], (0, 1)):
        coords = sorted({r["s_target_n"] if axis == 0 else r["l_target_n"] for r in results
                         if (r["l_target_n"] if axis == 0 else r["s_target_n"]) == 0})
        for original, color, label in ((True, "#888888", "Original terminal command"),
                                      (False, RED, "First-slip command (accepted)")):
            means, stds = [], []
            for value in coords:
                subset = [r for r in results if (r["s_target_n"] if axis == 0 else r["l_target_n"]) == value
                          and (r["l_target_n"] if axis == 0 else r["s_target_n"]) == 0
                          and (r["controller_result"] == "detached_normal" if original else r["normal_fit_eligible"])]
                vals = [r["controller_fz_cmd_n"] if original else r["first_fz_cmd_n"] for r in subset]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.)
            ax.errorbar(coords, means, yerr=stds, fmt="o-", color=color, capsize=3, lw=1.2, ms=5, label=label)
        ax.set(xlabel="Rake target S [N]" if axis == 0 else "Lateral target L [N]", ylabel="Normal command [N]")
        panel(ax, "a" if axis == 0 else "b", "L = 0" if axis == 0 else "S = 0")
    axes[0].legend(fontsize=8, loc="upper left")
    ax = axes[2]
    eligible = [r for r in results if r["normal_fit_eligible"]]
    changes = [r["controller_minus_first_fz_n"] for r in eligible]
    ax.hist(changes, bins=np.arange(-.5, max(10.5, np.ceil(max(changes)) + .5) + .01, .5),
            color="#2A78D6", edgecolor="white")
    ax.set(xlabel="Terminal minus first-slip command [N]", ylabel="Accepted normal-stage trials")
    panel(ax, "c", f"Median difference: {np.median(changes):.2f} N")
    save_figure(fig, out, "04_curve_correction")


def audit_pdf(results, traces, all_events, out):
    with PdfPages(out / "all_trials_review.pdf") as pdf:
        for start in range(0, len(results), 3):
            fig, axes = plt.subplots(3, 2, figsize=(16, 10.2))
            fig.subplots_adjust(left=.065, right=.98, top=.90, bottom=.07, hspace=.65, wspace=.20)
            for axs, row in zip(axes, results[start:start + 3]):
                plot_trace(*axs, row, traces[row["trial"]], all_events[row["trial"]])
            for axs in axes[len(results[start:start + 3]):]:
                for ax in axs:
                    ax.axis("off")
            fig.suptitle(f"Trial audit | {start + 1}–{min(start + 3, len(results))} / {len(results)}", x=.065,
                         ha="left", fontsize=17, y=.98)
            fig.text(.065, .938, "Red: first event | pale red: subsequent events after re-engagement | "
                     "black dashed: controller stop | dotted: loading-stage transitions\n"
                     "Forces: blue rake, orange lateral, green normal. Colored dashed lines are commands.",
                     fontsize=9, color="#52514E")
            pdf.savefig(fig)
            plt.close(fig)


def json_clean(obj):
    if isinstance(obj, dict):
        return {k: json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_report(out, data, rows, results, cells, fit, config):
    counts = Counter(r["category"] for r in results)
    original_counts = Counter(r["result"] for r in rows)
    advance = [r["time_advance_s"] for r in results if "time_advance_s" in r]
    early_reclassified = [r["trial"] for r in results if r["controller_result"] == "detached_normal"
                          and r["first_phase"] in ("drag", "lateral")]
    accepted = [r for r in results if r["normal_fit_eligible"]]
    deltas = [r["controller_minus_first_fz_n"] for r in accepted]
    review = [r["trial"] for r in results if r["review_required"]]
    audit = {"data": str(data.relative_to(ROOT) if data.is_relative_to(ROOT) else data),
             "summary_sha256": sha256(data / "results.csv"), "n_trials": len(results),
             "n_detail_rows": sum(r["raw_rows"] for r in results),
             "original_categories": dict(original_counts), "first_event_categories": dict(counts),
             "n_reengaged": sum(r["reengaged"] for r in results),
             "review_trial_ids": review, "normal_to_early_trial_ids": early_reclassified,
             "n_normal_fit_trials": len(accepted), "n_supported_cells": fit["n_cells"],
             "median_time_advance_s": float(np.median(advance)),
             "normal_command_reduction_median_n": float(np.median(deltas)),
             "normal_command_reduction_max_n": float(np.max(deltas)),
             "detector_config": asdict(config), "sensitivity_threshold_scales": [.75, 1., 1.25],
             "fit": {k: v for k, v in fit.items() if k not in ("hull", "config")},
             "no_residual_outliers_removed": True,
             "force_frame": "S=(-Fx_tcp+Fy_tcp)/sqrt(2), L=(Fx_tcp+Fy_tcp)/sqrt(2), N=Fz_tcp",
             "mirror": "Negative L is assumed even symmetry, never an observed sample",
             "figure_provenance": {
                 "reference_image": "data/microspine/raw/image-1788640220142.png",
                 "source_status": "New visual reconstruction; original reference plotting source not found",
                 "force_quantity": "Measured Fz median 80–20 ms before the detected first-slip onset",
                 "plot_source": "src/scr_phybot_mjab/tools/microspine/first_failure_plot.py",
             },
             "script_sha256": sha256(Path(__file__)),
             "plot_script_sha256": sha256(ROOT / "src/scr_phybot_mjab/tools/microspine/first_failure_plot.py"),
             "detector_sha256": sha256(ROOT / "src/scr_phybot_mjab/tools/microspine/first_failure.py")}
    (out / "analysis.json").write_text(json.dumps(json_clean(audit), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    count_table = "\n".join(f"| {CATEGORIES[key][2]} | {counts[key]} |" for key in CATEGORIES)
    scores = ", ".join(f"{v['degree']} 阶 {v['loco_rmse_n']:.3f} N" for v in fit["cv_scores"])
    report = f"""# 2026-09-22 microspine 首次失效分析

输入：`{audit['data']}`。共 {len(results)} 次试验、{audit['n_detail_rows']:,} 条详细采样。
R1–R3 各 77 次；S=0 的正、负 side 是相同载荷目标的重复，合并为 70 个力单元格。
86 次来自恢复前的运行，145 次来自恢复后的运行；时间仅在每个详细文件内使用，不跨运行拼接。

## 图和结果

- [失效分类图](01_failure_classification.png)：各类数量、每个目标的重复试验、原判定与首次事件交叉表。
- [首次失效四联图](02_first_failure_surface.png)：按旧 PNG 恢复的四面板布局、红线、粉色带、三维视角和 ±10 N 色标。
- [代表试验时序](03_event_examples.png)：力、实际 TCP 位移、第一次及后续失效、控制器停止点。
- [曲线修正对比](04_curve_correction.png)：终点命令与首次事件命令的同口径比较。
- [拟合核查图](05_fit_diagnostics.png)：留出预测误差、常数基线对照、两个截面实际纳入的重复数。
- [全部试验复核 PDF](all_trials_review.pdf)：每页 3 次试验，按 T1–T231 排序。
- `trials.csv`：每次试验的首次点、原终点、信号依据、敏感性和复核标记。
- `events.csv`：首次及再次卡住后的后续事件；`cells.csv`：参与拟合的单元格均值与样本 SD。
- `review_queue.csv`：需人工复核的试验。所有主图同时提供 PDF / SVG。

## 本批结论

| 信号对应的加载阶段 | 次数 |
| --- | ---: |
{count_table}

原记录为 {original_counts.get('detached_drag', 0)} 次拖拽脱离、{original_counts.get('detached_lateral', 0)} 次侧向脱离、
{original_counts.get('detached_normal', 0)} 次法向脱离、{original_counts.get('no_contact', 0)} 次未接触。
其中 {len(early_reclassified)} 次原“法向失效”在更早的拖拽/侧向阶段出现可检测滑移。
{audit['n_reengaged']} 次在第一事件后重新达到稳定承载，后续峰值不能代表第一次失效。
首次点相对控制器结束加载的时间提前中位数为 {np.median(advance):.2f} s。
通过敏感性筛选的法向试验共 {len(accepted)} 次，其终点法向命令减首次法向命令的中位数为
{np.median(deltas):.2f} N，最大为 {np.max(deltas):.2f} N。这是同一试验、同一命令信号的比较。

本次预压力是 **3 N**；旧参考图使用的 20260905 数据为 **10 N**。两者加载条件不同，不能直接解释为性能提升或下降。

## 判定口径与信号处理

1. 从拖拽阶段开始检查全部加载阶段，允许正常初始就位；取最早失载/滑移，不使用全程最大力。
   仅在没有力回落、平滑随命令增长的就位过程不会触发失载规则。
2. 数据约 100 Hz，重采样为 10 ms，3 点中值去尖峰；姿态投影到加载开始时的固定指令坐标系。
   使用实际 TCP XYZ 重建位移，避免 `disp_mm` 在拖拽转侧向时清零造成假跳变。
3. 突发滑移：140 ms 净运动 ≥0.15 mm，且满足以下之一：承载方向力下降 ≥max(0.75 N, 该分量命令的 6%)；
   或此前 250 ms 速度中值 <0.25 mm/s，随后速度 >max(1 mm/s, 基线的 8 倍)。
   拖拽/侧向的纯位移规则额外要求命令已稳定。位移必须持续保留，排除回弹尖峰。
   拖拽/侧向不以压紧力的瞬态波动判失效。
4. 缓慢法向阶段滑移：1 s 内净漂移 ≥max(0.20 mm, 初始法向速度基线的 4 倍×1 s)，
   同时要求位移已开始、至少 70% 窗口持续运动、速度中值 >max(0.15 mm/s, 基线 3 倍)，且没有短时高速尖峰。
   这些条件避免把未来的突然滑移提前标成“缓慢滑移”。
   它没有唯一突变时刻，标为 `progressive_motion` 并进入复核队列，不用于主拟合。
5. 再次卡住：第一事件后速度 <0.40 mm/s，测量/命令误差 <max(1.5 N, 命令合力的 10%)，持续至少 0.35 s。
   只有重新稳定后才计为新的失效事件，同一次连续滑移不重复计数。
6. 首次载荷取速度起点前 80–20 ms 的测量力中值；命令力另存。事件定位精度约 0.1 s，不能把 10 ms 采样间隔当作真实判定精度。
   `fz_first_release_n` 只作对照，没有作为检测真值。控制器事件常落在第一条 `unload` 记录，保留该瞬间但不读取之后主动卸载的信号。
7. 位移、绝对力降和速度等检测阈值尺度分别取 0.75 / 1.00 / 1.25 复算；相对力降比例、时间窗和再卡住规则固定。
   首次阶段变化、法向命令跨度 >0.75 N、起点跨度 >1 s，
   以及渐进滑移或未识别，均需复核。本次共 {len(review)} 次。它们仍展示在分类与逐次表中，不作为确定的法向拟合点。

分类是**加载阶段及可观测失稳形式的推断**，不能凭力和末端位移确认断刺、岩面破坏、单根刺脱钩或机械结构损伤。
`preload_loss` 表示外加压紧力尚未完全卸掉就发生运动，不能据此称为正向拉脱。
滑移方向和末端转角另存表中；没有视频真值，阈值稳健不等于物理原因已确认。

## 四联图的统计含义

**复核结论：多项式计算可复现，但当前拟合不是经过验证的物理失效包络。**
按用户要求恢复旧图的出版排版：无大标题、四个字母面板、较大的三维图、红色回归线、粉色残差带、
蓝色均值与 SD、灰色镜像区域，以及以零为中心的 ±10 N 色标。图例 `Polynomial mean fit` 明确这是均值回归。
事件检测、纳入的试验、测量力口径和回归系数未因排版修正而更改。

参考图为 `data/microspine/raw/image-1788640220142.png`（2148×1320，Matplotlib 3.5.1）。
在项目、Git 历史和可用的旧记录中未找到其原绘图源码，旧记录中它已作为 PNG 输入。
当前绘图模块是**根据该 PNG 重新实现的源码**，不是找回的原脚本：
`src/scr_phybot_mjab/tools/microspine/first_failure_plot.py`。
旧记录说明原图使用控制器终点的 `fz_cmd_n`；本次仍使用首次滑移前的测量 Fz，
且预压力从旧实验的 10 N 改为本次 3 N，因此新图的曲面形状和数据点不会与旧图一致。
旧图的离群点编号、+10 N 未失效三角标记不适用于本批，不能随样式照搬。

图中的 `Drag / lateral slip`（前一版 `Early-stage slip` / `Slip before normal ramp`）指法向加载开始前，
在拖拽或侧向阶段已经检测到滑移；它是第一次失效记录，不是统计离群点。
这一阶段法向命令仍是 −3 N，所以这些叉多数落在 Fz≈−3 N。
法向卸压阶段虽然也可能在 Fz<0 时失效，但已开始法向 ramp，不属于此类。

仅使用“第一次事件发生在法向阶段且通过筛选”的 {len(accepted)} 次试验 / {fit['n_cells']} 个单元格。
统计量是首次失效前的测量 Fz；横轴是对应保持不变的目标 S、L。
早期失败单独画叉：二维目标图的叉表示该目标有提前失败，三维/截面的叉使用真正达到的失效力，
不把尚未达到的目标值当作失败坐标，也不把它们伪装成 −3 N 的法向失效样本。

用每个单元格等权拟合，避免 S=0 双倍重复数带来的权重偏差；候选为 L 偶对称 2/3/4 阶多项式。
逐单元格留一验证 RMSE：{scores}。选用 {fit['degree']} 阶；若低阶误差在最优值 5% 内则优先低阶。
单元格拟合 RMSE {fit['cell_rmse_n']:.3f} N，逐试验残差 SD {fit['residual_sd_n']:.3f} N。
单元格训练 R² 为 {fit['training_r2']:.3f}，留一验证 RMSE 为 {fit['loco_selected_rmse_n']:.3f} N；
同口径的常数基线 RMSE 为 {fit['loco_constant_rmse_n']:.3f} N。
回归比常数预测有所改善，但这不能验证物理边界，也不能把回归平均值解释成可靠承载上限。
{fit['n_cells']} 个拟合单元格中 {fit['n_singleton_cells']} 个只有一次有效观测；
例如 S=0、L=0 的 6 次试验仅有 1 次进入均值。因此局部样本筛选影响很大。
二次式的侧向平方项为正，造成 S=0 截面两端上翘；它是幸存样本上的全局回归结果，
不能据此认定侧向载荷增加会提高承载力。提高多项式阶数在当前留一验证中也没有改善。
误差棒为重复间样本 SD，n=1 不画 SD；色带为残差散布，不是置信区间或安全保证。
**没有按残差剔除离群点**，没有把早期失效当成应删除的异常值。

仅在有效单元格凸包内展示拟合；不向全是早期失败的区域补出一条确定曲线。
负 L 未测，灰底/空心点代表镜像假设。该面存在前序存活条件和选择效应，不能直接当作总体可靠性边界或运行时承载模型。

## 复现

```bash
/home/spider/miniconda3/envs/env_mjlab15/bin/python scripts/analyze_microspine_first_failure.py \\
  --data {audit['data']} \\
  --out {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}
```

`analysis.json` 保存规则、拟合系数、交叉验证和源文件哈希；每个试验表行保存详细 CSV 的 SHA256。
绘图模块只接收统计量与已有系数，不进行重新拟合；图的来源说明和绘图源码哈希也保存在 `analysis.json`。
未改动原始数据、旧图或仿真运行时参数。
"""
    (out / "README.md").write_text(report)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/microspine/raw/20260922-163807")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-audit-pdf", action="store_true", help="Skip the full per-trial review PDF during iteration")
    args = parser.parse_args()
    data = args.data.resolve()
    out = (args.out or LIMIT_SURFACE_ANALYSIS_DIR / data.name).resolve()
    if out == data or data in out.parents:
        raise ValueError("Analysis outputs must be outside the raw data directory")
    out.mkdir(parents=True, exist_ok=True)
    config = DetectorConfig()
    rows = csv_rows(data / "results.csv")
    ids = [int(row["trial"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trial IDs in summary")
    rows.sort(key=lambda row: int(row["trial"]))
    results, traces, all_events = [], {}, {}
    for row in rows:
        path = data / row["detail_csv"]
        trace = prepare_trace(path, config)
        result, events = analyze_trial(row, trace, config)
        if len(trace["t"]) and np.any(trace["step"] == "normal"):
            keep = trace["step"] == "normal"
            expected = [int(row["side"]) * float(row["s_n"]), float(row["l_n"])]
            if not np.allclose(np.median(trace["command"][keep, :2], axis=0), expected, atol=.002):
                raise ValueError(f"Force-frame mapping disagrees with targets: T{row['trial']}")
        results.append(result); traces[result["trial"]] = trace; all_events[result["trial"]] = events
        if len(results) % 30 == 0:
            print(f"Analyzed {len(results)}/{len(rows)} trials", flush=True)
    cells = cell_statistics(results)
    fit = fit_surface(cells, results)
    write_csv(out / "trials.csv", results)
    write_csv(out / "events.csv", [event for events in all_events.values() for event in events])
    write_csv(out / "cells.csv", cells)
    write_csv(out / "review_queue.csv", [r for r in results if r["review_required"]])
    style()
    plot_classification(results, out)
    plot_surface(results, cells, fit, out)
    plot_examples(results, traces, all_events, out)
    plot_correction(results, cells, fit, out)
    plot_fit_diagnostics(results, cells, fit, out)
    if not args.skip_audit_pdf:
        print("Writing all-trial review PDF", flush=True)
        audit_pdf(results, traces, all_events, out)
    audit = write_report(out, data, rows, results, cells, fit, config)
    print(json.dumps({k: v for k, v in audit.items() if k in (
        "first_event_categories", "n_reengaged", "n_normal_fit_trials", "n_supported_cells",
        "normal_command_reduction_median_n", "normal_command_reduction_max_n")}, indent=2))
    print(f"Artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
