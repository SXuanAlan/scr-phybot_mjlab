"""Wall -> pad wrench (force + moment) of every microspine EE over one rollout.

Rolls a climb checkpoint out for STEPS policy steps (one deterministic env,
play config, wall angle pinned) and records, per step and per end-effector,
the 6D wrench the WALL exerts on the pad:

  * latched phase — the weld/connect constraint reaction gathered from the
    equality efc rows (``MicrospineAffordance.anchor_load_tnl`` /
    ``anchor_moment_tnl``, the exact quantity the limit surface bills);
  * pressing phase — the sphere-grid contact compression against the wall
    (``pad_press_force``; the redundant contacts are OFF while latched, so
    the two channels never overlap).

Frame. The affordance judges loads in its engage basis (t̂ = rake / "down
the wall", l̂ = lateral, n̂_out = outward wall normal) as the load the PAD
puts on the wall: f_t > 0 = down-rake shear (hanging), f_n > 0 = pull-off
tension. The figure shows the wall->pad reaction in the right-handed wall
frame (û = -t̂ "up the wall", l̂, n̂_in = -n̂_out "into the wall"):

    F_up = f_t     F_lat = -f_l     F_in = f_n      (same for M with m_*)

so a hanging pad reads F_up > 0 (the wall carries it), a spine-tensioned pad
reads F_in > 0 (the wall pulls the pad in), a pressed pad reads
F_in = -press < 0 (the wall pushes back). The raw (t, l, n_out) loads are
kept in the .npz for anyone who wants the billing convention.

Usage (env_mjlab15, GPU):
  python scripts/plot_pad_wrench_timeseries.py <ckpt.pt> [--angle 90] \
      [--steps 1500] [--device cuda:0] [--out <prefix>]
  python scripts/plot_pad_wrench_timeseries.py --replot <prefix>.npz

PHYBOT_C2_PAD_GIMBAL=0 selects the legacy rigid-pad plant (the plant the
2026-08-02 deploypd-v4 checkpoints were trained on).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EE_SHORT = ("LH", "RH", "LF", "RF")
REASON_NAMES = {
  1: "reverse-shear(free)",
  2: "tension",
  3: "lateral",
  4: "ultimate",
  5: "tilt-peel",
  6: "dr-forced",
}
# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
C_UP, C_LAT, C_IN = "#2a78d6", "#eb6834", "#1baf7a"
C_CONTACT = "#52514e"
C_ENGAGED = "#1baf7a"
C_RELEASE = "#e34948"
C_WEIGHT = "#52514e"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("ckpt", nargs="?", type=Path, help="policy checkpoint (.pt)")
  p.add_argument("--task", default="Mjlab-Climb-Wall-Phybot-C2")
  p.add_argument("--angle", type=float, default=90.0, help="wall angle [deg]")
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--motion-term", default="climb_motion")
  p.add_argument("--aff-term", default="microspine_affordance")
  p.add_argument("--grip-event", default="microspine_grip")
  p.add_argument(
    "--out",
    type=Path,
    default=None,
    help="output prefix (default logs/eval/<run>/padwrench_<angle>deg_<ckpt>)",
  )
  p.add_argument(
    "--tilt-band",
    default=None,
    help=(
      "override the tilt-peel release band 'lo,hi' [deg] (legacy rigid-pad "
      "plant only; '0,0' disables). Play pins the FINAL ladder stage "
      "(3.5-4.5 deg), which a low-angle-era checkpoint never trained under "
      "(stage 0 is 12-15 deg)."
    ),
  )
  p.add_argument(
    "--replot", type=Path, default=None, help="re-render from a saved .npz"
  )
  args = p.parse_args()
  if args.replot is None and args.ckpt is None:
    p.error("a checkpoint (or --replot <npz>) is required")
  return args


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
def rollout(args: argparse.Namespace) -> dict:
  import torch

  import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
    PHYBOT_C2_PAD_GIMBAL,
  )

  ckpt = args.ckpt.resolve()
  device = args.device
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  if "wall_angle" in env_cfg.curriculum:
    env_cfg.curriculum["wall_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": args.angle}
    ]
  env_cfg.curriculum.pop("gravity_ladder", None)
  if args.tilt_band is not None:
    lo, hi = (float(x) for x in args.tilt_band.split(","))
    _ls = env_cfg.commands[args.aff_term].microspine_cfg.limit_surface
    _ls.tilt_peel_max_deg_range = (lo, hi)
    if "tilt_peel" in env_cfg.curriculum:
      env_cfg.curriculum["tilt_peel"].params["stages"] = [
        {"iter": 0, "deg_range": (lo, hi)}
      ]
    print(f"[tilt-band] tilt-peel release band pinned to {lo}-{hi} deg")
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  u = env.unwrapped
  motion = u.command_manager.get_term(args.motion_term)
  aff = u.command_manager.get_term(args.aff_term)
  executor = u.event_manager.get_term_cfg(args.grip_event).func
  robot = u.scene["robot"]
  dt = float(u.step_dt)
  ee_names = tuple(aff._mcfg.ee_body_names)
  H = len(ee_names)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  policy_set = "student" if hasattr(agent_cfg, "student") else "actor"
  runner.load(
    str(ckpt), load_cfg={policy_set: True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # Robot mass for the weight-support sanity line.
  mj = u.sim.mj_model
  robot_mass = 0.0
  for i in range(mj.nbody):
    name = mj.body(i).name
    if name.startswith("robot/") or name.startswith("robot_"):
      robot_mass += float(mj.body_mass[i])
  if robot_mass <= 0.0:
    robot_mass = float(
      sum(
        float(mj.body_mass[i])
        for i in range(mj.nbody)
        if "wall" not in mj.body(i).name and mj.body(i).name != "world"
      )
    )
  g = float(np.linalg.norm(np.asarray(env_cfg.sim.mujoco.gravity)))

  S = args.steps
  rec = {
    "load_tnl": np.zeros((S, H, 3), np.float32),  # pad->wall, engage basis
    "moment_tnl": np.zeros((S, H, 3), np.float32),
    "load_w": np.zeros((S, H, 3), np.float32),  # pad->wall, world
    "moment_w": np.zeros((S, H, 3), np.float32),
    "eff_load_tnl": np.zeros((S, H, 3), np.float32),  # worst-unit pad scale
    "press": np.zeros((S, H), np.float32),  # contact compression [N]
    "contact_count": np.zeros((S, H), np.float32),
    "engaged": np.zeros((S, H), bool),
    "ref_engaged": np.zeros((S, H), bool),
    "forced": np.zeros((S, H), bool),
    "reason": np.zeros((S, H), np.int8),
    "margin": np.ones((S, H), np.float32),
    "t_hat_w": np.zeros((S, H, 3), np.float32),
    "n_hat_w": np.zeros((S, H, 3), np.float32),
    "height": np.zeros(S, np.float32),
    "done": np.zeros(S, bool),
  }
  ret = env.get_observations()
  obs = ret[0] if isinstance(ret, (tuple, list)) else ret
  with torch.no_grad():
    for t in range(S):
      act = policy(obs)
      step_ret = env.step(act)
      obs, dones = step_ret[0], step_ret[2]
      t_hat, _, n_hat = aff._engage_frame()
      rec["load_tnl"][t] = aff.anchor_load_tnl[0].cpu().numpy()
      rec["moment_tnl"][t] = aff.anchor_moment_tnl[0].cpu().numpy()
      rec["load_w"][t] = executor.anchor_load_w[0].cpu().numpy()
      rec["moment_w"][t] = executor.anchor_moment_w[0].cpu().numpy()
      rec["eff_load_tnl"][t] = aff.anchor_effective_load_tnl[0].cpu().numpy()
      rec["press"][t] = aff.pad_press_force[0].cpu().numpy()
      rec["contact_count"][t] = aff.contact_count[0].cpu().numpy()
      rec["engaged"][t] = aff.engaged[0].bool().cpu().numpy()
      rec["ref_engaged"][t] = motion.ref_engaged[0].bool().cpu().numpy()
      rec["forced"][t] = aff.forced_release_this_step[0].bool().cpu().numpy()
      rec["reason"][t] = aff.release_reason[0].cpu().numpy()
      rec["margin"][t] = aff.hold_margin[0].cpu().numpy()
      rec["t_hat_w"][t] = t_hat[0].cpu().numpy()
      rec["n_hat_w"][t] = n_hat[0].cpu().numpy()
      rec["height"][t] = float(motion.height_gained[0].item())
      rec["done"][t] = bool(dones[0])
      if t % 250 == 0:
        print(f"[rollout] step {t}/{S}  t={t * dt:.1f}s", flush=True)
  env.close()

  rec.update(
    dt=np.float32(dt),
    ee_names=np.array(ee_names),
    angle_deg=np.float32(args.angle),
    ckpt=np.array(str(ckpt)),
    task=np.array(args.task),
    plant=np.array("gimbal" if PHYBOT_C2_PAD_GIMBAL else "rigid-pad"),
    robot_mass=np.float32(robot_mass),
    gravity=np.float32(g),
  )
  return rec


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
  """Contiguous True runs as (start, end_exclusive)."""
  out, start = [], None
  for i, v in enumerate(mask):
    if v and start is None:
      start = i
    elif not v and start is not None:
      out.append((start, i))
      start = None
  if start is not None:
    out.append((start, len(mask)))
  return out


def wall_to_pad(rec: dict) -> tuple[np.ndarray, np.ndarray]:
  """(F, M) wall->pad in the (up, lateral, into-wall) frame, (S, H, 3)."""
  L, Mo = rec["load_tnl"], rec["moment_tnl"]
  F = np.stack([L[..., 0], -L[..., 1], L[..., 2]], axis=-1)
  M = np.stack([Mo[..., 0], -Mo[..., 1], Mo[..., 2]], axis=-1)
  return F, M


def make_figure(rec: dict):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch

  dt = float(rec["dt"])
  S, H = rec["engaged"].shape
  time_s = np.arange(S) * dt
  F, M = wall_to_pad(rec)
  press = rec["press"]
  engaged = rec["engaged"]
  forced = rec["forced"]
  resets = np.where(rec["done"])[0]
  names = [str(n) for n in rec["ee_names"]]
  short = EE_SHORT if H == 4 else tuple(names)
  weight = float(rec["robot_mass"]) * float(rec["gravity"])

  plt.rcParams.update(
    {
      "font.family": ["DejaVu Sans", "Liberation Sans", "sans-serif"],
      "font.size": 9.5,
      "axes.titlesize": 10,
      "axes.labelsize": 9.5,
      "axes.edgecolor": "#b8b7b0",
      "axes.grid": True,
      "grid.color": "#e6e5df",
      "grid.linewidth": 0.6,
      "xtick.color": "#52514e",
      "ytick.color": "#52514e",
      "legend.frameon": False,
    }
  )
  fig, axes = plt.subplots(
    H + 1, 2, figsize=(17, 2.55 * (H + 1)), sharex=True,
    constrained_layout=True,
  )

  def shade(ax, h):
    for s0, e0 in _runs(engaged[:, h]):
      ax.axvspan(s0 * dt, e0 * dt, color=C_ENGAGED, alpha=0.10, lw=0)
    for r in resets:
      ax.axvline(r * dt, color="k", ls="--", lw=0.8)

  for h in range(H):
    axf, axm = axes[h]
    shade(axf, h)
    shade(axm, h)
    axf.plot(time_s, F[:, h, 0], color=C_UP, lw=1.5)
    axf.plot(time_s, F[:, h, 1], color=C_LAT, lw=1.5)
    axf.plot(time_s, F[:, h, 2], color=C_IN, lw=1.5)
    axf.plot(time_s, -press[:, h], color=C_CONTACT, lw=1.1, ls="--")
    axf.axhline(0.0, color="#b8b7b0", lw=0.8)
    axm.plot(time_s, M[:, h, 0], color=C_UP, lw=1.5)
    axm.plot(time_s, M[:, h, 1], color=C_LAT, lw=1.5)
    axm.plot(time_s, M[:, h, 2], color=C_IN, lw=1.5)
    axm.axhline(0.0, color="#b8b7b0", lw=0.8)
    # A single impact spike (a fall landing on a pad, a solver transient)
    # must not squash the whole trace: clip the y-range to the 99.5th
    # percentile when the peak is an outlier, and say so in the title.
    def _clip(ax, values: np.ndarray, unit: str, sym: str) -> str:
      v_abs = np.abs(values).ravel()
      v_p, v_max = float(np.percentile(v_abs, 99.5)), float(v_abs.max())
      if v_max > 3.0 * v_p and v_p > 0.0:
        ax.set_ylim(-1.15 * v_p, 1.15 * v_p)
        return f"   (y clipped at p99.5; peak |{sym}| = {v_max:.3g} {unit})"
      return ""

    clipped_f = _clip(
      axf, np.concatenate([F[:, h].ravel(), press[:, h]]), "N", "F"
    )
    clipped = _clip(axm, M[:, h], "N·m", "M")
    # Forced releases: a tick at the top of both panels.
    ts = np.where(forced[:, h])[0]
    if len(ts):
      for ax in (axf, axm):
        y_top = ax.get_ylim()[1]
        ax.plot(
          ts * dt, np.full(len(ts), y_top), "v", color=C_RELEASE, ms=5,
          ls="none", clip_on=False,
        )
    axf.set_ylabel("F wall→pad [N]")
    axm.set_ylabel("M wall→pad [N·m]")
    axf.set_title(f"{short[h]}  ({names[h]})  — force{clipped_f}", loc="left")
    axm.set_title(f"{short[h]}  ({names[h]})  — moment{clipped}", loc="left")

  # Bottom row: whole-robot support through the anchors + engaged count.
  axs, axc = axes[H]
  fz_anchor = -rec["load_w"][..., 2].sum(axis=1)  # world +z on the robot
  axs.plot(time_s, fz_anchor, color=C_UP, lw=1.5, label="Σ anchors: vertical force on robot (world +z)")
  axs.axhline(weight, color=C_WEIGHT, lw=1.1, ls="--", label=f"robot weight m·g = {weight:.0f} N")
  axs.axhline(0.0, color="#b8b7b0", lw=0.8)
  for r in resets:
    axs.axvline(r * dt, color="k", ls="--", lw=0.8)
  axs.set_ylabel("F_z [N]")
  axs.set_title("all pads — vertical support through the welds", loc="left")
  axs.set_xlabel("time [s]")
  axs.legend(loc="lower right", fontsize=8.5)

  axc.step(time_s, engaged.sum(axis=1), where="post", color=C_UP, lw=1.5, label="policy: pads latched")
  axc.step(time_s, rec["ref_engaged"].sum(axis=1), where="post", color=C_CONTACT, lw=1.1, ls="--", label="reference schedule")
  axc.set_ylim(-0.2, H + 0.4)
  axc.set_yticks(range(H + 1))
  axc.set_ylabel("# latched")
  axc.set_title("latched-pad count", loc="left")
  axc.set_xlabel("time [s]")
  axc.legend(loc="lower right", fontsize=8.5)
  for r in resets:
    axc.axvline(r * dt, color="k", ls="--", lw=0.8)

  axes[0, 0].set_xlim(0.0, S * dt)
  # One figure-level key above the panels (the same three colors carry the
  # force AND moment components), kept out of the data area.
  key = [
    Line2D([], [], color=C_UP, lw=2, label="along / about  û = −t̂  (up the wall, anti-rake)"),
    Line2D([], [], color=C_LAT, lw=2, label="along / about  l̂  (lateral)"),
    Line2D([], [], color=C_IN, lw=2, label="along / about  n̂_in = −n̂_out  (into the wall)"),
    Line2D([], [], color=C_CONTACT, lw=1.5, ls="--", label="sphere-grid contact compression  −press  (wall pushes pad out)"),
    Patch(facecolor=C_ENGAGED, alpha=0.25, label="pad latched (weld active)"),
    Line2D([], [], marker="v", color=C_RELEASE, ls="none", label="forced release (limit surface)"),
  ]
  fig.legend(handles=key, loc="outside lower center", ncol=3, fontsize=9)

  fig.suptitle(
    f"Wall → microspine-pad wrench per end-effector   |   "
    f"{Path(str(rec['ckpt'])).parent.name}/{Path(str(rec['ckpt'])).name}, "
    f"wall {float(rec['angle_deg']):.0f}°, plant={rec['plant']}, "
    f"dt={dt * 1e3:.0f} ms, {S * dt:.0f} s",
    fontsize=11,
  )
  return fig


def summarize(rec: dict) -> None:
  dt = float(rec["dt"])
  S, H = rec["engaged"].shape
  F, M = wall_to_pad(rec)
  names = EE_SHORT if H == 4 else tuple(str(n) for n in rec["ee_names"])
  weight = float(rec["robot_mass"]) * float(rec["gravity"])
  print(
    f"steps={S} ({S * dt:.1f}s) dt={dt:.3f}s angle={float(rec['angle_deg']):.0f}"
    f" plant={rec['plant']} ckpt={Path(str(rec['ckpt'])).name}"
  )
  print(f"robot mass={float(rec['robot_mass']):.2f} kg  weight={weight:.1f} N")
  print("per-EE wall->pad wrench while latched (mean / max|.|), N and N*m:")
  for h in range(H):
    on = rec["engaged"][:, h]
    if not on.any():
      print(f"  {names[h]}: never latched")
      continue
    f, m = F[on, h], M[on, h]
    print(
      f"  {names[h]}: occ={on.mean():.2f}"
      f" | F_up {f[:, 0].mean():7.1f}/{np.abs(f[:, 0]).max():6.1f}"
      f"  F_lat {f[:, 1].mean():6.1f}/{np.abs(f[:, 1]).max():6.1f}"
      f"  F_in {f[:, 2].mean():6.1f}/{np.abs(f[:, 2]).max():6.1f}"
      f" | M_up {m[:, 0].mean():6.2f}/{np.abs(m[:, 0]).max():5.2f}"
      f"  M_lat {m[:, 1].mean():6.2f}/{np.abs(m[:, 1]).max():5.2f}"
      f"  M_in {m[:, 2].mean():6.2f}/{np.abs(m[:, 2]).max():5.2f}"
      f" | press max {rec['press'][:, h].max():5.1f} N"
    )
  fz = -rec["load_w"][..., 2].sum(axis=1)
  all_on = rec["engaged"].all(axis=1)
  if all_on.any():
    print(
      f"vertical support through welds: mean {fz.mean():.1f} N over the run,"
      f" {fz[all_on].mean():.1f} N while all four latched (weight {weight:.1f} N)"
    )
  n_forced = int(rec["forced"].sum())
  hist = {}
  for r in rec["reason"][rec["forced"]]:
    hist[REASON_NAMES.get(int(r), str(int(r)))] = hist.get(
      REASON_NAMES.get(int(r), str(int(r))), 0
    ) + 1
  print(f"forced releases={n_forced} reasons={hist}")
  print(
    f"height gained (last)={rec['height'][-1]:.3f} m; resets at"
    f" {[round(i * dt, 1) for i in np.where(rec['done'])[0]]} s"
  )
  # Frame orientation readout: how "up" is -t_hat, in world z.
  up_z = -rec["t_hat_w"][..., 2]
  print(
    "frame check: mean world-z component of the up-wall axis (-t_hat) per EE ="
    + " ".join(f" {names[h]}={up_z[:, h].mean():.3f}" for h in range(H))
  )


def main() -> None:
  args = parse_args()
  if args.replot is not None:
    data = np.load(args.replot, allow_pickle=False)
    rec = {k: data[k] for k in data.files}
    prefix = args.out or args.replot.with_suffix("")
  else:
    rec = rollout(args)
    if args.out is not None:
      prefix = args.out
    else:
      run_name = args.ckpt.resolve().parent.name
      prefix = (
        REPO / "logs" / "eval" / run_name
        / f"padwrench_{args.angle:.0f}deg_{args.ckpt.stem}"
      )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prefix.with_suffix(".npz"), **rec)
    print(f"wrote {prefix.with_suffix('.npz')}")
  summarize(rec)
  fig = make_figure(rec)
  png = prefix.with_suffix(".png")
  pdf = prefix.with_suffix(".pdf")
  fig.savefig(png, dpi=150)
  fig.savefig(pdf)
  print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
  main()
