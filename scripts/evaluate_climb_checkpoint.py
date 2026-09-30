"""Evaluate the four-corner climbing actor without hiding resets or seed grips.

Example (CUDA_VISIBLE_DEVICES selects the physical evaluation GPU)::

  python scripts/evaluate_climb_checkpoint.py --checkpoint logs/.../model_250.pt \
    --angles 45 50 --num-envs 32 --duration-s 30 --output-dir logs/eval/model_250

Every angle produces metrics.json, trajectory.npz, engage_schedule.png and a
video of environment 0 from that same physical rollout. Terminations stay
enabled. We disable automatic reset only to record the terminal state, then
explicitly reset done worlds before the next physics step. No capacity,
reward, controller or failure threshold is changed.

The default canonical profile starts at reference frame 0 with ordinary DR
off; empirical per-unit strength/early-loading draws still run. The optional
training-rsi profile retains training DR and height offsets, with uniform
full-cycle phase coverage instead of the training failure-adaptive sampler.
Both profiles use strict actor loading, including observation normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASK = "Mjlab-Climb-Wall-Phybot-C2-Corners-A60"
PAD_NAMES = ("LH", "RH", "LF", "RF")
RELEASE_NAMES = {
  1: "reverse_shear", 2: "normal_capacity", 3: "lateral",
  4: "ultimate", 5: "tilt_peel", 6: "external_dr",
  7: "empirical_domain", 8: "empirical_early_loading",
}


def write_json(path: Path, value: dict) -> None:
  """Atomic publication: the monitor never consumes a partial result."""
  def clean(item):
    if isinstance(item, dict):
      return {key: clean(child) for key, child in item.items()}
    if isinstance(item, (list, tuple)):
      return [clean(child) for child in item]
    if isinstance(item, (float, np.floating)) and not math.isfinite(item):
      return None
    return item
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def finite_stats(values) -> dict:
  values = np.asarray(values, dtype=np.float64)
  values = values[np.isfinite(values)]
  if not values.size:
    return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
  return {"count": int(values.size), "mean": float(values.mean()),
          "median": float(np.median(values)), "min": float(values.min()),
          "max": float(values.max())}


def schedule_statistics(actual: np.ndarray, reference: np.ndarray) -> dict:
  """Balanced accuracy exposes all-four-latched policies despite high occupancy."""
  actual, reference = np.asarray(actual, bool), np.asarray(reference, bool)
  on_count, off_count = int(reference.sum()), int((~reference).sum())
  on_accuracy = float(actual[reference].mean()) if on_count else None
  off_accuracy = float((~actual[~reference]).mean()) if off_count else None
  balanced = (0.5 * (on_accuracy + off_accuracy)
              if on_accuracy is not None and off_accuracy is not None else None)
  return {"agreement": float((actual == reference).mean()) if actual.size else None,
          "reference_on_accuracy": on_accuracy, "reference_off_accuracy": off_accuracy,
          "balanced_accuracy": balanced, "reference_on_samples": on_count,
          "reference_off_samples": off_count}


def matched_schedule_edges(reference, actual_events, episode_id, tolerance_steps):
  """Match each real latch to at most one reference rising edge in its episode.

  Reset-to-seed transitions are excluded by episode_id. The reference rising
  edge is not itself evidence of a new physical grip.
  """
  reference, actual_events = np.asarray(reference, bool), np.asarray(actual_events, bool)
  expected = (~reference[:-1] & reference[1:]
              & (episode_id[1:] == episode_id[:-1])[:, None])
  expected_count = expected.sum(0).astype(int)
  matched = np.zeros(reference.shape[-1], dtype=int)
  for pad in range(reference.shape[-1]):
    available = set(np.flatnonzero(actual_events[:, pad]).tolist())
    for edge in np.flatnonzero(expected[:, pad]) + 1:
      candidates = [idx for idx in available
                    if abs(idx - edge) <= tolerance_steps and episode_id[idx] == episode_id[edge]]
      if candidates:
        selected = min(candidates, key=lambda idx: (abs(idx - edge), idx))
        available.remove(selected)
        matched[pad] += 1
  return expected_count, matched


def configure_evaluation(task, angle_deg, num_envs, profile, seed):
  """Build from the registered training task and pin only evaluation controls."""
  import scr_phybot_mjab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  if task != TASK:
    raise ValueError(f"This evaluation contract is for {TASK}; received {task}")
  train_cfg = load_env_cfg(task, play=False)
  cfg = load_env_cfg(task, play=(profile == "canonical"))
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.auto_reset = False
  # play uses an effectively infinite timeout; grade against the actual
  # training horizon instead. Failure terms themselves are unchanged.
  cfg.episode_length_s = train_cfg.episode_length_s
  cfg.curriculum["wall_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": float(angle_deg), "jitter_deg": 0.0}]
  cfg.curriculum["wall_angle"].params["angle_jitter_deg"] = 0.0
  motion = cfg.commands["climb_motion"]
  motion.sampling_mode = "start" if profile == "canonical" else "uniform"
  microspine = cfg.commands["microspine_affordance"].microspine_cfg
  event_microspine = cfg.events["microspine_grip"].params["microspine_cfg"]
  for owner in (microspine, event_microspine):
    if owner.limit_surface.model != "empirical_unit" or owner.limit_surface.unit_layout != "corners":
      raise ValueError("Evaluation requires the registered empirical four-corner hardware")
    owner.debug = False
  return cfg, load_rl_cfg(task)


def _numpy(tensor):
  return tensor.detach().cpu().numpy().copy()


def snapshot(env, motion, aff) -> dict:
  root = env.scene["robot"].data.root_link_pos_w
  reference = motion.ref_root_pos_w
  robot_altitude = ((root - motion.wall_pivot_w) * motion.wall_up_w).sum(-1)
  reference_altitude = ((reference - motion.wall_pivot_w) * motion.wall_up_w).sum(-1)
  return {
    "engaged": _numpy(aff.engaged), "reference_engaged": _numpy(motion.ref_engaged),
    "seeded": _numpy(aff.seeded), "new_latch": _numpy(aff.engaged_this_step & ~aff.seeded),
    "released_seed": _numpy(aff.released_was_seeded),
    "release_reason": _numpy(aff.release_reason), "phase_frame": _numpy(motion.time_steps),
    "robot_altitude_m": _numpy(robot_altitude), "reference_altitude_m": _numpy(reference_altitude),
    "unit_load_tln_N": _numpy(aff.unit_load_tln), "hold_margin": _numpy(aff.hold_margin),
    "episode_step": _numpy(env.episode_length_buf),
  }


def _episode_result(env_id, episode_id, start, last, elapsed_s, new_latches,
                    seed_releases, settled, end_type, termination_reasons):
  progress = float(last["robot_altitude_m"] - start["robot_altitude_m"])
  ref_progress = float(last["reference_altitude_m"] - start["reference_altitude_m"])
  settled_progress = (float(last["robot_altitude_m"] - settled["robot_altitude_m"])
                      if settled is not None else None)
  settled_ref_progress = (float(last["reference_altitude_m"] - settled["reference_altitude_m"])
                          if settled is not None else None)
  return {"env_id": env_id, "episode_id": episode_id, "duration_s": float(elapsed_s),
          "end_type": end_type, "termination_reasons": termination_reasons,
          "start_phase_frame": int(start["phase_frame"]),
          "end_phase_frame": int(last["phase_frame"]),
          "start_reference_engaged": start["reference_engaged"].tolist(),
          "initial_seed_grips": int(start["seeded"].sum()),
          "new_latches_by_pad": new_latches.tolist(),
          "released_seed_grips_by_pad": seed_releases.tolist(),
          "progress_from_birth_m": progress, "reference_progress_from_birth_m": ref_progress,
          "progress_after_settle_m": settled_progress,
          "reference_progress_after_settle_m": settled_ref_progress}


def _world(state, env_id):
  return {key: value[env_id].copy() if hasattr(value[env_id], "copy") else value[env_id]
          for key, value in state.items()}


def summarize(trajectory, episodes, num_envs, dt, duration_s, cycle_duration_s):
  """Use one initial episode per world for curriculum-facing success rates."""
  first = [ep for ep in episodes if ep["episode_id"] == 0]
  if len(first) != num_envs:
    raise ValueError("Every evaluation world must have exactly one initial episode outcome")
  survived = [ep["end_type"] != "terminated" and ep["duration_s"] >= duration_s - dt / 2 for ep in first]
  first_mask = trajectory["episode_id"] == 0
  schedule = schedule_statistics(trajectory["engaged"][first_mask], trajectory["reference_engaged"][first_mask])
  expected, matched = np.zeros(4, int), np.zeros(4, int)
  per_env = []
  cycle_frames = round(cycle_duration_s / dt)
  for env_id in range(num_envs):
    mask = first_mask[:, env_id]
    episode = next(ep for ep in first if ep["env_id"] == env_id)
    e, m = matched_schedule_edges(
      np.concatenate([np.asarray(episode["start_reference_engaged"])[None],
                      trajectory["reference_engaged"][mask, env_id]]),
      np.concatenate([np.zeros((1, 4), bool), trajectory["new_latch"][mask, env_id]]),
      np.zeros(int(mask.sum()) + 1, int), round(0.3 / dt))
    expected += e
    matched += m
    actual = trajectory["engaged"][mask, env_id]
    reference = trajectory["reference_engaged"][mask, env_id]
    local_schedule = schedule_statistics(actual, reference)
    occ_actual, occ_reference = actual.mean(0), reference.mean(0)
    phase = np.r_[episode["start_phase_frame"], trajectory["phase_frame"][mask, env_id]]
    advanced = int((np.diff(phase) % cycle_frames).sum())
    ref_progress = episode["reference_progress_after_settle_m"]
    progress = episode["progress_after_settle_m"]
    ratio = (float(progress / ref_progress)
             if progress is not None and ref_progress is not None
             and math.isfinite(progress) and math.isfinite(ref_progress) and ref_progress > .005 else None)
    per_env.append({
      "env_id": env_id, "duration_s": episode["duration_s"],
      "survived_horizon": episode["end_type"] != "terminated" and episode["duration_s"] >= duration_s - dt / 2,
      "engage_match": local_schedule["agreement"], "schedule": local_schedule,
      "actual_occupancy_by_pad": occ_actual.tolist(),
      "reference_occupancy_by_pad": occ_reference.tolist(),
      "pad_occupancy_excess_per_pad": (occ_actual - occ_reference).tolist(),
      "progress_ratio": ratio, "reference_cycles_completed": advanced // cycle_frames,
      "reference_frames_advanced": advanced,
      "true_regrips_per_pad": episode["new_latches_by_pad"],
      "true_regrip_cycles": min(episode["new_latches_by_pad"]),
      "scheduled_regrip_expected_per_pad": e.tolist(), "scheduled_regrip_matched_per_pad": m.tolist(),
    })
  new_by_pad = np.asarray([ep["new_latches_by_pad"] for ep in first]).sum(0)
  progress = [ep["progress_after_settle_m"] for ep in first if ep["progress_after_settle_m"] is not None]
  termination_counts = {}
  for ep in episodes:
    for term in ep["termination_reasons"]:
      termination_counts[term] = termination_counts.get(term, 0) + 1
  reason_counts = {name: int((trajectory["release_reason"] == reason).sum())
                   for reason, name in RELEASE_NAMES.items()}
  # This is explicitly an automatic candidate check, not a substitute for
  # the schedule plot and video review. A static seeded hang cannot pass.
  sufficient_horizon = duration_s >= 30. - dt / 2
  first_progress_ok = [ep["progress_ratio"] is not None and ep["progress_ratio"] >= .8 for ep in per_env]
  latch_ok = [ep["true_regrip_cycles"] >= 2 for ep in per_env]
  engage_match_min = min(ep["engage_match"] for ep in per_env)
  occupancy_excess_max = np.asarray([ep["pad_occupancy_excess_per_pad"] for ep in per_env]).max(0)
  valid_progress = [ep["progress_ratio"] for ep in per_env if ep["progress_ratio"] is not None]
  progress_ratio_min = min(valid_progress) if len(valid_progress) == num_envs else None
  reference_cycles_min = min(ep["reference_cycles_completed"] for ep in per_env)
  true_regrip_cycles_min = min(ep["true_regrip_cycles"] for ep in per_env)
  candidate_checks = {
    "at_least_30_seconds": sufficient_horizon,
    "all_first_episodes_survive_horizon": bool(all(survived)),
    "all_first_episodes_complete_two_reference_cycles": reference_cycles_min >= 2,
    "all_first_episodes_progress_ratio_at_least_80pct": bool(all(first_progress_ok)),
    "all_first_episodes_each_pad_regrips_at_least_twice": bool(all(latch_ok)),
    "worst_env_engage_match_at_least_95pct": engage_match_min >= .95,
    "worst_env_each_pad_occupancy_excess_below_4pct": bool((occupancy_excess_max < .04).all()),
  }
  return {
    "first_episode_count": num_envs,
    "first_episode_falls": sum(ep["end_type"] == "terminated" for ep in first),
    "first_episode_survival_rate": float(np.mean(survived)),
    "full_30s_survival_rate": float(np.mean([
      ep["end_type"] != "terminated" and ep["duration_s"] >= 30. - dt / 2 for ep in first])),
    "engage_match_min": engage_match_min,
    "engage_match_mean": float(np.mean([ep["engage_match"] for ep in per_env])),
    "pad_occupancy_excess_max_per_pad": occupancy_excess_max.tolist(),
    "actual_occupancy_mean_per_pad": np.asarray([ep["actual_occupancy_by_pad"] for ep in per_env]).mean(0).tolist(),
    "reference_occupancy_mean_per_pad": np.asarray([ep["reference_occupancy_by_pad"] for ep in per_env]).mean(0).tolist(),
    "progress_ratio_min": progress_ratio_min, "progress_ratio": finite_stats(valid_progress),
    "reference_cycles_completed_min": reference_cycles_min,
    "true_regrip_cycles_min": true_regrip_cycles_min,
    "per_env": per_env,
    "first_episode_duration_s": finite_stats([ep["duration_s"] for ep in first]),
    "first_episode_progress_from_birth_m": finite_stats([ep["progress_from_birth_m"] for ep in first]),
    "first_episode_progress_after_settle_m": finite_stats(progress),
    "first_episode_reference_progress_after_settle_m": finite_stats([
      ep["reference_progress_after_settle_m"] for ep in first if ep["reference_progress_after_settle_m"] is not None]),
    "first_episode_progress_success_rate": float(np.mean(first_progress_ok)),
    "first_episode_each_pad_regripped_rate": float(np.mean(latch_ok)),
    "new_latches_by_pad": dict(zip(PAD_NAMES, new_by_pad.tolist())),
    "initial_seed_grips": sum(ep["initial_seed_grips"] for ep in first),
    "released_seed_grips_by_pad": dict(zip(PAD_NAMES, np.asarray([
      ep.get("released_seed_grips_by_pad", [0] * 4) for ep in first]).sum(0).tolist())),
    "schedule_balanced_accuracy": schedule["balanced_accuracy"], "schedule": schedule,
    "scheduled_regrip_expected_by_pad": dict(zip(PAD_NAMES, expected.tolist())),
    "scheduled_regrip_matched_by_pad": dict(zip(PAD_NAMES, matched.tolist())),
    "termination_counts": termination_counts, "release_reason_counts": reason_counts,
    "total_episodes_observed": len(episodes),
    "automatic_candidate_checks": candidate_checks,
    "automatic_candidate_pass": bool(all(candidate_checks.values())),
    "visual_review_required": True,
  }


def plot_rollout(path: Path, trajectory: dict, dt: float, angle: float, profile: str):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap
  from matplotlib.patches import Patch

  time_s = np.arange(1, len(trajectory["episode_id"]) + 1) * dt
  fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3.2, 1.2, 1.2]}, layout="constrained")
  actual = trajectory["engaged"][:, 0]
  seed = trajectory["seeded"][:, 0]
  reference = trajectory["reference_engaged"][:, 0]
  raster = np.zeros((8, len(time_s)), dtype=int)
  labels = []
  for pad, name in enumerate(PAD_NAMES):
    raster[2 * pad] = reference[:, pad]
    raster[2 * pad + 1] = np.where(actual[:, pad], np.where(seed[:, pad], 2, 3), 0)
    labels.extend((f"{name} reference", f"{name} actual"))
  axes[0].imshow(raster, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                 extent=(0, time_s[-1], 7.5, -0.5),
                 cmap=ListedColormap(["#eeeeee", "#4178a8", "#d8a542", "#32975a"]))
  axes[0].set_yticks(range(8), labels)
  for pad in range(4):
    events = trajectory["new_latch"][:, 0, pad]
    axes[0].scatter(time_s[events], np.full(events.sum(), 2 * pad + 1), marker="^", c="#003d1e", s=16)
    released = trajectory["release_reason"][:, 0, pad] > 0
    axes[0].scatter(time_s[released], np.full(released.sum(), 2 * pad + 1), marker="x", c="#b32020", s=16)
  axes[0].legend(handles=[Patch(color="#4178a8", label="reference ON"),
                          Patch(color="#d8a542", label="seeded grip"),
                          Patch(color="#32975a", label="policy-created grip")], ncol=3, loc="upper right")
  axes[0].set_title(f"{angle:g} degrees | {profile} | environment 0 | triangles=new physical latch; x=release")
  axes[1].plot(time_s, trajectory["robot_altitude_m"][:, 0], label="robot along-wall altitude")
  axes[1].plot(time_s, trajectory["reference_altitude_m"][:, 0], label="reference", alpha=0.8)
  axes[1].set_ylabel("Altitude (m)")
  axes[1].legend(loc="upper left")
  axes[2].plot(time_s, trajectory["phase_frame"][:, 0], color="#7252a1")
  axes[2].set_ylabel("Reference frame")
  axes[2].set_xlabel("Rollout time (s); red dashed lines mark explicit episode resets")
  for t in time_s[trajectory["done"][:, 0]]:
    for ax in axes:
      ax.axvline(t, color="#b32020", linestyle="--", linewidth=0.7, alpha=0.7)
  for ax in axes[1:]:
    ax.grid(alpha=0.2)
  fig.savefig(path, dpi=140)
  plt.close(fig)


def annotate_frame(frame, *, t, episode, state, ended, angle, checkpoint):
  from PIL import Image, ImageDraw, ImageFont
  frame = Image.fromarray(frame)
  draw = ImageDraw.Draw(frame)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
  except OSError:
    font = ImageFont.load_default()
  actual = " ".join(str(int(v)) for v in state["engaged"][0])
  reference = " ".join(str(int(v)) for v in state["reference_engaged"][0])
  seeded = " ".join(str(int(v)) for v in state["seeded"][0])
  lines = [f"{checkpoint} | {angle:g} deg | t={t:.2f}s | episode={episode} | phase={int(state['phase_frame'][0])}",
           f"LH RH LF RF   actual: {actual}   reference: {reference}   seeds: {seeded}",
           f"Along-wall altitude: {state['robot_altitude_m'][0]:.3f} m" + ("  EPISODE ENDED" if ended else "")]
  draw.rectangle((0, 0, frame.width, 86), fill=(15, 20, 25))
  for row, line in enumerate(lines):
    draw.text((12, 6 + 26 * row), line, font=font, fill=(255, 130, 100) if ended else (245, 245, 245))
  return np.asarray(frame)


def save_contact_sheet(path, frames):
  """Regular video samples plus event stills; labels retain exact sim times."""
  from PIL import Image, ImageDraw
  width, height = 640, 360
  columns = 3
  rows = math.ceil(len(frames) / columns)
  sheet = Image.new("RGB", (columns * width, rows * (height + 25)), (235, 235, 235))
  draw = ImageDraw.Draw(sheet)
  for index, (label, frame) in enumerate(frames):
    x, y = index % columns * width, index // columns * (height + 25)
    sheet.paste(Image.fromarray(frame).resize((width, height)), (x, y + 25))
    draw.text((x + 8, y + 6), label, fill=(10, 10, 10))
  sheet.save(path)


def evaluate_angle(args, angle, checkpoint_info):
  import imageio.v2 as imageio
  import torch
  from dataclasses import asdict
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_runner_cls
  from mjlab.viewer import ViewerConfig

  out = args.output_dir / f"{args.profile}_{angle:g}deg"
  out.mkdir(parents=True, exist_ok=True)
  cfg, agent_cfg = configure_evaluation(args.task, angle, args.num_envs, args.profile, args.seed)
  cfg.viewer = ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                            entity_name="robot", distance=2.5, elevation=-12.0,
                            azimuth=70.0, height=args.height, width=args.width, max_extra_envs=0)
  env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array" if args.video else None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  motion = env.command_manager.get_term("climb_motion")
  aff = env.command_manager.get_term("microspine_affordance")
  if aff.unit_load_tln.shape[-2] != 4:
    raise RuntimeError("Expected four active units per pad in the physical runtime")
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  # Strict loading must include the actual actor state and normalization;
  # verify equality instead of relying on a permissive runner API.
  loaded_actor = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  runtime_actor = runner.alg._raw_actor.state_dict()
  if set(loaded_actor) != set(runtime_actor) or any(
      not torch.equal(value.cpu(), runtime_actor[key].detach().cpu()) for key, value in loaded_actor.items()):
    raise RuntimeError("Actor state verification failed after checkpoint load")
  dt = float(env.step_dt)
  steps = int(round(args.duration_s / dt))
  duration_s = steps * dt
  stride = max(1, int(round(1.0 / (args.video_fps * dt))))
  video_path = out / "rollout.mp4"
  writer = imageio.get_writer(video_path, fps=1.0 / (stride * dt), codec="libx264", quality=7) if args.video else None
  histories: dict[str, list] = {}
  sheet_frames = []
  seen_events = set()
  regular_stills = set(np.linspace(1, steps, 7, dtype=int).tolist())
  previous_video_frame = None
  episodes = []
  ids = np.zeros(args.num_envs, int)
  elapsed = np.zeros(args.num_envs, float)
  latches = np.zeros((args.num_envs, 4), int)
  released_seeds = np.zeros_like(latches)
  settled = [None] * args.num_envs
  peak_total_contacts, peak_constraint_rows, overflow_steps = 0, 0, 0
  started = time.monotonic()
  try:
    with torch.no_grad():
      # The wrapper already reset before actor construction. Re-seed and
      # reset after construction so policy initialization cannot change RSI
      # or empirical draws between different actor architectures.
      env.reset(seed=args.seed)
      obs = wrapped.get_observations()
      state = snapshot(env, motion, aff)
      actual_angles = _numpy(torch.rad2deg(torch.asin(motion.wall_up_w[:, 2].clamp(-1., 1.))))
      if not np.allclose(actual_angles, angle, atol=1.e-3, rtol=0.):
        raise RuntimeError(f"Requested {angle} degrees but runtime walls are {actual_angles.tolist()}")
      starts = [_world(state, idx) for idx in range(args.num_envs)]
      if writer is not None:
        initial_frame = annotate_frame(env.render(), t=0., episode=0, state=state,
                                        ended=False, angle=angle, checkpoint=args.checkpoint.name)
        writer.append_data(initial_frame)
        sheet_frames.append(("Initial seeded stance: t=0.00s", initial_frame))
        previous_video_frame = initial_frame
      for step in range(steps):
        actions = policy(obs)
        obs, _, done_tensor, _ = wrapped.step(actions)
        state = snapshot(env, motion, aff)
        done = _numpy(done_tensor).astype(bool)
        terminated = _numpy(env.reset_terminated).astype(bool)
        timeout = _numpy(env.reset_time_outs).astype(bool)
        elapsed += dt
        latches += state["new_latch"]
        released_seeds += state["released_seed"]
        for env_id in range(args.num_envs):
          if settled[env_id] is None and elapsed[env_id] >= args.settle_s - dt / 2:
            settled[env_id] = _world(state, env_id)
        terms = {name: _numpy(env.termination_manager.get_term(name))
                 for name in env.termination_manager.active_terms}
        for key, value in state.items():
          histories.setdefault(key, []).append(value)
        histories.setdefault("episode_id", []).append(ids.copy())
        histories.setdefault("done", []).append(done.copy())
        histories.setdefault("terminated", []).append(terminated.copy())
        histories.setdefault("timeout", []).append(timeout.copy())
        contacts, constraints = int(env.sim.data.nacon.max()), int(env.sim.data.nefc.max())
        peak_total_contacts = max(peak_total_contacts, contacts)
        peak_constraint_rows = max(peak_constraint_rows, constraints)
        overflow_steps += int(contacts > env.sim.wp_data.naconmax or constraints > cfg.sim.njmax)
        event_labels = []
        for event, happened in (("first release", bool(state["release_reason"][0].any())),
                                ("first physical new latch", bool(state["new_latch"][0].any())),
                                ("first episode termination", bool(terminated[0]))):
          if happened and event not in seen_events:
            seen_events.add(event)
            event_labels.append(event)
        if writer is not None and ((step + 1) % stride == 0 or step + 1 in regular_stills or event_labels):
          frame = annotate_frame(env.render(), t=(step + 1) * dt, episode=int(ids[0]),
                                 state=state, ended=bool(done[0]), angle=angle, checkpoint=args.checkpoint.name)
          if (step + 1) % stride == 0:
            writer.append_data(frame)
          if event_labels:
            if "first episode termination" in event_labels and previous_video_frame is not None:
              sheet_frames.append(("Video sample immediately before first termination", previous_video_frame))
            label = ", ".join(event_labels)
            sheet_frames.append((f"{label}: t={(step + 1) * dt:.2f}s", frame))
            from PIL import Image
            Image.fromarray(frame).save(out / f"event_{step + 1:05d}.png")
          elif step + 1 in regular_stills:
            sheet_frames.append((f"Regular rollout sample: t={(step + 1) * dt:.2f}s", frame))
          if (step + 1) % stride == 0:
            previous_video_frame = frame
        for env_id in np.flatnonzero(done):
          episodes.append(_episode_result(
            int(env_id), int(ids[env_id]), starts[env_id], _world(state, env_id), elapsed[env_id],
            latches[env_id], released_seeds[env_id], settled[env_id],
            "terminated" if terminated[env_id] else "timeout",
            [name for name, values in terms.items() if values[env_id] and not env.termination_manager.get_term_cfg(name).time_out]))
        if step + 1 < steps and done.any():
          reset_ids = torch.as_tensor(np.flatnonzero(done), device=args.device)
          env.reset(env_ids=reset_ids)
          obs = wrapped.get_observations()
          reset_state = snapshot(env, motion, aff)
          for env_id in np.flatnonzero(done):
            ids[env_id] += 1
            elapsed[env_id] = 0.
            latches[env_id] = 0
            released_seeds[env_id] = 0
            settled[env_id] = None
            starts[env_id] = _world(reset_state, env_id)
        if step + 1 == steps:
          for env_id in np.flatnonzero(~done):
            episodes.append(_episode_result(
              int(env_id), int(ids[env_id]), starts[env_id], _world(state, env_id), elapsed[env_id],
              latches[env_id], released_seeds[env_id], settled[env_id], "evaluation_horizon", []))
        if (step + 1) % max(1, round(5 / dt)) == 0:
          print(f"[eval] {angle:g}deg {args.profile}: {(step + 1) * dt:.1f}/{duration_s:.1f}s; "
                f"completed episodes={len(episodes)}", flush=True)
  finally:
    if writer is not None:
      writer.close()
    wrapped.close()
  trajectory = {key: np.stack(values) for key, values in histories.items()}
  np.savez_compressed(out / "trajectory.npz", **trajectory, dt=np.asarray(dt), pad_names=PAD_NAMES)
  plot_rollout(out / "engage_schedule.png", trajectory, dt, angle, args.profile)
  if sheet_frames:
    save_contact_sheet(out / "video_contact_sheet.png", sheet_frames)
  result = summarize(trajectory, episodes, args.num_envs, dt, duration_s,
                     motion.clip.cycle_frames / motion.clip.fps)
  result.update({"angle_deg": angle, "profile": args.profile, "seed": args.seed,
                 "task": args.task, "num_envs": args.num_envs, "duration_s": duration_s,
                 "actual_wall_angle_deg": finite_stats(actual_angles),
                 "settle_s": args.settle_s, "step_dt": dt, "wall_clock_s": time.monotonic() - started,
                 "checkpoint": checkpoint_info, "actor_state_verified": True,
                 "hardware": {"model": "empirical_unit", "unit_layout": "corners", "units_per_pad": 4},
                 "motion_file": motion.cfg.motion_file, "motion_sha256": sha256(Path(motion.cfg.motion_file)),
                 "ordinary_dr_enabled": args.profile == "training-rsi",
                 "empirical_draws_enabled": True, "auto_reset": False,
                 "active_termination_terms": list(terms),
                 "peak_total_contacts": peak_total_contacts, "peak_constraint_rows_per_world": peak_constraint_rows,
                 "buffer_overflow_observed_steps": overflow_steps,
                 "artifacts": {"metrics": str((out / "metrics.json").resolve()),
                               "trajectory": str((out / "trajectory.npz").resolve()),
                               "engage_schedule": str((out / "engage_schedule.png").resolve()),
                               "contact_sheet": str((out / "video_contact_sheet.png").resolve()) if sheet_frames else None,
                               "video": str(video_path.resolve()) if args.video else None},
                 "episodes": episodes})
  write_json(out / "metrics.json", result)
  print(f"[eval] result={out / 'metrics.json'} survival={result['first_episode_survival_rate']:.3f} "
        f"new_latches={result['new_latches_by_pad']} candidate={result['automatic_candidate_pass']}", flush=True)
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--angles", "--angle-deg", required=True, nargs="+", type=float,
                      help="Explicit current mean and band upper edge (e.g. 45 50); no default 60-degree exam")
  parser.add_argument("--task", default=TASK)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--duration-s", type=float, default=30.)
  parser.add_argument("--settle-s", type=float, default=0.5)
  parser.add_argument("--profile", choices=("canonical", "training-rsi"), default="canonical")
  parser.add_argument("--seed", type=int, default=20260908)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--video-fps", type=float, default=25.)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args(argv)
  if args.num_envs < 1 or args.duration_s <= 0 or args.settle_s < 0 or args.video_fps <= 0:
    parser.error("num-envs, duration-s and video-fps must be positive; settle-s must be nonnegative")
  if any(not math.isfinite(angle) or not 0 < angle <= 60 for angle in args.angles):
    parser.error("The reduced-hardware curriculum evaluation supports explicit angles in (0, 60]")
  args.checkpoint = args.checkpoint.resolve(strict=True)
  args.output_dir = args.output_dir.resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MUJOCO_GL", "egl")
  sys.path.insert(0, str(REPO / "src"))
  import torch
  from mjlab.utils.torch import configure_torch_backends
  configure_torch_backends()
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if "actor_state_dict" not in checkpoint:
    raise ValueError("This is not a PPO actor checkpoint; refusing a silent random-policy evaluation")
  match = re.fullmatch(r"model_(\d+)", args.checkpoint.stem)
  checkpoint_info = {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
                     "iteration": int(checkpoint.get("iter", match.group(1) if match else -1))}
  summary = {"schema_version": 1, "checkpoint": checkpoint_info, "checkpoint_iter": checkpoint_info["iteration"],
             "script_sha256": sha256(Path(__file__)), "results": [], "complete": False}
  write_json(args.output_dir / "summary.json", summary)
  for angle in dict.fromkeys(args.angles):
    summary["results"].append(evaluate_angle(args, angle, checkpoint_info))
    write_json(args.output_dir / "summary.json", summary)
  summary["complete"] = True
  write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
  main()
