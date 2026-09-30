"""Chained mount->climb exam: the Plan-C baseline gate.

Runs the mount policy from a DETERMINISTIC grid of standing spawn offsets
(the blind-delivery scatter, enumerated instead of sampled), switches each
env to the frozen climb policy the first time ``mount_handoff_ready``
fires, and scores survival through a hold window. This is the seam
measurement of docs/plans/teacher-curriculum plan §5 / stage T2: the
two-policy (Plan C) baseline every unified student must beat.

The climb side can run in TEACHER form (privileged obs, default) or in
DEPLOY form (--estimator: the actor's privileged slices are substituted by
phi estimates, mirroring train_finetune_est.py) — teacher chains gate the
specialists, deploy chains gate the ship pair.

Example (27-cell grid, teacher chain):
  python scripts/chained_mount_exam.py \\
    --mount-ckpt logs/.../transition-vX/model_NNNN.pt \\
    --climb-ckpt logs/.../deploypd-v4-rung85/model_5050.pt

Band tasks: --task Mjlab-Climb-Transition-Band80-Phybot-C2 (exam runs the
band's play center).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
  sys.path.insert(0, str(_REPO_SRC))

# -- pure helpers (unit-tested; keep torch-only, no env dependencies) ---------


def make_offset_grid(xs, ys, yaws_deg):
  """Enumerate the (dx, dy, dyaw_rad) grid, x-major then y then yaw.

  Returns a (N, 3) float tensor; ordering is deterministic so cell indices
  are stable across runs and reports.
  """
  import torch

  rows = []
  for dx in xs:
    for dy in ys:
      for dyaw in yaws_deg:
        rows.append((float(dx), float(dy), float(dyaw) * torch.pi / 180.0))
  return torch.tensor(rows, dtype=torch.float32)


def apply_offset_grid(root_pos, root_quat, offsets):
  """Apply per-env (dx, dy, dyaw) to root pose — world x/y/yaw semantics.

  The exam mirrors ``_apply_spawn_pose_noise``: the scatter models the walk
  delivery error of a robot standing on the horizontal floor, so world
  axes are correct at every wall angle. z untouched; yaw premultiplies.
  """
  from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

  import torch

  pos = root_pos.clone()
  pos[:, 0] += offsets[:, 0]
  pos[:, 1] += offsets[:, 1]
  axis = torch.zeros(offsets.shape[0], 3, device=root_pos.device)
  axis[:, 2] = 1.0
  quat = quat_mul(
    quat_from_angle_axis(offsets[:, 2].to(root_pos.device), axis), root_quat
  )
  return pos, quat


# -- exam ---------------------------------------------------------------------

_TASK_BUILDER_KWARGS = {
  "Mjlab-Climb-Transition-Phybot-C2": {},
  "Mjlab-Climb-Transition-Band80-Phybot-C2": {"wall_band": 80.0},
  "Mjlab-Climb-Transition-Band88-Phybot-C2": {"wall_band": 88.0},
}


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--mount-ckpt", type=Path, required=True)
  p.add_argument("--climb-ckpt", type=Path, required=True)
  p.add_argument("--estimator", type=Path, default=None,
                 help="phi .pt (+ sibling .json meta): climb side runs the "
                 "DEPLOY form with privileged slices substituted")
  p.add_argument("--task", default="Mjlab-Climb-Transition-Phybot-C2",
                 choices=sorted(_TASK_BUILDER_KWARGS))
  p.add_argument("--grid-x", default="-0.05,0.0,0.05",
                 help="comma list of wall-distance offsets (m)")
  p.add_argument("--grid-y", default="-0.03,0.0,0.03",
                 help="comma list of lateral offsets (m)")
  p.add_argument("--grid-yaw-deg", default="-5.0,0.0,5.0",
                 help="comma list of yaw offsets (deg)")
  p.add_argument("--mount-timeout-s", type=float, default=25.0)
  p.add_argument("--hold-s", type=float, default=10.0,
                 help="post-handoff survival window on the climb policy")
  p.add_argument("--gpu-id", type=int, default=0)
  p.add_argument("--out", type=Path, default=None)
  return p.parse_args()


def _floats(csv: str) -> list[float]:
  return [float(tok) for tok in csv.split(",") if tok.strip()]


def main() -> None:
  args = _parse_args()
  for f in (args.mount_ckpt, args.climb_ckpt):
    if not f.exists():
      raise FileNotFoundError(f)
  if args.estimator is not None and not args.estimator.exists():
    raise FileNotFoundError(args.estimator)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0"

  from dataclasses import asdict

  import torch

  import scr_phybot_mjab.tasks  # noqa: F401 - registers tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from scr_phybot_mjab.tasks.climb_transition.config.phybot_c2.env_cfgs import (
    phybot_c2_climb_transition_env_cfg,
  )
  from scr_phybot_mjab.tasks.climb_transition.mdp.metrics import (
    mount_handoff_ready,
  )

  configure_torch_backends()

  offsets = make_offset_grid(
    _floats(args.grid_x), _floats(args.grid_y), _floats(args.grid_yaw_deg)
  ).to(device)
  n_cells = offsets.shape[0]

  deploy_form = args.estimator is not None
  env_cfg = phybot_c2_climb_transition_env_cfg(
    play=True,
    student_obs=deploy_form,
    **_TASK_BUILDER_KWARGS[args.task],
  )
  env_cfg.scene.num_envs = n_cells
  # The chain must outlive mount timeout + hold window; never time out
  # mid-exam.
  env_cfg.episode_length_s = args.mount_timeout_s + args.hold_s + 5.0
  agent_cfg = load_rl_cfg("Mjlab-Climb-Transition-Phybot-C2")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  def _policy(ckpt: Path):
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return runner.get_inference_policy(device=device)

  policy_mount = _policy(args.mount_ckpt)
  policy_climb = _policy(args.climb_ckpt)

  substitute = None
  if deploy_form:
    phi = torch.jit.load(str(args.estimator), map_location=device).eval()
    meta = json.loads(args.estimator.with_suffix(".json").read_text())
    target_sl = [(t["actor_start"], t["actor_end"]) for t in meta["targets"]]
    zero_sl = [(z["actor_start"], z["actor_end"]) for z in meta["zero_slices"]]

    def substitute(obs):
      with torch.no_grad():
        est = phi(obs["student"])
      actor = obs["actor"].clone()
      ofs = 0
      for s, e in target_sl:
        actor[:, s:e] = est[:, ofs : ofs + (e - s)]
        ofs += e - s
      for s, e in zero_sl:
        actor[:, s:e] = 0.0
      out = obs.clone()
      out["actor"] = actor
      return out

  # -- reset to the canonical start, then stamp the grid offsets -------------
  env.reset()
  robot = env.scene["robot"]
  ids = torch.arange(n_cells, device=device)
  pos = robot.data.root_link_pos_w[ids].clone()
  quat = robot.data.root_link_quat_w[ids].clone()
  pos, quat = apply_offset_grid(pos, quat, offsets)
  zeros3 = torch.zeros(n_cells, 3, device=device)
  robot.write_root_state_to_sim(
    torch.cat([pos, quat, zeros3, zeros3], dim=-1), env_ids=ids
  )
  robot.clear_state(env_ids=ids)
  obs = wrapped.get_observations()

  dt = float(env.step_dt)
  mount_deadline = int(round(args.mount_timeout_s / dt))
  hold_steps = int(round(args.hold_s / dt))
  max_steps = mount_deadline + hold_steps + 10

  switched = torch.zeros(n_cells, dtype=torch.bool, device=device)
  finished = torch.zeros_like(switched)
  survived = torch.zeros_like(switched)
  reached = torch.zeros_like(switched)
  t_handoff = torch.full((n_cells,), -1, dtype=torch.long, device=device)
  t_switch = torch.zeros(n_cells, dtype=torch.long, device=device)

  with torch.no_grad():
    for step in range(max_steps):
      a_mount = policy_mount(obs)
      a_climb = policy_climb(substitute(obs) if substitute else obs)
      actions = torch.where(switched.unsqueeze(-1), a_climb, a_mount)
      obs, _, dones, _ = wrapped.step(actions)
      dones = dones.to(torch.bool).squeeze(-1) if dones.dim() > 1 else dones.to(torch.bool)

      # Handoff trigger: first frame with the terminal stance latched.
      ready = mount_handoff_ready(env) > 0.5
      newly = ready & ~switched & ~finished
      if bool(newly.any()):
        switched |= newly
        reached |= newly
        t_handoff = torch.where(newly, torch.full_like(t_handoff, step), t_handoff)
        t_switch = torch.where(newly, torch.full_like(t_switch, step), t_switch)

      # Terminations: a done env fell (episode length is beyond the exam).
      fell = dones & ~finished
      if bool(fell.any()):
        finished |= fell

      # Survived the hold window on the climb policy.
      done_hold = switched & ~finished & ((step - t_switch) >= hold_steps)
      if bool(done_hold.any()):
        survived |= done_hold
        finished |= done_hold

      # Mount timeout: never reached handoff in time.
      if step == mount_deadline:
        finished |= ~switched

      if bool(finished.all()):
        break

  # -- report ----------------------------------------------------------------
  rows = []
  for i in range(n_cells):
    rows.append(
      {
        "dx": round(float(offsets[i, 0]), 4),
        "dy": round(float(offsets[i, 1]), 4),
        "dyaw_deg": round(float(offsets[i, 2]) * 180.0 / torch.pi, 2),
        "handoff": bool(reached[i]),
        "t_handoff_s": round(float(t_handoff[i]) * dt, 2)
        if int(t_handoff[i]) >= 0
        else None,
        "survived": bool(survived[i]),
      }
    )
  summary = {
    "task": args.task,
    "mount_ckpt": str(args.mount_ckpt),
    "climb_ckpt": str(args.climb_ckpt),
    "estimator": str(args.estimator) if args.estimator else None,
    "cells": n_cells,
    "handoff_rate": float(reached.float().mean()),
    "survival_rate": float(survived.float().mean()),
    "hold_s": args.hold_s,
    "mount_timeout_s": args.mount_timeout_s,
    "grid": rows,
  }

  out = args.out
  if out is None:
    root = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"chained_exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
  out.write_text(json.dumps(summary, indent=2))

  print(f"\n[chained] {args.task}  cells={n_cells}")
  print(f"[chained] handoff  {summary['handoff_rate']:.3f}")
  print(f"[chained] survival {summary['survival_rate']:.3f}")
  print(f"[chained] report -> {out}")
  header = f"{'dx':>7} {'dy':>7} {'yaw':>6} {'handoff':>8} {'t_h':>6} {'surv':>5}"
  print(header)
  for r in rows:
    print(
      f"{r['dx']:>7.3f} {r['dy']:>7.3f} {r['dyaw_deg']:>6.1f} "
      f"{str(r['handoff']):>8} "
      f"{(r['t_handoff_s'] if r['t_handoff_s'] is not None else '-'):>6} "
      f"{str(r['survived']):>5}"
    )

  env.close()


if __name__ == "__main__":
  main()
