"""Route-B RMA phase 2: train phi(student history) -> privileged estimates.

Step-0 ablation (2026-08-03, model_5300 @ 90 deg, zeroing) cut the actor's
privileged set down to three load-bearing groups:

  microspine_engaged_state (4)   zeroed -> 0.419 / 34 falls   (the beam)
  height_above_spawn       (1)   zeroed -> 0.643 / 22 falls
  microspine_contact_fraction(4) zeroed -> 0.896 /  4 falls

phi regresses exactly these 9 dims — in OBS space, post term scale/clip, so
the estimates substitute 1:1 into the teacher's actor vector — from the
deployable student observation (89 dims x 50-frame history incl. the pad
load-cell channel). The remaining privileged channels (disengage_time,
cooldown, base_lin_vel) cost <= 0.015 match when zeroed and are simply
zeroed at deploy.

Modes:
  2a (default)   the TorchScript teacher drives the env through its TRUE
                 privileged obs; phi trains on the teacher's distribution.
  2b --closed-loop  the teacher instead receives phi's estimates (privileged
                 slices overwritten, droppable slices zeroed) — phi learns on
                 the state distribution its own errors induce (DAgger).

The teacher is the exported TorchScript (normalization baked, feed RAW obs),
so no rsl_rl runner surgery is needed. Outputs: phi.pt (TorchScript:
normalizer + MLP) and phi_meta.json (input contract + target/zero slices).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TARGET_TERMS = (
  "microspine_engaged_state",
  "microspine_contact_fraction",
  "height_above_spawn",
)
ZERO_TERMS = ("microspine_disengage_time", "microspine_cooldown", "base_lin_vel")


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("task", nargs="?", default="Mjlab-Climb-Wall-Phybot-C2-Distill")
  p.add_argument("--teacher-jit", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=20_000)
  p.add_argument("--lr", type=float, default=3.0e-4)
  p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
  p.add_argument("--closed-loop", action="store_true")
  p.add_argument("--no-contact", action="store_true",
                 help="comparison arm: drop the pad_normal_force block from "
                      "phi's input (sliced INSIDE the module, so the raster "
                      "estimator hook keeps feeding the full student vector)")
  p.add_argument("--warmup-steps", type=int, default=500,
                 help="closed-loop only: drive with TRUE privileged obs this "
                      "long before switching to phi estimates")
  p.add_argument("--init-phi", type=Path, default=None,
                 help="warm-start phi weights (e.g. the 2a result for 2b)")
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--gpu-id", type=int, default=0)
  p.add_argument("--log-every", type=int, default=200)
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0"

  import torch
  from torch import nn

  import scr_phybot_mjab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # Slice tables straight from the live observation manager — the same
  # source export_policy uses, so the contract cannot drift.
  om = env.observation_manager

  def slices(group: str) -> dict[str, tuple[int, int]]:
    out, start = {}, 0
    for name, dim in zip(om.active_terms[group], om.group_obs_term_dim[group]):
      width = int(torch.tensor(dim).prod().item()) if not isinstance(dim, int) else dim
      out[name] = (start, start + width)
      start += width
    return out

  actor_sl = slices("actor")
  student_sl = slices("student")
  target_sl = [actor_sl[t] for t in TARGET_TERMS]
  zero_sl = [actor_sl[t] for t in ZERO_TERMS]
  tgt_dim = sum(e - s for s, e in target_sl)

  teacher = torch.jit.load(str(args.teacher_jit), map_location=device).eval()

  obs0 = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
  stu_dim = obs0["student"].shape[-1]
  keep_dim = stu_dim
  if args.no_contact:
    pad_s, pad_e = student_sl["pad_normal_force"]
    if pad_e != stu_dim:
      raise AssertionError(
        f"pad_normal_force must be the tail block, got [{pad_s}:{pad_e}) "
        f"of {stu_dim}"
      )
    keep_dim = pad_s
  print(f"[phi] student in = {stu_dim} (keep {keep_dim}"
        f"{', NO CONTACT' if args.no_contact else ''}), targets = {tgt_dim} "
        f"({', '.join(TARGET_TERMS)}), mode = "
        f"{'2b closed-loop' if args.closed_loop else '2a teacher-driven'}")

  from rsl_rl.modules import EmpiricalNormalization

  class Phi(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.keep_dim = keep_dim
      self.norm = EmpiricalNormalization(shape=[keep_dim])
      layers: list[nn.Module] = []
      last = keep_dim
      for h in args.hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
      layers.append(nn.Linear(last, tgt_dim))
      self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.mlp(self.norm(x[:, : self.keep_dim]))

  phi = Phi().to(device)
  if args.init_phi is not None:
    loaded = torch.jit.load(str(args.init_phi), map_location=device)
    phi.load_state_dict(loaded.state_dict(), strict=True)
    print(f"[phi] warm-started from {args.init_phi}")
  opt = torch.optim.Adam(phi.parameters(), lr=args.lr)

  def gather_targets(actor_obs: torch.Tensor) -> torch.Tensor:
    return torch.cat([actor_obs[:, s:e] for s, e in target_sl], dim=-1)

  def substitute(actor_obs: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    out = actor_obs.clone()
    ofs = 0
    for s, e in target_sl:
      out[:, s:e] = est[:, ofs : ofs + (e - s)]
      ofs += e - s
    for s, e in zero_sl:
      out[:, s:e] = 0.0
    return out

  obs = obs0
  eng_w = target_sl[0][1] - target_sl[0][0]  # engaged block leads the output
  for step in range(args.steps):
    actor_obs = obs["actor"]
    with torch.no_grad():
      est = phi(obs["student"])
      use_est = args.closed_loop and step >= args.warmup_steps
      teacher_in = substitute(actor_obs, est) if use_est else actor_obs
      actions = teacher(teacher_in)

    pred = phi(obs["student"])
    tgt = gather_targets(actor_obs).detach()
    loss = nn.functional.mse_loss(pred, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    phi.norm.update(obs["student"][:, :keep_dim].detach())

    obs, *_ = env.step(actions)
    if isinstance(obs, tuple):
      obs = obs[0]

    if step % args.log_every == 0 or step == args.steps - 1:
      with torch.no_grad():
        eng_acc = ((pred[:, :eng_w] > 0.5) == (tgt[:, :eng_w] > 0.5)).float().mean()
        h_mae = (pred[:, -1] - tgt[:, -1]).abs().mean()
      print(
        f"[phi] step {step:6d} mse={loss.item():.5f} "
        f"engaged_acc={eng_acc.item():.4f} height_mae={h_mae.item():.4f}",
        flush=True,
      )

  out = args.out or (args.teacher_jit.parent / (
    "phi_2b.pt" if args.closed_loop else "phi_2a.pt"
  ))
  phi.eval()
  scripted = torch.jit.script(phi)
  scripted.save(str(out))
  meta = {
    "input_group": "student",
    "input_dim": stu_dim,
    "input_slice_end": keep_dim,
    "no_contact": bool(args.no_contact),
    "targets": [
      {"term": t, "actor_start": s, "actor_end": e}
      for t, (s, e) in zip(TARGET_TERMS, target_sl)
    ],
    "zero_slices": [
      {"term": t, "actor_start": s, "actor_end": e}
      for t, (s, e) in zip(ZERO_TERMS, zero_sl)
    ],
    "note": "phi outputs OBS-space values; substitute into the actor vector "
            "at the listed slices, zero the zero_slices, feed RAW to the "
            "teacher TorchScript.",
  }
  out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
  print(f"[phi] saved {out} + {out.with_suffix('.json')}")
  env.close()


if __name__ == "__main__":
  main()
