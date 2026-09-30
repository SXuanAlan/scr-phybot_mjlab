"""L-E 合跑: concurrent pi (PPO finetune) + phi (privileged regression).

One run replaces the 2a/2b/2c/Phase-3 four-stage pipeline
(docs/plans/2026-08-10_unified_mount_climb_policy_plan.md §7). Per rollout
step the wrapper (a) captures (student obs, privileged truth) rows for
phi's regression and (b) scheduled-samples the actor's privileged slices —
each env row independently keeps the TRUTH with probability p (annealed
1 -> 0 over --sched-iters), otherwise reads phi's live estimates with the
deploy-dropped slices zeroed. The first --freeze-pi-iters iterations train
phi only (pi is warm-started and waits for a sane ê).

phi's network, TorchScript artifact and meta json are byte-compatible with
scripts/train_estimator.py's output: the raster ``estimator=`` token,
estfreeze and the C++ packer consume the 合跑 phi as-is.

Example (climb 90, ship-lineage warm start, no-contact arm):
  python scripts/train_concurrent_est.py \\
    --init logs/.../deploypd-v4-rung85/model_5050.pt \\
    --no-contact --run-name conc-est-v1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import scr_phybot_mjab.tasks  # noqa: F401 - registers SCR PHYBot tasks

TARGET_TERMS = (
  "microspine_engaged_state",
  "microspine_contact_fraction",
  "height_above_spawn",
)
ZERO_TERMS = ("microspine_disengage_time", "microspine_cooldown", "base_lin_vel")

_BUILDERS = ("registry", "unified", "transition", "band80", "band88")


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--env-builder", choices=_BUILDERS, default="registry",
                 help="registry = --task id (default climb -Distill); the "
                 "rest call the transition-package builders directly with "
                 "student_obs=True")
  p.add_argument("--task", default="Mjlab-Climb-Wall-Phybot-C2-Distill")
  p.add_argument("--rl-task", default=None,
                 help="runner-cfg task id (default: climb base for "
                 "registry/climb, transition otherwise)")
  p.add_argument("--init", type=Path, required=True,
                 help="pi warm-start PPO checkpoint (actor+critic strict)")
  p.add_argument("--phi-init", type=Path, default=None,
                 help="optional phi warm start (TorchScript .pt, e.g. the "
                 "shipped phi_2b_nc)")
  p.add_argument("--no-contact", action="store_true",
                 help="drop the pad load-cell tail INSIDE phi (ship arm)")
  p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
  p.add_argument("--sched-iters", type=int, default=1000,
                 help="truth->ê anneal length (iterations)")
  p.add_argument("--freeze-pi-iters", type=int, default=100)
  p.add_argument("--phi-lr", type=float, default=3.0e-4)
  p.add_argument("--lr", type=float, default=1.0e-4, help="pi finetune LR")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--max-iterations", type=int, default=1500)
  p.add_argument("--run-name", default="conc-est")
  p.add_argument("--gpu-id", type=int, default=0)
  p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if not args.init.exists():
    raise FileNotFoundError(args.init)
  if args.phi_init is not None and not args.phi_init.exists():
    raise FileNotFoundError(args.phi_init)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0"

  import dataclasses

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wandb import add_wandb_tags

  from scr_phybot_mjab.rl.concurrent_est import (
    ConcurrentCfg,
    ConcurrentEstPPO,
    EstBuffer,
    PhiNet,
    ScheduleState,
  )

  configure_torch_backends()

  # -- env cfg -----------------------------------------------------------------
  if args.env_builder == "registry":
    env_cfg = load_env_cfg(args.task)
    rl_task = args.rl_task or "Mjlab-Climb-Wall-Phybot-C2"
    experiment = "phybot_c2_climb_wall_estconc"
  else:
    from scr_phybot_mjab.tasks.climb_transition.config.phybot_c2.env_cfgs import (
      phybot_c2_climb_transition_env_cfg,
      phybot_c2_climb_unified_env_cfg,
    )

    if args.env_builder == "unified":
      env_cfg = phybot_c2_climb_unified_env_cfg(student_obs=True)
    elif args.env_builder == "transition":
      env_cfg = phybot_c2_climb_transition_env_cfg(student_obs=True)
    else:
      band = 80.0 if args.env_builder == "band80" else 88.0
      env_cfg = phybot_c2_climb_transition_env_cfg(
        student_obs=True, wall_band=band
      )
    # Distill-variant convention: labels come from clean privileged state;
    # the student group keeps its own noise + history.
    env_cfg.observations["actor"].enable_corruption = False
    rl_task = args.rl_task or "Mjlab-Climb-Transition-Phybot-C2"
    experiment = f"phybot_c2_climb_{args.env_builder}_estconc"

  agent_cfg = load_rl_cfg(rl_task)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.run_name = args.run_name
  agent_cfg.logger = args.logger
  agent_cfg.upload_model = False
  agent_cfg.experiment_name = experiment
  agent_cfg.wandb_tags = ("phybot_c2", "concurrent-est")
  # Warm-started pi on its own rollouts: fixed low LR (the adaptive-KL
  # controller fights it — same call as the staged finetune).
  agent_cfg.algorithm = dataclasses.replace(
    agent_cfg.algorithm,
    learning_rate=args.lr,
    schedule="fixed",
    class_name="scr_phybot_mjab.rl.concurrent_est:ConcurrentEstPPO",
  )

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_dir = log_root / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
  )
  log_dir.mkdir(parents=True, exist_ok=False)
  print(f"[conc] run: {log_dir}")
  print(f"[conc] pi init: {args.init}")
  print(f"[conc] phi init: {args.phi_init or 'from scratch'}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # -- slice tables from the live observation manager (export parity) ---------
  om = env.observation_manager

  def slices(group: str) -> dict[str, tuple[int, int]]:
    out, start = {}, 0
    for name, dim in zip(om.active_terms[group], om.group_obs_term_dim[group]):
      width = (
        int(torch.tensor(dim).prod().item())
        if not isinstance(dim, int)
        else dim
      )
      out[name] = (start, start + width)
      start += width
    return out

  actor_sl = slices("actor")
  student_sl = slices("student")
  target_sl = [actor_sl[t] for t in TARGET_TERMS]
  zero_sl = [actor_sl[t] for t in ZERO_TERMS]
  tgt_dim = sum(e - s for s, e in target_sl)

  probe = env.reset()
  obs0 = probe[0] if isinstance(probe, tuple) else probe
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
  print(
    f"[conc] student in = {stu_dim} (keep {keep_dim}"
    f"{', NO CONTACT' if args.no_contact else ''}), targets = {tgt_dim}, "
    f"sched {args.sched_iters} it, freeze-pi {args.freeze_pi_iters} it"
  )

  phi = PhiNet(keep_dim, tgt_dim, hidden=tuple(args.hidden)).to(device)
  if args.phi_init is not None:
    loaded = torch.jit.load(str(args.phi_init), map_location=device)
    phi.load_state_dict(loaded.state_dict(), strict=True)
    print(f"[conc] phi warm-started from {args.phi_init}")

  buffer = EstBuffer()
  state = ScheduleState()

  def gather_targets(actor_obs: torch.Tensor) -> torch.Tensor:
    return torch.cat([actor_obs[:, s:e] for s, e in target_sl], dim=-1)

  class SchedSampleWrapper(RslRlVecEnvWrapper):
    """Capture phi rows; scheduled-sample the actor's privileged slices."""

    def _process(self, obs):
      actor = obs["actor"]
      student = obs["student"]
      buffer.append(
        student.detach().clone(), gather_targets(actor).detach().clone()
      )
      with torch.no_grad():
        est = phi(student)
      sub = actor.clone()
      ofs = 0
      for s, e in target_sl:
        sub[:, s:e] = est[:, ofs : ofs + (e - s)]
        ofs += e - s
      for s, e in zero_sl:
        sub[:, s:e] = 0.0
      use_truth = (
        torch.rand(actor.shape[0], device=actor.device) < state.p_truth
      )
      obs = obs.clone()
      obs["actor"] = torch.where(use_truth.unsqueeze(-1), actor, sub)
      return obs

    def get_observations(self):
      return self._process(super().get_observations())

    def step(self, actions):
      obs, rew, dones, extras = super().step(actions)
      return self._process(obs), rew, dones, extras

  wrapped = SchedSampleWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), str(log_dir), device)
  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  runner.load(
    str(args.init),
    load_cfg={"actor": True, "critic": True},
    strict=True,
    map_location=device,
  )
  runner.current_learning_iteration = 0
  env.unwrapped.common_step_counter = 0

  alg = runner.alg
  if not isinstance(alg, ConcurrentEstPPO):
    raise RuntimeError(
      f"Runner built {type(alg).__name__}, expected ConcurrentEstPPO — "
      "check algorithm.class_name plumbing."
    )
  alg.attach_estimation(
    buffer,
    phi,
    ConcurrentCfg(
      sched_iters=args.sched_iters,
      freeze_pi_iters=args.freeze_pi_iters,
      phi_lr=args.phi_lr,
    ),
    state,
  )

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )

  # -- phi artifact, template-identical ---------------------------------------
  out = log_dir / "phi_conc.pt"
  phi.eval()
  torch.jit.script(phi).save(str(out))
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
  print(f"[conc] saved {out} + {out.with_suffix('.json')}")
  env.close()


if __name__ == "__main__":
  main()
