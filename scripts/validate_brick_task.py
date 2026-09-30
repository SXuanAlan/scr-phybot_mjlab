"""Finite-state/GPU smoke check, reference PD replay or policy evaluation.

Reference replay is diagnostic and must not be reported as trained-policy success.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from scr_phybot_mjab.tasks.brick import mdp
from scr_phybot_mjab.tasks.brick.env_cfg import brick_env_cfg


@torch.inference_mode()
def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--stage", choices=("full", "hold", "engage", "place"), default="full")
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--episodes", type=int, default=0, help="Evaluate at least this many completed episodes")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=17)
  p.add_argument("--dr", action="store_true")
  p.add_argument("--no-grip", action="store_true")
  p.add_argument("--policy", type=Path, help="Exported TorchScript policy, actor normalization included")
  p.add_argument("--report", type=Path)
  a = p.parse_args()
  cfg = brick_env_cfg(play=not a.dr, stage=a.stage, randomize=a.dr, grip_enabled=not a.no_grip)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  cfg.auto_reset = False
  policy = torch.jit.load(str(a.policy), map_location=a.device).eval() if a.policy else None
  env = ManagerBasedRlEnv(cfg, device=a.device)
  counts = {
    "episodes": 0,
    "successes": 0,
    "stage_completions": 0,
    "failures": 0,
    "timeouts": 0,
    "lifted": 0,
    "carried": 0,
  }
  maximum_force, maximum_foot_drift = 0.0, 0.0
  phase_counts = [0] * 7
  release_counts = {}
  try:
    obs, _ = env.reset()
    c = env.command_manager.get_term("brick_task")
    g = env.command_manager.get_term("microspine_affordance")
    step = 0
    while step < a.steps or counts["episodes"] < a.episodes:
      with torch.inference_mode():
        action = policy(obs["actor"]) if policy else torch.zeros_like(c.joint_pos_ref)
        obs, reward, terminated, truncated, _ = env.step(action)
      assert torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values())
      maximum_force = max(maximum_force, float(g.anchor_load_tnl.abs().max()))
      maximum_foot_drift = max(maximum_foot_drift, float(c.metrics["foot_drift"].max()))
      for reason in g.release_reason[g.forced_release_this_step].tolist():
        release_counts[str(reason)] = release_counts.get(str(reason), 0) + 1
      done = terminated | truncated
      ids = done.nonzero().flatten()
      if len(ids):
        counts["episodes"] += len(ids)
        failure = mdp.failed(env)[ids]
        counts["successes"] += int((c.success[ids] & ~failure).sum())
        counts["stage_completions"] += int((c.training_complete[ids] & ~failure).sum())
        counts["timeouts"] += int(truncated[ids].sum())
        counts["failures"] += int(failure.sum())
        counts["lifted"] += int(c.lifted[ids].sum())
        counts["carried"] += int(c.carried[ids].sum())
        for phase in c.phase[ids].tolist():
          phase_counts[phase] += 1
        obs, _ = env.reset(env_ids=ids)
      step += 1
    report = dict(
      kind="policy_evaluation" if policy else "reference_pd_diagnostic",
      stage=a.stage,
      unit_layout=c.cfg.unit_layout,
      reference_sha256=hashlib.sha256(Path(c.cfg.reference_file).read_bytes()).hexdigest(),
      policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest() if a.policy else None,
      seed=a.seed,
      randomize=a.dr,
      grip_enabled=not a.no_grip,
      steps=step,
      **counts,
      success_rate=counts["successes"] / max(counts["episodes"], 1),
      terminal_phase_counts=phase_counts,
      max_anchor_component_n=maximum_force,
      release_reason_counts=release_counts,
      max_foot_drift_m=maximum_foot_drift,
      actor_dim=obs["actor"].shape[-1],
      critic_dim=obs["critic"].shape[-1],
    )
    if a.report:
      a.report.parent.mkdir(parents=True, exist_ok=True)
      a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
