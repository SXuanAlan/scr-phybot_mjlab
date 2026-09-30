"""Export a trained policy to TorchScript with its observation/action contract.

The exported .pt embeds the observation normalizer (rsl_rl EmpiricalNormalization),
so the deploy side must feed RAW observations (no obs_scales, no clip_obs) in the
layout documented in docs/walk_policy_deploy_interface.md and the JSON sidecar
written next to the model.

  python scripts/export_policy.py Mjlab-Walk-Phybot-C2 \
      --checkpoint-file logs/rsl_rl/phybot_c2_walk/<run>/model_30000.pt
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
  sys.path.insert(0, str(_REPO_SRC))

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  checkpoint_file: str
  """Path to the rsl_rl checkpoint (model_<iter>.pt)."""
  output_dir: str | None = None
  """Output directory. Defaults to `<checkpoint dir>/exported/`."""
  onnx: bool = False
  """Also export an ONNX copy."""
  self_test_steps: int = 100
  """Env steps to roll for the jit-vs-runner consistency check (0 disables)."""
  device: str | None = None


def _obs_slice_table(env: ManagerBasedRlEnv, group: str = "actor"):
  """Per-term slice offsets in the flat observation vector."""
  names = env.observation_manager.active_terms[group]
  dims = env.observation_manager.group_obs_term_dim[group]
  table = []
  offset = 0
  for name, dim in zip(names, dims):
    size = 1
    for d in dim:
      size *= d
    table.append({"term": name, "start": offset, "end": offset + size})
    offset += size
  return table, offset


def run_export(task_id: str, cfg: ExportConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  out_dir = Path(cfg.output_dir) if cfg.output_dir else resume_path.parent / "exported"
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  policy_set = "actor"
  obs_group = agent_cfg.obs_groups[policy_set][0]
  env_cfg.observations[obs_group].enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), load_cfg={policy_set: True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  jit_name = "policy.pt"
  runner.export_policy_to_jit(str(out_dir), jit_name)
  jit_path = out_dir / jit_name
  print(f"[INFO] TorchScript policy written to {jit_path}")
  if cfg.onnx:
    runner.export_policy_to_onnx(str(out_dir), "policy.onnx")
    print(f"[INFO] ONNX policy written to {out_dir / 'policy.onnx'}")

  # --- Metadata sidecar -----------------------------------------------------
  slice_table, obs_dim = _obs_slice_table(env, group=obs_group)
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("joint_pos")
  joint_names = [n.removeprefix("robot/") for n in action_term.target_names]
  m = env.sim.mj_model
  kp, kd, torque_limit = [], [], []
  for name in joint_names:
    act = m.actuator(f"robot/{name}")
    kp.append(float(m.actuator_gainprm[act.id, 0]))
    kd.append(float(-m.actuator_biasprm[act.id, 2]))
    torque_limit.append(float(m.actuator_forcerange[act.id, 1]))
  joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
  default_dof_pos = robot.data.default_joint_pos[0, joint_ids].cpu().tolist()
  scale = action_term.scale
  action_scale = scale if isinstance(scale, float) else scale[0].cpu().tolist()

  history_length = env_cfg.observations[obs_group].history_length
  meta = {
    "task_id": task_id,
    "checkpoint": str(resume_path),
    "control_freq_hz": 1.0 / env.step_dt,
    "obs_group": obs_group,
    "obs_dim": obs_dim,
    "obs_history_length": history_length,
    "obs_layout": slice_table,
    "obs_note": (
      "Per-term blocks; within each block the term's frames are stacked "
      "oldest->newest (history_length frames, each frame contiguous). "
      "Feed RAW values: normalization is baked into the TorchScript model. "
      "On reset, backfill all history slots with the first frame."
    ),
    "action_dim": len(joint_names),
    "action_note": "q_target = default_dof_pos + action_scale * action",
    "action_scale": action_scale,
    "joint_names": joint_names,
    "default_dof_pos": default_dof_pos,
    "kp": kp,
    "kd": kd,
    "torque_limits": torque_limit,
  }
  meta_path = out_dir / "policy_meta.json"
  if "brick_task" in (env_cfg.commands or {}):
    meta["action_note"] = "q_target = q_reference(current_phase, progress) + action_scale * action"
    meta["action_offset_source"] = "brick_reference.npz; use the measured-pose phase controller in brick_task_meta.json"
  meta_path.write_text(json.dumps(meta, indent=2))
  print(f"[INFO] Metadata written to {meta_path}")
  print(f"[INFO] obs_group={obs_group}, obs_dim={obs_dim}, action_dim={len(joint_names)}")
  for row in slice_table:
    print(f"       [{row['start']:4d}:{row['end']:4d})  {row['term']}")

  # --- Climb motion-command table sidecar -----------------------------------
  # The motion_command observation has no producer on the robot: it is a pure
  # function of the 50 Hz cycle clock. Export the full cycle so the deploy
  # side replays row[t % cycle_frames] instead of reimplementing the
  # schedule/countdown logic.
  if "climb_motion" in (env_cfg.commands or {}):
    motion = env.command_manager.get_term("climb_motion")
    saved_steps = motion.time_steps.clone()
    rows = []
    with torch.no_grad():
      for frame in range(int(motion.clip.cycle_frames)):
        motion.time_steps[:] = frame
        rows.append(motion.command[0].detach().float().cpu().clone())
    motion.time_steps[:] = saved_steps
    cmd_table = torch.stack(rows)
    cmd_path = out_dir / "motion_command_table.json"
    cmd_path.write_text(
      json.dumps(
        {
          "command_name": "climb_motion",
          "cycle_frames": int(cmd_table.shape[0]),
          "dim": int(cmd_table.shape[1]),
          "rate_hz": 1.0 / env.step_dt,
          "note": (
            "motion_command observation row per clip frame. Deploy replays "
            "row[t % cycle_frames] on the policy clock; layout matches the "
            "motion_command block in obs_layout."
          ),
          "table": cmd_table.tolist(),
        },
        indent=None,
      )
    )
    print(
      f"[INFO] Motion command table ({cmd_table.shape[0]}x{cmd_table.shape[1]}) "
      f"written to {cmd_path}"
    )

  # --- Deploy-simulation self-test -------------------------------------------
  if "brick_task" in (env_cfg.commands or {}):
    from scr_phybot_mjab.tasks.brick.export import export_brick_sidecars

    print(f"[INFO] Brick command contract written to {export_brick_sidecars(env, out_dir)}")

  if cfg.self_test_steps > 0:
    jit_policy = torch.jit.load(str(jit_path), map_location="cpu")
    jit_policy.eval()
    obs = wrapped.get_observations()
    max_err = 0.0
    with torch.no_grad():
      for _ in range(cfg.self_test_steps):
        actions = policy(obs)
        jit_actions = jit_policy(obs[obs_group].cpu().float())
        err = (actions.cpu() - jit_actions).abs().max().item()
        max_err = max(max_err, err)
        obs, _, _, _ = wrapped.step(actions)
    print(f"[INFO] Self-test over {cfg.self_test_steps} steps: "
          f"max |runner - jit| = {max_err:.2e}")
    if max_err > 1e-5:
      raise RuntimeError(f"Exported policy diverges from runner policy: {max_err}")
    print("[INFO] Self-test PASSED")

  env.close()


def main():
  import scr_phybot_mjab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
