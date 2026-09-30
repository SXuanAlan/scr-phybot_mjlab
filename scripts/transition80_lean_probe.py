"""Does the hand REST on the wall without hard contact? Per-step LH/RH trace
from frame 0 (deterministic policy): min sphere gap to the wall surface,
hard-sphere contact count, press, down-shear, pad->anchor distance, cone
violations. Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_lean_probe.py CKPT [steps=400]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
import mjlab  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import scr_phybot_mjab.tasks  # noqa
from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import _SphereGeomCache

TASK = "Mjlab-Climb-Transition80-Phybot-C2"
CKPT = Path(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
device = "cuda:0"
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), "/tmp/.t80_lean_runner", device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion"); aff = u.command_manager.get_term("microspine_affordance"); ex = aff._get_executor()
obs = wrapped.get_observations()
from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import _engage_match_gate
print("step  t(s)  frame | LH: gap_mm contact press_N shear_N d_anchor_cm | pad-ORIGIN err LH/RH cm | gate | RH: gap_mm d_anchor_cm | ref_eng LH RH | reset")
for step in range(STEPS):
  with torch.no_grad():
    act = policy(obs)
  obs, rew, dones, extras = wrapped.step(act)
  gaps = _SphereGeomCache.surface_distances(u).view(1, 4, 4)  # (E, ee, sphere)
  gmin = gaps.min(dim=-1).values[0]
  d = (ex.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)[0]
  oerr = (motion.body_pos_target_w[:, motion.pad_slots] - motion.robot_body_pos_w[:, motion.pad_slots]).norm(dim=-1)[0]
  gate = float(_engage_match_gate(motion, 0.25, 0.25, 0.3)[0])
  if step % 20 == 0 or bool(dones[0]):
    print(f"{step:4d} {step*0.02:5.2f} {int(motion.time_steps[0]):5d} | {gmin[0]*1000:7.1f} {int(aff.contact_count[0,0]):3d} {float(aff.pad_press_force[0,0]):6.1f} {float(aff.pad_shear_down[0,0]):6.1f} {d[0]*100:6.1f} | {oerr[0]*100:5.1f}/{oerr[1]*100:5.1f} | {gate:4.2f} | {gmin[1]*1000:7.1f} {d[1]*100:6.1f} | {int(motion.ref_engaged[0,0])} {int(motion.ref_engaged[0,1])} | {'RESET' if bool(dones[0]) else ''}")
