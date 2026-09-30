"""Training-replica probe: run a checkpoint under the UNMODIFIED training cfg
(adaptive RSI + classroom, every termination) with auto-resets for STEPS
steps and reproduce the training log's episode statistics from the env's
own episode-end log (Episode_Metrics / Episode_Termination, weighted by the
number of resets) — plus the seed fate the log cannot show: per completed
episode, whether the birth seeds were lost, at which step, and whether that
loss fell inside the settle exemption (free). Use it to decide whether the
training numbers and the birth-fate probe describe the same policy.
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_replica_probe.py CKPT [nenv=1024] [steps=400] [mode=1]
"""
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
from scr_phybot_mjab.tasks.climb_transition80.config.phybot_c2.env_cfgs import TRANSITION80_SEED_SETTLE_EXEMPT_S

TASK = "Mjlab-Climb-Transition80-Phybot-C2"
CKPT = Path(sys.argv[1])
NENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
MODE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
device = "cuda:0"
cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NENVS
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), "/tmp/.t80_replica_runner", device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy_det = runner.get_inference_policy(device=device)
actor = runner.alg.actor
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
term = u.termination_manager
EXEMPT = int(round(TRANSITION80_SEED_SETTLE_EXEMPT_S / float(env.step_dt)))
C = motion.clip.cycle_frames; nbins = motion._bin_count
seg_d = np.load(cfg.commands["climb_motion"].motion_file, allow_pickle=True).item()
names = list(seg_d["segment_names"]); labels = np.asarray(seg_d["phase_segment"])
def seg(f): return names[labels[min(int(f), len(labels) - 1)]]
tnames = list(term.active_terms)
env.reset()
obs = wrapped.get_observations()
N = NENVS
seed = aff.engaged.clone()
ep_step = torch.zeros(N, dtype=torch.long, device=device)
first_loss = torch.full((N,), -1, dtype=torch.long, device=device)
birth = motion._birth_frames.clone()
log_sum, log_w = {}, {}
rec = {k: [] for k in ("seeded", "first", "len", "bin", "cause")}
for step in range(STEPS):
  with torch.no_grad():
    a = actor(obs, stochastic_output=True) if MODE == 1 else policy_det(obs)
  obs, rew, dones, extras = wrapped.step(a)
  d = dones.bool()
  rel_any = (aff.forced_release_this_step & seed).any(1) & ~d
  first_loss = torch.where(rel_any & (first_loss < 0), ep_step, first_loss)
  ep_step += 1
  nd = int(d.sum())
  if nd:
    if "log" in extras:
      for k, v in extras["log"].items():
        log_sum[k] = log_sum.get(k, 0.0) + float(v) * nd; log_w[k] = log_w.get(k, 0) + nd
    cause = torch.full((N,), -1, dtype=torch.long, device=device)
    for k, n in enumerate(tnames):
      f = term.get_term(n) & d & (cause < 0); cause[f] = k
    rec["seeded"].append(seed[d].any(1).cpu()); rec["first"].append(first_loss[d].cpu()); rec["len"].append(ep_step[d].cpu())
    rec["bin"].append(((birth[d] * nbins) // C).cpu()); rec["cause"].append(cause[d].cpu())
    seed[d] = aff.engaged[d]; ep_step[d] = 0; first_loss[d] = -1; birth[d] = motion._birth_frames[d]
R = {k: torch.cat(v) for k, v in rec.items()}
n = int(R["len"].numel())
mode = "stochastic" if MODE == 1 else "deterministic"
print(f"\n=== replica {CKPT.name} {mode} envs {NENVS} steps {STEPS}: completed episodes {n}, mean eplen {R['len'].float().mean():.1f}, seeded births {R['seeded'].float().mean():.2f} ===")
print("death shares:", "  ".join(f"{tnames[k]} {float((R['cause'] == k).float().mean()):.0%}" for k in range(len(tnames)) if int((R['cause'] == k).sum())))
keys = ["Episode_Metrics/left_hand_engagement_occupancy", "Episode_Metrics/right_hand_engagement_occupancy", "Episode_Metrics/left_foot_engagement_occupancy",
        "Episode_Metrics/right_foot_engagement_occupancy", "Episode_Metrics/microspine_forced_release_rate_hz", "Episode_Metrics/microspine_engage_event_rate_hz",
        "Episode_Metrics/microspine_hold_margin_engaged", "Episode_Metrics/scheduled_hold_margin", "Episode_Metrics/action_saturation_fraction",
        "Metrics/climb_motion/error_pad_pos", "Metrics/climb_motion/birth_in_classroom"]
print("episode-end log (reset-weighted means, compare with the training log):")
for k in keys:
  if k in log_sum: print(f"  {k.split('/')[-1]:36s} {log_sum[k] / log_w[k]:.4f}")
s = R["seeded"]; f = R["first"]; L = R["len"]
lost = (f >= 0)
print(f"seed fate over {int(s.sum())} completed seeded episodes: lost-any {float(lost[s].float().mean()):.2f} | lost by step {EXEMPT - 1} (free) {float(((f >= 0) & (f < EXEMPT))[s].float().mean()):.2f} | by 25 {float(((f >= 0) & (f <= 25))[s].float().mean()):.2f} | median first-loss step {int(f[s & lost].float().median()) if int((s & lost).sum()) else -1}")
print("per birth bin: n  mean eplen  seeded  lost-any  lost-free(<exempt)  rip  closs  cone")
ri = tnames.index("anchor_rip") if "anchor_rip" in tnames else -1; ci = tnames.index("contact_loss") if "contact_loss" in tnames else -1; ki = tnames.index("engage_constraint") if "engage_constraint" in tnames else -1
for b in range(nbins):
  m = R["bin"] == b
  nb = int(m.sum())
  if nb == 0: continue
  sb = m & s
  la = float(lost[sb].float().mean()) if int(sb.sum()) else 0.0
  lf = float(((f >= 0) & (f < EXEMPT))[sb].float().mean()) if int(sb.sum()) else 0.0
  f0 = int(b * C // nbins)
  print(f"  bin {b:2d} f~{f0:4d} {seg(f0):17s} n={nb:4d} eplen {float(L[m].float().mean()):6.1f} seeded {float(s[m].float().mean()):.2f} lost {la:.2f} free {lf:.2f} | rip {float((R['cause'][m] == ri).float().mean()):.2f} closs {float((R['cause'][m] == ci).float().mean()):.2f} cone {float((R['cause'][m] == ki).float().mean()):.2f}")
