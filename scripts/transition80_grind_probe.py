"""Where does the engage-cone 'grind' come from? Per-EE / per-clip-bin tally.

Runs a checkpoint's policy (stochastic = training distribution, or the
deterministic mean) over uniform births and tallies, per pad and per clip
bin: wall-touch rate of UNENGAGED pads, the cone's billed grind/force/tilt
violation rates (after persistence), tangential speed percentiles of
touching pads, and the resulting termination causes.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_grind_probe.py CKPT [nenv=512] [steps=200] [stochastic=1]
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
from mjlab.utils.lab_api.math import quat_apply
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))
import scr_phybot_mjab.tasks  # noqa

TASK = "Mjlab-Climb-Transition80-Phybot-C2"
CKPT = Path(sys.argv[1])
NENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
STOCH = int(sys.argv[4]) if len(sys.argv) > 4 else 1
device = "cuda:0"
cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NENVS
cfg.commands["climb_motion"].sampling_mode = "uniform"
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), "/tmp/.t80_grind_runner", device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy_det = runner.get_inference_policy(device=device)
actor = runner.alg.actor  # rsl_rl 5.x: the actor model; (obs, stochastic_output=True) samples
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
ex = aff._get_executor()
robot = u.scene["robot"]
cone = u.termination_manager.get_term_cfg("engage_constraint").func
term = u.termination_manager
C = motion.clip.cycle_frames; nbins = motion._bin_count
EE = ["LH", "RH", "LF", "RF"]
obs = wrapped.get_observations()
touch_n = torch.zeros(nbins, 4, device=device); grind_n = torch.zeros_like(touch_n); force_n = torch.zeros_like(touch_n); tilt_n = torch.zeros_like(touch_n); steps_n = torch.zeros(nbins, device=device)
cond_n = torch.zeros(4, 5, device=device); touch_steps = torch.zeros(4, device=device); count_hist = torch.zeros(4, 6, device=device)
vt_samples = [[] for _ in range(4)]; press_samples = [[] for _ in range(4)]; shear_samples = [[] for _ in range(4)]; near_n = torch.zeros(4, device=device); win_n = torch.zeros(4, device=device); unsched_n = torch.zeros_like(touch_n)
deaths = {}
for step in range(STEPS):
  with torch.no_grad():
    if STOCH:
      act = actor(obs, stochastic_output=True)
    else:
      act = policy_det(obs)
  obs, rew, dones, extras = wrapped.step(act)
  bins = ((motion.time_steps * nbins) // C).clamp(0, nbins - 1)
  engaged = aff.engaged; touch = (aff.contact_count > 0) & ~engaged
  vel = robot.data.body_link_lin_vel_w[:, ex._ee_body_ids, :]
  n_local = torch.zeros(NENVS, 3, device=device); n_local[:, 0] = -1.0
  n_w = quat_apply(motion.wall_quat, n_local).unsqueeze(1)
  v_tan = (vel - (vel * n_w).sum(dim=-1, keepdim=True) * n_w).norm(dim=-1)
  for b in range(nbins):
    m = bins == b
    if not m.any(): continue
    steps_n[b] += m.sum()
    touch_n[b] += touch[m].float().sum(0)
    v = cone.last_violations
    grind_n[b] += (v["grind"][m] > 0).float().sum(0)
    force_n[b] += (v["force"][m] > 0).float().sum(0)
    tilt_n[b] += (v["tilt"][m] > 0).float().sum(0)
    unsched_n[b] += (v["unscheduled"][m] > 0).float().sum(0)
  for e in range(4):
    sel = touch[:, e]
    if sel.any():
      vt_samples[e].append(v_tan[sel, e].detach().cpu()); press_samples[e].append(aff.pad_press_force[sel, e].detach().cpu()); shear_samples[e].append(aff.pad_shear_down[sel, e].detach().cpu())
  tau = float(aff._mcfg.engage_shear_threshold_n); cos_thr = float(aff._mcfg.engage_parallel_cos_threshold); vmax = float(aff._mcfg.engage_max_antirake_vel)
  t_hat, _, _ = aff._engage_frame()
  v_rake = (vel * t_hat).sum(dim=-1)
  c_shear = aff.pad_shear_down >= tau; c_face = aff.face_cos >= cos_thr; c_rake = v_rake >= -vmax; c_cool = aff.cooldown_steps <= 0
  c_all = c_shear & c_face & c_rake & c_cool
  for e in range(4):
    sel = touch[:, e]
    touch_steps[e] += sel.float().sum()
    for k, c in enumerate((c_shear, c_face, c_rake, c_cool, c_all)):
      cond_n[e, k] += (c[:, e] & sel).float().sum()
    cnt = aff._engage_count[sel, e].clamp(0, 5)
    if cnt.numel(): count_hist[e] += torch.bincount(cnt.long(), minlength=6).float()
  lead = 2.0 / motion.cycle_time_s
  window = (motion.ref_engaged | (motion.ref_edge_countdowns[..., 1] <= lead)) & ~engaged
  dist = (ex.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
  win_n += window.float().sum(0); near_n += (window & (dist < 0.10)).float().sum(0)
  for n in term.active_terms:
    flag = term.get_term(n) & dones.bool()
    if flag.any(): deaths[n] = deaths.get(n, 0) + int(flag.sum())
seg_d = np.load(cfg.commands["climb_motion"].motion_file, allow_pickle=True).item()
names = list(seg_d["segment_names"]); labels = np.asarray(seg_d["phase_segment"])
def seg(f): return names[labels[min(int(f), len(labels) - 1)]]
mode = "stochastic (training distribution)" if STOCH else "deterministic mean"
std_val = float(getattr(actor, "output_std", torch.zeros(1)).mean())
print(f"\n=== {CKPT.name}  {mode}  envs {NENVS}  steps {STEPS}  std {std_val:.3f} ===")
tot = sum(deaths.values()) or 1
print("deaths:", "  ".join(f"{k} {v/tot:.0%}" for k, v in sorted(deaths.items(), key=lambda kv: -kv[1])))
print("\nper EE (rates per env-step): touch(unengaged)  grind-bill  force-bill  tilt-bill  unsched-bill | v_tan while touching p50/p90/p99 (m/s) | press while touching p50/p90 (N) | shear_down p50/p90 (N) | in-window unengaged: within 10 cm of anchor")
for e in range(4):
  t = float(touch_n[:, e].sum() / steps_n.sum()); g = float(grind_n[:, e].sum() / steps_n.sum()); f = float(force_n[:, e].sum() / steps_n.sum()); ti = float(tilt_n[:, e].sum() / steps_n.sum())
  vt = torch.cat(vt_samples[e]).numpy() if vt_samples[e] else np.zeros(1)
  pr = torch.cat(press_samples[e]).numpy() if press_samples[e] else np.zeros(1)
  un = float(unsched_n[:, e].sum() / steps_n.sum())
  sh = torch.cat(shear_samples[e]).numpy() if shear_samples[e] else np.zeros(1)
  near = float(near_n[e] / max(float(win_n[e]), 1.0))
  print(f"  {EE[e]}: {t:6.3f} {g:10.3f} {f:10.3f} {ti:9.3f} {un:12.3f} | {np.percentile(vt,50):.2f}/{np.percentile(vt,90):.2f}/{np.percentile(vt,99):.2f} | {np.percentile(pr,50):.1f}/{np.percentile(pr,90):.1f} | {np.percentile(sh,50):.1f}/{np.percentile(sh,90):.1f} | {near:.2f} (n={int(win_n[e])})")
print("\nweld sub-conditions on TOUCHING unengaged pads (fraction of touching steps): shear>=tau  face_cos>=thr  antirake_ok  cooldown_ok  ALL | engage_count histogram 0..5 (5 = weld)")
for e in range(4):
  n = max(float(touch_steps[e]), 1.0)
  fr = [float(cond_n[e, k] / n) for k in range(5)]
  hist = [int(x) for x in count_hist[e].tolist()]
  print(f"  {EE[e]}: " + "  ".join(f"{x:.3f}" for x in fr) + f" | {hist}")
print("\nper clip bin: steps  touch-rate(any pad)  grind-rate(any pad)  [per-EE grind rates LH RH LF RF]")
for b in range(nbins):
  if steps_n[b] == 0: continue
  f0 = int(b * C // nbins)
  tr = float((touch_n[b].sum()) / steps_n[b]); gr = float(grind_n[b].sum() / steps_n[b])
  per = " ".join(f"{float(grind_n[b, e] / steps_n[b]):.3f}" for e in range(4))
  print(f"  bin {b:2d} frame~{f0:4d} {seg(f0):17s} n={int(steps_n[b]):6d}  touch {tr:5.3f}  grind {gr:5.3f}  [{per}]")
