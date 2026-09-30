"""Seed fate under a POLICY, per birth clip-bin (or pinned frames).
Runs a checkpoint (stochastic = training distribution, deterministic mean,
or a passive hold of the birth pose) over the ORIGINAL birth cohort only
(auto-reset re-births are ignored) and reports, per birth bin / frame: when
the scheduled seeds are physically released (free = inside the settle
exemption, billed = later), the release reasons, per-EE losses, and the
death cause within STEPS. Answers "does the policy hold the births the
passive map calls untenable, or lose them for free and pantomime?"
Usage: CUDA_VISIBLE_DEVICES=2 python scripts/transition80_birth_fate_probe.py CKPT [nenv=1024] [steps=150] [mode=1] [frames=a,b,..]
  mode: 1 stochastic policy, 0 deterministic mean, -1 hold birth pose (no policy)
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
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 150
MODE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
FRAMES = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
device = "cuda:0"
# plant/envelope what-ifs (passive or policy): FATE_GIMBAL_DEG widens the pad gimbal travel
# (plant rebuild), FATE_NO_MOMENT=1 turns pad-moment billing off, FATE_CAP_SCALE scales the
# per-unit catch capacity band (envelope size).
import math
import scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants as _K
if os.environ.get("FATE_GIMBAL_DEG"):
  _K.PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD = math.radians(float(os.environ["FATE_GIMBAL_DEG"]))
cfg = load_env_cfg(TASK, play=False)
_ls = cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
if os.environ.get("FATE_GUARD"):
  _ls.seed_hold_judge_guard_steps = int(os.environ["FATE_GUARD"])
if os.environ.get("FATE_NO_MOMENT") == "1":
  _ls.moment_capacity_enable = False
if os.environ.get("FATE_CAP_SCALE"):
  _c = float(os.environ["FATE_CAP_SCALE"]); _ls.unit_catch_capacity_range = tuple(v * _c for v in _ls.unit_catch_capacity_range)
cfg.scene.num_envs = NENVS
mc = cfg.commands["climb_motion"]
mc.sampling_mode = "uniform"
if os.environ.get("FATE_CLIP"):
  mc.motion_file = str(Path(os.environ["FATE_CLIP"]).resolve())  # audit a draft clip
if FRAMES:
  mc.spawn_early_fraction = 1.0; mc.spawn_bias_auto_rake_frames = 0; mc.spawn_bias_frame_windows = ()
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), "/tmp/.t80_fate_runner", device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy_det = runner.get_inference_policy(device=device)
actor = runner.alg.actor
u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
act = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
term = u.termination_manager
C = motion.clip.cycle_frames; nbins = motion._bin_count
EE = ["LH", "RH", "LF", "RF"]
EEJ = ["left_hand", "right_hand", "left_foot", "right_foot"]
GIM = {}
REASON = {0: "none", 1: "rev-shear", 2: "tension", 3: "lateral", 4: "ultimate", 5: "tilt", 6: "dr"}
EXEMPT = int(round(TRANSITION80_SEED_SETTLE_EXEMPT_S / float(env.step_dt)))
FREEZE = int(mc.birth_freeze_steps)
seg_d = np.load(mc.motion_file, allow_pickle=True).item()
names = list(seg_d["segment_names"]); labels = np.asarray(seg_d["phase_segment"])
def seg(f): return names[labels[min(int(f), len(labels) - 1)]]
tnames = list(term.active_terms)
SHORT = {"anchor_rip": "rip", "contact_loss": "closs", "engage_constraint": "cone", "body_pos": "bpos", "anchor_pos": "apos",
         "anchor_ori": "aori", "wall_contact": "wall", "base_height": "fall", "bad_state": "bad", "time_out": "tout"}

def run_cohort():
  env.reset()
  obs = wrapped.get_observations()
  birth = motion._birth_frames.clone()
  seed = motion.clip.ee_engaged[birth].bool()
  ids = act._target_ids
  hold = (robot.data.joint_pos[:, ids] - act._offset) / act._scale
  alive = torch.ones(NENVS, dtype=torch.bool, device=device)
  first_rel = torch.full((NENVS, 4), -1, dtype=torch.long, device=device)
  rel_reason = torch.zeros((NENVS, 4), dtype=torch.long, device=device)
  death_cause = torch.full((NENVS,), -1, dtype=torch.long, device=device)
  death_step = torch.full((NENVS,), -1, dtype=torch.long, device=device)
  gim_ids = [i for i, n in enumerate(robot.joint_names) if "gimbal" in n]
  gim_max = torch.zeros(NENVS, 4, device=device)
  gim_ee = torch.tensor([[k for k, i in enumerate(gim_ids) if EEJ[e] in robot.joint_names[i]] for e in range(4)], device=device) if gim_ids else None
  for step in range(STEPS):
    with torch.no_grad():
      if MODE == 1: a = actor(obs, stochastic_output=True)
      elif MODE == 0: a = policy_det(obs)
      else: a = hold
    obs, rew, dones, extras = wrapped.step(a)
    if gim_ids and gim_ee is not None:
      g = robot.data.joint_pos[:, gim_ids].abs()
      for e in range(4):
        gim_max[:, e] = torch.where(alive, torch.maximum(gim_max[:, e], g[:, gim_ee[e]].max(dim=1).values), gim_max[:, e])
    rel = aff.forced_release_this_step & seed & alive.unsqueeze(1) & (first_rel < 0)
    first_rel[rel] = step
    rel_reason[rel] = aff.release_reason[rel].long()
    d = dones.bool() & alive
    for k, n in enumerate(tnames):
      f = term.get_term(n) & d & (death_cause < 0)
      death_cause[f] = k; death_step[f] = step
    alive &= ~dones.bool()
  GIM["last"] = gim_max
  return birth, seed, first_rel, rel_reason, death_cause, death_step, alive

def report(groups, birth, seed, first_rel, rel_reason, death_cause, death_step, alive):
  print(f"  columns: n seeded | seeded envs losing ANY seed by step {EXEMPT}(free)/{FREEZE}/50/{STEPS} | median first-loss step | reasons | per-EE lost/seeded | deaths (fraction of n) | alive")
  for label, m in groups:
    n = int(m.sum())
    if n == 0: continue
    s = seed & m.unsqueeze(1)
    seeded_env = s.any(1)
    ns = int(seeded_env.sum())
    lost = (first_rel >= 0) & s
    def by(k):
      if ns == 0: return 0.0
      return float(((first_rel <= k) & lost).any(1)[seeded_env].float().mean())
    steps = first_rel[lost]
    med = f"{int(steps.float().median())}" if steps.numel() else "-"
    reasons = torch.bincount(rel_reason[lost], minlength=7).tolist()
    rtxt = " ".join(f"{REASON[i]} {c}" for i, c in sorted(enumerate(reasons), key=lambda kv: -kv[1]) if c)
    ee = " ".join(f"{EE[e]} {int(lost[:, e].sum())}/{int(s[:, e].sum())}" for e in range(4) if int(s[:, e].sum()))
    dd = " ".join(f"{SHORT.get(tnames[k], tnames[k])} {int(((death_cause == k) & m).sum()) / n:.2f}" for k in range(len(tnames)) if int(((death_cause == k) & m).sum()))
    gm = GIM.get("last")
    gtxt = ""
    if gm is not None:
      gtxt = " | gimbal max p50/p90 deg " + " ".join(f"{EE[e]} {math.degrees(float(gm[m, e].median())):.1f}/{math.degrees(float(torch.quantile(gm[m, e], 0.9))):.1f}" for e in range(4) if int(s[:, e].sum()))
    print(f"  {label:28s} n={n:4d} seeded {ns:4d}{gtxt} | lost {by(EXEMPT - 1):.2f}/{by(FREEZE):.2f}/{by(50):.2f}/{by(STEPS):.2f} | med {med:>3s} | {rtxt} | {ee} | {dd} | alive {float(alive[m].float().mean()):.2f}")

mode = {1: "stochastic policy", 0: "deterministic mean", -1: "hold birth pose (passive)"}[MODE]
try:
  import mujoco as _mj
  _m = u.sim.mj_model; _j = [i for i in range(_m.njnt) if "gimbal" in _mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_JOINT, i)]
  print(f"gimbal joints {len(_j)} travel +-{math.degrees(float(_m.jnt_range[_j[0], 1])):.1f} deg | moment billing {_ls.moment_capacity_enable} | unit capacity {_ls.unit_catch_capacity_range}")
except Exception as e:
  print("gimbal readout failed:", e)
print(f"\n=== {CKPT.name}  {mode}  envs {NENVS}  steps {STEPS}  exempt {EXEMPT} steps  freeze {FREEZE}  guard {cfg.commands['microspine_affordance'].microspine_cfg.limit_surface.seed_hold_judge_guard_steps} ===")
if FRAMES:
  groups = []; packs = []
  for f in FRAMES:
    motion._bias_window = (f, f + 1)
    pack = run_cohort()
    birth = pack[0]
    assert int((birth - f).abs().max()) <= 1, (f, birth[:4].tolist())
    print(f"\n--- pinned frame {f} {seg(f)} sched {dict(zip(EE, motion.clip.ee_engaged[f].int().tolist()))}")
    report([(f"frame {f} {seg(f)}", torch.ones(NENVS, dtype=torch.bool, device=device))], *pack)
else:
  pack = run_cohort()
  birth = pack[0]
  bins = (birth * nbins) // C
  groups = [(f"bin {b:2d} f~{int(b * C // nbins):4d} {seg(int(b * C // nbins))}", bins == b) for b in range(nbins)]
  print("\nper birth bin (uniform births):")
  report(groups, *pack)
  masked = ((birth >= 270) & (birth < 380)) | ((birth >= 510) & (birth < 570)) | ((birth >= 790) & (birth < 800))
  print("\nformer mask vs rest:")
  report([("FORMER-MASK 270-380/510-570/790", masked), ("REST of the clip", ~masked)], *pack)
