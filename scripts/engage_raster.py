"""Roll out a climb checkpoint; produce the standard visualization trio.

One deterministic rollout produces, exactly aligned on the same time axis:
  <out_prefix>.png         — engage raster: policy latch bars heat-colored by
                             the limit-surface hold margin (green = unloaded,
                             red = about to fail) vs the reference schedule
                             (blue); forced releases as ticks colored by
                             failure type; episode resets as dashed lines.
  <out_prefix>_torque.png  — per-actuator |torque|/limit heatmap over time
                             plus the worst-joint margin-to-limit curve.
  <out_prefix>.mp4         — 720p side tracking-camera video of the SAME
                             rollout (timestamp burned in so video seconds
                             map 1:1 onto both plots).

Also prints occupancy / match / release stats, the limit-surface release
reason histogram (1=reverse shear [free by design], 2=tension, 3=lateral,
4=ultimate), per-EE engaged hold-margin stats and the torque-margin table.

Reference-replay mode (`reference` instead of a checkpoint) swaps the learned
policy for a scripted actor whose PD target is the motion clip itself — a
dynamic existence proof that the reference choreography is executable under
the live engage pipeline (cooldown/debounce, recruitment ramp, F0 lottery,
moment billing). Prints a PASS/FAIL verdict and writes it as JSON next to the
plots. `worstf0` (anywhere on the command line) collapses the per-grip F0
lottery to its worst draw.

Usage:
  python scripts/engage_raster.py <ckpt.pt> [angle_deg] [steps] [out_prefix]
  python scripts/engage_raster.py reference [angle_deg] [steps] [out_prefix] [worstf0]
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

TASK = next(
  (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("task=")),
  "Mjlab-Climb-Wall-Phybot-C2",
)
WORST_F0 = "worstf0" in sys.argv[1:]
SCRIPT_RELEASE = "scriptrelease" in sys.argv[1:]
GANTRY = "gantry" in sys.argv[1:]
LEASH_N = next(
  (float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("leash=")),
  0.0,
)
CLIP_OVERRIDE = next(
  (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("clip=")), None
)
TILT_BAND = next(
  (
    tuple(float(x) for x in a.split("=", 1)[1].split(","))
    for a in sys.argv[1:]
    if a.startswith("tiltband=")
  ),
  None,
)
# upperpd=kp,kd — override the upper-body actuator gains (waist_yaw,
# shoulder p/r/y, elbow) for a PD sweep. The arm chain carries the hang, so
# its stiffness sets both the tracking error under load and the pad rotation
# the anchors see; this token makes that a one-line experiment.
UPPER_PD = next(
  (
    tuple(float(x) for x in a.split("=", 1)[1].split(","))
    for a in sys.argv[1:]
    if a.startswith("upperpd=")
  ),
  None,
)
# DR-pinned bench tokens (2026-08-02): play strips all DR, so the clean exam
# says nothing about the lottery corners. These re-arm individual pieces at
# PINNED draws: drfriction=X (abs geom friction), drmass=Y (torso delta kg),
# drbias=Z (encoder-bias half-range in rad; the per-joint draw inside +-Z
# stays random, so bias cells carry draw-to-draw variance), drnoise (actor
# obs corruption ON — the training noise cfgs are already on the terms).
DR_FRICTION = next(
  (float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("drfriction=")),
  None,
)
DR_MASS = next(
  (float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("drmass=")),
  None,
)
DR_BIAS = next(
  (float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("drbias=")),
  None,
)
DR_NOISE = "drnoise" in sys.argv[1:]
# drrelease=T1[,T2...] — R4 recovery drill: at each sim-time (s) force-release
# one engaged, schedule-anchored EE (reason=dr-forced, report=True), which
# arms the reseat hold and freezes the motion clock until the pad re-seats
# or the hold times out. The verdict gains a "reseat drills" line.
DR_RELEASE_TIMES = next(
  (
    [float(x) for x in a.split("=", 1)[1].split(",")]
    for a in sys.argv[1:]
    if a.startswith("drrelease=")
  ),
  [],
)
# ablate=term1,term2 — zero the named ACTOR observation terms (scale=0.0,
# applied before the policy's baked normalizer, so the model sees the
# z-scored "wrong" value). Route-B RMA step 0: measures the policy's true
# dependence on each privileged channel — channels that survive zeroing
# need no estimator; channels that crater define phi's regression targets.
ABLATE_TERMS = next(
  (
    [t for t in a.split("=", 1)[1].split(",") if t]
    for a in sys.argv[1:]
    if a.startswith("ablate=")
  ),
  [],
)
# estimator=<phi.pt> — Route-B RMA in-the-loop exam: run the TEACHER actor
# with its privileged slices fed from the phi estimator (phi_meta.json next
# to the .pt names the slices; the droppable channels are zeroed). Use with
# task=...-Distill so the env carries the student group phi reads; the
# policy/agent cfg is forced to the BASE task's PPO cfg (the checkpoint is
# a teacher actor, not a distillation student).
ESTIMATOR_PATH = next(
  (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("estimator=")),
  None,
)
_argv = [
  a
  for a in sys.argv[1:]
  if a not in ("worstf0", "scriptrelease", "gantry", "drnoise")
  and not a.startswith(
    (
      "clip=",
      "leash=",
      "tiltband=",
      "task=",
      "upperpd=",
      "drfriction=",
      "drmass=",
      "drbias=",
      "drrelease=",
      "ablate=",
      "estimator=",
    )
  )
]
REFERENCE = bool(_argv) and _argv[0] == "reference"
CKPT = None if REFERENCE else Path(_argv[0]).resolve()
ANGLE = float(_argv[1]) if len(_argv) > 1 else 90.0
STEPS = int(_argv[2]) if len(_argv) > 2 else 1500
if len(_argv) > 3:
  PREFIX = Path(_argv[3])
elif REFERENCE:
  # Existence-proof replays share one archive dir; the clip identity is
  # whatever the task config currently trains on (recorded in the verdict).
  _suffix = "_worstf0" if WORST_F0 else ""
  PREFIX = (
    REPO / "logs" / "eval" / "replay_reference" / f"replay_{ANGLE:.0f}deg{_suffix}"
  )
else:
  # Default: archive under the checkpoint's own run, e.g.
  # logs/eval/<run_name>/raster_45deg_model_1400.
  run_name = CKPT.parent.name
  PREFIX = REPO / "logs" / "eval" / run_name / f"raster_{ANGLE:.0f}deg_{CKPT.stem}"
PREFIX.parent.mkdir(parents=True, exist_ok=True)
LABEL = "reference-replay" if REFERENCE else CKPT.name
REASON_NAMES = {
  1: "reverse-shear(free)",
  2: "tension",
  3: "lateral",
  4: "ultimate",
  5: "tilt-peel",
  6: "dr-forced",
}

device = "cuda:0"
env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
# Pin the play gravity stage to the requested angle, whichever curriculum
# flavor the config carries. (Play configs already disable the angle jitter.)
if "climb_difficulty" in env_cfg.curriculum:
  env_cfg.curriculum["climb_difficulty"].params["stages"] = [
    {"angle_deg": ANGLE, "min_height_gain": 0.18}
  ]
else:
  env_cfg.curriculum["gravity_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": ANGLE}
  ]
if WORST_F0:
  # Pin the per-grip F0 lottery at its worst draw. A collapsed range redraws
  # the same value at every latch (bench determinism, see
  # MicrospineLimitSurfaceCfg.redraw_f_t_max_per_latch).
  _ls = env_cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
  _ls.f_t_max_range = (_ls.f_t_max_range[0], _ls.f_t_max_range[0])
if CLIP_OVERRIDE:
  # Audit a candidate clip without flipping the task's training default.
  from scr_phybot_mjab.tasks.climb_wall import mdp as _mdp

  _clip_path = str(Path(CLIP_OVERRIDE).resolve())
  env_cfg.commands["climb_motion"].motion_file = _clip_path
  _vz = _mdp.clip_average_climb_rate(_clip_path)
  env_cfg.commands["climb_velocity"].ranges.lin_vel_z = (_vz, _vz)
if TILT_BAND is not None:
  # Sensitivity sweeps over the peel-band calibration: override both the
  # static cfg AND the pinned curriculum stage (the schedule rewrites the
  # cfg each step, so both must agree). "tiltband=0,0" disables the check.
  _band = (TILT_BAND[0], TILT_BAND[-1])
  env_cfg.commands[
    "microspine_affordance"
  ].microspine_cfg.limit_surface.tilt_peel_max_deg_range = _band
  if "tilt_peel" in env_cfg.curriculum:
    env_cfg.curriculum["tilt_peel"].params["stages"] = [
      {"iter": 0, "deg_range": _band}
    ]
if UPPER_PD is not None:
  # Rebuild the actuator tuple (the cfgs are shared module singletons —
  # never mutate in place, other tasks reuse them).
  import dataclasses as _dc

  _kp, _kd = UPPER_PD[0], UPPER_PD[-1]
  _upper = (
    "waist_yaw",
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
    ".*_elbow_pitch",
  )
  _art = env_cfg.scene.entities["robot"].articulation
  env_cfg.scene.entities["robot"].articulation = _dc.replace(
    _art,
    actuators=tuple(
      _dc.replace(a, stiffness=_kp, damping=_kd)
      if a.target_names_expr[0] in _upper
      else a
      for a in _art.actuators
    ),
  )
  print(f"[upperpd] upper-body gains overridden to kp={_kp} kd={_kd}")
if DR_FRICTION is not None or DR_MASS is not None or DR_BIAS is not None or DR_NOISE:
  # Rebuild the pinned events from the TRAINING cfg so the bench can never
  # drift from what training actually randomizes (same funcs, same asset
  # regexes) — only the ranges collapse to the requested draw.
  _train_cfg = load_env_cfg(TASK)
  if DR_FRICTION is not None:
    _ev = _train_cfg.events["body_friction"]
    _ev.params["ranges"] = (DR_FRICTION, DR_FRICTION)
    env_cfg.events["body_friction"] = _ev
  if DR_MASS is not None:
    _ev = _train_cfg.events["base_mass"]
    _ev.params["ranges"] = (DR_MASS, DR_MASS)
    env_cfg.events["base_mass"] = _ev
  if DR_BIAS is not None:
    _ev = _train_cfg.events["encoder_bias"]
    _ev.params["bias_range"] = (-DR_BIAS, DR_BIAS)
    env_cfg.events["encoder_bias"] = _ev
  if DR_NOISE:
    env_cfg.observations["actor"].enable_corruption = True
  print(
    f"[drbench] pinned DR: friction={DR_FRICTION} mass={DR_MASS}"
    f" bias=+-{DR_BIAS} noise={'on' if DR_NOISE else 'off'}"
  )
if ABLATE_TERMS:
  _grp = env_cfg.observations["actor"]
  for _t in ABLATE_TERMS:
    if _t not in _grp.terms:
      raise KeyError(
        f"ablate: '{_t}' not an actor term; have {list(_grp.terms)}"
      )
    _grp.terms[_t].scale = 0.0
  print(f"[ablate] zeroed actor terms: {ABLATE_TERMS}")
env_cfg.viewer = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_ROOT,
  entity_name="robot",
  distance=3.0,
  elevation=-12.0,
  azimuth=90.0,
  height=720,
  width=1280,
)

video_dir = PREFIX.parent / f".{PREFIX.stem}_videotmp"
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(
  env,
  video_folder=video_dir,
  step_trigger=lambda step: step == 0,
  video_length=STEPS,
  disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

u = env.unwrapped
motion = u.command_manager.get_term("climb_motion")
aff = u.command_manager.get_term("microspine_affordance")
executor = u.event_manager.get_term_cfg("microspine_grip").func
robot = u.scene["robot"]
dt = float(u.step_dt)

if REFERENCE:
  # Scripted reference-tracking actor: the PD target is the clip frame being
  # ENTERED (time_steps advances inside env.step, so +1 avoids a one-step
  # phase lag). Inverts the SoftJointPositionAction map exactly like
  # validate_microspine_constraint.hold_pose_actions:
  # raw = (q_target + encoder_bias - offset) / scale. Clip joints and action
  # columns are both PHYBOT_C2_POLICY_JOINT_NAMES in preserved order.
  _term = u.action_manager._terms["joint_pos"]

  def policy(_obs):
    nxt = (motion.time_steps + 1) % motion.clip.cycle_frames
    target = motion.clip.joint_pos[nxt]
    bias = robot.data.encoder_bias[:, _term._target_ids]
    return (target + bias - _term.offset) / _term.scale
else:
  if ESTIMATOR_PATH is not None:
    # Teacher-with-estimator: the checkpoint is a PPO teacher actor, so the
    # runner must be built from the BASE task's PPO cfg even when the env
    # (and TASK token) is the -Distill/-Polish variant that carries the
    # student group.
    _base_task = TASK
    for _suffix in ("-Distill", "-Polish"):
      _base_task = _base_task.removesuffix(_suffix)
    agent_cfg = load_rl_cfg(_base_task)
    runner_cls = load_runner_cls(_base_task) or MjlabOnPolicyRunner
  else:
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  # Distillation ckpts hold student_state_dict; {"actor": True} would be a
  # silent no-op on them (random student weights walking off the wall).
  _policy_set = "student" if hasattr(agent_cfg, "student") else "actor"
  runner.load(
    str(CKPT), load_cfg={_policy_set: True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  if ESTIMATOR_PATH is not None:
    import json as _json

    _phi = torch.jit.load(ESTIMATOR_PATH, map_location=device).eval()
    _pm = _json.loads(Path(ESTIMATOR_PATH).with_suffix(".json").read_text())
    _tsl = [(t["actor_start"], t["actor_end"]) for t in _pm["targets"]]
    _zsl = [(z["actor_start"], z["actor_end"]) for z in _pm["zero_slices"]]
    _teacher_policy = policy

    def policy(_obs):  # noqa: F811 - deliberate estimator wrapper
      with torch.no_grad():
        _est = _phi(_obs["student"])
      _a = _obs["actor"].clone()
      _ofs = 0
      for _s, _e in _tsl:
        _a[:, _s:_e] = _est[:, _ofs : _ofs + (_e - _s)]
        _ofs += _e - _s
      for _s, _e in _zsl:
        _a[:, _s:_e] = 0.0
      _o2 = _obs.clone()
      _o2["actor"] = _a
      return _teacher_policy(_o2)

    print(f"[estimator] privileged slices fed from {ESTIMATOR_PATH}")

act_names = list(robot.actuator_names)
n_act = len(act_names)
# Model-level force ranges, sliced to this robot's actuators (ctrl_ids); the
# bridge returns (nu, 2) or a per-world (nworld, nu, 2).
ctrl_ids = robot.indexing.ctrl_ids
fr = torch.as_tensor(u.sim.model.actuator_forcerange)
if fr.ndim == 3:
  fr = fr[0]
tau_limit = fr[ctrl_ids].abs().amax(dim=-1).cpu().numpy()
tau_limit = np.where(tau_limit > 0, tau_limit, np.inf)

ret = env.get_observations()
obs = ret[0] if isinstance(ret, (tuple, list)) else ret

if LEASH_N > 0.0:
  _leash_body = robot.find_bodies(["torso"])[0]
  _leash_zero_torque = torch.zeros(1, 1, 3, device=device)
leash_hist = np.zeros(STEPS, dtype=np.float32)

pol = np.zeros((STEPS, 4), dtype=bool)
ref = np.zeros((STEPS, 4), dtype=bool)
forced = np.zeros((STEPS, 4), dtype=bool)
reason = np.zeros((STEPS, 4), dtype=np.int8)
margin = np.ones((STEPS, 4), dtype=np.float32)
margin_raw = np.ones((STEPS, 4), dtype=np.float32)
tau = np.zeros((STEPS, n_act), dtype=np.float32)
gain = np.zeros(STEPS, dtype=np.float32)
face_cos = np.ones((STEPS, 4), dtype=np.float32)
tilt_cap = np.zeros((STEPS, 4), dtype=np.float32)
# Capture-relative rotation split into PRY (about in-plane axes; the soft
# suspension-travel regime the 4-deg peel calibrates) and YAW (about the
# wall normal; tip-shear regime — near-rigid on hardware).
tilt_yaw = np.zeros((STEPS, 4), dtype=np.float32)
tilt_pry = np.zeros((STEPS, 4), dtype=np.float32)


def _quat_mul_np(a, b):
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


resets = []
prev_ref = motion.ref_engaged[0].clone()
_drill_idx = 0
_drill_log: list[tuple[int, int]] = []  # (step, ee_index)
with torch.no_grad():
  for t in range(STEPS):
    if LEASH_N > 0.0:
      # Soft leash: a force-capped PD pull of the base toward the ghost,
      # standing in for the balance/weight-shift control a policy learns
      # (ref15 demonstrated it) but open-loop joint replay lacks. Unlike
      # the gantry rail it cannot teleport — the cap (< body weight) keeps
      # anchor loads physical, and the recorded leash usage shows where
      # the choreography leans on balance control hardest.
      pos_err = motion.ref_root_pos_w - robot.data.root_link_pos_w
      vel_err = (
        motion.clip.root_lin_vel[motion.time_steps]
        - robot.data.root_link_lin_vel_w
      )
      f = 1200.0 * pos_err + 120.0 * vel_err
      norm = f.norm(dim=-1, keepdim=True).clamp(min=1.0e-9)
      f = f * (norm.clamp(max=LEASH_N) / norm)
      leash_hist[t] = float(f.norm())
      robot.write_external_wrench_to_sim(
        forces=f.unsqueeze(1),
        torques=_leash_zero_torque,
        body_ids=_leash_body,
      )
    step_ret = env.step(policy(obs))
    obs, dones = step_ret[0], step_ret[2]
    if GANTRY:
      # Timing-isolation rail: pin the base to the ghost trajectory each
      # policy step. Open-loop joint replay cannot reproduce the clip's
      # weight shifts (no feedback), and the resulting torso sag both
      # spuriously unhooks pads (45-deg spawn: reverse-sheared hands) and
      # misplaces relatch presses — control confounds, not timing ones.
      # With the base on the ghost rail the run isolates exactly what this
      # proof is for: window lengths vs the engage pipeline, relatch seat
      # quality, and fresh-anchor recruitment vs the next scheduled load.
      # Anchor-load margins recorded under the rail are indicative only
      # (the rail carries part of the couple); the static and
      # reference-hold audits own load-path realism.
      root_state = torch.cat(
        [
          motion.ref_root_pos_w,
          motion.ref_root_quat_w,
          motion.clip.root_lin_vel[motion.time_steps],
          motion.clip.root_ang_vel[motion.time_steps],
        ],
        dim=-1,
      )
      robot.write_root_state_to_sim(root_state)
    if SCRIPT_RELEASE:
      # Commanded release at the schedule's falling edges: physically a
      # microspine unhook is the limb loading the pad in reverse shear,
      # a force-control skill in its own right (ref15's policy learned it;
      # open-loop joint replay cannot — the weight shift that unloads the
      # pad first needs feedback). Scripting ONLY the release isolates the
      # question this proof exists for: can the pads RE-latch and HOLD on
      # the clip's timing under the live pipeline? Cooldown is applied
      # exactly as a real disengage would, and contact spheres return.
      cur_ref = motion.ref_engaged[0]
      to_release = prev_ref & ~cur_ref & aff.engaged[0]
      if bool(to_release.any()):
        # report=False: a scheduled release is not a fault and must not
        # pollute the forced-release statistics in the verdict.
        aff.force_release(
          torch.tensor([0], device=aff.engaged.device),
          to_release.unsqueeze(0),
          report=False,
        )
      prev_ref = cur_ref.clone()
    if _drill_idx < len(DR_RELEASE_TIMES) and t * dt >= DR_RELEASE_TIMES[_drill_idx]:
      # Fire at the first opportunity >= T: pick the engaged EE the schedule
      # wants anchored with the most run time left — the fully off-schedule
      # loss, the hardest recovery. report=True arms the reseat hold.
      cand = aff.engaged[0] & motion.ref_engaged[0]
      if bool(cand.any()):
        run_left = torch.where(
          cand, motion.ref_run_time[0], torch.zeros_like(motion.ref_run_time[0])
        )
        ee = int(torch.argmax(run_left).item())
        m = torch.zeros_like(aff.engaged[:1])
        m[0, ee] = True
        aff.force_release(torch.tensor([0], device=aff.engaged.device), m)
        _drill_log.append((t, ee))
        _drill_idx += 1
    pol[t] = motion.robot_engaged[0].bool().cpu().numpy()
    ref[t] = motion.ref_engaged[0].bool().cpu().numpy()
    forced[t] = aff.forced_release_this_step[0].bool().cpu().numpy()
    reason[t] = aff.release_reason[0].cpu().numpy()
    margin[t] = aff.hold_margin[0].float().cpu().numpy()
    margin_raw[t] = aff.hold_margin_raw[0].float().cpu().numpy()
    tau[t] = robot.data.actuator_force[0].float().cpu().numpy()
    gain[t] = float(motion.height_gained[0].item())
    face_cos[t] = aff.face_cos[0].float().cpu().numpy()
    tilt_cap[t] = aff.tilt_from_capture[0].float().cpu().numpy()
    pad_q = (
      robot.data.body_link_quat_w[0, executor._ee_body_ids].float().cpu().numpy()
    )
    anc_q = executor.anchor_quat_w[0].float().cpu().numpy()
    anc_conj = anc_q * np.array([1.0, -1.0, -1.0, -1.0])
    q_rel = _quat_mul_np(pad_q, anc_conj)
    q_rel *= np.where(q_rel[:, 0:1] >= 0.0, 1.0, -1.0)
    vec_n = np.linalg.norm(q_rel[:, 1:4], axis=-1)
    ang = 2.0 * np.arctan2(vec_n, q_rel[:, 0])
    axis = q_rel[:, 1:4] / np.maximum(vec_n, 1.0e-9)[:, None]
    rotvec = axis * ang[:, None]
    tilt_yaw[t] = np.abs(rotvec[:, 0])  # wall normal = world x
    tilt_pry[t] = np.linalg.norm(rotvec[:, 1:3], axis=-1)
    if bool(dones[0]):
      resets.append(t)
env.close()

names = ["LH", "RH", "LF", "RF"]
match = float((pol == ref).mean())
occ_p, occ_r = pol.mean(0), ref.mean(0)
util = np.abs(tau) / tau_limit  # (STEPS, n_act) torque utilization


def _events(mask_col):
  d = np.diff(np.r_[0, mask_col.astype(int)])
  return np.where(d == -1)[0]


print(f"steps={STEPS} ({STEPS * dt:.1f}s) angle={ANGLE:.0f} actor={LABEL}")
print(f"match(step x EE agreement)={match:.3f}")
rel_tot, forc_tot = 0, 0
for i, n in enumerate(names):
  rel = len(_events(pol[:, i]))
  fr_n = int(forced[:, i].sum())
  rel_tot += rel
  forc_tot += fr_n
  print(
    f"  {n}: occ pol={occ_p[i]:.2f} ref={occ_r[i]:.2f} | releases={rel} "
    f"(forced={fr_n}) | ref releases={len(_events(ref[:, i]))}"
  )
print(f"total releases={rel_tot} forced={forc_tot}")
hist = {
  REASON_NAMES[rid]: int((forced & (reason == rid)).sum())
  for rid in (1, 2, 3, 4, 5)
}
print(f"forced-release reasons: {hist}")
if DR_RELEASE_TIMES:
  _EE = ("LH", "RH", "LF", "RF")
  _timeout = aff._mcfg.dr_reseat_hold_timeout_steps
  _rows = []
  for t0, ee in _drill_log:
    _reseat = next((tt for tt in range(t0 + 1, STEPS) if pol[tt, ee]), None)
    _ok = _reseat is not None and (_reseat - t0) <= _timeout
    _rows.append(
      f"[t={t0 * dt:.1f}s ee={_EE[ee]} "
      f"reseat={'-' if _reseat is None else f'{(_reseat - t0) * dt:.2f}s'} "
      f"{'OK' if _ok else 'FAIL'}]"
    )
  _n_ok = sum(1 for r in _rows if r.endswith("OK]"))
  print(
    f"reseat drills: n={len(_drill_log)} ok={_n_ok} "
    f"requested={len(DR_RELEASE_TIMES)} " + " ".join(_rows)
  )
# Moment attribution: the envelope judged moment-billed EFFECTIVE loads; a
# release whose force-only margin was still positive was caused by the
# moment fold alone (billed edge prying, not raw force overload).
moment_caused = forced & (margin_raw > 0.0)
print(
  f"moment-billed releases (force-only margin still positive): "
  f"{int(moment_caused.sum())}"
)
for i, n in enumerate(names):
  m = margin[pol[:, i], i]
  if len(m):
    print(
      f"  {n}: engaged hold margin min={m.min():.2f} "
      f"p5={np.percentile(m, 5):.2f} mean={m.mean():.2f}"
    )
peak = util.max(axis=0)
order = np.argsort(-peak)
print("torque peak |τ|/limit  top5: " + ", ".join(
  f"{act_names[i]}={peak[i]:.2f}" for i in order[:5]
))
print(
  f"height gained (last)={gain[-1]:.3f} m; "
  f"resets at {[round(r * dt, 1) for r in resets]} s"
)
# Release-dip: base drop within 0.2 s after each release event. A properly
# unloaded (weight-shift-then-peel) unhook shows ~zero dip — the visible
# sag is the signature of releasing a still-loaded anchor (the ~85 N
# support withdrawal, not the ~20 N threshold push, is what jolts the
# body). Companion metric to the release-step torque spike.
_dip_win = max(1, int(round(0.2 / dt)))
_dips = []
for i in range(4):
  for t_ev in np.where(np.diff(np.r_[0, pol[:, i].astype(int)]) == -1)[0]:
    hi = min(STEPS, t_ev + _dip_win)
    if t_ev >= 1 and hi > t_ev:
      _dips.append(gain[t_ev - 1] - gain[t_ev:hi].min())
if _dips:
  print(
    f"release dip (0.2 s window): mean={np.mean(_dips) * 1000:.1f} mm, "
    f"max={np.max(_dips) * 1000:.1f} mm over {len(_dips)} releases"
  )
# Engaged-pad tilt: angle between the pad face axis and the wall normal
# while latched. The weld's rotation rows are deliberately soft
# (anchor_torquescale 0.01 — a stiff lock trapped phantom PD moments), so
# tilt is where the sim under-constrains reality: a REAL hooked patch
# holds orientation up to edge peel. Small angles here = the policy is not
# exploiting the softness and the sim2real rotation gap stays benign.
_tilts = []
for i in range(4):
  m = pol[:, i]
  if m.any():
    _tilts.append(
      np.degrees(np.arccos(np.clip(face_cos[m, i], -1.0, 1.0)))
    )
if _tilts:
  _tilt_all = np.concatenate(_tilts)
  print(
    f"engaged-pad tilt vs wall normal: mean={_tilt_all.mean():.1f} deg, "
    f"p95={np.percentile(_tilt_all, 95):.1f}, max={_tilt_all.max():.1f}"
  )
_caps = np.degrees(tilt_cap[pol])
if _caps.size:
  print(
    f"engaged-pad rotation from capture: mean={_caps.mean():.1f} deg, "
    f"p95={np.percentile(_caps, 95):.1f}, max={_caps.max():.1f} "
    f"(tilt-peel DR band 3.5-4.5)"
  )
  _yaw = np.degrees(tilt_yaw[pol])
  _pry = np.degrees(tilt_pry[pol])
  print(
    f"  split: PRY (suspension axes) mean={_pry.mean():.1f} "
    f"p95={np.percentile(_pry, 95):.1f} max={_pry.max():.1f} | "
    f"YAW (wall-normal, tip-shear axis) mean={_yaw.mean():.1f} "
    f"p95={np.percentile(_yaw, 95):.1f} max={_yaw.max():.1f}"
  )

# Torque-spike attribution: near-limit steps classified by their distance
# to the nearest latch-state edge (release vs engage, any EE). Locates
# WHERE the peak effort is spent — the unhook push or the fresh-anchor
# down-charge (recruitment needs >= 10 N down-shear; a violent charge is a
# pull-up through the shoulder, same joints as a violent unhook).
_spike_steps = np.where(util.max(axis=1) > 0.8)[0]
if len(_spike_steps):
  rel_edges, eng_edges = [], []
  for i in range(4):
    d = np.diff(np.r_[0, pol[:, i].astype(int)])
    rel_edges.extend(np.where(d == -1)[0])
    eng_edges.extend(np.where(d == 1)[0])
  rel_edges = np.array(sorted(rel_edges)) if rel_edges else np.array([-1e9])
  eng_edges = np.array(sorted(eng_edges)) if eng_edges else np.array([-1e9])
  near_rel = near_eng = tied = 0
  for t_ev in _spike_steps:
    dr = np.abs(rel_edges - t_ev).min() * dt
    de = np.abs(eng_edges - t_ev).min() * dt
    if abs(dr - de) < 0.02:
      tied += 1
    elif dr < de:
      near_rel += 1
    else:
      near_eng += 1
  print(
    f"torque spikes (worst util > 0.8): {len(_spike_steps)} steps — "
    f"nearest edge: release {near_rel}, engage {near_eng}, tied {tied}"
  )

if REFERENCE:
  import json

  # Existence-proof verdict: the reference is "executable" iff the scripted
  # replay climbs without structural failures — no falls, >= 2 rungs gained,
  # no overload (tension/lateral/ultimate) releases, reverse-shear unhooks
  # only near their scheduled release edges, and the latch chart followed.
  grace_s = 0.3
  grace_steps = int(round(grace_s / dt))
  off_edge_reverse = 0
  for i in range(4):
    fall_edges = np.where(np.diff(np.r_[ref[:, i].astype(int), 0]) == -1)[0]
    for t_ev in np.where(forced[:, i] & (reason[:, i] == 1))[0]:
      if len(fall_edges) == 0 or np.abs(fall_edges - t_ev).min() > grace_steps:
        off_edge_reverse += 1
  overload = {
    name: int((forced & (reason == rid)).sum())
    for rid, name in ((2, "tension"), (3, "lateral"), (4, "ultimate"))
  }
  checks = {
    "angle_deg": ANGLE,
    "steps": STEPS,
    "seconds": round(STEPS * dt, 2),
    "worst_f0": WORST_F0,
    "script_release": SCRIPT_RELEASE,
    "gantry": GANTRY,
    "leash_n_cap": LEASH_N,
    "leash_n_mean": round(float(leash_hist.mean()), 2),
    "leash_n_max": round(float(leash_hist.max()), 2),
    "motion_file": str(motion.cfg.motion_file),
    "no_resets": len(resets) == 0,
    "height_gained_m": round(float(gain[-1]), 4),
    # On the gantry rail the base is pinned to the ghost, so height is
    # assumed rather than proven — the criterion drops out there.
    "height_ok": bool(GANTRY or gain[-1] >= 0.06),
    "overload_counts": overload,
    "no_overload_releases": sum(overload.values()) == 0,
    "reverse_releases_off_schedule": int(off_edge_reverse),
    "reverse_ok": off_edge_reverse == 0,
    "match": round(match, 4),
    "match_ok": bool(match >= 0.9),
  }
  checks["verdict"] = (
    "PASS"
    if (
      checks["no_resets"]
      and checks["height_ok"]
      and checks["no_overload_releases"]
      and checks["reverse_ok"]
      and checks["match_ok"]
    )
    else "FAIL"
  )
  verdict_path = Path(str(PREFIX) + ".verdict.json")
  verdict_path.write_text(json.dumps(checks, indent=2))
  print(f"reference replay verdict: {checks['verdict']}  -> {verdict_path}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

time_s = np.arange(STEPS) * dt

# ---------------------------------------------------------------- raster ----
fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True, constrained_layout=True)
margin_cmap = plt.get_cmap("RdYlGn")
margin_norm = Normalize(vmin=0.0, vmax=1.0)
# Policy panel: each latch bar is split into two lanes. The top lane is the
# FORCE-ONLY hold margin (moment billing excluded); the bottom lane is the
# BILLED margin the release logic actually judges (forces + worst-edge
# moment fold). A bottom lane visibly redder than its top lane means the
# moment tax is eating that anchor's headroom — the moment margin is
# exactly the bottom lane's remaining green.
for i in range(4):
  for lane, (values, y_hi, y_lo) in enumerate(
    (
      (margin_raw[:, i], i - 0.38, i - 0.02),
      (margin[:, i], i + 0.02, i + 0.38),
    )
  ):
    rgba = margin_cmap(margin_norm(np.clip(values, 0.0, 1.0)))
    rgba[:, 3] = np.where(pol[:, i], 0.95, 0.0)
    axes[0].imshow(
      rgba[None, :, :],
      aspect="auto",
      interpolation="nearest",
      extent=(0.0, STEPS * dt, y_lo, y_hi),
      zorder=1,
    )
axes[0].set_title(
  f"policy  {LABEL} @ {ANGLE:.0f}°   (per row: top lane = force-only "
  "margin, bottom lane = billed margin incl. moment fold — bottom redder "
  "than top = moment tax; ▾ = forced release by type, ◆ = moment-billed)",
  loc="left",
  fontsize=10,
)
for i in range(4):
  e = ref[:, i].astype(int)
  d = np.diff(np.r_[0, e, 0])
  for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    axes[1].fill_between(
      [s0 * dt, e0 * dt], i - 0.38, i + 0.38, color="#3b6fd4", lw=0, alpha=0.9
    )
axes[1].set_title(
  "reference clip engage schedule (phase-aligned)", loc="left", fontsize=11
)
for ax in axes:
  ax.set_yticks(range(4), names)
  ax.set_ylim(3.6, -0.6)
  ax.set_xlim(0.0, STEPS * dt)
  ax.grid(axis="x", alpha=0.3)
fig.colorbar(
  cm.ScalarMappable(norm=margin_norm, cmap=margin_cmap),
  ax=list(axes),
  pad=0.01,
  label="hold margin (1 unloaded → 0 release boundary)",
)
# Forced releases split by limit-surface failure type. Reverse shear is the
# pad's intended unhook direction (free while supported); the warm colors are
# genuine overload failures.
REASON_STYLE = {
  1: ("#4c9be8", "reverse-shear (free unhook)"),
  2: ("#e03131", "tension pull-off"),
  3: ("#f59f00", "lateral peel"),
  4: ("#9c36b5", "ultimate overload"),
  5: ("#0ca678", "tilt-peel (rotated off grip)"),
}
for rid, (color, _label) in REASON_STYLE.items():
  for i in range(4):
    ts = np.where(forced[:, i] & (reason[:, i] == rid))[0]
    if len(ts):
      axes[0].plot(
        ts * dt, np.full(len(ts), i - 0.45), "v", color=color, ms=4, ls="none"
      )
# Moment-billed releases (force-only margin still positive at the kill step):
# the moment fold alone pushed the anchor out — overplotted as diamonds so
# they stand out from whatever direction code the effective load landed on.
for i in range(4):
  ts = np.where(moment_caused[:, i])[0]
  if len(ts):
    axes[0].plot(
      ts * dt,
      np.full(len(ts), i - 0.45),
      "D",
      color="#212529",
      ms=4.5,
      ls="none",
      markerfacecolor="none",
      markeredgewidth=1.4,
    )
legend_handles = [
  Line2D(
    [], [], marker="v", color=color, ls="none", ms=5,
    label=f"{label}  ×{int((forced & (reason == rid)).sum())}",
  )
  for rid, (color, label) in REASON_STYLE.items()
]
legend_handles.append(
  Line2D(
    [], [], marker="D", color="#212529", ls="none", ms=5,
    markerfacecolor="none", markeredgewidth=1.4,
    label=f"moment-billed  ×{int(moment_caused.sum())}",
  )
)
axes[0].legend(
  handles=legend_handles, loc="upper right", fontsize=8, ncol=2, framealpha=0.9
)
for r in resets:
  for ax in axes:
    ax.axvline(r * dt, color="k", ls="--", lw=0.9)
axes[1].set_xlabel("time (s)")
fig.suptitle(
  f"microspine engage raster — match {match:.2f} | occupancy pol "
  f"[{', '.join(f'{v:.2f}' for v in occ_p)}] vs ref [{', '.join(f'{v:.2f}' for v in occ_r)}]"
  f" | height +{gain[-1]:.2f} m",
  fontsize=12,
)
png_path = PREFIX.with_suffix(".png")
fig.savefig(png_path, dpi=130)
print(f"wrote {png_path}")

# ------------------------------------------------------- torque margin ------
fig2, (axh, axm) = plt.subplots(
  2,
  1,
  figsize=(16, 9),
  sharex=True,
  constrained_layout=True,
  gridspec_kw={"height_ratios": [3, 1]},
)
im = axh.imshow(
  util.T,
  aspect="auto",
  interpolation="nearest",
  cmap="RdYlGn_r",
  vmin=0.0,
  vmax=1.0,
  extent=(0.0, STEPS * dt, n_act - 0.5, -0.5),
)
axh.set_yticks(range(n_act), act_names, fontsize=7)
axh.set_title(
  f"|torque| / limit per actuator — peak {act_names[order[0]]} "
  f"{peak[order[0]]:.2f}",
  loc="left",
  fontsize=11,
)
fig2.colorbar(im, ax=[axh, axm], pad=0.01, label="|torque| / limit")
worst_margin = 1.0 - util.max(axis=1)
axm.plot(time_s, worst_margin, color="#1f77b4", lw=1.2)
axm.axhline(0.0, color="#e03131", ls="--", lw=1.0)
axm.set_ylabel("worst-joint margin")
axm.set_ylim(min(-0.05, float(worst_margin.min()) - 0.05), 1.0)
axm.set_xlabel("time (s)")
axm.set_title(
  "margin to torque limit, min over joints (0 = saturated)",
  loc="left",
  fontsize=11,
)
axm.grid(axis="both", alpha=0.3)
for r in resets:
  for ax in (axh, axm):
    ax.axvline(r * dt, color="k", ls="--", lw=0.9)
torque_path = Path(str(PREFIX) + "_torque.png")
fig2.savefig(torque_path, dpi=130)
print(f"wrote {torque_path}")

# ---------------------------------------------------------------- video -----
vids = sorted(video_dir.glob("*.mp4"))
if vids:
  import shutil
  import subprocess

  mp4_path = PREFIX.with_suffix(".mp4")
  # Burn a sim-time stamp so video seconds map 1:1 onto the raster x-axis.
  burn = subprocess.run(
    [
      "ffmpeg", "-v", "error", "-y", "-i", str(vids[0]),
      "-vf",
      "drawtext=text='t=%{eif\\:t\\:d}.%{eif\\:mod(t*100,100)\\:d\\:2}s':"
      "x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5",
      "-an", str(mp4_path),
    ],
    capture_output=True,
    text=True,
  )
  if burn.returncode != 0:
    print(f"timestamp burn failed ({burn.stderr.strip()[:200]}); copying raw video")
    shutil.copy(vids[0], mp4_path)
  print(f"wrote {mp4_path}")
  shutil.rmtree(video_dir, ignore_errors=True)
else:
  print(f"WARNING: no video produced under {video_dir}")
