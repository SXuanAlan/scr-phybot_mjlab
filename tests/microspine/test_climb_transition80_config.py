"""Config contract of the 80-degree transition task (``tasks/climb_transition80``).

A package built by mutating the climb builder's output. Promises asserted here: the real-robot PD table, the
explicit v2 observation contract (actor 238 / critic 419, dead climb lanes
retired, BeyondMimic reference lanes, pad faces, wall face corners), the BeyondMimic-minimal
reward table, the termination-first table with the engage cone in CaT
mode, curricula limited to wall pin + anti-rake ladder + anneals, the a80 clip
on a native 80-degree wall — and ZERO edits to either sibling package.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.climb_transition80.config.phybot_c2.env_cfgs import (
  TRANSITION80_ACTOR_DIM,
  TRANSITION80_ACTOR_TERMS,
  TRANSITION80_ANCHOR_RADIUS_STAGES,
  TRANSITION80_CRITIC_DIM,
  TRANSITION80_CRITIC_TAIL,
  TRANSITION80_OBS_DIMS,
  TRANSITION80_REMOVED_OBS,
  TRANSITION80_SEED_GUARD_STEPS,
  TRANSITION80_SPAWN_XY_NOISE,
  TRANSITION80_SPAWN_YAW_NOISE,
  phybot_c2_climb_transition80_env_cfg,
)
from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import (
  DEFAULT_TRACKING_BODY_NAMES,
  TransitionMotionCommandCfg,
  adaptive_bin_probabilities,
  build_fk_tables,
  finite_difference_velocities,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionCommandCfg

_ASSETS = Path(__file__).resolve().parents[2] / "src/scr_phybot_mjab/assets/motions"
_A80 = _ASSETS / "climb_mount_prologue_a80.npy"

_EXPECTED_REWARDS = {
  "motion_anchor_pos": 0.5,
  "motion_anchor_ori": 0.5,
  "motion_body_pos": 1.0,
  "motion_body_ori": 1.0,
  "motion_pad_pos": 1.0,
  "motion_body_lin_vel": 1.0,
  "motion_body_ang_vel": 1.0,
  "engage_progress": 3.0,
  "hold_margin_wage": 0.5,
  "anchor_rip_penalty": -1.0,
  "action_rate_l2": -0.05,
  "dof_pos_limits": -10.0,
}
_EXPECTED_TERMINATIONS = (
  "time_out",
  "base_height",
  "bad_state",
  "anchor_pos",
  "anchor_ori",
  "body_pos",
  "contact_loss",
  "wall_contact",
  "engage_constraint",
)
_NOISY_PROPRIO = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos_rel",
  "joint_vel_rel",
)


def test_transition80_observation_contract() -> None:
  """v2 (2026-08-21): the contract is written out in env_cfgs, not inherited."""
  cfg = phybot_c2_climb_transition80_env_cfg()
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  assert tuple(actor.terms) == TRANSITION80_ACTOR_TERMS
  assert tuple(critic.terms) == TRANSITION80_ACTOR_TERMS + TRANSITION80_CRITIC_TAIL
  assert TRANSITION80_ACTOR_DIM == 238 and TRANSITION80_CRITIC_DIM == 419
  assert set(TRANSITION80_OBS_DIMS) == set(critic.terms)
  for name in TRANSITION80_REMOVED_OBS:
    assert name not in actor.terms and name not in critic.terms, name
  assert tuple(cfg.commands) == ("microspine_affordance", "climb_motion")
  assert "climb_velocity" not in cfg.commands
  assert tuple(cfg.actions) == ("joint_pos",)
  assert tuple(cfg.actions["joint_pos"].actuator_names) == tuple(PHYBOT_C2_POLICY_JOINT_NAMES)
  # The reference lanes read the motion command; the pad lanes the affordance.
  for name in TRANSITION80_ACTOR_TERMS[11:17]:
    assert actor.terms[name].params == {"command_name": "climb_motion"}, name
  assert actor.terms["microspine_pad_center_world"].params == {"command_name": "microspine_affordance"}
  assert actor.terms["motion_command"].func.__name__ == "generated_commands"
  assert actor.terms["motion_command"].params["command_name"] == "climb_motion"
  for name, scale in (
    ("microspine_anchor_load", 0.01),
    ("microspine_pad_normal_force", 0.01),
  ):
    assert critic.terms[name].scale == pytest.approx(scale), name
  assert critic.terms["microspine_unit_preload_fraction"].scale is None  # already in [0, 1]


def test_transition80_obs_noise_and_critic_copies() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  assert actor.enable_corruption is True
  assert critic.enable_corruption is False
  for name in _NOISY_PROPRIO:
    assert actor.terms[name].noise is not None, name
  for name in TRANSITION80_ACTOR_TERMS[11:]:
    assert actor.terms[name].noise is None, name
  for name in TRANSITION80_ACTOR_TERMS:
    assert critic.terms[name] is not actor.terms[name], name
    assert critic.terms[name].noise is None, name
    assert critic.terms[name].func is actor.terms[name].func, name
  play = phybot_c2_climb_transition80_env_cfg(play=True)
  assert play.observations["actor"].enable_corruption is False
  assert tuple(play.observations["actor"].terms) == TRANSITION80_ACTOR_TERMS


def test_wall_face_corner_constants_match_the_asset() -> None:
  """The corners come from WALL_HALF_SIZE (the climb scene's collision geom
  is a plane whose size lanes are visual-only), pinned against the box the
  asset actually compiles."""
  import mujoco

  from scr_phybot_mjab.assets.wall.wall_constants import (
    WALL_HALF_SIZE,
    get_wall_plane_collision_spec,
  )
  from scr_phybot_mjab.tasks.climb_transition80.mdp.observations import (
    _WALL_FACE_CORNERS_LOCAL,
  )

  hx, hy, hz = (float(v) for v in WALL_HALF_SIZE)
  assert _WALL_FACE_CORNERS_LOCAL == (
    (-hx, -hy, -hz),
    (-hx, hy, -hz),
    (-hx, -hy, hz),
    (-hx, hy, hz),
  )
  model = get_wall_plane_collision_spec().compile()
  box = model.geom("wall_visual_geom")
  assert int(box.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX)
  assert tuple(float(v) for v in box.size) == pytest.approx((hx, hy, hz))
  assert (hx, hy, hz) == (0.5, 0.6, 1.0)


def test_transition80_command_is_the_tracking_subclass() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert type(motion) is TransitionMotionCommandCfg
  assert motion.motion_file.endswith("climb_mount_prologue_a80.npy")
  assert motion.sampling_mode == "adaptive"
  assert motion.relative_frame == "world"
  assert motion.spawn_match_vertical_height is False
  assert motion.spawn_early_fraction == pytest.approx(0.45)  # v6e classroom share
  assert motion.spawn_bias_auto_rake_frames == 0  # v6e: explicit windows replace auto rake
  assert motion.spawn_bias_segments == ()
  assert not hasattr(motion, "angle_clip_files")  # no band machinery here
  assert motion.spawn_xy_noise == TRANSITION80_SPAWN_XY_NOISE
  assert motion.spawn_yaw_noise == TRANSITION80_SPAWN_YAW_NOISE
  for pad in motion.ee_body_names:
    assert pad in motion.body_names
  assert motion.body_names == DEFAULT_TRACKING_BODY_NAMES


def test_transition80_reward_table_is_beyondmimic_minimal() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  assert tuple(cfg.rewards) == tuple(_EXPECTED_REWARDS)
  for name, weight in _EXPECTED_REWARDS.items():
    assert cfg.rewards[name].weight == pytest.approx(weight), name
  positive = sum(w for w in _EXPECTED_REWARDS.values() if w > 0)
  assert positive == pytest.approx(9.5)
  wage = cfg.rewards["hold_margin_wage"].params
  assert wage["full_margin"] == pytest.approx(0.3)
  assert wage["affordance_command_name"] == "microspine_affordance"
  negatives = [n for n, w in _EXPECTED_REWARDS.items() if w < 0]
  assert negatives == ["anchor_rip_penalty", "action_rate_l2", "dof_pos_limits"]
  phi = cfg.rewards["engage_progress"].params
  assert phi["anchor_radius_m"] == pytest.approx(TRANSITION80_ANCHOR_RADIUS_STAGES[0]["value"])
  assert phi["edge_lead_time_s"] == pytest.approx(2.0)
  assert cfg.rewards["motion_pad_pos"].func.__name__ == "motion_pad_center_error_exp"
  assert cfg.rewards["motion_pad_pos"].params["std"] == pytest.approx(0.08)
  assert phi["reach_radius_m"] == pytest.approx(0.5)
  # v3e hover-trap shape: approach + press lanes, monotone plateaus.
  assert phi["approach_share"] == pytest.approx(0.19) and phi["approach_radius_m"] == pytest.approx(0.10)
  assert phi["press_share"] == pytest.approx(0.35) and phi["press_ref_n"] == pytest.approx(20.0)
  assert phi["load_share"] == pytest.approx(0.10)
  assert phi["reach_share"] == pytest.approx(0.25) and phi["touch_level"] == pytest.approx(0.45)
  assert phi["reach_share"] + phi["approach_share"] < phi["touch_level"]
  assert phi["touch_level"] + phi["load_share"] + phi["press_share"] <= 0.9
  # Nothing from the transition wage/bonus/tax ledger leaks in.
  for banned in (
    "motion_scheduled_press",
    "motion_predrag_wage",
    "first_weld",
    "first_hold",
    "first_charge",
    "microspine_support_loss",
    "motion_engage_mismatch",
    "termination_penalty",
    "standing_wall_fence",
  ):
    assert banned not in cfg.rewards


def test_asym_arm_tracking_knob_adds_the_arm_split() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg(asym_arm_tracking=True)
  assert "motion_body_pos_arms" in cfg.rewards
  assert "motion_body_ori_arms" in cfg.rewards
  assert cfg.rewards["motion_body_pos_arms"].weight == pytest.approx(0.5)
  body = cfg.rewards["motion_body_pos"].params["body_names"]
  assert "left_elbow_pitch" not in body and "left_hand_spine" in body


def test_transition80_termination_table() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  assert tuple(cfg.terminations) == _EXPECTED_TERMINATIONS
  assert cfg.terminations["time_out"].time_out is True
  for name in _EXPECTED_TERMINATIONS[1:]:
    assert cfg.terminations[name].time_out is False, name
  cone = cfg.terminations["engage_constraint"].params
  assert cone["mode"] == "cat"
  assert cone["tilt_limit_deg"] == pytest.approx(15.0)
  # v3f: the cone's CaT probability starts near zero and anneals 0.05 -> 0.5 -> 1.0.
  assert cone["p_force"] == pytest.approx(0.05)
  assert cone["p_tilt"] == pytest.approx(0.05)
  assert [st["value"] for st in cfg.curriculum["constraint_p_force"].params["stages"]] == [0.05, 0.5, 1.0]
  law = cfg.terminations["contact_loss"].params
  # v6 dare: 1 s flat (LadderMan discipline).
  assert law["grace_s"] == pytest.approx(1.0)
  assert law["anchor_radius_m"] == pytest.approx(0.15)
  assert cfg.terminations["anchor_pos"].params["persist_steps"] == 10
  assert cfg.terminations["body_pos"].params["threshold"] == pytest.approx(0.30)
  assert cfg.terminations["body_pos"].func.__qualname__.startswith("bad_pad_center_termination")
  assert cfg.terminations["wall_contact"].params["gate"] == "none"
  # v6 dare: the four shoulder geoms left the fence for ordinary contact.
  assert len(cfg.terminations["wall_contact"].params["geom_names"]) == 9
  assert not any("shoulder" in g for g in cfg.terminations["wall_contact"].params["geom_names"])
  arm_geoms = cfg.commands["microspine_affordance"].microspine_cfg.arm_wall_contact_geoms
  assert len(arm_geoms) == 6 and all("shoulder" in g or "elbow" in g for g in arm_geoms)
  # v5: the rip death is retired; the rip is a per-pad bill on a ramp.
  assert "anchor_rip" not in cfg.terminations
  bill = cfg.rewards["anchor_rip_penalty"].params
  assert bill["release_lead_time_s"] == 0.5
  assert bill["seed_settle_exempt_s"] == pytest.approx(0.12)
  assert bill["weight_per_pad"] == pytest.approx(0.5)
  ramp = cfg.curriculum["rip_bill"].params
  assert ramp["reward_name"] == "anchor_rip_penalty" and ramp["interpolate"] is True
  assert [st["value"] for st in ramp["stages"]] == [0.5, 1.5, 3.0]
  assert [st["iter"] for st in ramp["stages"]] == [0, 1000, 2500]
  # v5: gimbal travel ladder 15 -> 3 deg, written by startup + reset events.
  for name, mode in (("pad_gimbal_travel_init", "startup"), ("pad_gimbal_travel", "reset")):
    ev = cfg.events[name]
    assert ev.mode == mode
    assert [st["iter"] for st in ev.params["stages"]] == [0, 7900, 8100, 10000, 10200, 16000, 16200]
    assert [st["value"] for st in ev.params["stages"]] == [15.0, 15.0, 8.0, 8.0, 5.0, 5.0, 3.0]
  assert cfg.metrics["anchor_rips"].params["reward_name"] == "anchor_rip_penalty"
  # v5e: adaptive sampler tightened (uniform floor 0.5, per-bin cap 0.08).
  motion = cfg.commands["climb_motion"]
  assert motion.adaptive_uniform_ratio == pytest.approx(0.5)
  assert motion.adaptive_max_bin_prob == pytest.approx(0.08)
  # v6e: explicit classroom = ignition rake windows + the first 30 frames of
  # each closing hand step; quota 0.45; 35 s episode cap.
  assert motion.spawn_early_fraction == pytest.approx(0.45)
  assert motion.spawn_bias_auto_rake_frames == 0
  wins = motion.spawn_bias_frame_windows
  assert len(wins) >= 8
  handup = [w for w in wins if w[1] - w[0] == 30]
  assert len(handup) == 3  # v6g: hand_up_right + the left hand's two steps
  import numpy as np

  data = np.load(motion.motion_file, allow_pickle=True).item()
  names = list(data["segment_names"])
  labels = np.asarray(data["phase_segment"])
  for seg, (a0, b0) in zip(("hand_up_right", "hand_up_left_mid", "hand_up_left"), sorted(handup)):
    start = int(np.flatnonzero(labels == names.index(seg))[0])
    assert (a0, b0) == (start, start + 30), seg
  assert cfg.episode_length_s == pytest.approx(35.0)
  cone = cfg.terminations["engage_constraint"].params
  assert cone["persist_steps"] == 5
  assert cone["seed_settle_exempt_s"] == pytest.approx(0.12)
  hard = phybot_c2_climb_transition80_env_cfg(constraint_mode="hard")
  assert hard.terminations["engage_constraint"].params["mode"] == "hard"
  with pytest.raises(ValueError):
    phybot_c2_climb_transition80_env_cfg(constraint_mode="soft")


def test_transition80_curriculum_table() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  names = tuple(cfg.curriculum)
  assert names[:2] == ("wall_angle", "engage_antirake_ladder")
  assert "engage_shear_ladder" not in names  # the empirical gate has no shear threshold
  assert "spawn_bias_window" not in names
  assert "gravity_ladder" not in names
  assert "tilt_peel" not in names
  assert "engage_gate_anneal" not in names
  for name, term in cfg.curriculum.items():
    assert term.params.get("steps_per_iter") == 96, name
  assert cfg.curriculum["wall_angle"].params["angle_stages"] == [{"iter": 0, "angle_deg": 80.0}]
  assert cfg.curriculum["wall_angle"].params["angle_jitter_deg"] == 0.0
  rake = cfg.curriculum["engage_antirake_ladder"].params
  assert rake["param_name"] == "engage_max_antirake_vel"
  assert [s["value"] for s in rake["stages"]] == [0.05, 0.02]
  # v5f: the loosened regime lasts to iter 7500 on every ladder / anneal (+3000 vs v4).
  assert [s["iter"] for s in rake["stages"]] == [0, 50000]
  assert [s["iter"] for s in cfg.curriculum["constraint_p_force"].params["stages"]] == [0, 50000, 52000]
  assert [s["iter"] for s in cfg.curriculum["contact_law_grace"].params["stages"]] == [0]
  # Phi's anchor radius anneals in lockstep with the contact law.
  assert (
    cfg.curriculum["engage_progress_radius"].params["stages"] == cfg.curriculum["contact_law_radius"].params["stages"]
  )
  assert cfg.curriculum["constraint_tilt"].params["termination_name"] == ("engage_constraint")


def test_transition80_plant_flags() -> None:
  cfg = phybot_c2_climb_transition80_env_cfg()
  ms = cfg.commands["microspine_affordance"].microspine_cfg
  # The climb task's empirical unit plant: every active unit preloads 10 N
  # before a new engage, and the contact buffers fit the 12-unit pads.
  assert ms.limit_surface.model == "empirical_unit"
  assert not ms.spring_field_enabled
  assert ms.engage_unit_preload_n == 10.0
  assert cfg.sim.nconmax >= 256 and cfg.sim.njmax >= 512
  assert ms.limit_surface.seed_hold_judge_guard_steps == TRANSITION80_SEED_GUARD_STEPS
  # v3b: the clock freeze (25) and the judge guard (6, the v2 regime) are decoupled —
  # a guard spanning the freeze let the policy lean on unbreakable seeds (91 % rips at step 26).
  assert TRANSITION80_SEED_GUARD_STEPS == 6
  motion = cfg.commands["climb_motion"]
  assert motion.birth_freeze_steps == 25
  assert motion.birth_exclude_frames == ()  # v4b: mask off (user decision 2026-08-22); re-arm via the tyro flag
  assert "scheduled_hold_margin" in cfg.metrics and "rh_hold_margin" in cfg.metrics
  assert "random_release" not in cfg.events
  assert "mount_handoff_ready" in cfg.metrics
  for key in ("tilt", "grind", "slam", "force", "unscheduled"):
    assert f"constraint_{key}_violation" in cfg.metrics


def test_play_overrides() -> None:
  play = phybot_c2_climb_transition80_env_cfg(play=True)
  motion = play.commands["climb_motion"]
  assert motion.sampling_mode == "start"
  assert motion.spawn_xy_noise == (0.0, 0.0)
  assert motion.spawn_yaw_noise == 0.0
  assert motion.joint_pos_noise == 0.0
  assert motion.birth_exclude_frames == ()  # play starts at frame 0; no mask
  assert motion.spawn_early_fraction == 0.0 and motion.spawn_bias_auto_rake_frames == 0
  assert play.terminations["engage_constraint"].params["mode"] == "off"
  assert tuple(play.curriculum) == ("wall_angle", "engage_antirake_ladder")
  assert play.curriculum["engage_antirake_ladder"].params["stages"] == [{"iter": 0, "value": 0.02}]
  assert play.curriculum["wall_angle"].params["angle_stages"] == [{"iter": 0, "angle_deg": 80.0}]
  assert play.scene.num_envs == 1


def test_sibling_packages_are_untouched() -> None:
  climb = phybot_c2_climb_wall_env_cfg()
  assert type(climb.commands["climb_motion"]) is ClimbMotionCommandCfg
  assert "engage_progress" not in climb.rewards
  assert "engage_constraint" not in climb.terminations
  assert "anchor_rip" not in climb.terminations
  from mjlab.tasks.registry import list_tasks

  assert "Mjlab-Climb-Mount80-Phybot-C2" not in list_tasks()


def test_registry_roundtrip() -> None:
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Climb-Transition80-Phybot-C2")
  assert type(cfg.commands["climb_motion"]) is TransitionMotionCommandCfg
  assert tuple(cfg.rewards) == tuple(_EXPECTED_REWARDS)


def test_adaptive_bin_probabilities_contract() -> None:
  n = 18
  flat = adaptive_bin_probabilities(torch.zeros(n), 0.1, 0.35)
  assert torch.allclose(flat, torch.full((n,), 1.0 / n))
  failed = torch.zeros(n)
  failed[4] = 100.0
  capped = adaptive_bin_probabilities(failed, 0.1, 0.35)
  assert capped.sum().item() == pytest.approx(1.0)
  assert capped[4].item() <= 0.35 + 1e-5
  assert capped.argmax().item() == 4
  uncapped = adaptive_bin_probabilities(failed, 0.1, None)
  assert uncapped[4].item() > 0.9


@pytest.mark.skipif(not _A80.exists(), reason="a80 mount clip not generated")
def test_fk_tables_match_the_clip_ee_positions() -> None:
  """The FK path reproduces the generator's own pad body positions."""
  import mujoco

  import scr_phybot_mjab.tasks  # noqa: F401  # registry import order
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec

  data = np.load(_A80, allow_pickle=True).item()
  model = get_spec().compile()
  assert int(model.joint("floating_base_joint").qposadr[0]) == 0
  joint_qadr = np.array([int(model.joint(n).qposadr[0]) for n in data["joints_list"]])
  body_ids = np.array([model.body(n).id for n in data["ee_list"]])
  root_pos = np.asarray(data["root_position"], np.float64)
  root_quat = np.asarray(data["root_quaternion"], np.float64)[:, [3, 0, 1, 2]]
  joint_pos = np.asarray(data["joint_positions"], np.float64)
  passive_qadr = np.array([int(model.joint(n).qposadr[0]) for n in data["passive_joint_names"]])
  pos, quat = build_fk_tables(
    model,
    np.array(model.qpos0, dtype=np.float64),
    np.arange(7),
    joint_qadr,
    body_ids,
    root_pos,
    root_quat,
    joint_pos,
    passive_qadr=passive_qadr,
    passive_pos=np.asarray(data["passive_joint_positions"], np.float64),
  )
  ref = np.asarray(data["ee_positions"], np.float64)
  assert np.abs(pos - ref).max() < 2.0e-3
  assert np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-6)
  ref_quat = np.asarray(data["ee_quaternions"], np.float64)[..., [3, 0, 1, 2]]
  assert np.allclose(np.abs(np.sum(quat * ref_quat, axis=-1)), 1.0, atol=1e-6)
  # C is offset from the pad body B; the saved contact targets must also
  # agree with FK, including the passive gimbal pose at the handoff.
  centers = np.empty_like(pos)
  for i, name in enumerate(data["ee_list"]):
    local_c = model.site(f"{name}_contact_center").pos
    for t in range(len(pos)):
      offset = np.empty(3)
      mujoco.mju_rotVecQuat(offset, local_c, quat[t, i])
      centers[t, i] = pos[t, i] + offset
  assert np.abs(centers - data["pad_positions"]).max() < 2.0e-3
  lin, ang = finite_difference_velocities(pos, quat, float(data["fps"]))
  assert np.isfinite(lin).all() and np.isfinite(ang).all()
  assert np.all(lin[-1] == 0.0) and np.all(ang[-1] == 0.0)


def test_cfg_survives_cloudpickle_for_multi_gpu_launch() -> None:
  """torchrunx ships the env cfg to worker ranks with cloudpickle, which
  serialises closure-based terms BY VALUE together with the globals they
  reference; a module-level torch.jit ScriptFunction in that set kills the
  launch (2026-08-21 first mount80 launch: 'ScriptFunction cannot be
  pickled'). Guard every closure factory in the table."""
  cloudpickle = pytest.importorskip("cloudpickle")
  for cfg in (
    phybot_c2_climb_transition80_env_cfg(),
    phybot_c2_climb_transition80_env_cfg(play=True),
  ):
    blob = cloudpickle.dumps(cfg)
    back = cloudpickle.loads(blob)
    assert tuple(back.terminations) == tuple(cfg.terminations)
    assert tuple(back.rewards) == tuple(cfg.rewards)
    assert tuple(back.observations["actor"].terms) == TRANSITION80_ACTOR_TERMS
    assert tuple(back.observations["critic"].terms) == (TRANSITION80_ACTOR_TERMS + TRANSITION80_CRITIC_TAIL)


def _gains(cfg) -> dict[str, tuple[float, float]]:
  return {
    act.target_names_expr[0]: (act.stiffness, act.damping) for act in cfg.scene.entities["robot"].articulation.actuators
  }


def test_pd_table_is_the_real_robot_firmware_table() -> None:
  """Explicit PD (user call 2026-08-21): the whole table equals the motor
  firmware (MotorList phybot_mini_{1,2}.yaml) = walk-AMP/squat — legs 100/10,
  ankles 50/5, waist 20/10, arms 50/5 — with climb's 0.98 soft limit."""
  gains = _gains(phybot_c2_climb_transition80_env_cfg())
  for arm in (".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow_pitch"):
    assert gains[arm] == (50.0, 5.0), arm
  assert gains["waist_yaw"] == (20.0, 10.0)
  for leg in (".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw", ".*_knee"):
    assert gains[leg] == (100.0, 10.0), leg
  for ankle in (".*_ankle_pitch", ".*_ankle_roll"):
    assert gains[ankle] == (50.0, 5.0), ankle
  cfg = phybot_c2_climb_transition80_env_cfg()
  assert cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor == 0.98
  # The shared actuator singletons were not mutated (asset defaults intact)
  # and the climb table is untouched (its own test pins pd100).
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
    PHYBOT_C2_ACTUATOR_HIP_PITCH,
    PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
    PHYBOT_C2_ACTUATOR_WAIST_YAW,
  )

  assert (PHYBOT_C2_ACTUATOR_SHOULDER_PITCH.stiffness, PHYBOT_C2_ACTUATOR_SHOULDER_PITCH.damping) == (50.0, 5.0)
  assert (PHYBOT_C2_ACTUATOR_WAIST_YAW.stiffness, PHYBOT_C2_ACTUATOR_WAIST_YAW.damping) == (20.0, 10.0)
  assert (PHYBOT_C2_ACTUATOR_HIP_PITCH.stiffness, PHYBOT_C2_ACTUATOR_HIP_PITCH.damping) == (100.0, 5.0)
  climb = _gains(phybot_c2_climb_wall_env_cfg())
  assert climb[".*_shoulder_pitch"] == (100.0, 10.0)
  assert climb["waist_yaw"] == (100.0, 10.0)
  custom = _gains(phybot_c2_climb_transition80_env_cfg(arm_pd=(75.0, 7.5), waist_pd=(50.0, 10.0)))
  assert custom[".*_elbow_pitch"] == (75.0, 7.5) and custom["waist_yaw"] == (50.0, 10.0)


def test_pose_kernels_carry_the_climb_mismatch_gate() -> None:
  """v1e: climb's schedule-adherence discount on the five pose kernels only."""
  cfg = phybot_c2_climb_transition80_env_cfg()
  for name in ("motion_anchor_pos", "motion_anchor_ori", "motion_body_pos", "motion_body_ori"):
    params = cfg.rewards[name].params
    assert params["engage_gate_floor"] == pytest.approx(0.10), name
    assert params["engage_gate_grace_s"] == pytest.approx(0.5), name
    assert params["engage_gate_grace_fraction"] is None, name
  # The pad-face term is the path to compliance: never discounted.
  for name in ("motion_pad_pos", "motion_body_lin_vel", "motion_body_ang_vel", "engage_progress"):
    assert "engage_gate_floor" not in cfg.rewards[name].params, name
  ungated = phybot_c2_climb_transition80_env_cfg(engage_gate_floor=None)
  assert ungated.rewards["motion_body_pos"].params["engage_gate_floor"] is None
