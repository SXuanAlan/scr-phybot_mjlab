"""Unit tests for the transition80 task's reward/termination functions on stubs.

No env, no GPU: every term reads a handful of tensors off the command /
affordance objects, so SimpleNamespace stand-ins exercise the exact
branch logic (the Phi ladder, the CaT cone, the rip law, persistence).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scr_phybot_mjab.tasks.climb_transition80.mdp.rewards import (
  engage_progress,
  motion_anchor_position_error_exp,
)
from scr_phybot_mjab.tasks.climb_transition80.mdp.terminations import (
  bad_anchor_pos_termination,
  bad_pad_center_termination,
  engage_constraint_termination,
  illegal_wall_contact_termination,
  scheduled_anchor_rip,
)

E, H = 2, 4
CYCLE_S = 10.0


def _sec(seconds: float) -> float:
  return seconds / CYCLE_S


def _stub(
  *,
  engaged=None,
  touch=None,
  press=None,
  shear=None,
  face_cos=None,
  ref_engaged=None,
  countdowns=None,
  dist=None,
  charge=None,
  released=None,
  pad_vel=None,
  episode_len=10,
  tau=4.0,
):
  def z(dtype=torch.float32):
    return torch.zeros(E, H, dtype=dtype)

  engaged = z(torch.bool) if engaged is None else engaged
  touch = z(torch.bool) if touch is None else touch
  press = z() if press is None else press
  shear = z() if shear is None else shear
  face_cos = torch.ones(E, H) if face_cos is None else face_cos
  ref_engaged = z(torch.bool) if ref_engaged is None else ref_engaged
  # countdowns are normalised by the clip length (cycle_time_s = 10 s here,
  # so seconds = value * 10); the default "never" sentinel 1.0 on both
  # lanes reads as 10 s away — outside every lead window.
  countdowns = torch.ones(E, H, 2) if countdowns is None else countdowns
  dist = torch.full((E, H), 1.0) if dist is None else dist
  charge = torch.ones(E, H) if charge is None else charge
  released = z(torch.bool) if released is None else released
  pad_vel = torch.zeros(E, H, 3) if pad_vel is None else pad_vel

  anchors = torch.zeros(E, H, 3)
  centers = anchors.clone()
  centers[..., 2] = dist  # pad center sits `dist` above its anchor
  executor = SimpleNamespace(
    pad_center_world=lambda: centers,
    _ee_body_ids=torch.arange(H),
  )
  aff = SimpleNamespace(
    engaged=engaged,
    contact_count=touch.float(),
    pad_press_force=press,
    pad_shear_down=shear,
    face_cos=face_cos,
    dr_release_pending=z(torch.bool),
    forced_release_this_step=released,
    released_was_seeded=z(torch.bool),
    seat_charge_fraction=charge,
    _mcfg=SimpleNamespace(engage_shear_threshold_n=tau),
    _get_executor=lambda: executor,
  )
  wall_quat = torch.zeros(E, 4)
  wall_quat[:, 0] = 1.0
  robot = SimpleNamespace(
    data=SimpleNamespace(body_link_lin_vel_w=pad_vel)
  )
  motion = SimpleNamespace(
    ref_engaged=ref_engaged,
    ref_edge_countdowns=countdowns,
    ref_anchor_pos_w=anchors,
    cycle_time_s=CYCLE_S,
    wall_quat=wall_quat,
    robot=robot,
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_term=lambda n: {"climb_motion": motion, "microspine_affordance": aff}[n]
    ),
    episode_length_buf=torch.full((E,), episode_len, dtype=torch.long),
    step_dt=0.02,
    num_envs=E,
    device="cpu",
  )
  return env, motion, aff


# --- Phi -------------------------------------------------------------------


def test_phi_off_schedule_pays_for_being_disengaged():
  env, motion, aff = _stub()
  assert torch.allclose(engage_progress(env), torch.ones(E))
  aff.engaged[0, 1] = True  # welded while the chart says OFF
  phi = engage_progress(env)
  assert phi[0].item() == pytest.approx(0.75)  # 3 of 4 EEs match
  assert phi[1].item() == pytest.approx(1.0)


def test_phi_ladder_is_monotone_through_the_chain():
  env, motion, aff = _stub(dist=torch.full((E, H), 0.30))
  # Scheduled ON for LH on env0 only; other EEs stay OFF (Phi = 1 each).
  motion.ref_engaged[0, 0] = True

  def lh():
    return engage_progress(env)[0].item() * H - 3.0  # isolate LH's share

  assert lh() == pytest.approx(0.0)  # 30 cm away: reach floor
  aff._get_executor().pad_center_world()[0, 0, 2] = 0.10
  assert lh() == pytest.approx(0.3 * (1 - 0.10 / 0.25))  # reach ramp
  aff._get_executor().pad_center_world()[0, 0, 2] = 0.0
  assert lh() == pytest.approx(0.30)  # at the anchor, not touching
  aff.contact_count[0, 0] = 1.0
  assert lh() == pytest.approx(0.50)  # touching, no shear
  aff.pad_shear_down[0, 0] = 2.0  # half the 4 N rung
  assert lh() == pytest.approx(0.65)
  aff.pad_shear_down[0, 0] = 40.0  # saturates at the rung
  assert lh() == pytest.approx(0.80)
  aff.engaged[0, 0] = True
  aff.seat_charge_fraction[0, 0] = 0.0
  assert lh() == pytest.approx(0.90)  # welded, uncharged
  aff.seat_charge_fraction[0, 0] = 1.0
  assert lh() == pytest.approx(1.00)  # mature
  # Welded far from the anchor: back to the reach stage.
  aff._get_executor().pad_center_world()[0, 0, 2] = 0.5
  assert lh() == pytest.approx(0.0)


def test_phi_window_opens_two_seconds_before_the_edge():
  env, motion, aff = _stub(dist=torch.zeros(E, H))
  aff.contact_count[:] = 1.0
  # OFF and no edge in sight: touching is irrelevant to Phi, = 1[not engaged].
  assert torch.allclose(engage_progress(env), torch.ones(E))
  # Regrip edge 1.5 s away (lead 2 s): the window is open -> touch stage.
  motion.ref_edge_countdowns[0, 0, 1] = _sec(1.5)
  phi0 = engage_progress(env)[0].item() * H - 3.0
  assert phi0 == pytest.approx(0.50)
  motion.ref_edge_countdowns[0, 0, 1] = _sec(2.5)  # too early: OFF semantics
  assert engage_progress(env)[0].item() == pytest.approx(1.0)


# --- the engage cone -----------------------------------------------------------


def _cone(**kw):
  """The cone term plus stubs; single-step probes pass persist_steps=1."""
  term = engage_constraint_termination()
  env, motion, aff = _stub(**kw)
  # An engage edge is due within the 2 s lead by default, so C4
  # (unscheduled touch) stays silent unless a test re-arms it.
  motion.ref_edge_countdowns[..., 1] = _sec(1.0)
  return term, env, motion, aff


def test_cone_off_never_fires_but_still_reports():
  term, env, _, aff = _cone(touch=torch.ones(E, H, dtype=torch.bool),
                            face_cos=torch.zeros(E, H))  # 90 deg tilt
  aff.dr_release_pending = torch.zeros(E, H, dtype=torch.bool)
  out = term(env, mode="off", persist_steps=1)
  assert not out.any()
  assert term.last_violations["tilt"].max().item() == pytest.approx(1.0)
  assert term.last_delta.max().item() == pytest.approx(0.5)  # p_tilt 0.5


def test_cone_hard_mode_fires_on_any_violation():
  term, env, _, _ = _cone(touch=torch.ones(E, H, dtype=torch.bool),
                          face_cos=torch.full((E, H), 0.95))  # ~18 deg
  assert term(env, mode="hard", persist_steps=1, tilt_limit_deg=15.0).all()
  assert not term(env, mode="hard", persist_steps=1, tilt_limit_deg=20.0).any()


def test_cone_cat_mode_is_bernoulli_in_delta():
  torch.manual_seed(0)
  term, env, _, aff = _cone(touch=torch.ones(E, H, dtype=torch.bool),
                            press=torch.full((E, H), 160.0))  # 2x the 80 N cap
  out = term(env, mode="cat", persist_steps=1, p_force=1.0)
  assert out.all()  # v_force saturates at 1 -> delta 1 -> certain death
  aff.pad_press_force[:] = 0.0
  out = term(env, mode="cat", persist_steps=1)
  assert not out.any()  # delta 0 -> never


def test_cone_ignores_engaged_pads_and_counts_unscheduled_touch():
  term, env, motion, aff = _cone(touch=torch.ones(E, H, dtype=torch.bool),
                                 face_cos=torch.zeros(E, H))
  aff.engaged[:] = True
  term(env, mode="off", persist_steps=1)
  assert term.last_violations["tilt"].sum().item() == 0.0
  assert term.last_violations["unscheduled"].sum().item() == 0.0
  aff.engaged[:] = False
  motion.ref_edge_countdowns[..., 1] = 1.0  # no edge in sight
  term(env, mode="off", persist_steps=1)
  # Touching while the chart says OFF with no edge due = unscheduled.
  assert term.last_violations["unscheduled"].min().item() == 1.0
  motion.ref_edge_countdowns[..., 1] = _sec(1.0)  # edge due within the 2 s lead
  term(env, mode="off", persist_steps=1)
  assert term.last_violations["unscheduled"].sum().item() == 0.0


def test_cone_slam_reads_the_contact_rising_edge():
  vel = torch.zeros(E, H, 3)
  vel[0, 0, 0] = 1.0  # +x = INTO the wall (outward normal is -x)
  term, env, _, aff = _cone(pad_vel=vel)
  term(env, mode="off", persist_steps=1)  # not touching yet: no rising edge
  assert term.last_violations["slam"].sum().item() == 0.0
  aff.contact_count[0, 0] = 1.0  # touch begins this step
  term(env, mode="off", persist_steps=1, slam_limit_mps=0.6)
  assert term.last_violations["slam"][0, 0].item() == pytest.approx(
    min((1.0 - 0.6) / 0.6, 1.0)
  )
  term(env, mode="off", persist_steps=1, slam_limit_mps=0.6)  # still touching: no new edge
  assert term.last_violations["slam"][0, 0].item() == 0.0


# --- rip law -------------------------------------------------------------------


def test_rip_fires_only_for_scheduled_anchors_away_from_their_release_edge():
  env, motion, aff = _stub()
  aff.forced_release_this_step[0, 0] = True
  assert not scheduled_anchor_rip(env).any()  # chart says OFF: not a rip
  motion.ref_engaged[0, 0] = True
  assert scheduled_anchor_rip(env)[0] and not scheduled_anchor_rip(env)[1]
  motion.ref_edge_countdowns[0, 0, 0] = _sec(0.3)  # release due in 0.3 s (lead 0.5)
  assert not scheduled_anchor_rip(env).any()
  motion.ref_edge_countdowns[0, 0, 0] = 1.0
  aff.dr_release_pending[0, 0] = True
  assert not scheduled_anchor_rip(env).any()
  aff.dr_release_pending[0, 0] = False
  assert scheduled_anchor_rip(env)[0]
  # A seeded anchor lost inside the birth-settle window is exempt; the same
  # loss later in the episode is a rip.
  aff.released_was_seeded[0, 0] = True
  env.episode_length_buf[:] = 20  # 0.4 s < 0.5 s settle window
  assert not scheduled_anchor_rip(env).any()
  env.episode_length_buf[:] = 40
  assert scheduled_anchor_rip(env)[0]
  assert not scheduled_anchor_rip(env, seed_settle_exempt_s=1.0)[0]


# --- persistence ---------------------------------------------------------------


def test_anchor_pos_termination_needs_persistence():
  term = bad_anchor_pos_termination()
  ref = torch.zeros(E, 3)
  rob = torch.zeros(E, 3)
  rob[0, 0] = 1.0  # env0 one metre off
  cmd = SimpleNamespace(anchor_pos_w=ref, robot_anchor_pos_w=rob)
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda n: cmd),
    episode_length_buf=torch.full((E,), 10, dtype=torch.long),
  )
  fired = [bool(term(env, threshold=0.35, persist_steps=3)[0]) for _ in range(4)]
  assert fired == [False, False, True, True]
  assert not term(env, threshold=0.35, persist_steps=3)[1]
  env.episode_length_buf[:] = 1  # fresh episode resets the counter
  assert not term(env, threshold=0.35, persist_steps=3)[0]


# --- wall contact --------------------------------------------------------------


def test_illegal_wall_contact_uses_geom_surfaces_and_has_no_gate():
  term = illegal_wall_contact_termination()
  wall_quat = torch.zeros(E, 4)
  wall_quat[:, 0] = 1.0
  pivot = torch.tensor([[1.5, 0.0, 0.0]] * E)
  motion = SimpleNamespace(
    wall_quat=wall_quat, wall_pivot_w=pivot, time_steps=torch.zeros(E),
    _first_engage_frame=100,
  )
  geom_xpos = torch.zeros(E, 1, 3)
  geom_xpos[0, 0, 0] = 1.5 - 0.05  # center 5 cm outside a 7 cm sphere
  geom_xpos[1, 0, 0] = 1.5 - 0.20  # 20 cm outside: clear
  mj = SimpleNamespace(
    geom=lambda name: SimpleNamespace(id=0),
    geom_size=np.array([[0.07, 0.0, 0.0]]),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda n: motion),
    sim=SimpleNamespace(mj_model=mj, data=SimpleNamespace(geom_xpos=geom_xpos)),
    device="cpu",
  )
  out = term(env, geom_names=("head",), gate="none")
  assert bool(out[0]) and not bool(out[1])
  # Standing gate: past first engage the term is silent.
  motion.time_steps[:] = 500
  assert not term(env, geom_names=("head",), gate="standing").any()


def test_cone_persistence_ignores_spikes_and_bills_sustained_violations():
  term = engage_constraint_termination()
  env, motion, aff = _stub(touch=torch.ones(E, H, dtype=torch.bool),
                           face_cos=torch.zeros(E, H))  # 90 deg tilt, sustained
  motion.ref_edge_countdowns[..., 1] = _sec(1.0)  # scheduled soon: C4 silent
  fired = []
  for _ in range(6):
    term(env, mode="hard", persist_steps=5)
    fired.append(bool(term.last_violations["tilt"].sum() > 0))
  assert fired == [False] * 4 + [True, True]
  # One clean step resets the counter.
  aff.face_cos[:] = 1.0
  term(env, mode="hard", persist_steps=5)
  aff.face_cos[:] = 0.0
  term(env, mode="hard", persist_steps=5)
  assert term.last_violations["tilt"].sum().item() == 0.0
  # Episode start resets too.
  for _ in range(4):
    term(env, mode="hard", persist_steps=5)
  env.episode_length_buf[:] = 1
  term(env, mode="hard", persist_steps=5)
  assert term.last_violations["tilt"].sum().item() == 0.0


def test_cone_seed_settle_exemption_mirrors_the_rip_law():
  """A seeded weld torn off inside the settle window leaves a free pad on
  the wall; the cone must not bill it until the window closes."""
  term = engage_constraint_termination()
  env, motion, aff = _stub(touch=torch.ones(E, H, dtype=torch.bool),
                           press=torch.full((E, H), 160.0))  # 2x the force cap
  motion.ref_edge_countdowns[..., 1] = _sec(1.0)
  aff.released_was_seeded[0, 0] = True  # LH seed lost this step (env0)
  env.episode_length_buf[:] = 10  # inside the 0.5 s window
  term(env, mode="off", persist_steps=1)
  assert term.last_violations["force"][0, 0].item() == 0.0  # exempt
  assert term.last_violations["force"][1, 0].item() == 1.0  # env1: no seed lost
  aff.released_was_seeded[:] = False  # the flag is one-step; the memory persists
  term(env, mode="off", persist_steps=1)
  assert term.last_violations["force"][0, 0].item() == 0.0
  env.episode_length_buf[:] = 30  # window closed
  term(env, mode="off", persist_steps=1)
  assert term.last_violations["force"][0, 0].item() == 1.0


def test_pose_kernel_is_discounted_by_schedule_mismatch():
  """Matched latch state -> full pay; a pad welded/unwelded against the chart,
  far from any scheduled edge -> the floor; None -> ungated."""
  aff = SimpleNamespace(dr_release_pending=torch.zeros(E, H, dtype=torch.bool))
  cmd = SimpleNamespace(
    anchor_pos_w=torch.zeros(E, 3),
    robot_anchor_pos_w=torch.zeros(E, 3),
    ref_engaged=torch.zeros(E, H, dtype=torch.bool),
    robot_engaged=torch.zeros(E, H, dtype=torch.bool),
    ref_edge_distance=torch.ones(E, H),  # "never" sentinel: far from edges
    ref_run_time=torch.ones(E, H),
    cycle_time_s=CYCLE_S,
    _affordance=lambda: aff,
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: cmd))
  full = motion_anchor_position_error_exp(env, engage_gate_floor=0.25, engage_gate_grace_fraction=0.3)
  assert torch.allclose(full, torch.ones(E))
  cmd.ref_engaged[0, :] = True  # env0: all four scheduled ON, none welded
  gated = motion_anchor_position_error_exp(env, engage_gate_floor=0.25, engage_gate_grace_fraction=0.3)
  assert gated[0].item() == pytest.approx(0.25) and gated[1].item() == pytest.approx(1.0)
  cmd.ref_engaged[0, :2] = False  # two of four mismatched -> halfway
  half = motion_anchor_position_error_exp(env, engage_gate_floor=0.25, engage_gate_grace_fraction=0.3)
  assert half[0].item() == pytest.approx(0.25 + 0.75 * 0.5)
  assert motion_anchor_position_error_exp(env, engage_gate_floor=None)[0].item() == pytest.approx(1.0)


def test_adaptive_failures_are_charged_to_the_birth_bin():
  """v1f: on a clamped one-shot clip the death frame piles every late death
  into the last bin; failures are charged to the frame the episode started
  from instead (cfg switch keeps BeyondMimic's death attribution available)."""
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import TransitionMotionCommand

  cmd = object.__new__(TransitionMotionCommand)
  cmd.cfg = SimpleNamespace(adaptive_attribution="birth")
  cmd.clip = SimpleNamespace(cycle_frames=840)
  cmd._bin_count = 17
  cmd._current_bin_failed = torch.zeros(17)
  cmd.time_steps = torch.tensor([839, 839, 100])       # all clamped/late at death
  cmd._birth_frames = torch.tensor([230, 560, 100])    # born at LH / RH windows
  cmd._env = SimpleNamespace(
    termination_manager=SimpleNamespace(terminated=torch.tensor([True, True, False]))
  )
  cmd._record_failures(torch.arange(3))
  hist = cmd._current_bin_failed
  assert hist[230 * 17 // 840].item() == 1.0 and hist[560 * 17 // 840].item() == 1.0
  assert hist[16].item() == 0.0  # nothing charged to the clamp bin
  cmd.cfg.adaptive_attribution = "death"
  cmd._current_bin_failed.zero_()
  cmd._record_failures(torch.arange(3))
  assert cmd._current_bin_failed[16].item() == 2.0


def test_pad_center_termination_judges_the_face_not_the_origin():
  term = bad_pad_center_termination()
  ref = torch.zeros(E, H, 3)
  rob = torch.zeros(E, H, 3)
  rob[0, 0, 2] = 0.46  # env0 LH face 46 cm off while (hypothetically) the origin is fine
  cmd = SimpleNamespace(ref_pad_center_w=ref, robot_pad_center_w=rob)
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda n: cmd),
    episode_length_buf=torch.full((E,), 10, dtype=torch.long),
  )
  fired = [bool(term(env, threshold=0.30, persist_steps=3)[0]) for _ in range(3)]
  assert fired == [False, False, True]
  assert not term(env, threshold=0.30, persist_steps=3)[1]


# -- v2 observation lanes ------------------------------------------------------

import math  # noqa: E402

from scr_phybot_mjab.tasks.climb_transition80.mdp.observations import (  # noqa: E402
  _WALL_FACE_CORNERS_LOCAL,
  microspine_pad_center_world,
  motion_anchor_ori_b,
  motion_anchor_pos_b,
  motion_anchor_relative,
  motion_pad_center_relative,
  motion_ref_joint_pos_rel,
  robot_body_ori_b,
  robot_body_pos_b,
  wall_face_corners,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import wall_pose_for_angle  # noqa: E402


def _cmd_env(cmd):
  return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: cmd))


def _yaw(deg: float) -> torch.Tensor:
  half = math.radians(deg) / 2.0
  return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


def test_motion_command_is_ten_dims_and_picks_the_next_edge():
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import TransitionMotionCommand

  cmd = object.__new__(TransitionMotionCommand)
  edges = torch.rand(5, 4, 2)
  engaged = torch.zeros(5, 4, dtype=torch.bool)
  engaged[3, 0] = True  # frame 3: LH scheduled engaged, the rest free
  cmd.clip = SimpleNamespace(cycle_frames=5, ee_engaged=engaged, ee_edge_times=edges)
  cmd.time_steps = torch.tensor([0, 3])
  out = cmd.command
  assert out.shape == (2, 10)
  assert out[0, 0].item() == pytest.approx(1.0) and out[0, 1].item() == pytest.approx(0.0)
  assert out[1, 2].item() == 1.0 and out[1, 3:6].sum().item() == 0.0
  assert out[1, 6].item() == pytest.approx(edges[3, 0, 0].item())  # engaged -> t_release
  assert out[1, 7].item() == pytest.approx(edges[3, 1, 1].item())  # free -> t_regrip
  assert out[0, 6].item() == pytest.approx(edges[0, 0, 1].item())


def test_motion_ref_joint_pos_rel_uses_the_joint_pos_rel_basis():
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import TransitionMotionCommand

  cmd = object.__new__(TransitionMotionCommand)
  cmd.clip = SimpleNamespace(joint_pos=torch.arange(3 * 21, dtype=torch.float32).view(3, 21))
  cmd.time_steps = torch.tensor([2, 0])
  cmd._joint_ids = torch.arange(21) + 1  # the robot carries a leading extra joint
  cmd.robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.ones(2, 22)))
  rel = motion_ref_joint_pos_rel(_cmd_env(cmd))
  assert rel.shape == (2, 21)
  assert torch.allclose(rel, cmd.clip.joint_pos[[2, 0]] - 1.0)


def test_motion_anchor_pos_b_and_ori_b_are_root_relative():
  cmd = SimpleNamespace(
    robot_anchor_pos_w=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    robot_anchor_quat_w=torch.stack([_yaw(90.0), _yaw(0.0)]),
    anchor_pos_w=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    anchor_quat_w=torch.stack([_yaw(90.0), _yaw(0.0)]),
  )
  env = _cmd_env(cmd)
  pos = motion_anchor_pos_b(env)
  ori = motion_anchor_ori_b(env)
  assert pos.shape == (2, 3) and ori.shape == (2, 6)
  # world +y seen from a root yawed 90 deg is the root's +x
  assert torch.allclose(pos[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
  assert torch.allclose(pos[1], torch.zeros(3), atol=1e-6)
  identity6 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  assert torch.allclose(ori[0], identity6, atol=1e-6)
  assert torch.allclose(ori[1], identity6, atol=1e-6)
  cmd.anchor_quat_w = torch.stack([_yaw(180.0), _yaw(90.0)])  # +90 deg relative yaw on both
  ori = motion_anchor_ori_b(env)
  assert torch.allclose(ori[1], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_pad_center_lanes_rotate_into_the_base_frame():
  delta = torch.tensor([0.0, 1.0, 0.0]).expand(E, H, 3)
  cmd = SimpleNamespace(
    ref_pad_center_w=delta.clone(),
    robot_pad_center_w=torch.zeros(E, H, 3),
    ref_anchor_pos_w=delta.clone(),
    robot_anchor_quat_w=torch.stack([_yaw(90.0), _yaw(0.0)]),
  )
  env = _cmd_env(cmd)
  for fn in (motion_pad_center_relative, motion_anchor_relative):
    out = fn(env)
    assert out.shape == (E, 12)
    assert torch.allclose(out[0].view(H, 3), torch.tensor([1.0, 0.0, 0.0]).expand(H, 3), atol=1e-6)
    assert torch.allclose(out[1].view(H, 3), torch.tensor([0.0, 1.0, 0.0]).expand(H, 3), atol=1e-6)


def test_microspine_pad_center_world_is_env_local():
  centers = torch.arange(E * H * 3, dtype=torch.float32).view(E, H, 3)
  origins = torch.tensor([[10.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
  aff = SimpleNamespace(_get_executor=lambda: SimpleNamespace(pad_center_world=lambda: centers))
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda n: aff),
    scene=SimpleNamespace(env_origins=origins),
  )
  out = microspine_pad_center_world(env)
  assert out.shape == (E, 12)
  assert torch.allclose(out.view(E, H, 3), centers - origins.unsqueeze(1))


def test_wall_face_corners_closed_form():
  pos, quat = wall_pose_for_angle(torch.tensor([80.0, 90.0]))
  origins = torch.tensor([[10.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
  wall = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=pos + origins, root_link_quat_w=quat)
  )

  class _Scene:
    env_origins = origins

    def __getitem__(self, name):
      assert name == "wall"
      return wall

  out = wall_face_corners(SimpleNamespace(scene=_Scene())).view(2, 4, 3)
  c, s = math.cos(math.radians(80.0)), math.sin(math.radians(80.0))
  exp80 = torch.tensor(
    [[1.5, -0.6, 0.0], [1.5, 0.6, 0.0], [1.5 + 2 * c, -0.6, 2 * s], [1.5 + 2 * c, 0.6, 2 * s]]
  )
  exp90 = torch.tensor([[1.5, -0.6, 0.0], [1.5, 0.6, 0.0], [1.5, -0.6, 2.0], [1.5, 0.6, 2.0]])
  assert torch.allclose(out[0], exp80, atol=1e-4)
  assert torch.allclose(out[1], exp90, atol=1e-5)
  assert out[0, 2, 0].item() == pytest.approx(1.8473, abs=1e-3)
  assert out[0, 2, 2].item() == pytest.approx(1.9696, abs=1e-3)
  assert _WALL_FACE_CORNERS_LOCAL == (
    (-0.5, -0.6, -1.0),
    (-0.5, 0.6, -1.0),
    (-0.5, -0.6, 1.0),
    (-0.5, 0.6, 1.0),
  )


def test_robot_body_lanes_have_the_tracking_widths():
  num_bodies = 17
  identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
  body_pos = torch.rand(E, num_bodies, 3)
  cmd = SimpleNamespace(
    tracking_body_names=tuple(f"b{i}" for i in range(num_bodies)),
    robot_anchor_pos_w=torch.zeros(E, 3),
    robot_anchor_quat_w=identity.expand(E, 4).clone(),
    robot_body_pos_w=body_pos,
    robot_body_quat_w=identity.expand(E, num_bodies, 4).clone(),
  )
  env = _cmd_env(cmd)
  pos_b = robot_body_pos_b(env)
  ori_b = robot_body_ori_b(env)
  assert pos_b.shape == (E, 51) and ori_b.shape == (E, 102)
  assert torch.allclose(pos_b, body_pos.reshape(E, -1))
  identity6 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  assert torch.allclose(ori_b.view(E, num_bodies, 6), identity6.expand(E, num_bodies, 6))


# -- v3: wage, birth freeze, tenable-birth mask -----------------------------------

from scr_phybot_mjab.tasks.climb_transition80.mdp.rewards import scheduled_hold_margin_wage  # noqa: E402
from scr_phybot_mjab.tasks.climb_transition80.mdp.metrics import scheduled_hold_margin  # noqa: E402
from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import birth_mask_from_ranges  # noqa: E402


def test_hold_margin_wage_pays_only_welded_scheduled_pads():
  ref = torch.tensor([[True, True, False, False], [True, False, False, False]])
  eng = torch.tensor([[True, False, False, True], [False, False, False, False]])
  margin = torch.tensor([[0.15, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]])
  cmd = SimpleNamespace(ref_engaged=ref)
  aff = SimpleNamespace(engaged=eng, hold_margin=margin)
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: cmd if n == "climb_motion" else aff))
  wage = scheduled_hold_margin_wage(env, full_margin=0.3)
  # env0: LH welded at margin 0.15 -> half pay; RH scheduled but not welded -> 0;
  # RF welded but NOT scheduled -> ignored; averaged over the 2 scheduled pads.
  assert wage[0].item() == pytest.approx(0.25)
  assert wage[1].item() == pytest.approx(0.0)  # scheduled, not welded: nothing
  aff.hold_margin[0, 0] = 0.6
  assert scheduled_hold_margin_wage(env, full_margin=0.3)[0].item() == pytest.approx(0.5)
  aff.hold_margin[0, 0] = -0.2
  assert scheduled_hold_margin_wage(env, full_margin=0.3)[0].item() == pytest.approx(0.0)
  # The metric reads the same gate: env0 mean margin of welded scheduled pads.
  aff.hold_margin[0, 0] = 0.15
  assert scheduled_hold_margin(env)[0].item() == pytest.approx(0.15)
  assert scheduled_hold_margin(env)[1].item() == pytest.approx(1.0)
  assert scheduled_hold_margin(env, ee_index=1)[0].item() == pytest.approx(1.0)


def test_birth_freeze_holds_the_clock_then_releases():
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import TransitionMotionCommand

  cmd = object.__new__(TransitionMotionCommand)
  cmd.cfg = SimpleNamespace(sampling_mode="uniform", birth_freeze_steps=3)
  cmd.clip = SimpleNamespace(cycle_frames=100)
  cmd.time_steps = torch.tensor([10, 20])
  cmd._birth_freeze_left = torch.tensor([3, 0])  # env0 seeded birth, env1 standing
  cmd.metrics = {"birth_frozen": torch.zeros(2)}
  cmd._affordance = lambda: SimpleNamespace(reseat_hold=torch.tensor([False, False]))
  clocks = []
  for _ in range(5):
    cmd._update_command()
    clocks.append(cmd.time_steps.tolist())
  assert clocks == [[10, 21], [10, 22], [10, 23], [11, 24], [12, 25]]
  assert not bool(cmd.birth_frozen.any())


def test_birth_mask_ranges_and_draws():
  allowed = birth_mask_from_ranges(100, ((10, 20), (95, 200)))
  assert allowed.sum().item() == 100 - 10 - 5
  assert not allowed[10] and not allowed[19] and allowed[20] and not allowed[99]
  with pytest.raises(ValueError):
    birth_mask_from_ranges(10, ((0, 10),))
  # Adaptive draw: excluded frames are never produced, fully excluded bins get zero mass.
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import TransitionMotionCommand

  cmd = object.__new__(TransitionMotionCommand)
  C, B = 100, 5
  cmd.clip = SimpleNamespace(cycle_frames=C)
  cmd.cfg = SimpleNamespace(adaptive_uniform_ratio=0.1, adaptive_max_bin_prob=None)
  cmd._env = SimpleNamespace(device="cpu")  # ``device`` is a read-only property
  cmd._bin_count = B
  cmd._bin_failed = torch.zeros(B)
  cmd._adaptive_kernel = torch.tensor([1.0])
  cmd.metrics = {"sampling_entropy": torch.zeros(1), "sampling_top1_bin": torch.zeros(1)}
  allowed = birth_mask_from_ranges(C, ((20, 40), (55, 60)))  # bin 1 fully excluded, bin 2 partly
  cmd._birth_allowed = allowed
  bins_of = torch.arange(C) * B // C
  rows = [torch.nonzero((bins_of == b) & allowed).squeeze(-1) for b in range(B)]
  width = max(int(r.numel()) for r in rows)
  table = torch.zeros(B, width, dtype=torch.long); counts = torch.zeros(B, dtype=torch.long)
  for b, r in enumerate(rows):
    counts[b] = r.numel(); table[b, : r.numel()] = r
  cmd._bin_allowed_table, cmd._bin_allowed_count = table, counts
  frames = cmd._adaptive_draw(4000)
  assert bool(allowed[frames].all())
  assert not bool(((frames >= 20) & (frames < 40)).any())
  assert bool(((frames >= 40) & (frames < 55)).any()) and bool(((frames >= 60) & (frames < 80)).any())


def test_phi_v3e_pays_the_last_centimetres_and_the_press(monkeypatch):
  """Hover-trap fix: the wall-normal approach and the press earn Phi."""
  import scr_phybot_mjab.tasks.climb_transition80.mdp.rewards as rw

  env, motion, aff = _stub(dist=torch.full((E, H), 0.05))
  motion.ref_engaged[0, 0] = True
  gaps = torch.full((E, H * 4), 0.034)  # every sphere 3.4 cm off the wall
  monkeypatch.setattr(rw._SphereGeomCache, "surface_distances", classmethod(lambda cls, e: gaps))
  aff.pad_press_force = torch.zeros(E, H)
  kw = dict(reach_radius_m=0.5, load_share=0.2, approach_share=0.15, approach_radius_m=0.10, press_share=0.2, press_ref_n=20.0)

  def lh():
    return rw.engage_progress(env, **kw)[0].item() * H - 3.0

  assert lh() == pytest.approx(0.3 * (1 - 0.05 / 0.5) + 0.15 * (1 - 0.34))
  gaps[0, :4] = 0.0  # LH spheres at the wall, still no contact flag
  assert lh() == pytest.approx(0.27 + 0.15)
  aff.contact_count[0, 0] = 1.0  # touching: the press stage
  assert lh() == pytest.approx(0.50)
  aff.pad_press_force[0, 0] = 10.0
  assert lh() == pytest.approx(0.60)
  aff.pad_press_force[0, 0] = 40.0  # saturates at 20 N
  assert lh() == pytest.approx(0.70)
  aff.pad_shear_down[0, 0] = 2.0  # half the 4 N rung
  assert lh() == pytest.approx(0.80)
  aff.pad_shear_down[0, 0] = 40.0
  assert lh() == pytest.approx(0.90)  # = weld_level: monotone into the weld stage
  aff.engaged[0, 0] = True
  aff.seat_charge_fraction[0, 0] = 1.0
  assert lh() == pytest.approx(1.00)


def test_ignition_classroom_windows_and_draws():
  from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import (
    TransitionMotionCommand,
    rake_windows_from_engaged,
  )

  engaged = torch.zeros(100, 4, dtype=torch.bool)
  engaged[20:, 0] = True   # LH welds at 20
  engaged[60:, 1] = True   # RH welds at 60
  engaged[5:, 3] = True    # RF welds at 5 (window clipped at 0)
  assert rake_windows_from_engaged(engaged, 10) == ((0, 5), (10, 20), (50, 60))
  cmd = object.__new__(TransitionMotionCommand)
  cmd.cfg = SimpleNamespace(sampling_mode="uniform", spawn_early_fraction=1.0)
  cmd.clip = SimpleNamespace(cycle_frames=100)
  cmd._env = SimpleNamespace(device="cpu")
  cmd._classroom_windows = ((0, 5), (10, 20), (50, 60))
  cmask = torch.zeros(100, dtype=torch.bool)
  for a, b in cmd._classroom_windows:
    cmask[a:b] = True
  cmd._classroom_mask = cmask
  cmd._birth_allowed = torch.ones(100, dtype=torch.bool)
  cmd._birth_allowed[50:55] = False  # masked frames stay out even inside a window
  cmd._bias_window = (0, 100)
  frames = cmd._draw_rsi_frames(2000)
  assert bool(cmask[frames].all()) and bool(cmd._birth_allowed[frames].all())
  assert bool(((frames >= 55) & (frames < 60)).any()) and bool((frames < 5).any())
  # Without classroom windows the single segment window is used.
  cmd._classroom_windows = ()
  cmd._bias_window = (30, 40)
  frames = cmd._draw_rsi_frames(500)
  assert bool(((frames >= 30) & (frames < 40)).all())
