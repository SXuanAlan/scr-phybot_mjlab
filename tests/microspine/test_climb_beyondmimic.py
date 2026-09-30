"""Body kernels, loop velocities, failure attribution and early-stop contracts."""
import math
from types import SimpleNamespace

import numpy as np
import mujoco
import pytest
import torch

import scr_phybot_mjab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from scr_phybot_mjab.tasks.climb_wall.mdp.body_tracking import (
  align_body_reference, body_velocities_from_poses, build_body_pose_tables,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionCommand
from scr_phybot_mjab.tasks.climb_wall.mdp.phase_sampling import AdaptivePhaseSampler
from scr_phybot_mjab.tasks.climb_wall.mdp import rewards, terminations


@pytest.mark.parametrize("task", [
  "Mjlab-Climb-Wall-Phybot-C2", "Mjlab-Climb-Wall-Phybot-C2-Corners-A60",
])
def test_empirical_task_uses_agreed_sixteen_terms_and_scales(task):
  cfg = load_env_cfg(task)
  expected = {
    "motion_body_pos": (1., .1), "motion_body_ori": (1., .3),
    "motion_body_lin_vel": (1., 1.), "motion_body_ang_vel": (1., 3.14),
    "motion_root_ori": (.5, .3), "motion_ref_height": (2.5, .025),
    "motion_anchor_landing": (40., .03), "motion_swing_approach": (.5, .05),
    "motion_engage_mismatch": (-1., None), "microspine_support_loss": (-4., None),
    "unsafe_microspine_release": (-2.5, None), "microspine_release_load": (1., None),
    "action_rate_l2": (-.1, None), "dof_pos_limits": (-5., None),
    "termination_penalty": (-500., None), "undesired_contacts": (-.1, None),
  }
  assert cfg.rewards.keys() == expected.keys()
  for name, (weight, std) in expected.items():
    assert cfg.rewards[name].weight == weight
    if std is not None:
      assert cfg.rewards[name].params['std'] == std
  gated = {k for k, v in cfg.rewards.items() if 'engage_gate_floor' in v.params}
  assert gated == {'motion_body_pos', 'motion_body_ori', 'motion_root_ori'}
  motion = cfg.commands['climb_motion']
  assert motion.sampling_mode == 'adaptive'
  assert motion.synchronize_reset_phase and motion.initialize_seeded_anchors
  assert len(motion.tracking_body_names) == 17
  assert all(ee in motion.tracking_body_names for ee in motion.ee_body_names)
  sensor = next(s for s in cfg.scene.sensors if s.name == 'climb_undesired_contact')
  assert sensor.history_length == cfg.decimation
  assert sensor.reduce == 'maxforce'
  assert set(sensor.primary.exclude) == {'.*_spine.*', '.*_ankle_roll'}


def test_body_velocity_keeps_last_interval_and_ignores_quaternion_sign():
  positions = torch.zeros(4, 1, 3, dtype=torch.float64)
  positions[:, 0, 2] = torch.tensor([0., .02, .04, .06])
  theta = torch.arange(4, dtype=torch.float64) * .1
  quats = torch.stack((torch.cos(theta/2), theta*0, theta*0, torch.sin(theta/2)), -1)[:, None]
  quats[1] *= -1
  lin, ang = body_velocities_from_poses(positions, quats, 50.)
  torch.testing.assert_close(lin[:, 0, 2], torch.ones(3, dtype=torch.float64))
  torch.testing.assert_close(ang[:, 0, 2], torch.full((3,), 5., dtype=torch.float64))
  assert lin.shape == ang.shape == (3, 1, 3)


def test_fk_includes_passive_joint_geometry():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="root"><freejoint/><geom type="sphere" size=".1"/>
      <body name="hinge"><joint axis="0 0 1"/><geom type="sphere" size=".1"/>
        <body name="tip" pos="1 0 0"><geom type="sphere" size=".1"/></body>
      </body></body></worldbody></mujoco>''')
  pos, quat = build_body_pose_tables(
    model, model.qpos0, np.arange(7), np.array([], dtype=int), [model.body('tip').id],
    np.array([[0., 0., 1.], [0., 0., 1.02]]), np.array([[1., 0., 0., 0.]]*2),
    np.zeros((2, 0)), np.array([7]), np.array([[0.], [math.pi/2]]))
  np.testing.assert_allclose(pos[:, 0], [[1., 0., 1.], [0., 1., 1.02]], atol=1.e-12)
  np.testing.assert_allclose(quat[1, 0], [math.sqrt(.5), 0., 0., math.sqrt(.5)], atol=1.e-12)


def test_alignment_follows_xy_and_yaw_but_keeps_reference_height():
  identity = torch.tensor([[1., 0., 0., 0.]])
  yaw = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
  pos, quat = align_body_reference(
    torch.tensor([[[1., 0., 2.]]]), identity[:, None],
    torch.tensor([[0., 0., 2.]]), identity, torch.tensor([[4., 5., 9.]]), yaw)
  torch.testing.assert_close(pos, torch.tensor([[[4., 6., 2.]]]))
  torch.testing.assert_close(quat, yaw[:, None])


def _reward_env():
  q = torch.tensor([[[1., 0., 0., 0.]]])
  zero = torch.zeros(1, 1, 3)
  aff = SimpleNamespace(dr_release_pending=torch.zeros(1, 4, dtype=torch.bool))
  motion = SimpleNamespace(
    robot_body_pos_w=zero.clone(), robot_body_quat_w=q.clone(),
    robot_body_lin_vel_w=zero.clone(), robot_body_ang_vel_w=zero.clone(),
    ref_body_lin_vel_w=zero.clone(), ref_body_ang_vel_w=zero.clone(),
    aligned_body_reference=lambda: (zero, q),
    ref_engaged=torch.ones(1, 4, dtype=torch.bool),
    robot_engaged=torch.tensor([[False, True, True, True]]),
    ref_edge_distance=torch.ones(1, 4), ref_run_time=torch.ones(1, 4),
    cycle_time_s=9., _affordance=lambda: aff,
  )
  return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:motion)), motion


def test_pose_gate_applies_to_new_body_terms_and_velocities_remain_ungated():
  env, m = _reward_env()
  params = dict(engage_gate_floor=.25, engage_gate_grace_fraction=.3)
  m.robot_body_pos_w[0, 0, 0] = .1
  m.robot_body_quat_w[0, 0] = torch.tensor([math.cos(.15), 0., 0., math.sin(.15)])
  for f in (rewards.motion_body_position_error_exp, rewards.motion_body_orientation_error_exp):
    assert f(env, **params).item() == pytest.approx(math.exp(-1.) * .8125, rel=1.e-5)
  assert rewards.motion_body_linear_velocity_error_exp(env).item() == 1.
  assert rewards.motion_body_angular_velocity_error_exp(env).item() == 1.
  m.robot_body_lin_vel_w[0, 0, 0] = 1.
  m.robot_body_ang_vel_w[0, 0, 0] = 3.14
  assert rewards.motion_body_linear_velocity_error_exp(env).item() == pytest.approx(math.exp(-1.))
  assert rewards.motion_body_angular_velocity_error_exp(env).item() == pytest.approx(math.exp(-1.))


def test_unwanted_contact_reward_retains_a_brief_impact():
  history = torch.zeros(2, 3, 10, 3)
  history[0, 1, 3, 0] = 5.
  history[1, 2, :, 0] = 1.  # threshold is strictly exceeded
  env = SimpleNamespace(scene=SimpleNamespace(sensors={
    'climb_undesired_contact':SimpleNamespace(data=SimpleNamespace(force_history=history))}))
  torch.testing.assert_close(rewards.undesired_body_contacts(env), torch.tensor([1., 0.]))


def test_adaptive_sampler_keeps_coverage_and_accumulates_failures_once():
  sampler = AdaptivePhaseSampler(23, 5., 'cpu', alpha=.5)
  torch.manual_seed(5)
  draws = sampler.sample(10000)
  assert set(draws.tolist()) == set(range(23))
  sampler.record(torch.tensor([8, 8, 8]))
  sampler.record(torch.tensor([8, 8]))
  sampler.update()
  target = 8 * sampler.bin_count // sampler.frames
  assert sampler.failed.sum() == 2.5
  assert sampler.probabilities()[target] > .9
  assert torch.all(sampler.probabilities() > 0)
  sampler.update()
  assert sampler.failed.sum() == 1.25  # no double-counting


def test_failure_sampler_ignores_timeouts_and_initial_reset():
  sampler = AdaptivePhaseSampler(100, 50., 'cpu')
  motion = SimpleNamespace(phase_sampler=sampler, time_steps=torch.tensor([10, 60, 90]),
    _env=SimpleNamespace(episode_length_buf=torch.tensor([12, 30, 0]),
      termination_manager=SimpleNamespace(terminated=torch.tensor([True, False, True]))))
  ClimbMotionCommand._record_phase_failures(motion, torch.arange(3))
  torch.testing.assert_close(sampler.pending, torch.tensor([1., 0.]))


def test_support_loss_timer_allows_recovery_and_resets_only_selected_envs():
  aff = SimpleNamespace(engaged=torch.zeros(2, 4, dtype=torch.bool))
  env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02,
    command_manager=SimpleNamespace(get_term=lambda _:aff))
  terminate = terminations.prolonged_support_loss()
  for _ in range(19):
    assert not terminate(env).any()
  aff.engaged[1, 0] = True
  assert terminate(env).tolist() == [True, False]
  terminate.reset(torch.tensor([0]))
  assert not terminate(env).any()


def test_body_failure_checks_relative_pose_global_orientation_and_wall_height():
  env, m = _reward_env()
  m.robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))
  m.ref_root_pos_w = torch.zeros(1, 3)
  m.ref_root_quat_w = m.robot_root_quat_w = torch.tensor([[1., 0., 0., 0.]])
  m.wall_up_w = torch.tensor([[math.sqrt(.5), 0., math.sqrt(.5)]])
  assert not terminations.motion_body_tracking_divergence(env).any()
  m.robot_body_pos_w[0, 0, 0] = .301
  assert terminations.motion_body_tracking_divergence(env).item()
  m.robot_body_pos_w.zero_()
  m.robot_root_quat_w = torch.tensor([[math.cos(.51), 0., 0., math.sin(.51)]])
  assert terminations.motion_body_tracking_divergence(env).item()
  m.robot_root_quat_w = m.ref_root_quat_w
  m.robot.data.root_link_pos_w = m.wall_up_w * -.251
  assert terminations.motion_body_tracking_divergence(env).item()


def test_reset_phase_clock_skips_zero_time_and_only_holds_fresh_rows():
  m = SimpleNamespace(
    cfg=SimpleNamespace(synchronize_reset_phase=True, initialize_seeded_anchors=False),
    _affordance=lambda: SimpleNamespace(reseat_hold=torch.tensor([False, False])),
    time_steps=torch.tensor([449, 50]), _fresh=torch.tensor([False, True]),
    _advance_time=False, clip=SimpleNamespace(cycle_frames=450, step_height=.02),
    root_offset=torch.zeros(2, 3), phase_sampler=None,
  )
  ClimbMotionCommand._update_command(m)
  assert m.time_steps.tolist() == [449, 50]
  assert not m.root_offset.any()
  m._advance_time = True
  m._fresh[1] = True
  ClimbMotionCommand._update_command(m)
  assert m.time_steps.tolist() == [0, 50]
  torch.testing.assert_close(m.root_offset[:, 2], torch.tensor([.02, 0.]))
  ClimbMotionCommand._update_command(m)
  assert m.time_steps.tolist() == [1, 51]
