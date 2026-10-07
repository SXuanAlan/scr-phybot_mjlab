from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.climb_wall.mdp.metrics import microspine_engage_events
from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
  microspine_support_loss,
  motion_engage_mismatch,
  motion_ref_height_tracking,
  motion_root_ori_error_exp,
  motion_swing_approach,
  unsafe_microspine_release_penalty,
)


class FakeCommandManager:
  def __init__(self, command: torch.Tensor, affordance=None):
    self.command = command
    self.affordance = affordance

  def get_command(self, name: str) -> torch.Tensor:
    del name
    return self.command

  def get_term(self, name: str):
    del name
    return self.affordance


class FakeEventManager:
  def __init__(self, physics):
    self.physics = physics

  def get_term_cfg(self, name: str):
    del name
    return SimpleNamespace(func=self.physics)


def make_progress_env(z: float = 1.0, motion=None):
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
      root_link_lin_vel_w=torch.zeros(1, 3),
    )
  )
  return SimpleNamespace(
    scene={"robot": robot},
    command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.08]]), motion),
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    step_dt=0.02,
    num_envs=1,
    device="cpu",
  )


def test_ref_height_tracking_pays_on_profile_and_decays_both_ways():
  motion = SimpleNamespace(
    ref_root_pos_w=torch.tensor([[0.0, 0.0, 5.0]]),
    wall_up_w=torch.tensor([[0.0, 0.0, 1.0]]),  # vertical wall
  )
  env = make_progress_env(motion=motion)
  reward = motion_ref_height_tracking()
  kw = dict(asset_cfg=SceneEntityCfg("robot"), std=0.05)

  reward(env, **kw)
  env.episode_length_buf[:] = 25
  reward(env, **kw)  # last settling step re-bases robot and reference.

  # Both climb one step-height together: still on profile, full pay.
  env.episode_length_buf[:] = 26
  motion.ref_root_pos_w[:, 2] += 0.06
  env.scene["robot"].data.root_link_pos_w[:, 2] += 0.06
  assert reward(env, **kw).item() == pytest.approx(1.0)

  # Reference climbs one std further, robot stalls: exp(-1) — behind decays.
  env.episode_length_buf[:] = 27
  motion.ref_root_pos_w[:, 2] += 0.05
  assert reward(env, **kw).item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item(), rel=1e-4)

  # Robot overshoots to one std AHEAD: same exp(-1) — scrambling past the
  # ghost earns exactly as little as lagging it.
  env.episode_length_buf[:] = 28
  env.scene["robot"].data.root_link_pos_w[:, 2] += 0.10
  assert reward(env, **kw).item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item(), rel=1e-4)


def test_support_loss_penalizes_only_fewer_than_two_anchors():
  physics = SimpleNamespace(
    engaged=torch.tensor(
      [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
      ]
    )
  )
  env = SimpleNamespace(event_manager=FakeEventManager(physics))
  result = microspine_support_loss(env, minimum_supports=2)
  assert result.tolist() == [0.0, 1.0, 0.0]


def test_release_cost_distinguishes_controlled_and_failed_release():
  affordance = SimpleNamespace(
    engaged=torch.tensor(
      [
        [True, True, True, False],
        [True, True, True, False],
        [True, False, False, False],
      ]
    ),
    forced_release_this_step=torch.tensor(
      [
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
      ]
    ),
    release_reason=torch.tensor(
      [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [4, 0, 0, 0],
      ]
    ),
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.08]]), affordance),
    step_dt=0.02,
    episode_length_buf=torch.full((3,), 100, dtype=torch.long),
  )
  result = unsafe_microspine_release_penalty(env)
  assert result.tolist() == pytest.approx([0.0, 12.5, 100.0])


def test_empirical_domain_and_early_failures_are_not_free_unhooks():
  affordance = SimpleNamespace(
    engaged=torch.tensor([[False, True, True, True]] * 2),
    forced_release_this_step=torch.tensor([[True, False, False, False]] * 2),
    release_reason=torch.tensor([[7, 0, 0, 0], [8, 0, 0, 0]]),
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(torch.zeros(2, 3), affordance),
    step_dt=0.02,
    episode_length_buf=torch.full((2,), 100, dtype=torch.long),
  )
  cost = unsafe_microspine_release_penalty(env)
  assert torch.all(cost > 0)


def _make_motion_env(motion):
  return SimpleNamespace(
    command_manager=FakeCommandManager(torch.zeros(1, 3), motion),
  )


def test_motion_engage_mismatch_counts_per_ee_disagreements():
  motion = SimpleNamespace(
    ref_engaged=torch.tensor([[True, True, False, False]]),
    robot_engaged=torch.tensor([[True, False, True, False]]),
  )
  env = _make_motion_env(motion)
  assert motion_engage_mismatch(env).item() == 2.0


def test_motion_engage_mismatch_grace_discounts_near_schedule_edges():
  motion = SimpleNamespace(
    ref_engaged=torch.tensor([[True, True, True, True]]),
    robot_engaged=torch.tensor([[False, False, False, True]]),
    # Nearest-edge distances: 2.5 s, 0.25 s, and 0 s out on a 5 s cycle.
    ref_edge_distance=torch.tensor([[0.5, 0.05, 0.0, 0.0]]),
    cycle_time_s=5.0,
  )
  env = _make_motion_env(motion)
  # Full price deep in a segment, half price at half the grace window, free
  # at the flip itself; the matched EE bills nothing regardless of distance.
  assert motion_engage_mismatch(env, grace_time_s=0.5).item() == pytest.approx(1.5)


def test_motion_engage_mismatch_grace_fraction_scales_with_window():
  motion = SimpleNamespace(
    ref_engaged=torch.tensor([[True, True, True, True]]),
    robot_engaged=torch.tensor([[False, False, False, False]]),
    # Nearest-edge distances 2.5 / 0.3 / 0.15 / 0 s on a 5 s cycle...
    ref_edge_distance=torch.tensor([[0.5, 0.06, 0.03, 0.0]]),
    # ...inside windows of 2.0 / 1.0 / 1.0 / 1.0 s.
    ref_run_time=torch.tensor([[0.4, 0.2, 0.2, 0.2]]),
    cycle_time_s=5.0,
  )
  env = _make_motion_env(motion)
  # Grace = 0.3 x own window = 0.6 / 0.3 / 0.3 / 0.3 s: full price deep
  # inside (2.5 s out) and exactly at the grace boundary (0.3 s), half price
  # at half the grace (0.15 s), free at the flip. Fraction takes precedence
  # over any grace_time_s.
  result = motion_engage_mismatch(env, grace_time_s=99.0, grace_fraction=0.3)
  assert result.item() == pytest.approx(2.5)


def test_unsafe_release_bills_scheduled_reverse_unhooks():
  affordance = SimpleNamespace(
    engaged=torch.tensor([[True, True, False, False]] * 3),
    forced_release_this_step=torch.tensor([[True, False, False, False]] * 3),
    release_reason=torch.tensor([[1, 0, 0, 0]] * 3),
  )
  motion = SimpleNamespace(
    # env0: reverse unhook deep inside a scheduled ON window -> billed.
    # env1: same unhook AT the scheduled release edge -> free (on time).
    # env2: unhook while scheduled OFF -> free (reverse stays the free
    #       direction off schedule; only the mismatch bill applies there).
    ref_engaged=torch.tensor(
      [
        [True, True, True, True],
        [True, True, True, True],
        [False, True, True, True],
      ]
    ),
    ref_edge_distance=torch.tensor(
      [
        [0.25, 0.25, 0.25, 0.25],
        [0.0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
      ]
    ),
    cycle_time_s=4.0,
  )

  class _NamedManager:
    def __init__(self, terms):
      self._terms = terms

    def get_term(self, name):
      return self._terms[name]

  env = SimpleNamespace(
    command_manager=_NamedManager({"microspine_affordance": affordance, "climb_motion": motion}),
    step_dt=0.02,
    episode_length_buf=torch.full((3,), 100, dtype=torch.long),
  )
  result = unsafe_microspine_release_penalty(
    env,
    motion_command_name="climb_motion",
    scheduled_reverse_cost=0.3,
    scheduled_reverse_grace_s=0.25,
  )
  # env0: 0.3 / dt = 15 one-shot; the on-time and off-schedule unhooks are
  # free (two anchors remain, so no support-loss surcharge either).
  assert result.tolist() == pytest.approx([15.0, 0.0, 0.0])


def test_motion_swing_approach_pays_released_swings_toward_the_anchor():
  pad = torch.zeros(1, 4, 3)
  pad[0, 0, 2] = 0.05  # EE0 swinging 5 cm from its rung -> exp(-1).
  # EE1 swinging exactly at the rung -> 1.0. EE2 is latched during its swing
  # window -> no pay (the landing bonus owns that moment). EE3 is released
  # while scheduled ON -> no pay (that mismatch is billed, not rewarded).
  motion = SimpleNamespace(
    ref_engaged=torch.tensor([[False, False, False, True]]),
    robot_engaged=torch.tensor([[False, False, True, False]]),
    ref_anchor_pos_w=torch.zeros(1, 4, 3),
  )
  executor = SimpleNamespace(pad_center_world=lambda: pad)
  affordance = SimpleNamespace(_get_executor=lambda: executor)
  motion._affordance = lambda: affordance
  env = _make_motion_env(motion)
  expected = 1.0 + torch.exp(torch.tensor(-1.0)).item()
  assert motion_swing_approach(env, std=0.05).item() == pytest.approx(expected, rel=1e-4)


def test_engage_event_metric_counts_affordance_regrips_not_reset_anchors():
  affordance = SimpleNamespace(
    engaged_this_step=torch.tensor([[False, True, False, False], [False, False, False, False]])
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(torch.zeros(2, 3), affordance),
    step_dt=0.02,
  )

  assert microspine_engage_events(env).tolist() == pytest.approx([50.0, 0.0])


def test_engage_gate_scales_pose_income_by_schedule_adherence():
  # Zero pose error -> the kernel is 1.0 and the returned value IS the gate.
  ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

  def motion_with(match_pattern, edge_frac):
    return SimpleNamespace(
      ref_root_quat_w=ref,
      robot_root_quat_w=ref.clone(),
      ref_engaged=torch.tensor([[True, True, True, True]]),
      robot_engaged=torch.tensor([match_pattern]),
      ref_edge_distance=torch.full((1, 4), edge_frac),
      cycle_time_s=4.0,
    )

  # Full match -> full pay.
  env = _make_motion_env(motion_with([True, True, True, True], 0.5))
  assert motion_root_ori_error_exp(env, engage_gate_floor=0.25).item() == pytest.approx(1.0)

  # Half the limbs welded off-schedule, far from any edge: match 0.5 ->
  # gate = 0.25 + 0.75 * 0.5 (the statue keeps 62.5%, not 100%).
  env = _make_motion_env(motion_with([True, True, False, False], 0.5))
  assert motion_root_ori_error_exp(env, engage_gate_floor=0.25).item() == pytest.approx(0.625)

  # Same mismatch right at a scheduled flip (edge 0.125 s of a 0.25 s
  # grace): half forgiven -> billed 0.25 -> gate 0.8125. Same ramp as the
  # mismatch bill: physically impossible instant compliance is not taxed.
  env = _make_motion_env(motion_with([True, True, False, False], 0.125 / 4.0))
  assert motion_root_ori_error_exp(env, engage_gate_floor=0.25, engage_gate_grace_s=0.25).item() == pytest.approx(
    0.25 + 0.75 * 0.75
  )

  # Gate disabled (default None): mismatch does not touch pose income.
  env = _make_motion_env(motion_with([False, False, False, False], 0.5))
  assert motion_root_ori_error_exp(env).item() == pytest.approx(1.0)
