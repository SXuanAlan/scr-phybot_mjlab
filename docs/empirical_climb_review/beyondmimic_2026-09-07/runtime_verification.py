"""Verify reset/reward/termination plumbing and replay reference PD, without RL.

Uses real MuJoCo/Warp contacts and empirical draws. Reference-PD survival is
reported separately from implementation checks; it is not a trained policy.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_error_magnitude
import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
    phybot_c2_climb_wall_env_cfg, phybot_c2_climb_wall_corners60_env_cfg,
)


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'no_training': True, 'controller': 'reference joint-position PD', 'cases': []}
for layout, angle in (('all', 45.), ('all', 90.), ('corners', 45.), ('corners', 60.)):
    cfg = (phybot_c2_climb_wall_corners60_env_cfg() if layout == 'corners'
           else phybot_c2_climb_wall_env_cfg())
    cfg.scene.num_envs = 8
    cfg.curriculum['wall_angle'].params['angle_stages'] = [{'iter': 0, 'angle_deg': angle}]
    cfg.curriculum['wall_angle'].params['angle_jitter_deg'] = 0.
    # Keep actual training RSI/DR. The separate birth_reset_probe isolates
    # canonical frame-zero births with DR disabled.
    torch.manual_seed(20260907)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        obs, _ = env.reset()
        robot = env.scene['robot']
        motion = env.command_manager.get_term('climb_motion')
        aff = env.command_manager.get_term('microspine_affordance')
        grip = env.event_manager.get_term_cfg('microspine_grip').func
        action_term = env.action_manager.get_term('joint_pos')
        row = {'unit_layout': layout, 'wall_angle_deg': angle,
               'active_units_per_pad': aff.unit_load_tln.shape[2],
               'initial_frames': motion.time_steps.tolist(),
               'seeded_supports': aff.engaged.sum(-1).tolist(),
               'observation_shapes': {k: list(v.shape) for k, v in obs.items()},
               'physical_seed_anchors_correct': bool(torch.equal(
                   env.sim.data.eq_active[:, grip._eq_ids].bool(), aff.engaged)),
               'reset_pose_error_m': float((motion.robot_body_pos_w - motion.ref_body_pos_w).norm(dim=-1).max()),
               'reset_orientation_error_rad': float(quat_error_magnitude(
                   motion.robot_body_quat_w, motion.ref_body_quat_w).max()),
               'zero_birth_events': not bool(aff.engaged_this_step.any() | grip.engaged_this_step.any()),
               'zero_reset_guard_age': not bool(aff._steps_since_engage.any()),
               'contact_history_shape': list(env.scene.sensors['climb_undesired_contact'].data.force_history.shape),
               'finite': finite(obs), 'termination_events': 0, 'timeout_events': 0,
               'new_latches': 0, 'release_reasons': Counter(),
               'termination_reasons': Counter(), 'max_episode_steps_before_done': 0,
               'peak_total_contacts': 0, 'peak_constraint_rows_per_world': 0, 'overflow_substeps': 0}
        assert row['physical_seed_anchors_correct'] and row['zero_birth_events'] and row['zero_reset_guard_age']
        assert row['reset_pose_error_m'] < 1.e-5 and row['reset_orientation_error_rad'] < 1.e-3
        assert all(s in (3, 4) for s in row['seeded_supports'])
        # Reading each reward at birth cannot produce a landing/release event.
        for name in ('motion_anchor_landing', 'microspine_release_load'):
            reward_cfg = env.reward_manager.get_term_cfg(name)
            assert not bool(reward_cfg.func(env, **reward_cfg.params).any()), name

        untouched = {k: v[1:].clone() for k, v in {
            'frames': motion.time_steps, 'anchors': grip.anchor_pos_w,
            'anchor_quat': grip.anchor_quat_w, 'anchor_age': grip._engage_age,
            'hold_age': aff._steps_since_engage, 'debounce': aff._engage_count,
            'engaged': aff.engaged, 'physical_engaged': env.sim.data.eq_active[:, grip._eq_ids],
        }.items()}
        env.reset(env_ids=torch.tensor([0], device=env.device))
        after = {'frames': motion.time_steps, 'anchors': grip.anchor_pos_w,
                 'anchor_quat': grip.anchor_quat_w, 'anchor_age': grip._engage_age,
                 'hold_age': aff._steps_since_engage, 'debounce': aff._engage_count,
                 'engaged': aff.engaged, 'physical_engaged': env.sim.data.eq_active[:, grip._eq_ids]}
        row['partial_reset_preserves_other_envs'] = all(torch.equal(v, after[k][1:]) for k, v in untouched.items())
        assert row['partial_reset_preserves_other_envs']

        original_step = env.sim.step
        def recorded_step():
            original_step()
            contacts = int(env.sim.data.nacon.max())
            constraints = int(env.sim.data.nefc.max())
            row['peak_total_contacts'] = max(row['peak_total_contacts'], contacts)
            row['peak_constraint_rows_per_world'] = max(row['peak_constraint_rows_per_world'], constraints)
            row['overflow_substeps'] += int(contacts > env.sim.wp_data.naconmax or constraints > cfg.sim.njmax)
        env.sim.step = recorded_step

        original_update = aff._update_command
        def recorded_update():
            original_update()
            on = aff.engaged_this_step
            row['new_latches'] += int(on.sum())
            if bool(on.any()):
                assert bool(torch.isfinite(aff.unit_preload_n[on]).all())
                assert bool((aff.unit_preload_n[on] >= 10.).all())
            row['release_reasons'].update(map(str, aff.release_reason[aff.forced_release_this_step].tolist()))
        aff._update_command = recorded_update

        original_compute = env.termination_manager.compute
        def recorded_terminations():
            done = original_compute()
            for name in env.termination_manager.active_terms:
                row['termination_reasons'][name] += int(env.termination_manager.get_term(name).sum())
            if bool(done.any()):
                row['max_episode_steps_before_done'] = max(row['max_episode_steps_before_done'], int(env.episode_length_buf[done].max()))
            return done
        env.termination_manager.compute = recorded_terminations

        with torch.no_grad():
            for _ in range(100):
                target = robot.data.default_joint_pos.clone()
                target[:, motion._joint_ids] = motion.ref_joint_pos
                actions = (target[:, action_term._target_ids] - action_term._offset) / action_term._scale
                obs, reward, terminated, truncated, _ = env.step(actions)
                row['finite'] &= finite((obs, reward, robot.data.joint_pos, aff.unit_load_tln))
                row['termination_events'] += int(terminated.sum())
                row['timeout_events'] += int(truncated.sum())
                if bool(terminated.any() | truncated.any()):
                    ids = (terminated | truncated).nonzero().flatten()
                    assert torch.equal(env.sim.data.eq_active[:, grip._eq_ids].bool()[ids], aff.engaged[ids])
                    assert not bool(aff._steps_since_engage[ids].any())
                    assert not bool(aff._engage_count[ids].any())
        row['failure_histogram'] = motion.phase_sampler.failed.tolist()
        row['phase_probabilities'] = motion.phase_sampler.probabilities().tolist()
        row['final_supports_after_possible_resets'] = aff.engaged.sum(-1).tolist()
        row['policy_steps'] = 100
        row['implementation_checks_passed'] = row['finite'] and row['overflow_substeps'] == 0
        assert row['implementation_checks_passed']
        assert row['termination_events'] == 0 or sum(row['failure_histogram']) > 0
        report['cases'].append(row)
        Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT', json.dumps(row), flush=True)
    finally:
        env.close()
