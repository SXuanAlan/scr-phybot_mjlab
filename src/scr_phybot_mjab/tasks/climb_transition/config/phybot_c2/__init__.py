"""PHYBot C2 walk->climb transition task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  phybot_c2_climb_transition_env_cfg,
  phybot_c2_climb_unified_env_cfg,
)
from .rl_cfg import phybot_c2_climb_transition_ppo_runner_cfg

# The climb contract verbatim against the acyclic mount prologue: floor on,
# terminal-hold clock, blind-delivery spawn scatter. Same obs/action dims as
# Mjlab-Climb-Wall-Phybot-C2 by construction, so climb checkpoints
# warm-start via scripts/train_warmstart.py.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Phybot-C2",
  env_cfg=phybot_c2_climb_transition_env_cfg(),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)

# Band-teacher variants (docs/plans/ 2026-08-10 matrix): same contract, the
# mount prologue re-generated at native band angles (grid + <=2 deg runtime
# rotation) with the wall_angle curriculum pinned to the band. The clip
# assets are produced by gen_mount_clip --wall-angle-deg; the cfgs only
# carry paths, so registration is safe before the assets exist.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Band80-Phybot-C2",
  env_cfg=phybot_c2_climb_transition_env_cfg(wall_band=80.0),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True, wall_band=80.0),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Band88-Phybot-C2",
  env_cfg=phybot_c2_climb_transition_env_cfg(wall_band=88.0),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True, wall_band=88.0),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)

# From-scratch variant (mirrors the climb -From0 precedent): keeps the
# inherited tilt-peel anneal so cold-start latches survive long enough to
# be valued (v8 probe: 7/8 forced releases were tilt-peel at the
# calibrated band). Same contract; checkpoints interchange freely. Mature
# warm-started lineages stay on the plain task (calibrated band, no
# re-anneal on relaunch).
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Phybot-C2-From0",
  env_cfg=phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True
  ),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)

# From0 with ground-only spawns (v26): the FULL engage economy and the
# marching clock, but every RSI birth lands on a disengaged (standing)
# frame — the v25 one-pot run showed seeded-hanging spawns stall a fresh
# policy (1.4-1.6 Hz rips, v8 pattern) while diluting the standing skill;
# here hanging experience arrives through actual mounts first. Widen back
# to the plain -From0 task (uniform full clip) once the latch ignites.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Phybot-C2-From0-Ground",
  env_cfg=phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True,
    engage_gate=True,
    ground_spawns=True,
    # v53: LadderMan Eq (1) parity — strict legs/root (the support that
    # must not improvise), loose arms (the error-absorbing organ: they
    # carry the blind-delivery offset pre-latch and the closed-chain
    # conflict post-weld; the environment-true anchor economy owns their
    # precision). Paired with the scheduled_contact_loss executioner.
    asym_arm_tracking=True,
  ),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)

# Stand-first phase of the From0 line: engage economy offline, window
# pinned to the stand/raise/place stage — standing mastery IS the task;
# graduate by lane call, then resume on the plain -From0 task.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition-Phybot-C2-From0-Stand",
  env_cfg=phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True, stand_first=True
  ),
  play_env_cfg=phybot_c2_climb_transition_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)

# L-D unified full-chain task (standing -> mount -> seam -> cycles) for the
# KL-distillation student; scripts/train_unified_kl.py owns the trainer.
register_mjlab_task(
  task_id="Mjlab-Climb-Unified-Phybot-C2",
  env_cfg=phybot_c2_climb_unified_env_cfg(),
  play_env_cfg=phybot_c2_climb_unified_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition_ppo_runner_cfg(),
)
