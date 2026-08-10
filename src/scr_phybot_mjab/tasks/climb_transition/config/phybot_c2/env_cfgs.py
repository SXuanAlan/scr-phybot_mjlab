"""PHYBot C2 walk->climb transition (mount) environment.

Parallel to the climb task, not a fork of it: the config is BUILT by calling
the climb builder and mutating the returned object, so observations, rewards,
PD table and the s2r DR package — and hence the 150/170-dim contract — are
inherited verbatim and climb checkpoints warm-start strict. Everything
transition-specific lives in THIS package; climb_wall's sources are never
modified.

The task: the robot stands at the wall base (delivered by the walk policy),
replays the acyclic mount prologue — latch hands low, mount the feet, walk
the hands up — and holds the cycle's frame-0 stance for the climb policy to
take over. The robot is BLIND to its delivery error: standing RSI spawns
scatter the base pose while the reference and its wall anchors stay
canonical, so the policy must absorb the offset through contact and square
the yaw back up during the mount.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_STANDING_JOINT_POS,
  configure_phybot_c2_floor_geom,
)
from scr_phybot_mjab.tasks.climb_transition import mdp
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"
# Acyclic walk->climb mount prologue (scripts/gen_mount_clip.py): stand at
# standoff 0.23 -> lateral arm raise -> hands latch LOW (1.44 m, standing
# reach ceiling) -> feet mount -> toe braces -> two hand steps to 1.53 m ->
# handoff_hold at EXACTLY the cycle's frame-0 stance shifted -0.2234 m.
# Gates 2026-08-09: floor-aware static audit all-frames PASS, handoff joint
# delta vs state-1 4e-08 rad, contract tests in
# tests/microspine/test_mount_clip.py.
_MOUNT_MOTION_FILE = _MOTIONS_DIR / "climb_mount_prologue.npy"

# Blind-delivery spawn scatter: walk hands the robot over with tape-mark
# accuracy, and the mount policy must absorb at least this much base-pose
# error with no exteroception. Initial values; the mount-raster basin
# calibration will refine them against the real delivery spec.
_TRANSITION_SPAWN_XY_NOISE = (0.05, 0.03)  # +-wall-distance, +-lateral (m)
_TRANSITION_SPAWN_YAW_NOISE = 0.0873  # +-5 deg (rad)
# RSI spawn bias (v2, iter ~1200): half the resets land in a biased window
# so the gate skill gets real experience share; plain uniform gave the
# standing stretch ~7% and the run plateaued into "stand five seconds,
# refuse the raise, topple".
_TRANSITION_SPAWN_EARLY_FRACTION = 0.5
# v4 (iter ~5100): the window re-aims from [0, first-engage) to the
# place -> set LATCH stretch. v2/v3 taught standing, horizontal arms and
# leaning in, but neither the reach-up (narrow exp tracking kernels pay ~0
# beyond ~1 rad — no local gradient across the horizontal -> overhead gap)
# nor the press-to-latch ever ignited. Spawning AT the latch moment (arms
# on the anchors, one 3 N press away) trains the press directly; the
# reach-up then only has to bridge into a state the policy already values.
_TRANSITION_SPAWN_BIAS_SEGMENTS = ("hand_place", "hand_set_hold")
# Press-bootstrap (v4, paired with the window): the scheduled-press shaping
# at the climb-parity weight (0.4, income ~0.002) cannot teach pressing
# from scratch — the cycle never needed it, its teacher already pressed.
# Boosted 7.5x at v4 launch and annealed back toward parity by curriculum
# (iterations counted from the v4 process start; the env step counter
# resets on every launch).
_TRANSITION_PRESS_BOOTSTRAP_STAGES = [
  {"iter": 0, "weight": 3.0},
  {"iter": 2000, "weight": 1.5},
  {"iter": 4000, "weight": 0.4},
]


def _transition_spec_fn(spec: mujoco.MjSpec, base_fn) -> None:
  """The climb scene spec plus a contact-enabled floor.

  The climb builder's own spec_fn disables the terrain plane (gravity-tilt
  curriculum replaces the floor there); running it first and re-enabling
  contact afterwards means the later write wins without touching climb's
  closure. The mount stands on the floor until the hands take the wall —
  and latching is unaffected: the microspine sensors are pad-vs-WALL pairs,
  floor contact never reaches them.
  """
  base_fn(spec)
  configure_phybot_c2_floor_geom(spec, enable_contact=True)


def phybot_c2_climb_transition_env_cfg(
  play: bool = False,
  domain_rand: bool = True,
  student_obs: bool = False,
) -> ManagerBasedRlEnvCfg:
  """walk->climb transition (mount): stand at the wall, latch all four.

  Deltas against the climb cfg it is built from:

  * motion command swapped for :class:`mdp.MountMotionCommandCfg` on the
    mount prologue — clock clamps on the handoff frame, engage tables are
    acyclic, standing spawns carry the blind-delivery scatter;
  * floor contact enabled (spec_fn wrap);
  * no gravity ladder (default gravity IS the vertical-wall operating
    point) and no tilt-peel annealing (the static config already carries
    the calibrated band);
  * no random-release drill in v1 — an R-series rung for later; this also
    removes the only reseat_hold producer, so the clock cannot be frozen
    short of the handoff frame;
  * climb-rate command pinned to zero (a mount has no climb rate).
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    play=play,
    domain_rand=domain_rand,
    student_obs=student_obs,
    robustness_release=False,
  )

  base_spec_fn = cfg.scene.spec_fn
  cfg.scene.spec_fn = functools.partial(
    _transition_spec_fn, base_fn=base_spec_fn
  )

  # Swap the command IN PLACE (assignment to an existing key preserves the
  # test-enforced command order). The climb fields are carried over; play
  # mode mirrors the base play block's determinism and adds spawn-noise
  # zeroing — evaluation delivers the canonical frame-0 stance.
  cycle_motion = cfg.commands["climb_motion"]
  cfg.commands["climb_motion"] = mdp.MountMotionCommandCfg(
    entity_name=cycle_motion.entity_name,
    motion_file=str(_MOUNT_MOTION_FILE),
    affordance_command_name=cycle_motion.affordance_command_name,
    ee_body_names=cycle_motion.ee_body_names,
    sampling_mode="start" if play else "uniform",
    joint_pos_noise=0.0 if play else cycle_motion.joint_pos_noise,
    spawn_xy_noise=(0.0, 0.0) if play else _TRANSITION_SPAWN_XY_NOISE,
    spawn_yaw_noise=0.0 if play else _TRANSITION_SPAWN_YAW_NOISE,
    spawn_early_fraction=0.0 if play else _TRANSITION_SPAWN_EARLY_FRACTION,
    spawn_bias_segments=() if play else _TRANSITION_SPAWN_BIAS_SEGMENTS,
  )

  if not play:
    cfg.rewards["motion_scheduled_press"].weight = (
      _TRANSITION_PRESS_BOOTSTRAP_STAGES[0]["weight"]
    )
    cfg.curriculum["press_bootstrap_anneal"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "motion_scheduled_press",
        "weight_stages": _TRANSITION_PRESS_BOOTSTRAP_STAGES,
        "steps_per_iter": 96,
      },
    )

  # A mount has no climb rate (the clip's step_height is 0); the command
  # obs lane reads an honest constant zero. Hardcoded rather than derived:
  # clip_average_climb_rate falls back to the CYCLE's nominal rate when the
  # asset is absent at config-build time.
  cfg.commands["climb_velocity"].ranges.lin_vel_z = (0.0, 0.0)

  cfg.curriculum.pop("wall_angle", None)
  cfg.curriculum.pop("tilt_peel", None)

  # Cosmetic: the pre-reset init pose is the mount's standing start (RSI
  # overwrites it on every reset). Base x = wall 1.5 - standoff 0.23.
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.27, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=dict(PHYBOT_C2_STANDING_JOINT_POS),
    joint_vel={".*": 0.0},
  )

  cfg.metrics["mount_handoff_ready"] = MetricsTermCfg(
    func=mdp.mount_handoff_ready,
    params={
      "command_name": "climb_motion",
      "affordance_command_name": "microspine_affordance",
    },
  )

  return cfg
