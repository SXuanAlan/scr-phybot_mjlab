"""Transition commands: the grip affordance and the mount reference with failure-adaptive RSI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mjlab.managers.command_manager import CommandTermCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_transition80.motions import CLIMB_MOUNT80_MOTION
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.commands import affordance_command

# Walk hands the robot over with tape-mark accuracy: standing births scatter
# by +-wall distance and +-lateral offset (m), and by +-yaw (rad, 5 deg).
SPAWN_XY_NOISE = (0.05, 0.03)
SPAWN_YAW_NOISE = 0.0873


def classroom_windows(motion_file: Path, rake_frames: int, handup_frames: int) -> tuple[tuple[int, int], ...]:
  """Birth windows (start, end exclusive) of the ignition classroom.

  The pre-landing rake [edge - rake_frames, edge) of every engage edge,
  where the pad touches the wall and a weld is one press-and-hold away, and
  the first ``handup_frames`` of each closing hand step, so a hand-up birth
  has runway before its landing deadline.
  """
  data = np.load(motion_file, allow_pickle=True).item()
  windows = list(mdp.rake_windows_from_engaged(np.asarray(data["ee_engaged"]), rake_frames))
  names = list(data["segment_names"])
  labels = np.asarray(data["phase_segment"])
  steps = ["hand_up_right", "hand_up_left"]
  # A two-step left-hand walk-up has a middle step.
  if "hand_up_left_mid" in names and np.any(labels == names.index("hand_up_left_mid")):
    steps.append("hand_up_left_mid")
  for step in steps:
    first = int(np.flatnonzero(labels == names.index(step))[0])
    windows.append((first, first + handup_frames))
  return tuple(sorted(windows))


def transition80_commands(microspine_cfg: MicrospineCfg) -> dict[str, CommandTermCfg]:
  return {
    MICROSPINE_AFFORDANCE_COMMAND: affordance_command(microspine_cfg),
    # After the affordance: its reset (RSI) seeds the per-env latch state and
    # must run after the affordance's own reset.
    MOTION_COMMAND: mdp.TransitionMotionCommandCfg(
      entity_name="robot",
      motion_file=str(CLIMB_MOUNT80_MOTION),
      affordance_command_name=MICROSPINE_AFFORDANCE_COMMAND,
      ee_body_names=microspine_cfg.ee_body_names,
      sampling_mode="adaptive",
      spawn_xy_noise=SPAWN_XY_NOISE,
      spawn_yaw_noise=SPAWN_YAW_NOISE,
      # The reference clock holds a seeded birth frame for 0.5 s, so the
      # birth settle transient passes while the reference stands still.
      birth_freeze_steps=25,
      # No frames are excluded from RSI: the births a passive hold cannot
      # keep are the policy's to learn with active loading.
      birth_exclude_frames=(),
      # Births in the ignition classroom; the 25-frame rake is the clip's
      # 0.5 s hand rake.
      spawn_early_fraction=0.45,
      spawn_bias_frame_windows=classroom_windows(CLIMB_MOUNT80_MOTION, rake_frames=25, handup_frames=30),
      # A uniform floor and a per-bin cap keep the failure-weighted sampler
      # from piling births onto the windows where episodes die fastest.
      adaptive_uniform_ratio=0.5,
      adaptive_max_bin_prob=0.08,
    ),
  }
