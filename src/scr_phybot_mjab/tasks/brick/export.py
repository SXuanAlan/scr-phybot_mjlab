"""Export the measured-pose command contract alongside a normalized actor."""

import hashlib
import json
import shutil

from .reference import PHASE_NAMES

__all__ = ("export_brick_sidecars",)


def export_brick_sidecars(env, output_dir):
  command = env.command_manager.get_term("brick_task")
  cfg = command.cfg
  reference = output_dir / "brick_reference.npz"
  shutil.copyfile(cfg.reference_file, reference)
  offset, layout = 0, []
  for name, size in (
    ("measured_brick_pose_in_base", 7),
    ("goal_pose_in_base", 7),
    ("filtered_measured_brick_linear_velocity_in_base", 3),
    ("pose_valid", 1),
    ("pose_age_seconds", 1),
    ("phase_one_hot", 7),
    ("phase_progress", 1),
    ("reference_minus_measured_joint_position", 21),
  ):
    layout.append({"term": name, "start": offset, "end": offset + size})
    offset += size
  assert offset == command.command.shape[-1]
  contract = {
    "schema_version": 1,
    "reference_file": reference.name,
    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    "command_dim": offset,
    "command_layout_per_frame": layout,
    "phase_names": PHASE_NAMES,
    "phase_durations_s": command.reference["durations"].cpu().tolist(),
    "gate_dwell_s": [0.1, 0.2, 1.0, 0.2, 1.0, 0.3, 1.0],
    "measured_gate_thresholds": {
      "both_hands_max_position_error_m": 0.025,
      "reach_outward_offset_m": 0.025,
      "stable_linear_speed_m_s": 0.05,
      "lift_height_m": 0.10,
      "carry_min_height_m": 0.08,
      "goal_xy_error_m": 0.03,
      "goal_orientation_error_rad": 0.174533,
      "ground_height_error_m": 0.015,
      "release_min_outward_hand_clearance_m": 0.02,
      "stand_joint_rms_error_rad": 0.18,
      "stand_upright_error_rad": 0.174533,
    },
    "measured_phase_gates": [
      "Both nominal hand centers near the measured side faces plus reach_outward_offset.",
      "Both nominal hand centers near the measured side faces.",
      "Measured height above pickup >= lift_height and stable linear speed.",
      "Goal xy/orientation tolerances and measured height above pickup >= carry_min_height.",
      "Goal xy/orientation, ground height and stable linear speed.",
      "Placement gate plus both hands beyond release_min_outward_hand_clearance.",
      "Joint RMS error relative to joint_pos[0,0] and base upright error within tolerance.",
    ],
    "phase_controller": "scr_phybot_mjab.tasks.brick.command.measured_phase_gates",
    "phase_clock": (
      "Hold at the last reference knot until the measured gate has remained true for its dwell. "
      "Freeze elapsed time and reset gate dwell on an invalid pose. Start real pickups at phase 0."
    ),
    "reference_interpolation": (
      "Linear interpolation of joint_pos within the current phase; progress=clamp(elapsed/duration,0,1). "
      "Joint order is policy_meta.json joint_names."
    ),
    "pose_format": (
      "[x,y,z,qw,qx,qy,qz]; metres, unit quaternions, right-handed gravity-aligned world; base x forward, y left, z up."
    ),
    "required_measurements": [
      "Joint encoders, joint velocities, previous policy action, base gyro and gravity direction.",
      "Timestamped brick pose and base pose in the same calibrated world frame, from external tracking/estimation.",
      "Goal pose and floor height in that same world frame.",
    ],
    "velocity_estimator": (
      "World-frame pose differences divided by time since last valid packet, component-clipped to +/-2 m/s, "
      "filtered v=0.95*v_previous+0.05*v_sample; then rotate into current base frame. Zero on reset."
    ),
    "dropout": (
      "Hold the last measured pose and velocity; set valid=0 and advance packet age. "
      "Phase gates never consume simulator contact truth."
    ),
    "hand_geometry": "Nominal FK from active forearm and fixed pad offset; passive gimbal angles are not actor inputs.",
    "hand_offsets_in_forearm_m": command.hand_offsets.cpu().tolist(),
    "brick_size_m": cfg.brick.size,
    "brick_mass_kg": cfg.brick.mass,
    "unit_layout": cfg.unit_layout,
    "unit_preload_n": 10.0,
    "start_xy_m": cfg.brick.start_xy,
    "goal_offset_xy_m": cfg.brick.goal_offset_xy,
    "lift_height_m": cfg.brick.lift_height,
    "control_rate_hz": 1.0 / env.step_dt,
    "integration_status": (
      "Observation contract and actor export. "
      "A hardware pose adapter and closed-loop hardware validation are still required."
    ),
  }
  path = output_dir / "brick_task_meta.json"
  path.write_text(json.dumps(contract, indent=2) + "\n")
  return path
