"""PHYBot C2 climb-test environment: friction-only climbing baseline.

Differences vs ``Mjlab-Climb-Wall-Phybot-C2``:
  1. Microspine adhesion DISABLED — no event, no adhesion actuators, no
     engagement-dependent rewards/observations/terminations.
  2. Wall–spine contact pair uses MuJoCo defaults except sliding friction = 0.5
     (the climb_wall task uses friction=30 + custom solver params).
  3. Curriculum frozen — gravity tilt held at 30° from horizontal for the
     entire run (no ramp to vertical).
  4. ``ee_air_time`` driven by physical wall contact (mjlab ContactSensor)
     instead of the microspine engagement latch.
  5. Termination ``base_height`` triggers when the robot drops 0.3 m below
     the spawn height (spawn z = 1.0 → minimum_height = 0.7).
"""

from __future__ import annotations

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg

from spider_rl_mjlab.assets import (
  PHYBOT_C2_ACTION_SCALE,
  get_phybot_c2_robot_cfg,
  get_wall_cfg,
)
from spider_rl_mjlab.tasks.climb_wall import mdp
from spider_rl_mjlab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg
from spider_rl_mjlab.tasks.climb_wall.mdp.curriculums import gravity_angle


_EE_BODY_NAMES: tuple[str, ...] = (
  "left_hand_spine",
  "right_hand_spine",
  "left_foot_spine",
  "right_foot_spine",
)
_SPHERES_PER_EE_ROWS = 2
_SPHERES_PER_EE_COLS = 2
_VZ_CMD = 0.08
_FIXED_ANGLE_DEG = 30.0
_PAIR_FRICTION = 0.5

# Spawn z. base_height termination = spawn z - 0.3 = 0.7 → triggers when the
# robot has dropped 30 cm below where it started.
_SPAWN_Z = 1.0
_FALL_TOLERANCE = 0.3

_EE_CONTACT_SENSOR_NAME = "ee_wall_contact"

_EE_GEOMS = [
  f"{ee}_c_{i}_{j}"
  for ee in _EE_BODY_NAMES
  for i in range(_SPHERES_PER_EE_ROWS)
  for j in range(_SPHERES_PER_EE_COLS)
]


def ee_contact_air_time(
  env: ManagerBasedRlEnv,
  *,
  sensor_name: str = _EE_CONTACT_SENSOR_NAME,
  threshold: float = 0.3,
  vz_cmd: float = 0.08,
  cmd_eps: float = 0.0,
) -> torch.Tensor:
  """Climb-style ee_air_time using mjlab's ContactSensor air-time tracker.

  On each step an EE transitions from air → wall contact (``first_contact``),
  add ``(last_air_time - threshold)`` to the reward. Mirrors the reward shape
  of climb_wall's ``ee_air_time`` but reads physical wall contact instead of
  the microspine engagement latch (which is disabled in this task).

  Gated to zero when ``|vz_cmd| <= cmd_eps``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)  # (E, N) bool
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None
  reward = torch.sum(
    (last_air_time - threshold) * first_contact.float(), dim=1
  )
  if abs(vz_cmd) <= cmd_eps:
    reward = torch.zeros_like(reward)
  return reward


def phybot_c2_climb_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_climb_wall_env_cfg()

  robot_cfg = get_phybot_c2_robot_cfg()
  wall_cfg = get_wall_cfg()

  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.245, 0.0, _SPAWN_Z),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.815,
      "left_hip_roll": 0.09,
      "left_hip_yaw": -0.265,
      "left_knee": 0.765,
      "left_ankle_pitch": 1.34,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.815,
      "right_hip_roll": -0.09,
      "right_hip_yaw": 0.265,
      "right_knee": 0.765,
      "right_ankle_pitch": 1.34,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "neck_yaw": 0.0,
      "neck_pitch": 0.0,
      "left_shoulder_pitch": -2.0,
      "left_shoulder_roll": -0.17,
      "left_shoulder_yaw": 0.0,
      "left_elbow_pitch": -1.14,
      "right_shoulder_pitch": -2.0,
      "right_shoulder_roll": 0.17,
      "right_shoulder_yaw": 0.0,
      "right_elbow_pitch": -1.14,
    },
  )

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }

  # Wall–spine pairs: defaults everywhere except sliding friction.
  # MuJoCo default friction = [1.0, 0.005, 0.0001]; we override slide to 0.5
  # and leave torsional / rolling at defaults. NO adhesion actuators are
  # added — the only force keeping the robot on the wall is friction.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    for gname in _EE_GEOMS:
      pair = spec.add_pair()
      pair.geomname1 = f"robot/{gname}"
      pair.geomname2 = "wall/wall_geom"
      pair.friction = [_PAIR_FRICTION, _PAIR_FRICTION, 0.005, 0.0001, 0.0001]
    # Match climb_wall: disable terrain so gravity-tilt fully governs the
    # gravitational scenario without a floor under the robot.
    try:
      terrain_geom = spec.geom("terrain/terrain")
      terrain_geom.contype = 0
      terrain_geom.conaffinity = 0
    except Exception:
      pass

  cfg.scene.spec_fn = _spec_fn

  # ContactSensor over each spine subtree (covers the spine body's box geom
  # AND its child sphere bodies) against the wall's single geom.
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=_EE_CONTACT_SENSOR_NAME,
      primary=ContactMatch(
        mode="subtree",
        pattern=_EE_BODY_NAMES,
        entity="robot",
      ),
      secondary=ContactMatch(
        mode="geom",
        pattern="wall_geom",
        entity="wall",
      ),
      fields=("found",),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
  )

  # No microspine adhesion event — strip any base-level reference defensively.
  cfg.events.pop("microspine_adhesion", None)

  # Drop observations/terminations that depend on the engagement latch
  # (they read from the now-removed event).
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("hand_engaged_state", None)

  cfg.terminations.pop("both_hands_unlatched", None)

  # Tighten the fall-off termination: trigger when root drops 0.3 m below spawn.
  base_height_term = cfg.terminations.get("base_height")
  if base_height_term is not None:
    base_height_term.params["minimum_height"] = _SPAWN_Z - _FALL_TOLERANCE

  robot_cfg_e = SceneEntityCfg("robot")
  # Weights mirror IsaacLab's quadruped walking config
  # (isaaclab_tasks/.../velocity/velocity_env_cfg.py:RewardsCfg).
  # σ on track_vz stays at 0.05 (not 0.5 like IsaacLab) because target=0.08,
  # not ~1.0 — scaling σ with target keeps the "zero-velocity reward ≈ 0.02"
  # shape; σ=0.5 here would make doing nothing yield ~0.97.
  cfg.rewards = {
    "track_vz": RewardTermCfg(
      func=mdp.climb_vz_gauss,
      weight=1.0,
      params={"asset_cfg": robot_cfg_e, "target": _VZ_CMD, "sigma": 0.05},
    ),
    "lin_vel_into_wall_l2": RewardTermCfg(
      func=mdp.lin_vel_into_wall_l2,
      weight=-2.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.ang_vel_xy_l2,
      weight=-0.05,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": robot_cfg_e},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "ee_air_time": RewardTermCfg(
      func=ee_contact_air_time,
      weight=0.125,
      params={
        "sensor_name": _EE_CONTACT_SENSOR_NAME,
        "threshold": 0.5,
        "vz_cmd": _VZ_CMD,
        "cmd_eps": 0.0,
      },
    ),
    "ee_face_parallel_to_wall": RewardTermCfg(
      func=mdp.ee_face_parallel_to_wall,
      weight=0.0,
      params={
        "asset_cfg": robot_cfg_e,
        "ee_body_names": _EE_BODY_NAMES,
        "face_local_axis": (0.0, -1.0, 0.0),
      },
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  # Curriculum: keep ONLY a single-stage gravity_angle pinned at 30°.
  # Single stage → _interp_angle returns the constant; no tilt ramp.
  cfg.curriculum = {
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0, "angle_deg": _FIXED_ANGLE_DEG},
        ],
      },
    ),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
