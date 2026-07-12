"""Flexiv Rizon4 microspine pad test environment."""

from __future__ import annotations

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.rizon4.rizon4_constants import (
  RIZON4_HOME_JOINT_POS,
  RIZON4_JOINT_NAMES,
  RIZON4_MICROSPINE_EE_BODY,
  RIZON4_MICROSPINE_TCP_SITE,
  get_rizon4_robot_cfg,
)
from scr_phybot_mjab.assets.wall import WALL_OFFSET, get_wall_cfg
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg
from scr_phybot_mjab.microspine.contact import (
  add_contact_pairs,
  get_sphere_geom_names,
)
from scr_phybot_mjab.microspine.constraint import (
  add_attachment_equalities,
  make_constraint_event_cfg,
)
from scr_phybot_mjab.tasks.climb_wall import mdp as climb_mdp


def rizon4_microspine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the constraint-based Rizon4 microspine test task."""
  microspine_affordance_name = "microspine_affordance"
  microspine_contact_sensor_name = "microspine_wall_contact"
  microspine_event_name = "microspine_grip"

  microspine_cfg = MicrospineCfg(
    ee_body_names=(RIZON4_MICROSPINE_EE_BODY,),
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    engage_normal_force_threshold=3.0,
    engage_min_spheres=2,
    engage_parallel_cos_threshold=0.25,
    engage_consecutive_steps=5,
    reengage_cooldown_steps=30,
    face_local_axis=(0.0, -1.0, 0.0),
    initial_on=True,
    debug=play,
  )
  microspine_cfg.anchor_lock_rotation = True
  microspine_cfg.limit_surface.rake_mode = "ee_local"
  microspine_cfg.anchor_torquescale = 5.0
  # The arm spawns off the wall; bench flow starts disengaged and latches on a
  # light press.
  microspine_cfg.initial_on = False
  microspine_cfg.engage_min_spheres = 1
  microspine_cfg.engage_normal_force_threshold = 1.0
  microspine_cfg.engage_consecutive_steps = 5
  microspine_cfg.engage_parallel_cos_threshold = 0.0
  microspine_cfg.limit_surface.hold_grace_steps = 4
  microspine_cfg.reengage_cooldown_steps = 10
  microspine_cfg.anchor_solref = (0.03, 1.0)
  # Keep casual bench interaction forgiving; envelope tools restore calibrated
  # defaults before measuring the limit surface.
  microspine_cfg.limit_surface.f_n0_range = (200.0, 200.0)
  microspine_cfg.limit_surface.f_l0_range = (200.0, 200.0)

  robot_cfg = get_rizon4_robot_cfg()
  wall_cfg = get_wall_cfg()
  wall_cfg.init_state = type(wall_cfg.init_state)(
    pos=(1.30, WALL_OFFSET[1], WALL_OFFSET[2]),
    rot=wall_cfg.init_state.rot,
  )

  robot_entity = SceneEntityCfg("robot")
  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos_rel": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "microspine_state": ObservationTermCfg(
          func=climb_mdp.hand_engaged_state,
          params={"event_term_name": microspine_event_name},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions = {
    "cartesian_pose": DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=RIZON4_JOINT_NAMES,
      frame_type="site",
      frame_name=RIZON4_MICROSPINE_TCP_SITE,
      use_relative_mode=False,
      damping=0.04,
      max_dq=0.20,
      position_weight=1.0,
      # Weights enter the DLS cost squared; 0.4 gave the orientation error a
      # 0.16x cost and contact torques visibly tilted the pad on press.
      orientation_weight=1.0,
      joint_limit_weight=0.05,
      # Posture regularization is load-bearing: without it the redundant arm
      # wanders toward singular configurations and stalls mid-approach. Its
      # parasitic pull on an anchored pad is a few N through the softened
      # bench anchor — acceptable.
      posture_weight=0.01,
      posture_target=RIZON4_HOME_JOINT_POS,
    ),
  }

  commands = {
    microspine_affordance_name: MicrospineAffordanceCfg(
      microspine_cfg=microspine_cfg,
      contact_sensor_name=microspine_contact_sensor_name,
      physics_event_name=microspine_event_name,
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": robot_entity,
      },
    ),
  }

  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_attachment_equalities(spec, microspine_cfg)
    # Gravity-compensate the arm, like the real Flexiv controller's gravity
    # feedforward. Without it the position servos sag under gravity and the
    # arm leans ~50 N of permanent down-wall shear on a latched pad — biasing
    # every envelope readout and putting the reverse-shear release out of
    # reach of the teleop leash.
    for body in spec.bodies:
      if body.name.startswith("robot/"):
        body.gravcomp = 1.0

  events[microspine_event_name] = make_constraint_event_cfg(
    microspine_cfg,
    affordance_name=microspine_affordance_name,
  )

  return ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=120.0 if play else 30.0,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=3.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": robot_cfg,
        "wall": wall_cfg,
      },
      sensors=(
        ContactSensorCfg(
          name=microspine_contact_sensor_name,
          primary=ContactMatch(
            mode="geom",
            pattern=get_sphere_geom_names(microspine_cfg),
            entity="robot",
          ),
          secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
          fields=("found", "force"),
          reduce="maxforce",
          num_slots=1,
          global_frame=False,
          history_length=10,
        ),
      ),
      spec_fn=_spec_fn,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.002,
        cone="elliptic",
        integrator="implicitfast",
        impratio=10.0,
        ccd_iterations=100,
      ),
      njmax=180,
      nconmax=80,
      contact_sensor_maxmatch=256,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.55, 0.0, 0.9),
      distance=2.6,
      elevation=-20.0,
      azimuth=140.0,
      width=1280,
      height=720,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards={},
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    },
  )
