"""Phybot dual-pad floor pickup, carry and placement."""

import copy
import math

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg, MicrospineSurfaceCfg
from scr_phybot_mjab.microspine.constraint import add_attachment_equalities, make_constraint_event_cfg
from scr_phybot_mjab.microspine.contact import add_contact_pairs, get_sphere_geom_names
from scr_phybot_mjab.microspine.contact_sensor import WrenchContactSensorCfg
from scr_phybot_mjab.tasks.common.phybot_c2 import (
  BASE_ANG_VEL_SENSOR,
  BASE_QUAT_SENSOR,
  apply_real_pd_damping,
  policy_joint_cfg,
)
from scr_phybot_mjab.tasks.walk.mdp import projected_gravity_from_framequat_sensor

from . import mdp
from .actions import BrickJointPositionActionCfg
from .command import BrickTaskCommandCfg
from .geometry import DEFAULT_UNIT_LAYOUT, active_unit_count, brick_robot_collision


def brick_env_cfg(
  play=False, *, stage="full", randomize=False, grip_enabled=True, brick=None, unit_layout=DEFAULT_UNIT_LAYOUT
):
  active_unit_count(unit_layout)
  brick = BrickCfg() if brick is None else brick
  hands = ("left_hand_spine", "right_hand_spine")
  surfaces = tuple(
    MicrospineSurfaceCfg(
      ee_body_name=hand,
      entity_name="brick",
      body_name="brick",
      geom_name="brick_geom",
      point=(0.0, sign * brick.size[1] / 2, 0.0),
      normal=(0.0, sign, 0.0),
      tangent=(1.0, 0.0, 0.0),
      half_extents=(brick.size[0] / 2, brick.size[2] / 2),
    )
    for hand, sign in zip(hands, (1.0, -1.0))
  )
  mc = MicrospineCfg(
    ee_body_names=hands,
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
    surfaces=surfaces,
    initial_on=False,
    grip_enabled=grip_enabled,
    contact_margin=0.0,
    contact_gap=0.0,
    contact_friction=(0.2, 0.2, 0.005, 0.0001, 0.0001),
    anchor_lock_rotation=True,
    anchor_torquescale=1.0,
    reanchor_after_grace=False,
    anchor_project_to_wall=False,
    engage_unit_preload_n=10.0,
    engage_min_spine_fraction=1.0,
    engage_parallel_cos_threshold=math.cos(math.radians(5)),
    engage_consecutive_steps=3,
    reengage_cooldown_steps=10,
    disable_contacts_while_latched=True,
  )
  ls = mc.limit_surface
  ls.unit_layout = unit_layout
  ls.hold_grace_steps = 1
  robot = get_phybot_c2_robot_cfg()
  robot.collisions = (brick_robot_collision(),)
  asset = policy_joint_cfg()
  actor = {
    "base_ang_vel": ObservationTermCfg(func=base_mdp.builtin_sensor, params={"sensor_name": BASE_ANG_VEL_SENSOR}),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor, params={"sensor_name": BASE_QUAT_SENSOR}
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel, params={"asset_cfg": asset}, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel, params={"asset_cfg": asset}, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
    "task_command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": "brick_task"}),
  }
  proxy_names = get_sphere_geom_names(mc)
  sensors = (
    WrenchContactSensorCfg(
      name="microspine_brick_contact",
      primary=ContactMatch(mode="geom", pattern=proxy_names, entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      history_length=10,
      num_slots=1,
      reference_sites={name: f"robot/{name.split('_c_')[0]}_contact_center" for name in proxy_names},
    ),
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="subtree", pattern=r"^(left|right)_ankle_roll$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="brick_ground_contact",
      primary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="illegal_ground_contact",
      primary=ContactMatch(mode="body", pattern=r"^(?!(?:left|right)_(?:ankle_roll|foot_spine.*)$).*$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=10,
    ),
    ContactSensorCfg(
      name="illegal_brick_contact",
      primary=ContactMatch(mode="body", pattern=r"^(?!.*hand_spine).*$", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=10,
    ),
  )

  def spec_fn(spec: mujoco.MjSpec):
    configure_phybot_c2_floor_geom(spec)
    add_contact_pairs(spec, mc)
    add_attachment_equalities(spec, mc)
    for hand in hands:
      # The separate four face proxies own brick compression; retain the
      # physical backing plate against ground and other robot bodies.
      spec.add_exclude(bodyname1=f"robot/{hand}", bodyname2="brick/brick")
    spec.worldbody.add_site(
      name="brick_goal",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[brick.start_xy[0] + brick.goal_offset_xy[0], brick.start_xy[1] + brick.goal_offset_xy[1], 0.002],
      size=[brick.size[0] / 2 + 0.03, brick.size[1] / 2 + 0.03, 0.002],
      rgba=[0.2, 0.8, 0.3, 0.25],
    )

  command = BrickTaskCommandCfg(
    brick=brick,
    training_stage=stage,
    randomize=randomize,
    unit_layout=unit_layout,
    pose_position_noise_m=0.0 if play else 0.003,
    pose_rotation_noise_rad=0.0 if play else 0.01,
    max_delay_steps=0 if play else 2,
    dropout_probability=0.01 if randomize and not play else 0.0,
    reset_xy_noise_m=0.01 if randomize else 0.0,
    reset_yaw_noise_rad=0.03 if randomize else 0.0,
  )
  events = {
    "reset_scene": EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset"),
    "microspine_grip": make_constraint_event_cfg(mc, affordance_name="microspine_affordance"),
  }
  if randomize:
    events["brick_density"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("brick", body_names=("brick",)),
        "alpha_range": (0.5 * math.log(0.5), 0.5 * math.log(1.5)),
      },
    )
    events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_ankle_roll_collision$",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    )
    events["servo_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    )
  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=30.0,
    scene=SceneCfg(
      num_envs=1 if play else 1024,
      env_spacing=3.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot, "brick": brick.entity()},
      sensors=sensors,
      spec_fn=spec_fn,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", cone="elliptic", impratio=10.0),
      njmax=640,
      nconmax=256,
      contact_sensor_maxmatch=512,
    ),
    viewer=ViewerConfig(body_name="torso", distance=2.5, elevation=-15.0, azimuth=135.0),
    observations={
      "actor": ObservationGroupCfg(terms=actor, history_length=10, concatenate_terms=True, enable_corruption=not play),
      "critic": ObservationGroupCfg(
        terms={**copy.deepcopy(actor), "physical_state": ObservationTermCfg(func=mdp.privileged_state)},
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions={
      "joint_pos": BrickJointPositionActionCfg(
        entity_name="robot", actuator_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True, scale=0.25
      )
    },
    commands={
      "microspine_affordance": MicrospineAffordanceCfg(
        microspine_cfg=mc, skip_zero_time_update=True, contact_sensor_name="microspine_brick_contact"
      ),
      "brick_task": command,
    },
    events=events,
    rewards={
      "reference": RewardTermCfg(func=mdp.tracking, weight=1.0),
      "object": RewardTermCfg(func=mdp.object_tracking, weight=2.0),
      "milestone": RewardTermCfg(func=mdp.milestone, weight=2.0),
      "success": RewardTermCfg(func=mdp.completion, weight=20.0),
      "grip": RewardTermCfg(func=mdp.grip_quality, weight=0.5),
      "squeeze": RewardTermCfg(func=mdp.squeeze, weight=-0.2),
      "overload": RewardTermCfg(func=mdp.overload, weight=-2.0),
      "slip": RewardTermCfg(func=mdp.slip, weight=-1.0),
      "foot_drift": RewardTermCfg(func=mdp.foot_drift, weight=-100.0),
      "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.005),
      "failure": RewardTermCfg(func=mdp.failed, weight=-20.0 / 0.02),
    },
    terminations={
      "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
      "failure": TerminationTermCfg(func=mdp.failed),
      "success": TerminationTermCfg(func=mdp.finished),
    },
  )
  apply_real_pd_damping(cfg)
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  return cfg
