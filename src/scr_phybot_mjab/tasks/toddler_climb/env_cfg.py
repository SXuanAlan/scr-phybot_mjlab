"""Independent ToddlerBot climbing task; no PHYBot actuator/pad assumptions."""

from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import MOTOR_NAMES, STIFFNESS, UNIT_NAMES, get_robot_cfg
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.constraint import add_attachment_equalities, make_constraint_event_cfg
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import PIVOT, REFERENCE_PATH, wall_rotation

from .command import ToddlerClimbCommandCfg


def reference(env):
    return env.command_manager.get_term("climb_reference")


def observation(env, privileged=False):
    c = reference(env)
    r = c.robot
    phase = c.elapsed.remainder(c.duration) / c.duration * 2 * torch.pi
    quat = r.data.root_link_quat_w
    values = [
        r.data.joint_pos[:, c.motor_ids] - c.joint_pos_ref,
        r.data.joint_vel[:, c.motor_ids],
        quat,
        r.data.root_link_ang_vel_w,
        env.action_manager.action,
        torch.stack((phase.sin(), phase.cos()), -1),
        c.sample("engaged").float(),
    ]
    if privileged:
        values.extend(
            [c.engaged.float(), c.loads.flatten(1), c.margin, r.data.root_link_pos_w - c.sample("qpos")[:, :3]]
        )
    return torch.cat(values, -1)


def tracking(env):
    c = reference(env)
    q = (c.robot.data.joint_pos[:, c.motor_ids] - c.joint_pos_ref).square().mean(-1)
    root = (c.robot.data.root_link_pos_w - c.sample("qpos")[:, :3]).square().sum(-1)
    sites = (c.robot.data.site_pos_w[:, c.site_ids] - c.sample("unit_pos")).square().sum(-1).mean(-1)
    return 0.4 * torch.exp(-q / 0.15) + 0.3 * torch.exp(-root / 0.0025) + 0.3 * torch.exp(-sites / 0.0004)


def failed(env):
    c = reference(env)
    return torch.linalg.vector_norm(c.robot.data.root_link_pos_w - c.sample("qpos")[:, :3], dim=-1) > 0.15


@dataclass(kw_only=True)
class ReferenceActionCfg(JointPositionActionCfg):
    use_default_offset: bool = False

    def build(self, env):
        return ReferenceAction(self, env)


class ReferenceAction(JointPositionAction):
    def _find_targets(self, cfg):
        # MJLab 1.5 ignores preserve_order for joint actions by default.
        return self._entity.find_joints(MOTOR_NAMES, preserve_order=True)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if tuple(self.target_names) != MOTOR_NAMES:
            raise ValueError("ToddlerBot motor order mismatch")
        self.kp = torch.tensor(STIFFNESS, dtype=torch.float32, device=env.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        c = reference(self._env)
        # Audited gravity/contact feedforward expressed as a bounded PD offset.
        # Actual output is still clamped by the physical actuator model.
        self._processed_actions += c.joint_pos_ref + c.sample("motor_torque") / self.kp


def toddler_climb_env_cfg(play=False, *, angle_deg=90.0, stochastic=True, reference_file=str(REFERENCE_PATH)):
    rotation = wall_rotation(angle_deg)

    def wall_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="wall_body")
        body.add_geom(
            name="wall_geom",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            quat=[0.7071067812, 0, -0.7071067812, 0],
            size=[0.6, 0.8, 0.02],
            rgba=[0.47, 0.5, 0.54, 1],
            contype=1,
            conaffinity=1,
            condim=3,
            friction=[0.2, 0.005, 0.0001],
            solref=[0.004, 1],
        )
        return spec

    wall = EntityCfg(
        spec_fn=wall_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(PIVOT), rot=tuple(rotation.as_quat(scalar_first=True))),
    )
    mc = MicrospineCfg(
        ee_body_names=UNIT_NAMES,
        spheres_per_ee_rows=1,
        spheres_per_ee_cols=1,
        anchor_lock_rotation=False,
        anchor_project_to_wall=False,
        reanchor_after_grace=False,
        disable_contacts_while_latched=False,
        anchor_solref=(0.004, 1.0),
        engage_unit_preload_n=1.5,
    )
    mc.limit_surface.model = "empirical_unit"
    mc.seat_quality_scales_capacity = False

    def scene_spec(spec):
        add_attachment_equalities(spec, mc)

    return ManagerBasedRlEnvCfg(
        decimation=10,
        episode_length_s=30.0,
        scene=SceneCfg(
            num_envs=1 if play else 512,
            env_spacing=3.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_robot_cfg(), "wall": wall},
            spec_fn=scene_spec,
            sensors=(
                ContactSensorCfg(
                    name="unit_contact",
                    primary=ContactMatch(mode="geom", pattern=tuple(n + "_c_0_0" for n in UNIT_NAMES), entity="robot"),
                    secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    history_length=10,
                ),
            ),
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", cone="elliptic", iterations=100),
            njmax=800,
            nconmax=128,
            contact_sensor_maxmatch=256,
        ),
        viewer=ViewerConfig(body_name="torso", distance=1.4, elevation=-12, azimuth=25),
        observations={
            "actor": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=observation)}, concatenate_terms=True),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=observation, params={"privileged": True})},
                concatenate_terms=True,
            ),
        },
        actions={
            "joint_pos": ReferenceActionCfg(
                entity_name="robot", actuator_names=MOTOR_NAMES, preserve_order=True, scale=0.15
            )
        },
        commands={
            "climb_reference": ToddlerClimbCommandCfg(
                reference_file=reference_file, wall_angle_deg=angle_deg, stochastic=stochastic
            )
        },
        events={
            "reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
            "microspine_grip": make_constraint_event_cfg(mc, affordance_name="climb_reference"),
        },
        rewards={
            "tracking": RewardTermCfg(func=tracking, weight=1.0),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.002),
            "failure": RewardTermCfg(func=failed, weight=-5.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "failure": TerminationTermCfg(func=failed),
        },
    )
