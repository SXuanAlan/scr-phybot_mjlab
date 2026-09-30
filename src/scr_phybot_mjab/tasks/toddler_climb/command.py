"""Reference phase and a single-unit engagement/holding state machine."""

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import MOTOR_NAMES, UNIT_NAMES, UNIT_PRELOAD
from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface
from scr_phybot_mjab.microspine.surfaces import rotate
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import PIVOT, REFERENCE_PATH, load_reference, wall_rotation


@dataclass(kw_only=True)
class ToddlerClimbCommandCfg(CommandTermCfg):
    reference_file: str = str(REFERENCE_PATH)
    wall_angle_deg: float = 90.0
    preload_n: float = UNIT_PRELOAD
    stochastic: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env):
        return ToddlerClimbCommand(self, env)


class ToddlerClimbCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.archive = load_reference(cfg.reference_file)
        if not np.isclose(float(self.archive["wall_angle_deg"]), cfg.wall_angle_deg):
            raise ValueError("Task wall angle and motion reference must agree")
        if not np.isclose(cfg.preload_n, UNIT_PRELOAD):
            raise ValueError("Regenerate the reference to change the preload assumption")
        self.motor_ids, _ = self.robot.find_joints(MOTOR_NAMES, preserve_order=True)
        self.all_joint_ids, _ = self.robot.find_joints(tuple(self.archive["joint_names"]), preserve_order=True)
        self.site_ids, _ = self.robot.find_sites(tuple(n + "_contact_center" for n in UNIT_NAMES), preserve_order=True)
        self.body_ids, _ = self.robot.find_bodies(UNIT_NAMES, preserve_order=True)
        self.reference = {
            k: torch.as_tensor(self.archive[k], dtype=torch.float32, device=self.device)
            for k in ("qpos", "motor_pos", "motor_torque", "unit_pos", "engaged", "press_unit", "cycle_delta")
        }
        self.fps = float(self.archive["fps"])
        self.duration = (len(self.archive["qpos"]) - 1) / self.fps
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.fresh = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        shape = (self.num_envs, 4)
        self.engaged = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.age = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.preload_count = self.age.clone()
        self.cooldown = self.age.clone()
        self.strength = torch.rand(shape, device=self.device)
        self.loading = torch.rand(shape, device=self.device)
        self.preload = torch.zeros(shape, device=self.device)
        self.margin = self.preload.clone()
        self.loads = torch.zeros((*shape, 3), device=self.device)
        self.table = EmpiricalLimitSurface(device=self.device)
        self.normal = torch.tensor(
            wall_rotation(cfg.wall_angle_deg).apply([-1.0, 0.0, 0.0]), dtype=torch.float32, device=self.device
        )
        self.pivot = torch.tensor(PIVOT, dtype=torch.float32, device=self.device)
        sensor = env.scene["unit_contact"]
        self.sensor_ids = [
            next(i for i, n in enumerate(sensor.primary_names) if n.endswith(u + "_c_0_0")) for u in UNIT_NAMES
        ]
        for key in ("attached_units", "minimum_margin", "reference_error_m", "failed_grips"):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        self.failed_grips = torch.zeros(self.num_envs, device=self.device)
        self._dt = 0.0

    @property
    def command(self):
        return self.engaged

    def sample(self, key):
        frame = (self.elapsed.remainder(self.duration) * self.fps).clamp(max=len(self.archive["qpos"]) - 1.00001)
        i = frame.long()
        mix = (frame - i).view(-1, *([1] * (self.reference[key].ndim - 1)))
        if key == "engaged":
            return self.reference[key][i] > 0.5
        if key == "press_unit":
            return self.reference[key][i].long()
        result = torch.lerp(self.reference[key][i], self.reference[key][i + 1], mix)
        cycles = (self.elapsed / self.duration).floor()
        delta = self.reference["cycle_delta"]
        if key == "qpos":
            result[:, :3] += cycles[:, None] * delta + self._env.scene.env_origins
        elif key == "unit_pos":
            result += cycles[:, None, None] * delta + self._env.scene.env_origins[:, None]
        return result

    @property
    def joint_pos_ref(self):
        return self.sample("motor_pos")

    def compute(self, dt):
        self._dt = dt
        super().compute(dt)

    def _resample_command(self, env_ids):
        self.elapsed[env_ids] = 0
        self.fresh[env_ids] = True
        self.engaged[env_ids] = self.reference["engaged"][0] > 0.5
        self.age[env_ids] = 100
        self.preload_count[env_ids] = 0
        self.cooldown[env_ids] = 0
        self.failed_grips[env_ids] = 0
        self.margin[env_ids] = 0
        self.preload[env_ids] = 0
        self.loads[env_ids] = 0
        self.strength[env_ids] = torch.rand((len(env_ids), 4), device=self.device) if self.cfg.stochastic else 0.5
        self.loading[env_ids] = torch.rand((len(env_ids), 4), device=self.device) if self.cfg.stochastic else 1.0
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.all_joint_ids] = self.reference["qpos"][0, 7:]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        root = self.reference["qpos"][0, :7].expand(len(env_ids), -1).clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(
            torch.cat((root, torch.zeros((len(env_ids), 6), device=self.device)), -1), env_ids=env_ids
        )

    def _update_command(self):
        grip = self._env.event_manager.get_term_cfg("microspine_grip").func
        fresh_ids = self.fresh.nonzero().flatten()
        if len(fresh_ids):
            # This task starts on a wall with mature contacts; it is not a mount task.
            # Strengths are not screened/resampled to make this pose survive.
            grip.initialize_seeded_anchors(fresh_ids)
        if self._dt == 0:
            self.fresh[:] = False
            return
        contact = self._env.scene["unit_contact"].data.force[:, self.sensor_ids].reshape(self.num_envs, 4, 3)
        # MuJoCo netforce points primary (unit) -> secondary (wall): it is
        # already the load on the wall, like compute_anchor_loads().
        history = self._env.scene["unit_contact"].data.force_history[:, self.sensor_ids]
        self.preload = (-(history.mean(2) * self.normal).sum(-1)).clamp_min(0)
        loads = grip.compute_anchor_loads() + contact
        quat = self.robot.data.body_link_quat_w[:, self.body_ids]
        rake = rotate(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device))
        rake = rake - (rake * self.normal).sum(-1, keepdim=True) * self.normal
        rake = rake / torch.linalg.vector_norm(rake, dim=-1, keepdim=True).clamp_min(1e-8)
        lateral = torch.linalg.cross(self.normal.expand_as(rake), rake, dim=-1)
        self.loads = torch.stack(((loads * rake).sum(-1), (loads * lateral).sum(-1), (loads * self.normal).sum(-1)), -1)
        cap, early, valid, _ = self.table.query(self.loads[..., 0], self.loads[..., 1], self.strength)
        self.margin = cap - self.loads[..., 2]
        face = rotate(quat, torch.tensor([0.0, -1.0, 0.0], device=self.device))
        aligned = (face * (-self.normal)).sum(-1) > 0.98480775
        overload = (~valid) | (self.margin < 0) | (self.loading < early) | (~aligned) | (~torch.isfinite(loads).all(-1))
        failed = self.engaged & (self.age >= 1) & overload & (~self.fresh[:, None])
        self.failed_grips += failed.sum(-1)
        self.elapsed += self._dt
        desired = self.sample("engaged") | (self.sample("press_unit")[:, None] == torch.arange(4, device=self.device))
        release = self.engaged & (~desired | failed)
        self.cooldown = torch.where(release, torch.full_like(self.cooldown, 5), (self.cooldown - 1).clamp_min(0))
        self.engaged &= ~release
        positions = self.robot.data.site_pos_w[:, self.site_ids]
        gap = ((positions - self._env.scene.env_origins[:, None] - self.pivot) * self.normal).sum(-1)
        ready = (
            desired
            & ~self.engaged
            & (self.cooldown == 0)
            & aligned
            & (gap.abs() < 0.003)
            & (self.preload + 1e-6 >= self.cfg.preload_n)
        )
        self.preload_count = torch.where(ready, self.preload_count + 1, torch.zeros_like(self.preload_count))
        rising = self.preload_count >= 3
        if self.cfg.stochastic:
            self.strength = torch.where(rising, torch.rand_like(self.strength), self.strength)
            self.loading = torch.where(rising, torch.rand_like(self.loading), self.loading)
        self.engaged |= rising
        self.age = torch.where(rising, torch.zeros_like(self.age), self.age + 1)
        self.preload_count = torch.where(rising, torch.zeros_like(self.preload_count), self.preload_count)
        self.fresh[:] = False

    def _update_metrics(self):
        self.metrics["attached_units"][:] = self.engaged.sum(-1)
        self.metrics["minimum_margin"][:] = (
            torch.where(self.engaged, self.margin, torch.full_like(self.margin, 100)).min(-1).values
        )
        self.metrics["reference_error_m"][:] = torch.linalg.vector_norm(
            self.robot.data.root_link_pos_w - self.sample("qpos")[:, :3], dim=-1
        )
        self.metrics["failed_grips"][:] = self.failed_grips
