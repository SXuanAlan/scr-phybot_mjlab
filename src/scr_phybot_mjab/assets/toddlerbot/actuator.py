"""Conservative symmetric envelope from the pinned ToddlerBot servo fit.

The same bound applies while driving or braking. Thermal derating, controller
latency and the upstream friction compensation are not modeled here.
"""

from dataclasses import dataclass

import torch
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg


@dataclass(kw_only=True)
class ToddlerServoCfg(IdealPdActuatorCfg):
    corner_speed: float
    max_speed: float
    torque_at_max_speed: float

    def build(self, entity, target_ids, target_names):
        return ToddlerServo(self, entity, target_ids, target_names)


class ToddlerServo(IdealPdActuator):
    def compute(self, cmd):
        self.speed = cmd.vel.abs()
        return super().compute(cmd)

    def _clip_effort(self, effort):
        fraction = ((self.speed - self.cfg.corner_speed) / (self.cfg.max_speed - self.cfg.corner_speed)).clamp(0, 1)
        limit = self.force_limit - fraction * (self.force_limit - self.cfg.torque_at_max_speed)
        limit = torch.where(self.speed <= self.cfg.max_speed, limit, 0.0)
        return torch.clamp(effort, -limit, limit)
