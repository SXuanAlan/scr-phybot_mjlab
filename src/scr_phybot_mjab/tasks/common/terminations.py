"""Termination terms shared by the PHYBot C2 walk and squat tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ("illegal_contact",)


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """True where a sensed body touches with more than ``force_threshold`` newtons.

  Checks every substep of the force history when the sensor keeps one, and
  any contact at all otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)
