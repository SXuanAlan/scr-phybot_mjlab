"""Contact readings of the pad proxies against the climbing surface."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import get_sphere_geom_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PadContactReader:
  """Reads the pad proxies' contact sensor in EE-major proxy order.

  The contact sensor expands its primary pattern in model order (kinematic
  tree: feet before hands), not the configured EE order; the row
  permutation is resolved by name on first read. Without it the hand
  blocks would read the foot proxies.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    mcfg: MicrospineCfg,
    proxy_inward_normal: Callable[[], torch.Tensor],
  ):
    self._env = env
    self._sensor_name = sensor_name
    self._mcfg = mcfg
    self._proxy_inward_normal = proxy_inward_normal
    self._num_ee = len(mcfg.ee_body_names)
    self._proxies_per_ee = mcfg.spheres_per_ee_rows * mcfg.spheres_per_ee_cols
    self._row_perm: torch.Tensor | None = None

  def _sensor(self):
    return self._env.scene[self._sensor_name]

  def row_permutation(self, sensor) -> torch.Tensor:
    """Map the sensor's actual row order onto the EE-major proxy order."""
    if self._row_perm is None:
      names = getattr(sensor, "primary_names", None)
      if names is None:
        # Older ContactSensor API: row order lives in the slot list — one row
        # per unique primary, in slot order (num_slots=1 for this sensor).
        names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
      actual = [name.split("/")[-1] for name in names]
      expected = list(get_sphere_geom_names(self._mcfg))
      try:
        perm = [actual.index(name) for name in expected]
      except ValueError as exc:
        raise ValueError(
          f"Contact sensor '{self._sensor_name}' rows {actual} cannot be mapped onto the expected proxy order "
          f"{expected}."
        ) from exc
      self._row_perm = torch.tensor(perm, dtype=torch.long, device=self._env.device)
    return self._row_perm

  def proxy_normal_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-proxy wall-normal compression (E, H*S) and per-pad touching proxies (E, H)."""
    sensor = self._sensor()
    data = sensor.data
    if data.found is None or data.force is None:
      raise ValueError(f"Contact sensor '{self._sensor_name}' must provide 'found' and 'force' fields.")
    sensor_cfg = getattr(sensor, "cfg", None)
    world_force = getattr(sensor_cfg, "reduce", None) == "netforce" or getattr(sensor_cfg, "global_frame", False)

    perm = self.row_permutation(sensor)
    found = data.found[:, perm]
    found_mask = found > 0
    engage_force = data.force[:, perm]
    inward = self._proxy_inward_normal() if world_force else None
    if data.force_history is not None:
      # Average force history so one compression impulse cannot latch a spine.
      force_history = data.force_history[:, perm]
      engage_force = force_history.mean(dim=2)
      # MuJoCo contacts can be created and resolved within a decimation
      # window; the instantaneous ``found`` is then zero although the history
      # holds the compression. The history spans one policy step, so treating
      # any historical normal force as contact cannot leak across steps.
      if world_force:
        normal_history = (force_history * inward.unsqueeze(2)).sum(-1)
      else:
        normal_history = force_history[..., 0]
      found_mask |= (normal_history > 0.0).any(dim=2)

    self._validate_shapes(found, engage_force)

    # Netforce sums every box-wall contact and returns a world-frame vector
    # directed primary -> secondary; other sensors use contact coordinates.
    normal_component = (engage_force * inward).sum(-1) if world_force else engage_force[..., 0]
    normal_force = normal_component.clamp(min=0.0)
    normal_force = torch.where(found_mask, normal_force, torch.zeros_like(normal_force))

    num_envs = self._env.num_envs
    found_count = torch.zeros(num_envs, self._num_ee, device=self._env.device)
    for h in range(self._num_ee):
      s_start = h * self._proxies_per_ee
      s_end = s_start + self._proxies_per_ee
      found_count[:, h] = found_mask[:, s_start:s_end].sum(dim=1)
    return normal_force, found_count

  def pad_wrench(self, engaged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad-on-wall force and moment about each contact-array centre C, (E, H, 3) each.

    Netforce reports force and torque about each contact centroid P; every
    sample is shifted to C (M_C = torque_P + (P - C) x force) and averaged
    over the policy step. Latched pads read zero when their proxies are
    retracted, so stale pre-latch history is never reported.
    """
    sensor = self._sensor()
    data = sensor.data
    if sensor.cfg.reduce != "netforce" or any(
      getattr(data, field, None) is None for field in ("force", "torque", "pos", "reference_pos")
    ):
      raise ValueError("Unit preload engagement needs netforce contact force, torque, pos and reference_pos")
    values = [data.force, data.torque, data.pos, data.reference_pos]
    if data.force_history is not None:
      values = [
        data.force_history,
        data.torque_history,
        getattr(data, "pos_history", None),
        getattr(data, "reference_pos_history", None),
      ]
      if any(v is None or v.shape != values[0].shape for v in values):
        raise ValueError("Unit preload engagement needs synchronized force/torque/pos/reference_pos histories")
    else:
      values = [v.unsqueeze(2) for v in values]
    perm = self.row_permutation(sensor)
    num_envs = self._env.num_envs
    force, torque, position, center = (
      v[:, perm].reshape(num_envs, self._num_ee, self._proxies_per_ee, -1, 3) for v in values
    )
    moment = torque + torch.linalg.cross(position - center, force, dim=-1)
    force = force.sum(dim=2).mean(dim=2)
    moment = moment.sum(dim=2).mean(dim=2)
    if self._mcfg.disable_contacts_while_latched:
      force = torch.where(engaged[..., None], 0.0, force)
      moment = torch.where(engaged[..., None], 0.0, moment)
    return force, moment

  def _validate_shapes(self, found: torch.Tensor, force: torch.Tensor) -> None:
    # Sensor rows must match get_sphere_geom_names(): EE-major, then rows, then
    # columns. Otherwise force aggregation by EE would be wrong.
    expected = (self._env.num_envs, self._num_ee * self._proxies_per_ee)
    if found.shape != expected:
      raise ValueError(
        f"Contact sensor '{self._sensor_name}' found shape {tuple(found.shape)} does not match expected {expected}."
      )
    if force.shape != (*expected, 3):
      raise ValueError(
        f"Contact sensor '{self._sensor_name}' force shape {tuple(force.shape)} does not match expected "
        f"{(*expected, 3)}."
      )
