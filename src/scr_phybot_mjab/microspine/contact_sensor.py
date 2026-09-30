"""Net contact wrenches with synchronized substep application positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from mjlab.sensor.contact_sensor import ContactData, ContactSensor, ContactSensorCfg


@dataclass
class WrenchContactData(ContactData):
  reference_pos: torch.Tensor | None = None
  """World wrench reference (pad centre C) for each primary geom."""
  pos_history: torch.Tensor | None = None
  """World contact centroids at the same substeps as force/torque_history."""
  reference_pos_history: torch.Tensor | None = None
  """Reference positions at those same substeps, including pad motion."""


@dataclass
class WrenchContactSensorCfg(ContactSensorCfg):
  """Preserve the application point of every netforce history sample."""

  fields: tuple[str, ...] = ("found", "force", "torque", "pos")
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "netforce"
  reference_sites: dict[str, str] = field(default_factory=dict)
  """Name from ContactSensor.primary_names -> scoped wrench-reference site."""

  def build(self) -> WrenchContactSensor:
    return WrenchContactSensor(self)


class WrenchContactSensor(ContactSensor):
  """MuJoCo netforce torque is about its force-weighted contact centroid.

  mjlab stores force and torque histories, but no position history. Pairing
  an old force with the current centroid invents a bending moment (especially
  when the live contact has disappeared). Keep the three fields synchronized
  so consumers can shift each wrench before averaging over a policy step.
  """

  cfg: WrenchContactSensorCfg

  def __init__(self, cfg: WrenchContactSensorCfg):
    if cfg.reduce != "netforce" or cfg.num_slots != 1 or not {"found", "force", "torque", "pos"}.issubset(cfg.fields):
      raise ValueError("WrenchContactSensor requires netforce, one slot and found/force/torque/pos")
    super().__init__(cfg)

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    self._reference_site_ids = torch.tensor(
      [mj_model.site(self.cfg.reference_sites[name]).id for name in self.primary_names],
      dtype=torch.long,
      device=device,
    )
    if self._history_state is not None:
      # The base reset clears every entry, including both position histories.
      for name in ("pos", "reference_pos"):
        self._history_state[name] = torch.zeros_like(self._history_state["force"])

  def _extract_sensor_data(self) -> WrenchContactData:
    data = super()._extract_sensor_data()
    assert self._data is not None
    return WrenchContactData(**vars(data), reference_pos=self._data.site_xpos[:, self._reference_site_ids])

  def _update_history(self) -> None:
    assert self._history_state is not None
    data = self._extract_sensor_data()
    for field, history in self._history_state.items():
      history = history.roll(1, dims=2)
      history[:, :, 0] = getattr(data, field)
      self._history_state[field] = history

  def _compute_data(self) -> WrenchContactData:
    data = super()._compute_data()
    assert isinstance(data, WrenchContactData)
    if self._history_state is not None:
      data.pos_history = self._history_state["pos"]
      data.reference_pos_history = self._history_state["reference_pos"]
    return data
