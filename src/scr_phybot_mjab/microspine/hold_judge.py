"""Hold judgement of a latched pad against the empirical unit model."""

from __future__ import annotations

import dataclasses

import torch

from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface
from scr_phybot_mjab.microspine.pad_geometry import PadGeometry
from scr_phybot_mjab.microspine.release_reason import ReleaseReason
from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units


@dataclasses.dataclass(frozen=True)
class UnitVerdict:
  """The hold judgement of every pad, (E, H) unless noted."""

  margin: torch.Tensor
  """The worst unit's margin; negative means the pad releases."""
  violation: torch.Tensor
  reason: torch.Tensor
  """ReleaseReason of the worst unit (int8)."""
  worst_load_pad_scale: torch.Tensor
  """The worst unit's load times the unit count, (E, H, 3)."""
  unit_load_tln: torch.Tensor
  """Every unit's load in the engage frame, (E, H, U, 3)."""
  unit_normal_capacity: torch.Tensor
  unit_supported: torch.Tensor
  unit_early_failed: torch.Tensor
  unit_tail_capped: torch.Tensor


class HoldJudge:
  """Distributes a pad wrench onto its active units and judges each unit.

  The pad wrench about the contact-array centre is shared by the units with
  equal-stiffness bolt-group statics; each unit's load is judged against its
  persistent empirical draws, and the weakest unit releases the pad.
  """

  def __init__(self, geometry: PadGeometry, table_path, device: torch.device | str):
    self.surface = EmpiricalLimitSurface(table_path, device)
    self.unit_x = torch.tensor(geometry.x.copy(), dtype=torch.float32, device=device)
    self.unit_y = torch.tensor(geometry.y.copy(), dtype=torch.float32, device=device)
    self._sxx = float((self.unit_x**2).sum())
    self._syy = float((self.unit_y**2).sum())
    self._srr = self._sxx + self._syy

  @property
  def unit_count(self) -> int:
    return self.unit_x.numel()

  def distribute(self, f_t, f_l, f_n, m_t, m_l, m_n) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-unit (shear, lateral, normal) loads, each (E, H, U)."""
    return distribute_wrench_to_units(
      f_t, f_l, f_n, m_t, m_l, m_n, self.unit_x, self.unit_y, self._sxx, self._syy, self._srr
    )

  def judge(self, f_t, f_l, f_n, m_t, m_l, m_n, strength_q, loading_q) -> UnitVerdict:
    a, b, c = self.distribute(f_t, f_l, f_n, m_t, m_l, m_n)
    unit_load_tln = torch.stack((a, b, c), dim=-1)
    margins, capacity, domain, early, tail = self.surface.margins(a, b, c, strength_q, loading_q)
    margin, worst = margins.min(-1)
    idx = worst.unsqueeze(-1)
    worst_load = unit_load_tln.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, 3)).squeeze(-2)
    reason = torch.full(margin.shape, int(ReleaseReason.NORMAL), dtype=torch.int8, device=margin.device)
    reason = torch.where(early.gather(-1, idx).squeeze(-1), int(ReleaseReason.EMPIRICAL_EARLY), reason)
    # Domain rejection is explicitly distinct from a measured strength failure.
    reason = torch.where(~domain.gather(-1, idx).squeeze(-1), int(ReleaseReason.EMPIRICAL_DOMAIN), reason)
    return UnitVerdict(
      margin=margin,
      violation=margin < 0,
      reason=reason,
      worst_load_pad_scale=worst_load * self.unit_count,
      unit_load_tln=unit_load_tln,
      unit_normal_capacity=capacity,
      unit_supported=domain,
      unit_early_failed=early,
      unit_tail_capped=tail,
    )

  def force_only_margin(self, f_t, f_l, f_n, strength_q, loading_q) -> torch.Tensor:
    """Worst-unit margin with moments excluded (an even split), same draws."""
    count = self.unit_count
    return self.surface.margins(
      f_t.unsqueeze(-1).expand_as(strength_q) / count,
      f_l.unsqueeze(-1).expand_as(strength_q) / count,
      f_n.unsqueeze(-1).expand_as(strength_q) / count,
      strength_q,
      loading_q,
    )[0].amin(-1)
