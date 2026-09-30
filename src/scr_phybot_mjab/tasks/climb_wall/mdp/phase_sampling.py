"""Failure-weighted phase sampling, following BeyondMimic's EMA histogram.

The uniform ratio is an additive probability floor before normalization,
not a fixed mixture percentage. Failures are attributed to the death phase;
sampling stays uniform within each bin and reaches every loop frame.
"""

from __future__ import annotations

import math

import torch


class AdaptivePhaseSampler:
  def __init__(self, frames: int, fps: float, device, *, bin_time_s=1.0, alpha=0.001, uniform_ratio=0.1):
    if frames < 1 or fps <= 0 or bin_time_s <= 0:
      raise ValueError("Adaptive sampling requires positive frame and time scales")
    if not 0 < alpha <= 1 or uniform_ratio <= 0:
      raise ValueError("Adaptive sampling needs 0 < alpha <= 1 and a positive uniform floor")
    self.frames = frames
    self.bin_count = min(frames, max(1, math.ceil(frames / fps / bin_time_s)))
    self.alpha = alpha
    self.uniform_ratio = uniform_ratio
    self.failed = torch.zeros(self.bin_count, device=device)
    self.pending = torch.zeros_like(self.failed)
    # Frame -> bin uses floor(frame * bins / frames), so inverse boundaries
    # use ceil. Every bin is nonempty, including a partial final second.
    edges = torch.arange(self.bin_count + 1, device=device, dtype=torch.long)
    self.edges = (edges * frames + self.bin_count - 1) // self.bin_count

  def record(self, frames: torch.Tensor) -> None:
    bins = frames * self.bin_count // self.frames
    self.pending += torch.bincount(bins, minlength=self.bin_count).to(self.failed.dtype)

  def update(self) -> None:
    self.failed.mul_(1.0 - self.alpha).add_(self.pending, alpha=self.alpha)
    self.pending.zero_()

  def probabilities(self) -> torch.Tensor:
    probabilities = self.failed + self.uniform_ratio / self.bin_count
    return probabilities / probabilities.sum()

  def sample(self, count: int) -> torch.Tensor:
    bins = torch.multinomial(self.probabilities(), count, replacement=True)
    starts = self.edges[bins]
    lengths = self.edges[bins + 1] - starts
    offset = (torch.rand(count, device=self.failed.device) * lengths).long()
    return starts + offset
