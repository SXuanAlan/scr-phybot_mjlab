"""Per-frame tables of a regrip cycle's engage schedule.

The schedule is a loop: every table wraps around the cycle.
"""

from __future__ import annotations

import numpy as np

__all__ = ("edge_frame_countdowns", "run_lengths", "schedule_tables")


def edge_frame_countdowns(engaged: np.ndarray) -> np.ndarray:
  """Per-frame countdowns (in frames) to each EE's next schedule edges.

  For frame ``c`` and EE ``h``, entry ``[c, h, 0]`` counts the frames until
  the schedule next turns that EE OFF (its next release) and ``[c, h, 1]``
  until it next turns ON (its next regrip), both wrapping around the cycle.
  Together with the current engage bit these two edges reconstruct the EE's
  entire periodic schedule row — the whole raster chart ahead, not point
  samples of it. A row with no edge (an EE the clip never releases) reads
  the full cycle length: "a whole cycle away", forever.
  """
  cycle, num_ee = engaged.shape
  out = np.full((cycle, num_ee, 2), cycle, dtype=np.int64)
  frames = np.arange(cycle)
  for h in range(num_ee):
    now = engaged[:, h]
    nxt = np.roll(now, -1)
    for k, edge in enumerate((now & ~nxt, ~now & nxt)):  # falling, rising
      event_frames = (np.flatnonzero(edge) + 1) % cycle
      if event_frames.size == 0:
        continue
      dist = (event_frames[None, :] - frames[:, None]) % cycle
      dist[dist == 0] = cycle
      out[:, h, k] = dist.min(axis=1)
  return out


def run_lengths(engaged: np.ndarray) -> np.ndarray:
  """Per-frame length (frames) of the circular ON/OFF run containing it.

  For frame ``c`` and EE ``h`` this is the length of the maximal constant
  stretch of ``engaged[:, h]`` that frame ``c`` sits in, with wrap-around
  (the schedule is a loop). An EE the clip never flips reads the full cycle.
  Consumed by the window-proportional mismatch grace: a weld-through bill
  then costs a constant fraction of the violated window regardless of how
  the clip is timed.
  """
  cycle, num_ee = engaged.shape
  out = np.full((cycle, num_ee), cycle, dtype=np.int64)
  for h in range(num_ee):
    col = engaged[:, h]
    starts = np.flatnonzero(col != np.roll(col, 1))  # circular run starts
    if starts.size == 0:
      continue
    for k, s in enumerate(starts):
      e = starts[(k + 1) % starts.size]
      length = (e - s) % cycle
      if length == 0:
        length = cycle
      out[(np.arange(length) + s) % cycle, h] = length
  return out


def schedule_tables(engaged: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The schedule tables of one cycle's engage flags (C, H), normalized by the cycle.

  Returns the countdowns to each EE's next release and regrip (C, H, 2), the
  observation chart; the distance to the NEAREST edge, past or future
  (C, H), for the graded mismatch bill; and the length of the containing
  ON/OFF run (C, H), for the window-proportional grace. The backward pass is
  the same countdown on the time-reversed schedule, minus one because "0
  frames ago" reads as 1 in countdown form. EEs the clip never flips read
  1.0 everywhere: no edge is ever near, so their mismatch bills full price.
  """
  cycle = engaged.shape[0]
  future = edge_frame_countdowns(engaged)
  past = edge_frame_countdowns(engaged[::-1])[::-1]
  edge_distance = (np.minimum(future.min(axis=-1), past.min(axis=-1) - 1) / cycle).astype(np.float32)
  edge_distance[:, future.min(axis=(0, -1)) >= cycle] = 1.0
  run_time = (run_lengths(engaged) / cycle).astype(np.float32)
  return future.astype(np.float32) / cycle, edge_distance, run_time
