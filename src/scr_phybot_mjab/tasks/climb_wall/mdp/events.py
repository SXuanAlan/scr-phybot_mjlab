"""Event terms for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def random_microspine_release(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "microspine_affordance",
  only_if_num_engaged_ge: int = 3,
  max_release_per_env: int = 1,
) -> None:
  """Force-release a random latched EE in the selected envs (interval DR).

  Teaches unexpected-release recovery without an engaged-state observation:
  the policy feels the sag and re-presses the scheduled limb. The engaged
  guard is a correctness requirement, not a safety margin: the
  ``unsafe_support`` penalty is release-reason blind with
  ``minimum_supports=2``, so a release must leave >= 2 latches or the DR
  event itself bills the policy for a fault it did not commit.
  """
  if only_if_num_engaged_ge - max_release_per_env < 2:
    raise ValueError(
      "random_microspine_release must leave >= 2 latched EEs: got guard "
      f"{only_if_num_engaged_ge} - releases {max_release_per_env} < 2."
    )
  affordance = env.command_manager.get_term(command_name)
  engaged = affordance.engaged[env_ids]
  eligible = engaged.sum(dim=-1) >= only_if_num_engaged_ge
  if not bool(eligible.any()):
    return
  mask = torch.zeros_like(engaged)
  rows = eligible.nonzero(as_tuple=True)[0]
  choice = torch.multinomial(engaged[rows].float(), num_samples=max_release_per_env)
  mask[rows.unsqueeze(1), choice] = True
  affordance.force_release(env_ids, mask)
