"""Custom observation functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def microspine_privileged_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged observation from the microspine controller.

    Returns per-EE state for RMA training:
      [engaged_ratio, shear_capacity, normal_capacity, is_in_contact, surface_quality]
    repeated for each of the 4 end-effectors (20-dim total).
    """
    controller = env.event_manager.get_term_cfg("microspine_adhesion").func
    return controller.get_privileged_obs()
