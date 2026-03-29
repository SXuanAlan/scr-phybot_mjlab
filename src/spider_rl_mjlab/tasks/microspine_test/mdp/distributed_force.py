"""Distributed surface pressure event.

Applies a uniform pressure across sub-geom contact surface as an equivalent
body wrench (force + torque at body COM).

MuJoCo only supports body-level external forces. To simulate uniform pressure
on a contact surface, we:
  1. Divide total force equally among N sub-geom positions
  2. Compute the equivalent torque about the body COM from the distributed loads
  3. Apply the combined wrench to the body
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def apply_distributed_pressure(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    body_name: str,
    geom_prefix: str,
    total_force: tuple[float, float, float] = (10.0, 0.0, 0.0),
    duration: float = 1e9,
    total_force_after: tuple[float, float, float] = (10.0, 0.0, 0.0),
) -> None:
    """Apply uniform pressure across sub-geom positions as a body wrench.

    Args:
      env: The RL environment.
      env_ids: Environment indices (unused for step events).
      asset_cfg: Entity containing the body.
      body_name: Name of the body to apply wrench to.
      geom_prefix: Prefix to match sub-geom names (e.g. "right_hand_c_").
      total_force: Total force vector [N] during phase 1.
      duration: Duration of phase 1 [s].
      total_force_after: Total force vector [N] during phase 2.
    """
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    mj_model = env.sim.mj_model
    entity_name = asset_cfg.name

    # Cache sub-geom local positions (relative to body frame) on first call.
    cache_key = f"_dist_pressure_{body_name}_{geom_prefix}"
    if not hasattr(env, cache_key):
        body_id = mj_model.body(f"{entity_name}/{body_name}").id

        geom_positions = []
        for i in range(mj_model.ngeom):
            geom = mj_model.geom(i)
            if geom_prefix in geom.name and geom.bodyid == body_id:
                # Geom pos is relative to body frame.
                geom_positions.append(geom.pos.copy())

        if not geom_positions:
            raise ValueError(
                f"No geoms matching prefix '{geom_prefix}' found on body '{body_name}'"
            )

        import numpy as np
        geom_pos_array = np.stack(geom_positions, axis=0)  # (N_geoms, 3)
        setattr(env, cache_key, {
            "geom_pos": torch.tensor(geom_pos_array, device=env.device, dtype=torch.float32),
            "n_geoms": len(geom_positions),
        })

    cache = getattr(env, cache_key)
    geom_pos = cache["geom_pos"]  # (N_geoms, 3) in body local frame
    n_geoms = cache["n_geoms"]

    # Select phase.
    elapsed = env.episode_length_buf.float() * env.step_dt
    in_phase1 = (elapsed < duration).view(num_envs, 1)

    f1 = torch.tensor(total_force, device=env.device, dtype=torch.float32)
    f2 = torch.tensor(total_force_after, device=env.device, dtype=torch.float32)
    total_f = torch.where(in_phase1, f1.expand(num_envs, 3), f2.expand(num_envs, 3))

    # Force per sub-geom.
    f_per_geom = total_f / n_geoms  # (num_envs, 3)

    # Total force on body = sum of per-geom forces = total_f (unchanged).
    # Total torque = sum of (r_i × f_per_geom) where r_i is geom pos in body frame.
    # Since f_per_geom is the same for all geoms:
    #   torque = (sum of r_i) × f_per_geom = n_geoms * r_centroid × f_per_geom
    r_centroid = geom_pos.mean(dim=0)  # (3,) centroid of sub-geoms in body frame

    # Cross product: torque = r_centroid × total_f  (per env)
    # r_centroid is the same for all envs.
    torque = torch.cross(
        r_centroid.unsqueeze(0).expand(num_envs, 3),
        total_f,
        dim=-1,
    )  # (num_envs, 3)

    # Write wrench to body.
    body_ids, _ = asset.find_bodies([body_name])
    forces = total_f.unsqueeze(1)   # (num_envs, 1, 3)
    torques = torque.unsqueeze(1)   # (num_envs, 1, 3)
    asset.write_external_wrench_to_sim(forces, torques, body_ids=body_ids)
