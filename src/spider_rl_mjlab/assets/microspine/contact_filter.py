"""Anisotropic contact friction for microspine adhesion.

Modifies hand-wall contact parameters inside MuJoCo Warp's collision pipeline
via the contactfilter callback. Runs at every physics substep on GPU.

The kernel rotates the contact frame so tangent1 aligns with wall-up (world +z),
then sets high friction along tangent1 (resist downward sliding) and low friction
along tangent2 (lateral). This creates true anisotropic friction resolved by the
constraint solver with zero delay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

# Import vec5 type from mujoco_warp (5-element float vector for friction).
from mujoco_warp._src.types import vec5f as vec5

if TYPE_CHECKING:
    from mujoco_warp._src.types import Data, Model


@wp.kernel
def _microspine_contact_filter(
    # Contact arrays (from d.contact)
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_friction: wp.array(dtype=vec5),
    contact_frame: wp.array(dtype=wp.mat33),
    contact_dim: wp.array(dtype=int),
    contact_worldid: wp.array(dtype=int),
    nacon: wp.array(dtype=int),
    # Microspine state
    engaged: wp.array2d(dtype=int),
    geom_to_ee: wp.array(dtype=int),
    wall_geom_id: int,
    engaged_friction: float,
    lateral_friction: float,
):
    """Per-contact kernel: modify friction for hand-wall contacts.

    One thread per contact slot. Threads beyond nacon[0] early-exit.
    """
    i = wp.tid()

    # Skip inactive contact slots.
    if i >= nacon[0]:
        return

    g0 = contact_geom[i][0]
    g1 = contact_geom[i][1]
    worldid = contact_worldid[i]

    # O(1) lookup via precomputed geom_to_ee table.
    ee_idx = int(-1)
    if geom_to_ee[g0] >= 0 and g1 == wall_geom_id:
        ee_idx = geom_to_ee[g0]
    elif geom_to_ee[g1] >= 0 and g0 == wall_geom_id:
        ee_idx = geom_to_ee[g1]

    if ee_idx < 0:
        return  # Not a hand-wall contact.

    if engaged[worldid, ee_idx] == 0:
        # Not engaged: make frictionless so hand slides freely.
        contact_dim[i] = 1
        return

    # --- Engaged: apply anisotropic friction ---

    # Read current contact normal (row 0 of frame matrix).
    frame = contact_frame[i]
    normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])

    # Project world +z onto contact tangent plane to get wall-up direction.
    world_up = wp.vec3(0.0, 0.0, 1.0)
    dot_nz = wp.dot(normal, world_up)

    # Guard against degenerate case where normal ≈ world_up (horizontal surface).
    tangent1_raw = world_up - dot_nz * normal
    len_t1 = wp.length(tangent1_raw)
    if len_t1 < 1.0e-6:
        # Normal is nearly vertical — fall back to existing tangent directions.
        return

    tangent1 = tangent1_raw / len_t1
    tangent2 = wp.cross(normal, tangent1)

    # Write rotated frame: row0=normal, row1=tangent1(wall-up), row2=tangent2(lateral).
    contact_frame[i] = wp.mat33(
        normal[0], normal[1], normal[2],
        tangent1[0], tangent1[1], tangent1[2],
        tangent2[0], tangent2[1], tangent2[2],
    )

    # Set anisotropic friction coefficients.
    #   [0] tangent1 (wall-up) = HIGH friction — resists downward sliding
    #   [1] tangent2 (lateral) = LOW friction — allows lateral motion
    #   [2] spin, [3] roll1, [4] roll2 = minimal
    contact_friction[i] = vec5(
        engaged_friction,
        lateral_friction,
        0.001,
        0.001,
        0.001,
    )

    # Ensure condim=3 (normal + 2 tangent friction directions).
    contact_dim[i] = 3


class MicrospineContactFilter:
    """Manages the contactfilter callback and its shared state.

    Created by MicrospineController.__init__, registered on
    ``sim.wp_model.callback.contactfilter``.

    The engagement state is a warp array updated by the controller (PyTorch)
    via zero-copy ``wp.to_torch()`` bridging. The kernel reads this state
    at every physics substep to decide per-contact friction.
    """

    def __init__(
        self,
        geom_to_ee: list[int],
        wall_geom_id: int,
        num_envs: int,
        n_ee: int,
        engaged_friction: float,
        lateral_friction: float,
        device: str,
    ):
        """Initialize the contact filter.

        Args:
            geom_to_ee: List of length ngeom. geom_to_ee[geom_id] = ee_index
                        for hand geoms, -1 for all other geoms.
            wall_geom_id: MuJoCo geom ID of the wall collision geom.
            num_envs: Number of parallel environments (nworld).
            n_ee: Number of end effectors.
            engaged_friction: High friction coefficient for wall-up direction.
            lateral_friction: Low friction coefficient for lateral direction.
            device: Torch device string (e.g. "cuda:0").
        """
        self._wall_geom_id = wall_geom_id
        self._engaged_friction = engaged_friction
        self._lateral_friction = lateral_friction

        wp_device = wp.get_device(device)

        with wp.ScopedDevice(wp_device):
            self._geom_to_ee_wp = wp.array(geom_to_ee, dtype=int)
            self._engaged_wp = wp.zeros((num_envs, n_ee), dtype=int)

        # Zero-copy torch view for the controller to write engagement state.
        self._engaged_torch = wp.to_torch(self._engaged_wp)

    def update_engagement(self, engaged: torch.Tensor) -> None:
        """Sync engagement state from controller (PyTorch) to warp array.

        Args:
            engaged: (num_envs, n_ee) boolean or int tensor.
                     True/1 = spines engaged, False/0 = not engaged.
        """
        self._engaged_torch.copy_(engaged.int())

    def __call__(self, m: "Model", d: "Data") -> None:
        """contactfilter callback — launched by MuJoCo Warp at every substep."""
        wp.launch(
            _microspine_contact_filter,
            dim=(d.naconmax,),
            inputs=[
                d.contact.geom,
                d.contact.friction,
                d.contact.frame,
                d.contact.dim,
                d.contact.worldid,
                d.nacon,
                self._engaged_wp,
                self._geom_to_ee_wp,
                self._wall_geom_id,
                self._engaged_friction,
                self._lateral_friction,
            ],
        )
