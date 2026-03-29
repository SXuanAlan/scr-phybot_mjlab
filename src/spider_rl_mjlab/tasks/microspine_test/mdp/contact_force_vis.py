"""Debug visualization for contact forces.

A class-based step event that draws per-contact force arrows and a resultant
force arrow for contacts between two named geom groups.
Works independently of the MicrospineController.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class ContactForceVis:
    """Step-mode event that visualizes contact forces between two geom groups.

    Draws:
      - Blue arrows: per-contact force on the target body
      - Red arrow: resultant (summed) force

    Params:
      target_geom_patterns: list of geom name substrings to match (e.g. ["right_hand_c_"])
      wall_geom_name: full scoped geom name (e.g. "wall/wall_geom")
      force_scale: arrow length per Newton [m/N]
      ext_force_body_names: list of body names to show external forces for (optional)
      ema_alpha: EMA smoothing factor in [0,1]. 0=no update, 1=no smoothing. Default 0.3.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._num_envs = env.num_envs
        self._sim_data = env.sim.data
        self._device = env.device
        self._asset_name = cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name if "asset_cfg" in cfg.params else "robot"

        target_patterns: list[str] = cfg.params["target_geom_patterns"]
        wall_geom_name: str = cfg.params["wall_geom_name"]
        self._force_scale: float = cfg.params.get("force_scale", 0.003)
        self._ema_alpha: float = cfg.params.get("ema_alpha", 0.3)

        mj_model = env.sim.mj_model
        entity_name = self._asset_name

        # Resolve wall geom ID.
        self._wall_gid = mj_model.geom(wall_geom_name).id

        # Resolve target geom IDs (any geom whose name contains one of the patterns).
        self._target_gids = set()
        for i in range(mj_model.ngeom):
            name = mj_model.geom(i).name
            for pat in target_patterns:
                if pat in name:
                    self._target_gids.add(i)
                    break

        # Resolve external force body IDs for visualization.
        ext_body_names: list[str] = cfg.params.get("ext_force_body_names", [])
        self._ext_body_ids = []
        self._ext_body_labels = []
        asset = env.scene[entity_name]
        for bname in ext_body_names:
            ids, _ = asset.find_bodies([bname])
            self._ext_body_ids.append(ids[0])
            self._ext_body_labels.append(bname)
        self._asset = asset

        self._warp_cached = False

        # EMA state: per-geom filtered forces, keyed by geom id.
        self._ema_forces: dict[int, np.ndarray] = {}
        self._ema_resultant: np.ndarray = np.zeros(3)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        **kwargs,
    ) -> None:
        """No-op step function. Visualization happens in debug_vis."""
        pass

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear EMA state on episode reset."""
        self._ema_forces.clear()
        self._ema_resultant = np.zeros(3)

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draw per-contact and resultant force arrows."""
        import warp as wp

        env_indices = visualizer.get_env_indices(self._num_envs)
        if not env_indices:
            return

        if not self._warp_cached:
            raw = self._sim_data.struct
            self._c_geom = wp.to_torch(raw.contact.geom)
            self._c_wid = wp.to_torch(raw.contact.worldid)
            self._c_pos = wp.to_torch(raw.contact.pos)
            self._c_frame = wp.to_torch(raw.contact.frame)
            self._c_dim = wp.to_torch(raw.contact.dim)
            self._c_efc = wp.to_torch(raw.contact.efc_address)
            self._efc_force = wp.to_torch(raw.efc.force)
            self._nacon = wp.to_torch(raw.nacon)
            self._warp_cached = True

        nacon = int(self._nacon[0].item())
        if nacon <= 0:
            return

        wall_gid = self._wall_gid
        target_gids = self._target_gids
        scale = self._force_scale

        for batch in env_indices:
            resultant_force = np.zeros(3)
            resultant_pos = np.zeros(3)
            n_contacts = 0

            for ci in range(min(nacon, self._c_geom.shape[0])):
                wid = int(self._c_wid[ci].item())
                if wid != batch:
                    continue

                g0 = int(self._c_geom[ci, 0].item())
                g1 = int(self._c_geom[ci, 1].item())

                # Check if this is a target-wall contact.
                # Contact normal points from g0 → g1.
                # Force on g1 = +frame.T @ f_contact, force on g0 = -frame.T @ f_contact.
                if g0 == wall_gid and g1 in target_gids:
                    sign = 1.0   # target is g1: force along +normal
                elif g1 == wall_gid and g0 in target_gids:
                    sign = -1.0  # target is g0: force along -normal
                else:
                    continue

                frame = self._c_frame[ci].cpu().numpy().reshape(3, 3)
                condim = int(self._c_dim[ci].item())
                pos = self._c_pos[ci].cpu().numpy()

                f_contact = np.zeros(3)
                for k in range(min(condim, 3)):
                    addr = int(self._c_efc[ci, k].item())
                    if 0 <= addr < self._efc_force.shape[1]:
                        f_contact[k] = float(self._efc_force[wid, addr].item())

                f_world = sign * (frame.T @ f_contact)

                n_contacts += 1
                resultant_force += f_world
                resultant_pos += pos

                # EMA filter per-contact force.
                target_gid = g0 if g0 in target_gids else g1
                alpha = self._ema_alpha
                if target_gid in self._ema_forces:
                    self._ema_forces[target_gid] = alpha * f_world + (1 - alpha) * self._ema_forces[target_gid]
                else:
                    self._ema_forces[target_gid] = f_world.copy()

                f_smooth = self._ema_forces[target_gid]
                f_mag = np.linalg.norm(f_smooth)
                if f_mag < 1e-3:
                    continue

                # Per-contact arrow (blue).
                visualizer.add_arrow(
                    start=pos,
                    end=pos + f_smooth * scale,
                    color=(0.3, 0.5, 1.0, 0.7),
                    width=0.003,
                )

            # EMA filter resultant force.
            if n_contacts > 0:
                center = resultant_pos / n_contacts
                alpha = self._ema_alpha
                self._ema_resultant = alpha * resultant_force + (1 - alpha) * self._ema_resultant
                r_mag = np.linalg.norm(self._ema_resultant)
                if r_mag > 0.1:
                    visualizer.add_arrow(
                        start=center,
                        end=center + self._ema_resultant * scale,
                        color=(1.0, 0.1, 0.1, 1.0),
                        width=0.008,
                    )

            # External force arrows (green) on tracked bodies.
            # body_external_wrench: (num_envs, num_local_bodies, 6) = [fx,fy,fz,tx,ty,tz]
            ext_wrench = self._asset.data.body_external_wrench
            for body_id in self._ext_body_ids:
                body_pos = self._asset.data.body_link_pos_w[batch, body_id].cpu().numpy()
                ext_f = ext_wrench[batch, body_id, :3].cpu().numpy()
                ext_mag = np.linalg.norm(ext_f)
                if ext_mag > 0.1:
                    visualizer.add_arrow(
                        start=body_pos,
                        end=body_pos + ext_f * scale,
                        color=(0.1, 0.9, 0.2, 0.9),
                        width=0.006,
                    )
