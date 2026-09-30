"""Shared native model and coordinate helpers for offline reference audits."""
from __future__ import annotations

import mujoco
import numpy as np
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec, PHYBOT_C2_FOOT_TOES
from scr_phybot_mjab.microspine.contact import ToeContact, add_toe_geoms, toe_site_name

WALL_OUTWARD_WORLD = np.array([-1., 0., 0.], dtype=np.float64)
RAKE_LOCAL = np.array([0., 0., -1.], dtype=np.float64)

class WallGeometry:
  """Tilted-wall geometry for clips carrying a ``wall_angle`` key.

  Frame convention: clips are stored CANONICAL (vertical-wall frame); a
  native angle a means the physical scene was the wall rotated by
  Rot_y(90 - a) about its base line (x = wall_surface_x, z = 0), gravity
  physically down. The audit runs in that NATIVE world — the clip's root
  pose is rotated back before mj_forward, the wall plane is tilted, and the
  floor branch needs no changes at all (the floor is horizontal in every
  native world). ``identity`` (a == 90, or no key) keeps every original
  expression verbatim so legacy audits stay bitwise identical.
  """

  def __init__(self, angle_deg: float, wall_surface_x: float) -> None:
    self.angle_deg = float(angle_deg)
    self.identity = self.angle_deg == 90.0
    self.pivot = np.array([wall_surface_x, 0.0, 0.0], dtype=np.float64)
    if self.identity:
      self.R = np.eye(3)
      self.outward = WALL_OUTWARD_WORLD
      self.quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
      theta = np.radians(90.0 - self.angle_deg)
      c, s = float(np.cos(theta)), float(np.sin(theta))
      self.R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
      self.outward = self.R @ WALL_OUTWARD_WORLD
      half = 0.5 * theta
      self.quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])

  def rotate_point(self, p: np.ndarray) -> np.ndarray:
    return self.pivot + self.R @ (np.asarray(p, dtype=np.float64) - self.pivot)

  def distance(self, p: np.ndarray) -> float:
    """Signed clearance to the tilted surface (positive on the robot side)."""
    return float(
      self.outward @ (np.asarray(p, dtype=np.float64) - self.pivot)
    )


def joint_group(name: str) -> str:
  return name.removeprefix("left_").removeprefix("right_")


def build_reference_model(
  ee_names: tuple[str, ...],
) -> tuple[
  mujoco.MjModel,
  dict[str, int],
  dict[str, int],
  dict[str, tuple[int, float]],
  dict[str, tuple[int, list[np.ndarray]]],
]:
  """Compile the robot with pad-center sites and the foot bracing-toe sites.

  Returns (model, pad_site_ids, ee_body_ids, toe_sites, ground_points) where ``toe_sites``
  maps a foot EE to its (toe site id, toe sphere radius).
  """
  spec = get_spec()
  pad_site_names: dict[str, str] = {}
  for ee in ee_names:
    site_name = f"{ee}_static_audit_center"
    spec.body(ee).add_site(
      name=site_name,
      pos=spec.site(f"{ee}_contact_center").pos,
      size=[0.005, 0.005, 0.005],
    )
    pad_site_names[ee] = site_name

  toes = {
    ee: ToeContact(ee_body_name=ee, body_name=body, pos=pos, radius=radius)
    for ee, body, pos, radius in PHYBOT_C2_FOOT_TOES
    if ee in ee_names
  }
  add_toe_geoms(
    spec, tuple(toes.values()), robot_prefix="", with_sites=True
  )

  model = spec.compile()
  pad_site_ids = {
    ee: mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_SITE, pad_site_names[ee]
    )
    for ee in ee_names
  }
  ee_body_ids = {
    ee: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee)
    for ee in ee_names
  }
  toe_sites = {
    ee: (
      mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, toe_site_name(toe)
      ),
      toe.radius,
    )
    for ee, toe in toes.items()
  }

  # Floor-support sole points (mount prologue frames): heel and toe of each
  # foot at sole level, from the union AABB of the foot body's COLLISION
  # geoms in body frame. Two points let a flat foot carry a pitch couple,
  # which a single mid-sole point cannot.
  corner_signs = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
  )
  ground_points: dict[str, tuple[int, list[np.ndarray]]] = {}
  for ee, toe in toes.items():
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, toe.body_name)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
      if model.geom_bodyid[g] != bid:
        continue
      if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
        continue
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, model.geom_quat[g])
      corners = model.geom_pos[g] + (
        model.geom_aabb[g, :3] + model.geom_aabb[g, 3:] * corner_signs
      ) @ rot.reshape(3, 3).T
      lo = np.minimum(lo, corners.min(axis=0))
      hi = np.maximum(hi, corners.max(axis=0))
    sole_z = float(lo[2]) + 0.002
    ground_points[ee] = (
      bid,
      [
        np.array([float(lo[0]) + 0.01, 0.0, sole_z]),
        np.array([float(hi[0]) - 0.01, 0.0, sole_z]),
      ],
    )
  return model, pad_site_ids, ee_body_ids, toe_sites, ground_points


def engage_basis(
  data: mujoco.MjData,
  body_id: int,
  n_hat: np.ndarray = WALL_OUTWARD_WORLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  raw = data.xmat[body_id].reshape(3, 3) @ RAKE_LOCAL
  tangent = raw - raw.dot(n_hat) * n_hat
  norm = np.linalg.norm(tangent)
  if norm < 1.0e-9:
    tangent = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tangent -= tangent.dot(n_hat) * n_hat
    norm = np.linalg.norm(tangent)
  t_hat = tangent / norm
  l_hat = np.cross(n_hat, t_hat)
  return t_hat, l_hat, n_hat

