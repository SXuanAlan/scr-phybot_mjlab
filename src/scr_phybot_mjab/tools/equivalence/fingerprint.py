"""Fingerprint of the compiled MuJoCo model a task simulates."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

_SIZE_FIELDS = (
  "nq",
  "nv",
  "nu",
  "na",
  "nbody",
  "njnt",
  "ngeom",
  "nsite",
  "nsensor",
  "neq",
  "npair",
  "nexclude",
  "nmesh",
  "nmeshvert",
  "nkey",
  "ntendon",
  "nmocap",
)


def model_fingerprint(env_cfg: Any) -> tuple[dict, str | None]:
  """Compile the scene the way ``Simulation`` does and hash the result.

  Returns the fingerprint and the scene XML, which is only for reading a diff
  when the hashes differ.
  """
  from mjlab.scene import Scene

  scene = Scene(env_cfg.scene, device="cpu")
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  fingerprint = {
    "mjb_sha256": _mjb_digest(model),
    "sizes": {name: int(getattr(model, name)) for name in _SIZE_FIELDS},
    "options": _struct_text(model.opt),
    "arrays": _array_digests(model),
    "variants": len(scene.collect_variant_info() or ()),
  }
  return fingerprint, _scene_xml(scene)


def changed_parts(base: dict, new: dict) -> list[str]:
  """Human-readable list of what differs between two fingerprints."""
  changes = [f"size {k}: {v} -> {new['sizes'].get(k)}" for k, v in base["sizes"].items() if new["sizes"].get(k) != v]
  changes += [
    f"option {k}: {v} -> {new['options'].get(k)}" for k, v in base["options"].items() if new["options"].get(k) != v
  ]
  changes += [f"array {k}" for k, v in base["arrays"].items() if new["arrays"].get(k) != v]
  return changes


def _mjb_digest(model: Any) -> str:
  import mujoco

  buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
  mujoco.mj_saveModel(model, None, buffer)
  return hashlib.sha256(buffer.tobytes()).hexdigest()


def _array_digests(model: Any) -> dict[str, str]:
  digests = {}
  for name in sorted(dir(model)):
    if name.startswith("_"):
      continue
    value = getattr(model, name, None)
    if isinstance(value, np.ndarray):
      digests[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]
  return digests


def _struct_text(struct: Any) -> dict[str, str]:
  fields = {}
  for name in sorted(dir(struct)):
    if name.startswith("_"):
      continue
    value = getattr(struct, name, None)
    if isinstance(value, np.ndarray):
      fields[name] = repr(value.tolist())
    elif isinstance(value, (int, float)):
      fields[name] = repr(value)
  return fields


def _scene_xml(scene: Any) -> str | None:
  try:
    return scene.spec.to_xml()
  except Exception:  # noqa: BLE001  (XML export is a debugging aid only)
    return None
